# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT
"""Early-TTIR and hardware regressions for the indirect-load gather rule."""

import os

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import pytest
import triton
import triton.language as tl
from triton._C.libtriton import ascend, ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.backends.ascend.compiler import NPUOptions, make_ttir
from triton.compiler.code_generator import ast_to_ttir
from triton.compiler.compiler import ASTSource

pytestmark = pytest.mark.backend("none")


@triton.jit
def indirect_rows_kernel(src_ptr, idx_ptr, out_ptr, n_rows, WIDTH: tl.constexpr, K: tl.constexpr, ROW_BLK: tl.constexpr,
                         ROW_STEP: tl.constexpr, PER_LANE: tl.constexpr, VOLATILE: tl.constexpr):
    for rb in range(0, ROW_BLK, ROW_STEP):
        rows = tl.program_id(0) * ROW_BLK + rb + tl.arange(0, ROW_STEP)
        cols = tl.arange(0, K)
        row_mask = rows < n_rows
        positions = rows[:, None] * K + cols[None, :]
        # Deliberately out of range: inactive indices must be sanitized before
        # both the bounds reductions and the actual gather.
        indices = tl.load(idx_ptr + positions, row_mask[:, None], other=-123)
        mask = row_mask[:, None]
        if PER_LANE:
            mask = mask & (cols[None, :] % 2 == 0)
        # Leave one physical row before the logical source so negative pointer
        # offsets are valid and have a different answer from wrapped indices.
        base = (rows + 1)[:, None] * WIDTH
        value = tl.load(src_ptr + base + indices, mask, other=-7.0, volatile=VOLATILE)
        # Output includes padding rows: observe `other` rather than hiding it
        # behind a masked store. The host allocates ROW_BLK rows per program.
        tl.store(out_ptr + positions, value)


def make_indirect_ttir(rule_mask, *, per_lane=False, volatile=False, index_type="i32", value_type="fp32"):
    options = NPUOptions(arch="Ascend910B1", graph_optimize_rule_mask=rule_mask)
    source = ASTSource(
        indirect_rows_kernel,
        {"src_ptr": f"*{value_type}", "idx_ptr": f"*{index_type}", "out_ptr": f"*{value_type}", "n_rows": "i32"},
        {"WIDTH": 16, "K": 8, "ROW_BLK": 8, "ROW_STEP": 2, "PER_LANE": per_lane, "VOLATILE": volatile},
    )
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    module = ast_to_ttir(indirect_rows_kernel, source, context, options, {}, {})
    return make_ttir(module, {}, options)


@pytest.mark.parametrize("rule_mask,per_lane,volatile,index_type,expected", [
    (0, False, False, "i32", False),
    (511, False, False, "i32", False),
    (512, False, False, "i32", True),
    (1023, False, False, "i32", True),
    (512, True, False, "i32", False),
    (512, False, True, "i32", False),
    (512, False, False, "i64", False),
])
def test_gather_rule_in_make_ttir(tmp_path, rule_mask, per_lane, volatile, index_type, expected):
    module = make_indirect_ttir(rule_mask, per_lane=per_lane, volatile=volatile, index_type=index_type)
    text = str(module)
    assert ("tt.gather" in text) == expected
    if expected:
        assert 'gather.optimised.load = "source"' in text
        assert 'gather.optimised.load = "fallback"' in text
        assert "arith.shrsi" not in text
    # The real early pipeline must emit valid, reparsable IR.
    path = tmp_path / "gather.ttir.mlir"
    path.write_text(text)
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    ir.parse_mlir_module(str(path), context)
    # A second pass must leave the transformed source/fallback alone.
    pm = ir.pass_manager(module.context)
    ascend.passes.ttir.add_graph_optimize(pm, rule_mask=512, ub_capacity_bytes=96 * 1024)
    pm.run(module, "")
    if expected:
        assert str(module).count("tt.gather") == text.count("tt.gather")
        assert str(module).count("scf.if") == text.count("scf.if")


def test_gather_rule_mask_changes_cache_key():
    assert NPUOptions(graph_optimize_rule_mask=511).hash() != NPUOptions(graph_optimize_rule_mask=1023).hash()


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("case", ["tail", "negative", "upper_bound", "all_masked", "per_lane", "volatile", "i64"])
def test_gather_rule_npu_equivalence(monkeypatch, dtype, case):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu", exc_type=ImportError)
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    width, k, row_blk = 16, 8, 8
    n_rows = 0 if case == "all_masked" else 7
    # Only the upper-bound fallback case needs a legal extra source row.
    physical_rows = n_rows + (2 if case == "upper_bound" else 1)
    source = torch.arange(physical_rows * width, dtype=torch.float32).to(getattr(torch, dtype))
    index_dtype = torch.int64 if case == "i64" else torch.int32
    indices = torch.arange(max(n_rows, 1) * k, dtype=index_dtype).reshape(-1, k) % width
    if case == "negative":
        indices[0, 0] = -1
        indices[1, 1] = -width
    if case == "upper_bound":
        indices[-1, 0] = width
    reference = torch.full((row_blk, k), -7.0, dtype=getattr(torch, dtype))
    for row in range(n_rows):
        reference[row] = source[(row + 1) * width + indices[row].long()]
    if case == "per_lane":
        reference[:, 1::2] = -7.0
    # Keep stage markers visible with pytest -s: after a device exception,
    # subsequent cases may fail during input copies before launching a kernel.
    diagnostic = f"[gather] case={case} dtype={dtype}"
    print(f"{diagnostic} stage=copy-inputs", flush=True)
    source_npu, indices_npu = source.npu(), indices.npu()
    print(f"{diagnostic} stage=inputs-ready", flush=True)
    outputs = []
    for rule_mask in (511, 1023):
        indirect_rows_kernel.device_caches.clear()
        output = torch.empty((row_blk, k), dtype=getattr(torch, dtype), device="npu")
        print(f"{diagnostic} rule_mask={rule_mask} stage=compile-and-launch", flush=True)
        compiled = indirect_rows_kernel[(1, )](
            source_npu,
            indices_npu,
            output,
            n_rows,
            WIDTH=width,
            K=k,
            ROW_BLK=row_blk,
            ROW_STEP=2,
            PER_LANE=case == "per_lane",
            VOLATILE=case == "volatile",
            graph_optimize_rule_mask=rule_mask,
        )
        print(f"{diagnostic} rule_mask={rule_mask} stage=synchronize", flush=True)
        torch.npu.synchronize()
        expected_rewrite = rule_mask == 1023 and case not in ("per_lane", "volatile", "i64")
        assert ("tt.gather" in compiled.asm["ttir"]) == expected_rewrite
        outputs.append(output.cpu())
        torch.testing.assert_close(outputs[-1], reference, rtol=0, atol=0)
        print(f"{diagnostic} rule_mask={rule_mask} stage=verified", flush=True)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
