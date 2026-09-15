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
def indirect_rows_kernel(src_ptr, idx_ptr, out_ptr, readable_out_ptr, n_rows, WIDTH: tl.constexpr, K: tl.constexpr,
                         ROW_BLK: tl.constexpr, ROW_STEP: tl.constexpr, PER_LANE: tl.constexpr, VOLATILE: tl.constexpr,
                         READABLE: tl.constexpr, BASE_ROW: tl.constexpr):
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
        base = (rows + BASE_ROW)[:, None] * WIDTH
        # Keep the integer offset addition before the pointer addition. Nested
        # addptr operations with dynamic i32 offsets cannot generally be folded
        # together (the combined i32 sum could overflow), and are outside the
        # gather rule's direct-pointer matcher. Test addresses fit in i32.
        offsets = base + indices
        if READABLE:
            readable_cols = tl.arange(0, WIDTH)
            readable_offsets = base + readable_cols[None, :]
            readable = tl.load(src_ptr + readable_offsets, row_mask[:, None], other=0.0)
        value = tl.load(src_ptr + offsets, mask, other=-7.0, volatile=VOLATILE)
        if READABLE:
            tl.store(readable_out_ptr + rows[:, None] * WIDTH + readable_cols[None, :], readable)
        # Output includes padding rows: observe `other` rather than hiding it
        # behind a masked store. The host allocates ROW_BLK rows per program.
        tl.store(out_ptr + positions, value)


def make_indirect_ttir(rule_mask, *, per_lane=False, volatile=False, index_type="i32", value_type="fp32",
                       compile_mode="simd", arch="Ascend910B1", readable=True):
    options = NPUOptions(arch=arch, graph_optimize_rule_mask=rule_mask, compile_mode=compile_mode)
    source = ASTSource(
        indirect_rows_kernel,
        {
            "src_ptr": f"*{value_type}", "idx_ptr": f"*{index_type}", "out_ptr": f"*{value_type}", "readable_out_ptr":
            f"*{value_type}", "n_rows": "i32"
        },
        {
            "WIDTH": 16, "K": 8, "ROW_BLK": 8, "ROW_STEP": 2, "PER_LANE": per_lane, "VOLATILE": volatile, "READABLE":
            readable, "BASE_ROW": 1
        },
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
    ascend.passes.ttir.add_graph_optimize(pm, rule_mask=512, ub_capacity_bytes=96 * 1024, compile_mode="simd")
    pm.run(module, "")
    if expected:
        assert str(module).count("tt.gather") == text.count("tt.gather")
        assert str(module).count("scf.if") == text.count("scf.if")


def test_gather_rule_mask_changes_cache_key():
    assert NPUOptions(graph_optimize_rule_mask=511).hash() != NPUOptions(graph_optimize_rule_mask=1023).hash()


@pytest.mark.parametrize("rule_mask", [512, 1023])
def test_gather_rule_requires_readable_source_rows(rule_mask):
    text = str(make_indirect_ttir(rule_mask, readable=False))
    assert "tt.gather" not in text
    assert "gather.optimised.load" not in text


@pytest.mark.parametrize("rule_mask", [511, 512, 1023])
@pytest.mark.parametrize("arch,compile_mode", [
    ("Ascend910B1", "simd"),
    ("Ascend910B1", "simd_simt_template"),
    ("Ascend910B1", "unstructured_in_simt"),
    ("Ascend910_9589", "simd"),
    ("Ascend910_9589", "simd_simt_template"),
    ("Ascend910_9589", "unstructured_in_simt"),
    ("Ascend910_9589", "simt_only"),
])
def test_gather_rule_requires_explicit_simd(rule_mask, arch, compile_mode):
    text = str(make_indirect_ttir(rule_mask, arch=arch, compile_mode=compile_mode))
    expected = compile_mode == "simd" and bool(rule_mask & 512)
    assert ("tt.gather" in text) == expected
    assert ("gather.optimised.load" in text) == expected


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
    source_npu, indices_npu = source.npu(), indices.npu()
    outputs = []
    for rule_mask in (511, 1023):
        indirect_rows_kernel.device_caches.clear()
        output = torch.empty((row_blk, k), dtype=getattr(torch, dtype), device="npu")
        readable_output = torch.empty((row_blk, width), dtype=getattr(torch, dtype), device="npu")
        compiled = indirect_rows_kernel[(1, )](
            source_npu,
            indices_npu,
            output,
            readable_output,
            n_rows,
            WIDTH=width,
            K=k,
            ROW_BLK=row_blk,
            ROW_STEP=2,
            PER_LANE=case == "per_lane",
            VOLATILE=case == "volatile",
            READABLE=True,
            BASE_ROW=1,
            graph_optimize_rule_mask=rule_mask,
            compile_mode="simd",
        )
        torch.npu.synchronize()
        expected_rewrite = rule_mask == 1023 and case not in ("per_lane", "volatile", "i64")
        assert ("tt.gather" in compiled.asm["ttir"]) == expected_rewrite
        outputs.append(output.cpu())
        torch.testing.assert_close(outputs[-1], reference, rtol=0, atol=0)
        readable_reference = torch.zeros((row_blk, width), dtype=getattr(torch, dtype))
        readable_reference[:n_rows] = source[width:(n_rows + 1) * width].reshape(n_rows, width)
        torch.testing.assert_close(readable_output.cpu(), readable_reference, rtol=0, atol=0)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


def test_gather_rule_npu_offset_view_stays_indirect(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu", exc_type=ImportError)
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    # Original addresses are storage[1:9] and storage[17:25]. Reading the
    # entire second row from this view would include illegal storage[32].
    storage = torch.arange(32, dtype=torch.float32, device="npu")
    source = storage[1:]
    indices = torch.arange(8, dtype=torch.int32, device="npu").repeat(2, 1)
    reference = torch.stack((storage[1:9], storage[17:25])).cpu()
    for rule_mask in (511, 1023):
        indirect_rows_kernel.device_caches.clear()
        output = torch.empty((2, 8), dtype=torch.float32, device="npu")
        compiled = indirect_rows_kernel[(1, )](source, indices, output, output, 2, WIDTH=16, K=8, ROW_BLK=2, ROW_STEP=2,
                                               PER_LANE=False, VOLATILE=False, READABLE=False, BASE_ROW=0,
                                               graph_optimize_rule_mask=rule_mask, compile_mode="simd")
        torch.npu.synchronize()
        assert "tt.gather" not in compiled.asm["ttir"]
        assert "gather.optimised.load" not in compiled.asm["ttir"]
        torch.testing.assert_close(output.cpu(), reference, rtol=0, atol=0)
