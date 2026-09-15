# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native-pass regressions for proof of expanded Gather source reads."""

from pathlib import Path
import re

import pytest
from triton._C.libtriton import ascend, ir
from triton._C.libtriton.ascend import ir as ascend_ir

pytestmark = pytest.mark.backend("none")


def readable_fixture():
    path = Path(__file__).parents[1] / "Conversion/General/TritonToGraph/graph-optimize-gather-safety.mlir"
    return re.search(r"tt.func @row_mask_other\(.*?\n}", path.read_text(), re.S).group()


def run_gather_pass(tmp_path, text, **options):
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    path = tmp_path / "gather-input.mlir"
    path.write_text("module {\n" + text + "\n}")
    module = ir.parse_mlir_module(str(path), context)
    pm = ir.pass_manager(module.context)
    options.setdefault("target_arch", "Ascend910B1")
    ascend.passes.ttir.add_graph_optimize(pm, rule_mask=512, ub_capacity_bytes=96 * 1024, compile_mode="simd",
                                          **options)
    pm.run(module, "")
    result = str(module)
    path.write_text(result)
    ir.parse_mlir_module(str(path), context)
    return result


@pytest.mark.parametrize("case,expected", [
    ("same_mask", True),
    ("unmasked_witness", True),
    ("commuted_offsets", True),
    ("missing", False),
    ("unused", False),
    ("later", False),
    ("conditional", False),
    ("different_pointer", False),
    ("different_base", False),
    ("narrower_row", False),
    ("shifted_range", False),
    ("column_mask", False),
    ("different_row_mask", False),
    ("volatile", False),
    ("intervening_store", False),
    ("generated_load", False),
])
def test_full_row_readability(tmp_path, case, expected):
    text = readable_fixture()
    read = "  %readable = tt.load %readable_ptrs, %readable_mask, %readable_zero : tensor<2x16x!tt.ptr<f32>>\n"
    candidate = "  %value = tt.load %ptrs, %mask_grid, %other : tensor<2x8x!tt.ptr<f32>>\n"
    assert read in text and candidate in text
    if case == "unmasked_witness":
        text = text.replace(read, read.replace(", %readable_mask, %readable_zero", ""))
    elif case == "commuted_offsets":
        text = text.replace("arith.addi %readable_base, %readable_grid", "arith.addi %readable_grid, %readable_base")
    elif case in ("missing", "unused"):
        if case == "missing":
            text = text.replace(read, "")
        text = text.replace("-> (tensor<2x8xf32>, tensor<2x16xf32>)", "-> tensor<2x8xf32>")
        text = text.replace("tt.return %value, %readable : tensor<2x8xf32>, tensor<2x16xf32>",
                            "tt.return %value : tensor<2x8xf32>")
    elif case == "later":
        text = text.replace(read, "").replace(candidate, candidate + read)
    elif case == "conditional":
        text = text.replace("%src: !tt.ptr<f32>", "%condition: i1, %src: !tt.ptr<f32>")
        text = text.replace(
            read, "  %readable = scf.if %condition -> tensor<2x16xf32> {\n" +
            read.replace("%readable =", "%branch_readable =") + "    scf.yield %branch_readable : tensor<2x16xf32>\n" +
            "  } else {\n    scf.yield %readable_zero : tensor<2x16xf32>\n  }\n")
    elif case == "different_pointer":
        text = text.replace("%src: !tt.ptr<f32>", "%other_src: !tt.ptr<f32>, %src: !tt.ptr<f32>")
        text = text.replace("%readable_src = tt.splat %src", "%readable_src = tt.splat %other_src")
    elif case == "different_base":
        text = text.replace("  %readable_base =", "  %different_base = arith.addi %base, %width : tensor<2x1xi32>\n"
                            "  %readable_base =")
        text = text.replace("%readable_base = tt.broadcast %base", "%readable_base = tt.broadcast %different_base")
    elif case == "narrower_row":
        text = text.replace("end = 16 : i32", "end = 8 : i32").replace("16x", "8x")
    elif case == "shifted_range":
        text = text.replace("start = 0 : i32, end = 16 : i32", "start = 1 : i32, end = 17 : i32")
    elif case == "column_mask":
        text = text.replace("%src: !tt.ptr<f32>", "%column_mask: tensor<2x16xi1>, %src: !tt.ptr<f32>")
        text = text.replace(read, read.replace("%readable_mask", "%column_mask"))
    elif case == "different_row_mask":
        text = text.replace("%src: !tt.ptr<f32>", "%other_mask: tensor<2x1xi1>, %src: !tt.ptr<f32>")
        text = text.replace("%readable_mask = tt.broadcast %mask", "%readable_mask = tt.broadcast %other_mask")
    elif case == "volatile":
        text = text.replace(
            read, read.replace(" : tensor<2x16x!tt.ptr<f32>>", " {isVolatile = true} : tensor<2x16x!tt.ptr<f32>>"))
    elif case == "intervening_store":
        text = text.replace(read,
                            read + "  tt.store %readable_ptrs, %readable, %readable_mask : tensor<2x16x!tt.ptr<f32>>\n")
    elif case == "generated_load":
        text = text.replace(
            read,
            read.replace(" : tensor<2x16x!tt.ptr<f32>>",
                         ' {gather.optimised.load = "source"} : tensor<2x16x!tt.ptr<f32>>'))
    result = run_gather_pass(tmp_path, text)
    assert ("tt.gather" in result) == expected
    # The artificial source marker in one input must not produce a fallback.
    assert ('gather.optimised.load = "fallback"' in result) == expected


@pytest.mark.parametrize("arch,expected", [
    ("Ascend910B1", True),
    ("Ascend910_9391", True),
    ("Ascend910_9589", False),
    ("Ascend950", False),
    ("Ascend950PR_9599", False),
    ("", False),
    ("unknown", False),
])
def test_gather_target_eligibility(tmp_path, arch, expected):
    result = run_gather_pass(tmp_path, readable_fixture(), target_arch=arch)
    assert ("tt.gather" in result) == expected
    assert ("gather.optimised.load" in result) == expected
