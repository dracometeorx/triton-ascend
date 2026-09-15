# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT
"""Benefit-gate regressions at the short/long row-loop threshold."""

import pytest
from test_gather_readability import readable_fixture, run_gather_pass

pytestmark = pytest.mark.backend("none")


@pytest.mark.parametrize("lower,upper,step,dependency,expected", [
    (0, 4, 2, "row", False),
    (128, 132, 2, "row", False),  # two iterations, previously estimated as 66
    (128, 161, 2, "row", True),  # ceil(33 / 2) = 17, not 16
    (128, 160, 2, "row", False),  # exactly 16 iterations
    (128, 260, 2, "row", True),
    (-128, -124, 2, "row", False),
    (-128, 4, 2, "row", True),
    (132, 128, 2, "row", False),
    (128, 128, 2, "row", False),
    (None, 260, 2, "row", False),
    (0, None, 2, "row", False),
    (0, 256, 4, "row", False),  # loop step differs from the two-row tile
    (0, 256, 2, "unrelated", False),
    (0, 256, 2, "cancelled", False),
    (0, 256, 2, "nonlinear", False),
])
def test_gather_row_loop_iterations(tmp_path, lower, upper, step, dependency, expected):
    # FP32 source width 16 and index width 2: thresholds are 4 for <= 4
    # iterations, 3 for <= 16, and 2 for >= 17. Readability is proved in every
    # case so the test exercises the benefit gate, not the safety gate.
    text = readable_fixture().replace("2x8x", "2x2x")
    header, body = text.split("{\n", 1)
    text = header.replace("%src:", "%dynamic_lower: i32, %dynamic_upper: i32, %src:") + "{\n"
    lower_name, upper_name = "%dynamic_lower", "%dynamic_upper"
    if lower is not None:
        text += f"  %lower = arith.constant {lower} : i32\n"
        lower_name = "%lower"
    if upper is not None:
        text += f"  %upper = arith.constant {upper} : i32\n"
        upper_name = "%upper"
    text += f"  %step = arith.constant {step} : i32\n"
    text += "  %init = arith.constant dense<0.0> : tensor<2x2xf32>\n"
    text += "  %init_readable = arith.constant dense<0.0> : tensor<2x16xf32>\n"
    text += (f"  %result:2 = scf.for %iv = {lower_name} to {upper_name} step %step "
             "iter_args(%previous = %init, %previous_readable = %init_readable) "
             "-> (tensor<2x2xf32>, tensor<2x16xf32>) : i32 {\n")
    if dependency != "unrelated":
        carrier = "%iv"
        if dependency == "cancelled":
            text += "  %carrier = arith.subi %iv, %iv : i32\n"
            carrier = "%carrier"
        elif dependency == "nonlinear":
            text += "  %carrier = arith.muli %iv, %iv : i32\n"
            carrier = "%carrier"
        text += f"  %row_bias = arith.addi %bias, {carrier} : i32\n"
        body = body.replace("tt.splat %bias :", "tt.splat %row_bias :")
    body = body.replace("tt.return %value, %readable", "scf.yield %value, %readable")
    text += body + "\n  tt.return %result#0, %result#1 : tensor<2x2xf32>, tensor<2x16xf32>\n}"
    result = run_gather_pass(tmp_path, text)
    assert ("tt.gather" in result) == expected
    assert ("gather.optimised.load" in result) == expected
