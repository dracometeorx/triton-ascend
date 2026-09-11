# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT
"""ND kernel from PR #1027; preserve its addressing, masks and constexprs.

Source: https://github.com/user-attachments/files/30009694/test_gather_optimization.py
The kernel AST is checked against the original attachment by host tests.
"""

import triton
import triton.language as tl


@triton.jit
def scalar_loop_load_nd(
    src_ptr,
    idx_ptr,
    out_ptr,
    XB: tl.constexpr,
    YB: tl.constexpr,
    ZB: tl.constexpr,
    MB: tl.constexpr,
    NB: tl.constexpr,
    I_XB: tl.constexpr,
    I_YB: tl.constexpr,
    I_ZB: tl.constexpr,
    I_MB: tl.constexpr,
    I_NB: tl.constexpr,
    DIMS: tl.constexpr,
    AXIS: tl.constexpr,
    N_ROWS: tl.constexpr,
    ROW_BLK: tl.constexpr,
    ROW_STEP: tl.constexpr,
):
    row_begin = tl.program_id(0) * ROW_BLK
    row_end = tl.minimum(row_begin + ROW_BLK, N_ROWS)
    for rb in range(0, ROW_BLK, ROW_STEP):
        # for indirect load we need base offsets (without tl.arange())
        idx_stride = 1
        in_offsets_base = 0  # special case 1D
        if DIMS > 1:
            in_offsets_base = row_begin + rb
            in_offsets_base = in_offsets_base + tl.arange(0, ROW_STEP)
            in_offsets_base = in_offsets_base[:, None] * YB
            if AXIS != 1:
                in_offsets_base = in_offsets_base + tl.arange(0, I_YB)[None, :]
        if DIMS > 2:
            in_offsets_base = in_offsets_base[:, :, None] * ZB
            if AXIS != 2:
                in_offsets_base = in_offsets_base + tl.arange(0, I_ZB)[None, None, :]
                if AXIS < 2:
                    idx_stride = idx_stride * ZB
        if DIMS > 3:
            in_offsets_base = in_offsets_base[:, :, :, None] * MB
            if AXIS != 3:
                in_offsets_base = in_offsets_base + tl.arange(0, I_MB)[None, None, None, :]
                if AXIS < 3:
                    idx_stride = idx_stride * MB
        if DIMS > 4:
            in_offsets_base = in_offsets_base[:, :, :, :, None] * NB
            if AXIS != 4:
                in_offsets_base = in_offsets_base + tl.arange(0, I_NB)[None, None, None, None, :]
                if AXIS < 4:
                    idx_stride = idx_stride * NB

        idx_offsets = tl.arange(0, I_XB)  # special case 1D
        mask = idx_offsets < I_XB  # special case 1D
        if DIMS > 1:
            idx_offsets = row_begin + rb + tl.arange(0, ROW_STEP)
            mask = idx_offsets < row_end
            idx_offsets = idx_offsets[:, None] * I_YB + tl.arange(0, I_YB)[None, :]
            mask = mask[:, None]
        if DIMS > 2:
            idx_offsets = idx_offsets[:, :, None] * I_ZB + tl.arange(0, I_ZB)[None, None, :]
            mask = mask[:, :, None]
        if DIMS > 3:
            idx_offsets = idx_offsets[:, :, :, None] * I_MB + tl.arange(0, I_MB)[None, None, None, :]
            mask = mask[:, :, :, None]
        if DIMS > 4:
            idx_offsets = idx_offsets[:, :, :, :, None] * I_NB + tl.arange(0, I_NB)[None, None, None, None, :]
            mask = mask[:, :, :, :, None]

        idx = tl.load(idx_ptr + idx_offsets, mask=mask, other=0)
        in_offsets = in_offsets_base + (idx * idx_stride)
        out = tl.load(src_ptr + in_offsets, mask=mask, other=0)
        tl.store(out_ptr + idx_offsets, out, mask=mask)
