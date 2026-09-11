# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT
"""Run one kernel/mask per process and retain compiler artifacts before launch."""

import argparse
import json
import os
from pathlib import Path

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from triton.backends.ascend.utils import _get_npucompiler_path
from test_gather_graph_optimize import indirect_rows_kernel


@triton.jit
def copy_probe_kernel(src_ptr, out_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(out_ptr + offsets, tl.load(src_ptr + offsets))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("copy", "full", "tail"), default="tail")
    parser.add_argument("--rule-mask", type=int, choices=(0, 511, 512, 1023), required=True)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    compiler, _ = _get_npucompiler_path()
    run_info = {
        "case": args.case,
        "rule_mask": args.rule_mask,
        "dtype": args.dtype,
        "compiler": compiler,
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", "unknown"),
        "triton": triton.__file__,
        "environment": {
            key: os.getenv(key)
            for key in ("TRITON_ENABLE_SANITIZER", "TRITON_ALWAYS_COMPILE", "PYTORCH_NO_NPU_MEMORY_CACHING",
                        "ASCEND_LAUNCH_BLOCKING", "TRITON_ENABLE_TASKQUEUE", "ASCEND_HOME_PATH")
        },
    }
    print(json.dumps(run_info, indent=2), flush=True)
    (args.output_dir / "run.json").write_text(json.dumps(run_info, indent=2))
    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU is unavailable")
    dtype = getattr(torch, args.dtype)
    print("[probe] stage=copy-inputs", flush=True)
    if args.case == "copy":
        reference = torch.arange(128, dtype=torch.float32).to(dtype)
        source = reference.npu()
        output = torch.empty_like(source)
        kernel = copy_probe_kernel
        inputs = (source, output)
        options = {"BLOCK": 128}
    else:
        width, k, row_blk = 16, 8, 8
        n_rows = row_blk if args.case == "full" else row_blk - 1
        source = torch.arange((n_rows + 1) * width, dtype=torch.float32).to(dtype)
        indices = torch.arange(n_rows * k, dtype=torch.int32).reshape(n_rows, k) % width
        reference = torch.full((row_blk, k), -7.0, dtype=dtype)
        for row in range(n_rows):
            reference[row] = source[(row + 1) * width + indices[row].long()]
        source_npu, indices_npu = source.npu(), indices.npu()
        output = torch.empty((row_blk, k), dtype=dtype, device="npu")
        kernel = indirect_rows_kernel
        inputs = (source_npu, indices_npu, output, n_rows)
        options = {"WIDTH": width, "K": k, "ROW_BLK": row_blk, "ROW_STEP": 2, "PER_LANE": False, "VOLATILE": False}
    torch.npu.synchronize()
    options["graph_optimize_rule_mask"] = args.rule_mask
    kernel.device_caches.clear()
    print(f"[probe] rule_mask={args.rule_mask} stage=compile", flush=True)
    # JIT warmup compiles without launching; retain artifacts even if launch aborts.
    compiled = kernel.warmup(*inputs, grid=(1, ), **options)
    for stage, artifact in compiled.asm.items():
        path = args.output_dir / f"kernel.{Path(stage).name}"
        if isinstance(artifact, str):
            path.write_text(artifact)
        elif isinstance(artifact, bytes):
            path.write_bytes(artifact)
    has_gather = "tt.gather" in compiled.asm["ttir"]
    expected_gather = args.case != "copy" and bool(args.rule_mask & 512)
    print(f"[probe] tt.gather={has_gather} artifacts={args.output_dir.resolve()}", flush=True)
    assert has_gather == expected_gather, "Unexpected gather rule activation"
    print(f"[probe] rule_mask={args.rule_mask} stage=launch", flush=True)
    kernel[(1, )](*inputs, **options)
    print("[probe] stage=synchronize", flush=True)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), reference, rtol=0, atol=0)
    print("[probe] stage=verified", flush=True)


if __name__ == "__main__":
    main()
