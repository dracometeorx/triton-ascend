# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT
"""Profile indirect loads with gather disabled (511) and enabled (1023)."""

import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory; existing results are never overwritten")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", type=int, default=4096, help="Launched rows; tail uses one fewer logical row")
    parser.add_argument("--widths", type=int, nargs="+", default=[16])
    parser.add_argument("--index-widths", type=int, nargs="+", default=[8])
    parser.add_argument("--row-block", type=int, default=128)
    parser.add_argument("--row-step", type=int, default=4)
    parser.add_argument("--dtypes", nargs="+", choices=("float32", "float16", "bfloat16"), default=["float32"])
    parser.add_argument("--cases", nargs="+", choices=("full", "tail", "negative", "upper_bound"),
                        default=["full", "tail"])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5, help="Discarded launches per profiler session (count, not ms)")
    parser.add_argument("--repeat", type=int, default=30, help="Measured launches per profiler session")
    parser.add_argument("--clear-l2-cache", action="store_true",
                        help="Use the backend's cache-clearing operation before each launch")
    parser.add_argument("--require-rewrite", action="store_true", help="Fail if mask 1023 does not produce tt.gather")
    args = parser.parse_args()
    for name in ("rows", "row_block", "row_step", "rounds", "warmup", "repeat"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for value in [*args.widths, *args.index_widths, args.row_step]:
        if value <= 0 or value & (value - 1):
            parser.error("Widths, index widths and row-step must be positive powers of two")
    if args.rows % args.row_block or args.row_block % args.row_step:
        parser.error("rows must be divisible by row-block, and row-block by row-step")
    if args.rows < 2:
        parser.error("rows must be at least 2")
    # The kernel combines base + index as i32 before adding it to src_ptr.
    # Include the upper_bound case's last legal pointer and index/output grids.
    if any((args.rows + 1) * width > 2**31 - 1 for width in args.widths):
        parser.error("source element offsets must fit in signed i32")
    if any(args.rows * k - 1 > 2**31 - 1 for k in args.index_widths):
        parser.error("index/output element offsets must fit in signed i32")
    return args


def git_output(*args):
    return subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parent), *args], text=True).strip()


def main():
    args = parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    # A preceding sanitizer run must not supply binaries to a performance run.
    # Use a fresh cache, then keep it live throughout compilation and timing.
    os.environ["TRITON_CACHE_DIR"] = str(root / "cache")
    os.environ["TRITON_ENABLE_SANITIZER"] = "0"
    os.environ["TRITON_ALWAYS_COMPILE"] = "0"
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

    import torch
    import torch_npu
    import triton
    from triton.backends.ascend.testing import do_bench_npu
    from triton.backends.ascend.utils import _get_npucompiler_path
    import triton.backends.ascend.compiler as ascend_compiler
    import triton._C.libtriton as native
    from test_gather_graph_optimize import indirect_rows_kernel

    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU is unavailable")
    torch.npu.set_device(args.device)
    info = {
        "arguments": vars(args),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short"),
        "device": str(torch.npu.get_device_properties(args.device)),
        "target": str(triton.runtime.driver.active.get_current_target()),
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", "unknown"),
        "triton": triton.__file__,
        "ascend_backend": ascend_compiler.__file__,
        "native": native.__file__,
        "kernel_source": indirect_rows_kernel.fn.__code__.co_filename,
        "compiler": _get_npucompiler_path()[0],
        "environment": {
            key: os.getenv(key)
            for key in ("TRITON_CACHE_DIR", "TRITON_ENABLE_SANITIZER", "TRITON_ALWAYS_COMPILE",
                        "TRITON_ENABLE_TASKQUEUE", "ASCEND_LAUNCH_BLOCKING", "PYTORCH_NO_NPU_MEMORY_CACHING",
                        "TRITON_ALL_BLOCKS_PARALLEL", "ASCEND_HOME_PATH")
        },
        "timing": "do_bench_npu: mean device kernel duration per round; summary is the median of round means",
    }
    (root / "run.json").write_text(json.dumps(info, indent=2, default=str))
    print(json.dumps(info, indent=2, default=str), flush=True)
    sample_fields = [
        "point", "width", "index_width", "dtype", "case", "rows", "row_block", "row_step", "round", "rule_mask",
        "rewritten", "us", "kernel_name"
    ]
    summary_fields = [
        "point", "rewritten", "off_us", "on_us", "speedup", "off_min_us", "off_max_us", "on_min_us", "on_max_us"
    ]
    with (root / "samples.csv").open("w", newline="") as sample_file, (root / "summary.csv").open(
            "w", newline="") as summary_file:
        samples = csv.DictWriter(sample_file, fieldnames=sample_fields)
        summaries = csv.DictWriter(summary_file, fieldnames=summary_fields)
        samples.writeheader()
        summaries.writeheader()
        points = itertools.product(args.widths, args.index_widths, args.dtypes, args.cases)
        for point_index, (width, k, dtype_name, case) in enumerate(points):
            point = f"{point_index:04d}_w{width}_k{k}_{dtype_name}_{case}"
            print(f"[bench] {point} stage=compile-and-verify", flush=True)
            point_dir = root / point
            point_dir.mkdir()
            n_rows = args.rows - (case == "tail")
            dtype = getattr(torch, dtype_name)
            generator = torch.Generator().manual_seed(0)
            source = torch.randn((n_rows + 2, width), generator=generator).to(dtype)
            indices = torch.randint(width, (n_rows, k), dtype=torch.int32, generator=generator)
            # Every tile takes the fallback in these two scenarios, using legal
            # pointer addresses in the allocated preceding/following source row.
            if case == "negative":
                indices[:, 0] = -1
            elif case == "upper_bound":
                indices[:, 0] = width
            addresses = (torch.arange(n_rows) + 1)[:, None] * width + indices.long()
            reference = torch.full((args.rows, k), -7.0, dtype=dtype)
            reference[:n_rows] = source.flatten()[addresses]
            src_npu, idx_npu = source.npu(), indices.npu()
            grid = (args.rows // args.row_block, )
            launches, compiled_kernels, rewritten = {}, {}, {}
            for mask in (511, 1023):
                output = torch.empty((args.rows, k), dtype=dtype, device="npu")

                def launch(rule_mask=mask, result=output):
                    return indirect_rows_kernel[grid](
                        src_npu,
                        idx_npu,
                        result,
                        n_rows,
                        WIDTH=width,
                        K=k,
                        ROW_BLK=args.row_block,
                        ROW_STEP=args.row_step,
                        PER_LANE=False,
                        VOLATILE=False,
                        graph_optimize_rule_mask=rule_mask,
                    )

                compiled = launch()
                torch.npu.synchronize()
                launches[mask], compiled_kernels[mask] = launch, compiled
                ttir = compiled.asm["ttir"]
                (point_dir / f"mask{mask}.ttir").write_text(ttir)
                rewritten[mask] = "tt.gather" in ttir
                assert not (mask == 511 and rewritten[mask]), "Gather appeared with the rule disabled"
                if mask == 1023 and args.require_rewrite:
                    if not rewritten[mask]:
                        raise RuntimeError(
                            f"Gather did not fire at {point} (ROW_BLK={args.row_block}, ROW_STEP={args.row_step}). "
                            f"Inspect {point_dir / 'mask1023.ttir'} and {root / 'run.json'}; "
                            "no performance comparison has been recorded for this point.")
                torch.testing.assert_close(output.cpu(), reference, rtol=0, atol=0)
            timings = {511: [], 1023: []}
            for round_index in range(args.rounds):
                # Alternate within and across points to reduce order bias.
                order = (511, 1023) if (round_index + point_index) % 2 == 0 else (1023, 511)
                for mask in order:
                    name = compiled_kernels[mask].packed_metadata.get("kernel_name")
                    if not name:
                        raise RuntimeError("Compiled kernel has no profiler kernel_name; cannot filter samples safely")
                    print(f"[bench] {point} mask={mask} round={round_index + 1} rewritten={rewritten[mask]}",
                          flush=True)
                    ms = do_bench_npu(
                        launches[mask],
                        warmup=args.warmup,
                        active=args.repeat,
                        clear_l2_cache=args.clear_l2_cache,
                        prof_dir=str(point_dir / f"profile_r{round_index + 1}_mask{mask}"),
                        keep_res=True,
                        target_kernel_name=name,
                    )
                    if not math.isfinite(ms) or ms <= 0:
                        raise RuntimeError(f"Profiler returned an invalid duration: {ms}; inspect {point_dir}")
                    us = ms * 1000
                    timings[mask].append(us)
                    samples.writerow(
                        dict(point=point, width=width, index_width=k, dtype=dtype_name, case=case, rows=n_rows,
                             row_block=args.row_block, row_step=args.row_step, round=round_index + 1, rule_mask=mask,
                             rewritten=rewritten[mask], us=us, kernel_name=name))
                    sample_file.flush()
            off, on = statistics.median(timings[511]), statistics.median(timings[1023])
            summaries.writerow(
                dict(point=point, rewritten=rewritten[1023], off_us=off, on_us=on, speedup=off / on,
                     off_min_us=min(timings[511]), off_max_us=max(timings[511]), on_min_us=min(timings[1023]),
                     on_max_us=max(timings[1023])))
            summary_file.flush()
            print(f"[bench] {point} off={off:.3f} us on={on:.3f} us speedup={off / on:.3f}x", flush=True)
    print(f"Results: {root / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
