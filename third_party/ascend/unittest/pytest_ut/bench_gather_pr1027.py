# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT
"""Primary gather regression: PR #1027 shapes/kernel/tiling, masks 511 vs 1023."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import traceback

ORIGINAL_ATTACHMENT = "https://github.com/user-attachments/files/30009694/test_gather_optimization.py"
CORES = 40
UB_SIZE = 192 * 1024
CONFIGS = [
    {"src_shape": (65536, 4096), "idx_shape": (65536, 2048)},
    {"src_shape": (65536, 2048), "idx_shape": (65536, 3076)},
    {"src_shape": (65536, 1024), "idx_shape": (65536, 4096)},
    {"src_shape": (65536, 64, 64), "idx_shape": (65536, 64, 32)},
    {"src_shape": (65536, 64, 32), "idx_shape": (65536, 64, 32)},
    {"src_shape": (65536, 16, 64), "idx_shape": (65536, 16, 128)},
    {"src_shape": (65536, 16, 16, 16), "idx_shape": (65536, 16, 16, 8)},
    {"src_shape": (65536, 16, 8, 16), "idx_shape": (65536, 16, 8, 16)},
    {"src_shape": (65536, 4, 8, 32), "idx_shape": (65536, 4, 8, 64)},
    {"src_shape": (65536, 4, 8, 8, 8), "idx_shape": (65536, 4, 8, 8, 4)},
    {"src_shape": (65536, 4, 4, 2, 16), "idx_shape": (65536, 4, 4, 2, 16)},
    {"src_shape": (65536, 8, 2, 2, 8), "idx_shape": (65536, 8, 2, 2, 32)},
    {"src_shape": (65536, 2, 2, 1, 64), "idx_shape": (65536, 2, 2, 1, 256)},
    {"src_shape": (65536, 32, 2, 2, 2), "idx_shape": (65536, 32, 2, 2, 8)},
]
RULE_MASKS = (511, 1023)


def cdiv(numerator, denominator):
    return (numerator + denominator - 1) // denominator


def case_metadata(case_id):
    cfg = CONFIGS[case_id - 1]
    src_shape, idx_shape = cfg["src_shape"], cfg["idx_shape"]
    axis = len(src_shape) - 1
    row_blk = cdiv(idx_shape[0], CORES)
    # Exactly the original FP32/i32 heuristic, including ceil-div. Do not round
    # ROW_STEP to a power of two or make it divide ROW_BLK.
    ms = 4 if src_shape[axis] >= idx_shape[axis] else 0
    mi = 12 - ms
    slice_size = ms * math.prod(src_shape[1:]) * 4 + mi * math.prod(idx_shape[1:]) * 4
    row_step = min(row_blk, cdiv(UB_SIZE, slice_size))
    return dict(case_id=case_id, src_shape=src_shape, idx_shape=idx_shape, axis=axis, dtype="float32",
                index_dtype="int32", row_blk=row_blk, row_step=row_step, grid=(cdiv(idx_shape[0], row_blk), ),
                iterations=cdiv(row_blk, row_step))


def kernel_constants(meta):
    src_shape, idx_shape = meta["src_shape"], meta["idx_shape"]
    return (*src_shape, *([1] * (5 - len(src_shape))), *idx_shape, *([1] * (5 - len(idx_shape))), len(src_shape),
            meta["axis"], idx_shape[0], meta["row_blk"], meta["row_step"])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="New directory; required unless --list-cases is used")
    parser.add_argument("--list-cases", action="store_true",
                        help="Print the original 14 shapes and derived tiling without NPU imports")
    parser.add_argument("--case-ids", nargs="+", type=int, choices=range(1, 15), default=list(range(1, 15)))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5, help="Discarded launches per profiler session")
    parser.add_argument("--repeat", type=int, default=30, help="Measured launches per profiler session")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clear-l2-cache", action="store_true")
    parser.add_argument("--require-rewrite", action="store_true",
                        help="Return failure after the run if any enabled case did not rewrite")
    args = parser.parse_args(argv)
    if not args.list_cases and args.output_dir is None:
        parser.error("--output-dir is required")
    if len(set(args.case_ids)) != len(args.case_ids):
        parser.error("--case-ids must not contain duplicates")
    if args.device < 0 or min(args.rounds, args.warmup, args.repeat) <= 0:
        parser.error("device must be nonnegative; rounds, warmup and repeat must be positive")
    return args


def check_rewrite(mask, ttir):
    rewritten = "tt.gather" in ttir
    if mask == 511 and rewritten:
        raise RuntimeError("tt.gather appeared with the gather rule disabled")
    return rewritten


def profile_pair(launches, names, case_id, args, point_dir, profiler, record):
    timings = {mask: [] for mask in RULE_MASKS}
    for round_index in range(args.rounds):
        order = RULE_MASKS if (round_index + case_id - 1) % 2 == 0 else tuple(reversed(RULE_MASKS))
        for mask in order:
            print(f"[pr1027] case={case_id:02d} mask={mask} round={round_index + 1} stage=profile", flush=True)
            ms = profiler(launches[mask], warmup=args.warmup, active=args.repeat, clear_l2_cache=args.clear_l2_cache,
                          prof_dir=str(point_dir / f"profile_r{round_index + 1}_mask{mask}"), keep_res=True,
                          target_kernel_name=names[mask])
            if not math.isfinite(ms) or ms <= 0:
                raise RuntimeError(f"Invalid profiler duration {ms}; inspect {point_dir}")
            us = ms * 1000
            timings[mask].append(us)
            record(dict(case_id=case_id, round=round_index + 1, rule_mask=mask, us=us, kernel_name=names[mask]))
    return timings


def summarize(meta, timings, rewritten):
    off, on = (statistics.median(timings[mask]) for mask in RULE_MASKS)
    return dict(meta, status="rewritten" if rewritten else "not_rewritten", rewritten=rewritten, off_us=off, on_us=on,
                time_ratio=off / on, speedup=off / on if rewritten else "", off_min_us=min(timings[511]),
                off_max_us=max(timings[511]), on_min_us=min(timings[1023]), on_max_us=max(timings[1023]))


def benchmark_case(meta, args, point_dir, torch, kernel, profiler, record):
    # One case's allocations live only in this call. The two masks use the same
    # inputs AND output buffer, as in the original benchmark's repeated calls.
    torch.manual_seed(args.seed + meta["case_id"])
    src = torch.randn(meta["src_shape"], dtype=torch.float32, device="npu")
    idx = torch.randint(0, meta["src_shape"][meta["axis"]], meta["idx_shape"], dtype=torch.int32, device="npu")
    out = torch.empty(meta["idx_shape"], dtype=torch.float32, device="npu")
    inputs = (src, idx, out, *kernel_constants(meta))
    launches, names, rewritten = {}, {}, {}
    for mask in RULE_MASKS:
        print(f"[pr1027] case={meta['case_id']:02d} mask={mask} stage=compile", flush=True)
        compiled = kernel.warmup(*inputs, grid=meta["grid"], graph_optimize_rule_mask=mask)
        ttir = compiled.asm["ttir"]
        (point_dir / f"mask{mask}.ttir").write_text(ttir)
        rewritten[mask] = check_rewrite(mask, ttir)
        names[mask] = compiled.packed_metadata.get("kernel_name")
        if not names[mask]:
            raise RuntimeError("Compiled kernel has no profiler kernel_name")

        def launch(rule_mask=mask):
            return kernel[meta["grid"]](*inputs, graph_optimize_rule_mask=rule_mask)

        launches[mask] = launch
    (point_dir / "compile.json").write_text(json.dumps(dict(rewritten=rewritten, kernel_names=names), indent=2))
    if not rewritten[1023]:
        print(f"[pr1027] case={meta['case_id']:02d} status=not_rewritten; no gather speedup will be reported",
              flush=True)

    # torch.gather is a correctness reference only; its allocations and index
    # conversion are outside timing. The tested kernels still receive i32 idx.
    reference = torch.gather(src, dim=meta["axis"], index=idx.long())
    for mask in RULE_MASKS:
        # Poison the reused output so missing writes cannot inherit a previous
        # variant's correct output. Do not include this operation in timing.
        out.fill_(float("nan"))
        print(f"[pr1027] case={meta['case_id']:02d} mask={mask} stage=verify", flush=True)
        launches[mask]()
        torch.npu.synchronize()
        torch.testing.assert_close(out, reference, rtol=0, atol=0)
    del reference
    timings = profile_pair(launches, names, meta["case_id"], args, point_dir, profiler, record)
    return summarize(meta, timings, rewritten[1023])


def main():
    args = parse_args()
    cases = [case_metadata(case_id) for case_id in args.case_ids]
    if args.list_cases:
        print(json.dumps(cases, indent=2))
        return
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    os.environ["TRITON_CACHE_DIR"] = str(root / "cache")
    os.environ["TRITON_ENABLE_SANITIZER"] = "0"
    os.environ["TRITON_ALWAYS_COMPILE"] = "0"
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

    import torch
    import torch_npu
    import triton
    import triton._C.libtriton as native
    import triton.backends.ascend.compiler as ascend_compiler
    from triton.backends.ascend.testing import do_bench_npu
    from triton.backends.ascend.utils import _get_npucompiler_path
    from gather_pr1027_kernel import scalar_loop_load_nd

    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU is unavailable")
    torch.npu.set_device(args.device)

    def git_output(*git_args):
        return subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parent), *git_args],
                                       text=True).strip()

    info = dict(
        arguments=vars(args),
        cases=cases,
        original_attachment=ORIGINAL_ATTACHMENT,
        cores=CORES,
        ub_size=UB_SIZE,
        git_commit=git_output("rev-parse", "HEAD"),
        git_status=git_output("status", "--short"),
        device=str(torch.npu.get_device_properties(args.device)),
        target=str(triton.runtime.driver.active.get_current_target()),
        torch=torch.__version__,
        torch_npu=getattr(torch_npu, "__version__", "unknown"),
        triton=triton.__file__,
        native=native.__file__,
        ascend_backend=ascend_compiler.__file__,
        kernel_source=scalar_loop_load_nd.fn.__code__.co_filename,
        compiler=_get_npucompiler_path()[0],
        environment={
            key: os.getenv(key)
            for key in ("TRITON_CACHE_DIR", "TRITON_ENABLE_SANITIZER", "TRITON_ALWAYS_COMPILE",
                        "TRITON_ENABLE_TASKQUEUE", "ASCEND_LAUNCH_BLOCKING", "PYTORCH_NO_NPU_MEMORY_CACHING",
                        "TRITON_ALL_BLOCKS_PARALLEL", "ASCEND_HOME_PATH")
        },
        timing="do_bench_npu; microseconds; median of per-round mean kernel durations",
    )
    (root / "run.json").write_text(json.dumps(info, indent=2, default=str))
    print(json.dumps(info, indent=2, default=str), flush=True)
    sample_fields = ["case_id", "round", "rule_mask", "us", "kernel_name"]
    summary_fields = [
        *cases[0], "status", "rewritten", "off_us", "on_us", "time_ratio", "speedup", "off_min_us", "off_max_us",
        "on_min_us", "on_max_us"
    ]
    missed = []
    with (root / "samples.csv").open("w", newline="") as sample_file, (root / "summary.csv").open(
            "w", newline="") as summary_file:
        samples = csv.DictWriter(sample_file, fieldnames=sample_fields)
        summaries = csv.DictWriter(summary_file, fieldnames=summary_fields)
        samples.writeheader()
        summaries.writeheader()

        def record(sample):
            samples.writerow(sample)
            sample_file.flush()

        for meta in cases:
            point_dir = root / f"case_{meta['case_id']:02d}"
            point_dir.mkdir()
            (point_dir / "case.json").write_text(json.dumps(meta, indent=2))
            try:
                result = benchmark_case(meta, args, point_dir, torch, scalar_loop_load_nd, do_bench_npu, record)
            except Exception:
                (point_dir / "error.txt").write_text(traceback.format_exc())
                summaries.writerow(dict(meta, status="error"))
                summary_file.flush()
                raise
            summaries.writerow(result)
            summary_file.flush()
            if not result["rewritten"]:
                missed.append(meta["case_id"])
            print(
                f"[pr1027] case={meta['case_id']:02d} status={result['status']} "
                f"off={result['off_us']:.3f} us on={result['on_us']:.3f} us "
                f"time_ratio={result['time_ratio']:.3f}", flush=True)
    print(f"Results: {root / 'summary.csv'}; cases without gather: {missed}", flush=True)
    if args.require_rewrite and missed:
        raise SystemExit(f"Gather did not rewrite cases {missed}; original shapes and tiling were retained")


if __name__ == "__main__":
    main()
