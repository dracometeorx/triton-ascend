# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT
"""Host contracts for preserving and measuring the original PR #1027 workload."""

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import bench_gather_pr1027 as bench

pytestmark = pytest.mark.backend("none")


def test_original_shapes_and_kernel_are_preserved():
    # Digests of ast.dump(..., include_attributes=False) taken directly from
    # https://github.com/user-attachments/files/30009694/test_gather_optimization.py
    # Formatting/comments may change; expressions, signatures and shapes may not.
    module = ast.parse(Path(bench.__file__).read_text())
    configs = next(node.value for node in module.body if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "CONFIGS" for target in node.targets))
    kernel_module = ast.parse(Path(bench.__file__).with_name("gather_pr1027_kernel.py").read_text())
    kernel = next(node for node in kernel_module.body
                  if isinstance(node, ast.FunctionDef) and node.name == "scalar_loop_load_nd")
    assert hashlib.sha256(ast.dump(configs, include_attributes=False).encode()).hexdigest() == (
        "4386e9bd8db4dd6557e2b7d6b97c475b0851a902624a07cc1937962a154d226f")
    assert hashlib.sha256(ast.dump(kernel, include_attributes=False).encode()).hexdigest() == (
        "d48d7a45206c7000e3e08333fdafc4445a49a4e9af604e4cc45752d4d2c7cb2a")


@pytest.mark.parametrize("case_id,row_step,iterations", [
    (1, 2, 820),
    (2, 2, 820),
    (3, 1, 1639),
    (4, 2, 820),
    (5, 2, 820),
    (6, 2, 820),
    (7, 2, 820),
    (8, 2, 820),
    (9, 2, 820),
    (10, 3, 547),
    (11, 8, 205),
    (12, 4, 410),
    (13, 4, 410),
    (14, 4, 410),
])
def test_original_tiling_and_launch_constants(case_id, row_step, iterations):
    meta = bench.case_metadata(case_id)
    assert (meta["row_blk"], meta["row_step"], meta["grid"], meta["iterations"]) == (1639, row_step, (40, ), iterations)
    assert meta["dtype"] == "float32" and meta["index_dtype"] == "int32"
    constants = bench.kernel_constants(meta)
    rank = len(meta["src_shape"])
    assert constants[:rank] == meta["src_shape"]
    assert constants[rank:5] == (1, ) * (5 - rank)
    assert constants[5:5 + rank] == meta["idx_shape"]
    assert constants[5 + rank:10] == (1, ) * (5 - rank)
    assert constants[10:] == (rank, rank - 1, 65536, 1639, row_step)


def test_case_selection_preserves_ids_and_default_suite():
    assert bench.parse_args(["--list-cases"]).case_ids == list(range(1, 15))
    assert bench.parse_args(["--list-cases", "--case-ids", "10", "2"]).case_ids == [10, 2]
    assert bench.CONFIGS[1]["idx_shape"][-1] == 3076


@pytest.mark.parametrize("argv", [[], ["--list-cases", "--case-ids", "0"], ["--list-cases", "--case-ids", "1", "1"],
                                  ["--list-cases", "--rounds", "0"]])
def test_invalid_arguments(argv):
    with pytest.raises(SystemExit) as exc:
        bench.parse_args(argv)
    assert exc.value.code == 2


def test_disabled_rule_cannot_report_gather():
    with pytest.raises(RuntimeError, match="rule disabled"):
        bench.check_rewrite(511, "tt.gather")
    assert bench.check_rewrite(1023, "tt.gather")
    assert not bench.check_rewrite(1023, "tt.load")


def test_profiler_order_units_and_unmatched_summary(tmp_path):
    args = bench.parse_args(["--output-dir", str(tmp_path)])
    calls, records = [], []
    launches = {mask: object() for mask in bench.RULE_MASKS}
    names = {mask: f"kernel_{mask}" for mask in bench.RULE_MASKS}

    def profiler(fn, **kwargs):
        mask = next(mask for mask in launches if launches[mask] is fn)
        calls.append(mask)
        assert kwargs["target_kernel_name"] == names[mask]
        assert kwargs["warmup"] == 5 and kwargs["active"] == 30
        assert kwargs["keep_res"]
        assert kwargs["clear_l2_cache"] is False
        assert kwargs["prof_dir"].endswith(f"_mask{mask}")
        return 0.020 if mask == 511 else 0.010

    timings = bench.profile_pair(launches, names, 1, args, tmp_path, profiler, records.append)
    assert calls == [511, 1023, 1023, 511, 511, 1023]
    assert [record["us"] for record in records] == [20, 10, 10, 20, 20, 10]
    result = bench.summarize(bench.case_metadata(1), timings, True)
    assert (result["off_us"], result["on_us"], result["speedup"]) == (20, 10, 2)
    result = bench.summarize(bench.case_metadata(1), timings, False)
    assert result["status"] == "not_rewritten" and result["speedup"] == ""
    assert result["time_ratio"] == 2
    calls.clear()
    bench.profile_pair(launches, names, 2, args, tmp_path, profiler, records.append)
    assert calls == [1023, 511, 511, 1023, 1023, 511]


@pytest.mark.parametrize("duration", [float("inf"), float("nan"), 0, -1])
def test_invalid_profiler_result_does_not_record_a_sample(tmp_path, duration):
    args = bench.parse_args(["--output-dir", str(tmp_path)])
    records = []
    with pytest.raises(RuntimeError, match="Invalid profiler duration"):
        bench.profile_pair({511: object(), 1023: object()}, {511: "kernel", 1023: "kernel"}, 1, args, tmp_path,
                           lambda *args, **kwargs: duration, records.append)
    assert not records


@pytest.mark.parametrize("missing_write", [False, True])
def test_case_reuses_inputs_and_checks_each_output_before_timing(tmp_path, missing_write):
    # Metadata-only tensors exercise runner control flow without allocating the
    # original multi-GiB arrays. Native compilation/execution are not simulated.
    class Tensor:

        def __init__(self, shape, dtype):
            self.shape, self.dtype, self.state = shape, dtype, "unwritten"

        def long(self):
            return Tensor(self.shape, "int64")

        def fill_(self, _value):
            self.state = "poison"

    def assert_close(out, reference, **kwargs):
        assert out.shape == reference.shape
        assert out.state == "correct"
        assert kwargs == dict(rtol=0, atol=0)

    torch = SimpleNamespace(
        float32="float32",
        int32="int32",
        manual_seed=lambda seed: None,
        randn=lambda shape, dtype, device: Tensor(shape, dtype),
        randint=lambda low, high, shape, dtype, device: Tensor(shape, dtype),
        empty=lambda shape, dtype, device: Tensor(shape, dtype),
        gather=lambda src, dim, index: Tensor(index.shape, src.dtype),
        npu=SimpleNamespace(synchronize=lambda: None),
        testing=SimpleNamespace(assert_close=assert_close),
    )
    meta = bench.case_metadata(10)
    compiled_inputs, launches_seen = [], []

    class Kernel:

        def warmup(self, *inputs, grid, graph_optimize_rule_mask):
            assert grid == (40, ) and inputs[3:] == bench.kernel_constants(meta)
            assert inputs[0].shape == meta["src_shape"] and inputs[1].shape == meta["idx_shape"]
            assert inputs[1].dtype == "int32"
            compiled_inputs.append(inputs)
            return SimpleNamespace(asm={"ttir": "tt.gather" if graph_optimize_rule_mask == 1023 else "tt.load"},
                                   packed_metadata={"kernel_name": "scalar_loop_load_nd"})

        def __getitem__(self, grid):

            def launch(*inputs, graph_optimize_rule_mask):
                launches_seen.append(graph_optimize_rule_mask)
                if not (missing_write and graph_optimize_rule_mask == 1023):
                    inputs[2].state = "correct"

            return launch

    args = bench.parse_args(["--output-dir", str(tmp_path), "--rounds", "1"])
    records = []

    def profiler(fn, **kwargs):
        assert launches_seen[:2] == [511, 1023]
        fn()
        return 0.01

    if missing_write:
        with pytest.raises(AssertionError):
            bench.benchmark_case(meta, args, tmp_path, torch, Kernel(), profiler, records.append)
        assert not records
    else:
        result = bench.benchmark_case(meta, args, tmp_path, torch, Kernel(), profiler, records.append)
        assert result["rewritten"] and result["speedup"] == 1
        assert len(records) == 2
    assert all(left is right for left, right in zip(compiled_inputs[0][:3], compiled_inputs[1][:3]))
    assert (tmp_path / "mask511.ttir").read_text() == "tt.load"
    assert (tmp_path / "mask1023.ttir").read_text() == "tt.gather"
