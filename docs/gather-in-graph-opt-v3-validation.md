# Gather graph optimization v3 validation

Branch: `gather-in-graph-opt-v3`, based on v2 commit `8071bf98e`.

This change fixes the correctness issues found while reviewing the migration of
PR #1027 into the GraphOptimize framework. It does not recalibrate the performance
threshold tables or claim a new device speedup.

## Changes

- Preserve masks that are broadcast across the gather axis. Broadcast the
  original row/plane mask to the source tile, load inactive source lanes as zero,
  and restore the original load's `other` in the result. Sanitize inactive indices
  to zero before both bounds reductions and `tt.gather`.
- Use the fast path only for active indices in `[0, source_width)`. Negative and
  upper-bound indices execute the original pointer load in the fallback. No
  Python-style negative-index wrapping is introduced.
- Reuse the original axis-invariant source offsets instead of regenerating the
  outer dimensions from a guessed scalar row offset. Nonzero range starts and
  fixed pointer-offset biases retain their original addresses.
- Return offset-classification results by value and reacquire cache slots after
  recursion. DenseMap growth cannot invalidate outstanding classification values.
  Keep the rank of opaque tensor arguments and classify non-splat constants
  conservatively.
- Verify newly created operations before replacing the original load. On a
  verifier failure, erase the newly inserted operations in reverse order. Register
  the arith and scf dialect dependencies in the pass.
- Keep `graph_optimize_rule_mask` live; remove the five deprecated dataclass
  fields reintroduced by v2, and update the Python pipeline contract tests.

The matcher conservatively declines volatile accesses, masks varying along the
gather axis, non-i32 indices, encoded/dynamic tensors, nonpositive or overflowing
shapes, nested pointer-base forms, and partially covered non-gather dimensions.
These cases retain their original loads. A future extension must prove its
address and masking semantics before relaxing these restrictions.

## Local verification

On the development Mac:

- Nine modified Python contract cases passed in an isolated source harness using
  Python 3.12. The harness executes the actual `make_ttir`, `NPUOptions` class,
  and selected test bodies while replacing native imports; it is not a full
  backend test run.
- Python syntax, YAPF formatting, clang-format on modified C++ lines, and
  `git diff --check` passed.
- The native Ascend compiler, lit execution, ASan execution, and NPU tests were
  not available locally. They must be run on the Ascend build/test server.

## Server workflow

Rebuild and install this checkout using the server's normal Triton-Ascend build
procedure. Rebuilding only Python files is insufficient: the rule lives in C++.
Confirm the imported backend belongs to this checkout/build:

```bash
python -c 'import triton; import triton.backends.ascend.compiler as c; print(triton.__file__); print(c.__file__)'
```

Run the focused lit cases. Set `ASCEND_LIT_DIR` to the generated Ascend unittest
directory containing `lit.site.cfg.py` in the existing CMake build:

```bash
python -m lit -v "$ASCEND_LIT_DIR" --filter='graph-optimize-gather'
```

The three files cover the scalar-row and 3D tensor-row patterns, masked source
loads, nonnegative guards, `other`, exact source offsets, rejected volatile and
lane-mask cases, i64 rejection, negative extents, opaque tensor arguments, and a
96-operation expression chain that forces DenseMap growth. The safety file also
tests the existing capacity model's acceptance/rejection boundary at half-budgets
371/370 bytes. These are model-boundary tests, not device-memory measurements.

With an ASan-enabled build, run the same lit files, especially
`graph-optimize-gather-safety.mlir`. The `deep_offset_cache` case should neither
report a use-after-free nor crash in classification.

Run the changed Python configuration contracts and actual early-TTIR pipeline:

```bash
python -m pytest -q third_party/ascend/unittest/pytest_ut/test_compiler_layout_memory_contract.py -k 'make_ttir_passes_canonical_compile_mode_to_graph_optimize or npu_options_do_not_expose_deprecated_graph_switches'
python -m pytest -q third_party/ascend/unittest/pytest_ut/test_gather_graph_optimize.py -k 'not npu_equivalence'
```

Run the NPU numerical cases:

```bash
python -m pytest -q third_party/ascend/unittest/pytest_ut/test_gather_graph_optimize.py -k npu_equivalence
```

These compare masks 511 and 1023 on the same kernel, assert whether the rule
actually rewrote TTIR, and compare both outputs to explicit flat-pointer indexing
with zero tolerance. Cases include a tail row, all masked rows, nonzero `other`,
negative indices, a legal upper-bound fallback address, a per-lane mask, volatile
loads, and i64 indices, across f32/f16/bf16. Output padding is deliberately stored
so a lost `other` cannot be hidden by a masked store. Source/index allocations in
the tail case are not padded to the complete launch tile. Run these cases with
the server's supported device-memory sanitizer as well: numerical equality alone
does not prove that no inactive source lane was read.

`TRITON_ENABLE_SANITIZER=1` enables compiler instrumentation; it does not launch
the device checker. Run instrumented kernels through `mssanitizer`, and disable
the PyTorch NPU memory pool so allocation boundaries remain visible to the
checker. Upgrading bishengir alone does not configure the checker runtime. Follow
the installed tool's hardware/toolchain requirements and environment setup.
See the official [Triton checking example](https://github.com/Ascend/mssanitizer/blob/master/docs/zh/best_practices/mssanitizer_basic_cases.md#4-检测triton算子)
and [AscendNPU-IR debugging guide](https://github.com/Ascend/AscendNPU-IR/blob/master/docs/source/en/user_guide/debug_option.md).

Start with one case in a single process, retaining both pytest and checker output:

```bash
GATHER_TEST=third_party/ascend/unittest/pytest_ut/test_gather_graph_optimize.py
mkdir -p gather-v3-test-results
command -v mssanitizer
set -o pipefail
TRITON_ENABLE_SANITIZER=1 \
TRITON_ALWAYS_COMPILE=1 \
PYTORCH_NO_NPU_MEMORY_CACHING=1 \
ASCEND_LAUNCH_BLOCKING=1 \
mssanitizer --tool=memcheck -- python -m pytest -vv -s -x --tb=long \
  "$GATHER_TEST" -k 'npu_equivalence and tail and float32' \
  --junitxml=gather-v3-test-results/npu-sanitizer-smoke.xml \
  2>&1 | tee gather-v3-test-results/npu-sanitizer-smoke.log
```

After it passes without checker findings, repeat with `-k npu_equivalence` and
different log/XML filenames to cover all 21 cases. Keep the checker's own report
files as well; a passing pytest result alone does not establish a clean memory
check. Do not add pytest workers for this focused run.

For runtime diagnosis, the command also enables synchronous PyTorch execution
with `ASCEND_LAUNCH_BLOCKING=1`; see the official
[environment variable reference](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/apiref/envvar/envref_07_0050.html).
Keep this diagnostic setting out of performance measurements. The test already
explicitly synchronizes after each Triton launch.

An `ACL stream synchronize failed, error code:507035` with `vector core exception`
is a runtime failure, not by itself a sanitizer memory finding. When later cases
fail at `source.npu()` before their kernel launches, an earlier device exception
in the same process may be causing those failures. Exit that test process and
rerun with `-x` in a fresh process to retain the first failure. Stage markers from
`pytest -s` distinguish input copying, compile/launch, synchronization, and
successful verification for each rule mask. A new process is a diagnostic step,
not a guarantee of recovering a device that remains unhealthy.

If a compiler upgrade appears ineffective, inspect the executable selected by
the backend, which prefers its bundled bishengir over the shell's PATH:

```bash
python - <<'PY'
import subprocess
from triton.backends.ascend.utils import _get_npucompiler_path

compiler, compiler_env = _get_npucompiler_path()
print("Selected compiler:", compiler, flush=True)
subprocess.run([compiler, "--version"], env=compiler_env, check=True)
PY
```

On failure, retain the first compiler/runtime error and the rule mask from the
pytest traceback. Each case executes mask 511 before mask 1023; a failure at 511
already occurs with gather disabled and cannot by itself establish a gather
rewrite bug. Unknown compiler options, missing libraries, or checker startup
errors need toolchain/runtime diagnosis. An `illegal read`/`illegal write` report
needs address and generated-code analysis even when numerical tests pass. Do not
assume either category from the bishengir version alone.

For a hard abort inside the first launch, use the standalone probe to select one
kernel and one rule mask per process. It reuses the regression's indirect kernel,
compiles with JIT warmup, saves the available IR/binary artifacts before launch,
and verifies the selected rule's TTIR activation and numerical output. It does
not replace the paired 21-case correctness suite.

```bash
GATHER_PROBE=third_party/ascend/unittest/pytest_ut/gather_sanitizer_probe.py
set -o pipefail
gather_probe() {
  local probe_case="$1"
  local probe_mask="$2"
  local probe_dir="gather-v3-test-results/probe-${probe_case}-${probe_mask}"
  mkdir -p "$probe_dir"
  TRITON_ENABLE_SANITIZER=1 \
  TRITON_ALWAYS_COMPILE=1 \
  PYTORCH_NO_NPU_MEMORY_CACHING=1 \
  ASCEND_LAUNCH_BLOCKING=1 \
  mssanitizer --tool=memcheck -- python "$GATHER_PROBE" \
    --case "$probe_case" --rule-mask "$probe_mask" --output-dir "$probe_dir" \
    2>&1 | tee "$probe_dir/console.log"
}
gather_probe copy 0
```

Run `gather_probe tail 511` only after the simple contiguous copy passes without
checker findings, then `gather_probe tail 1023` after the disabled-gather tail
passes. If the tail fails with 511, compare `gather_probe full 511` (eight valid
rows, no masked tail) and `gather_probe tail 0` (all graph rules disabled), each in
a fresh invocation. Stop the failing process before the next diagnostic. A
failure with gather disabled is not evidence against the gather rewrite; if only
1023 fails, focus on the rewrite and its interaction with instrumentation.

The reported September 11 failure started msSanitizer from the CANN 9.0.0 tools
directory, registered one `indirect_rows_kernel`, and then aborted with an MPU
invalid-address exception at PC offset `0x56dc` from the reported kernel start.
The `No error detected` checker line cannot be treated as a pass when the device
subsequently faults and the process exits by signal 6. The supplied tool log has
host allocation/copy records and kernel-finish records, but no detailed kernel
memory-access report; it does not locate the offending source operation. Preserve
the matching binary and full tool/driver logs to map the device PC. The original
paired test runs 511 first, so the single-launch log points toward that baseline,
but the probe's explicit rule-mask output and TTIR are needed to confirm it.

After the focused cases pass, run the existing graph/layout regression suites:

```bash
python -m pytest -q third_party/ascend/unittest/pytest_ut/test_graph_optimize.py
cmake --build "$TRITON_BUILD_DIR" --target check-triton-ascend-lit-tests
```

## Performance follow-up

Compare `graph_optimize_rule_mask=511` (gather disabled) against `1023` (enabled)
with the same target, compile mode, tiling, inputs, and build. Record rule hits,
latencies, compiler/driver versions, and actual UB use. The mask-preserving IR and
source-address construction have changed, so the v2 empirical UB coefficients
and benefit tables require device validation. The existing global/liveness UB
model limitations and iteration-count heuristic are not resolved by this patch.
