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
