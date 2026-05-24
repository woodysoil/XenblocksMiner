# Codex Goal: Continuous Hash Throughput Optimization

## Goal Command

Use this exact command when starting or recreating the long-running goal:

```text
/goal Follow goal.md and docs/HASH_OPTIMIZATION_GOAL.md. Continuously optimize XenblocksMiner Hash API CUDA hashing throughput for fixed t=1, fixed p/s=1, and variable m=difficulty until the best verified warm steady-state rate is at least 1000% over the recorded baseline, or until evidence-backed plateau/practical-limit criteria are met. Verify progress with machine-readable Hash API benchmarks, golden hash checks, focused tests, and small validated commits. Preserve exact argon2id-xen semantics and the public Hash API contract. Work autonomously through benchmark, optimize, validate, document, and commit cycles without asking for approval except for the listed stop conditions. Keep all code, docs, tests, benchmark names, and commit messages in English. Never commit local paths, private machine details, raw benchmark reports, secrets, wallet private data, or local hardware identifiers.
```

This file is the compact persistent goal contract. `docs/HASH_OPTIMIZATION_GOAL.md` is the detailed operating manual, phase plan, and experiment ledger.

The goal is thread-scoped and evidence-based. Do not mark it complete because one iteration finished, the context is long, a benchmark is noisy, or the next step is uncertain. Completion requires concrete evidence from tests, benchmark output, changed files, and the documented completion rule.

## Goal Contract

This goal follows the strongest Codex Goal shape:

- Outcome: maximize real Hash API CUDA hashing throughput, with an aspirational 1000% improvement target, then continue until plateau or practical hardware-limit evidence is documented.
- Verification surface: focused Hash API tests, CUDA golden hash checks, machine-readable benchmark JSON, before/after comparison output, and committed diffs.
- Constraints: preserve exact `argon2id-xen` semantics, Hash API compatibility, mining result fields, target matching, privacy rules, and public-repo hygiene.
- Boundaries: focus on `src/hashapi/`, CUDA backend files, Argon2/Blake2b hot paths, benchmark scripts, tests, and narrowly related integration code.
- Iteration policy: choose the next experiment from measured timing bottlenecks, validate correctness before trusting speed, commit useful stable slices, and revert or document rejected experiments.
- Blocked stop condition: stop only when a listed stop condition is reached, no defensible optimization path remains, required tooling is unavailable, or public history rewrite decisions need the user.

## Outcome

Optimize the extracted Hash API and CUDA hashing path until one of these is true:

- throughput improves by at least 1000% over the initial measured baseline while correctness is preserved and no obvious low-risk improvements remain
- repeated well-scoped attempts plateau and the remaining bottleneck is documented with benchmark or profiler evidence
- profiler evidence shows the implementation is near the practical hardware limit for the tested GPU class

Until one of those outcomes is proven, keep iterating.

## Progress Accounting

Maintain progress against a named baseline rather than against memory or terminal prose.

- Baseline: the earliest trustworthy machine-readable CUDA benchmark for the selected scenario after the Hash API extraction, or a new documented baseline if no trustworthy report exists.
- Best result: the highest confirmed median warm throughput for the same scenario and correctness surface.
- Improvement: `(best_median_hps - baseline_median_hps) / baseline_median_hps * 100`.
- Main scenario for continuity: generated-key CUDA, main-target-only, difficulty `8`, batch size `2048`, warm-up `1`, repeat `3`, with the same seconds value for before/after comparisons.
- Supplemental scenarios: difficulty `1`, `64`, `256`, and `1024`, variable-difficulty sequences, and batch-size scans when the main scenario no longer explains the bottleneck.

Do not claim the 1000% target from a single noisy run. Confirm large claims with repeated runs or a stable scan, and keep the raw report ignored unless a sanitized summary is intentionally committed.

Plateau evidence requires at least three consecutive well-scoped optimization attempts against the current dominant bottleneck with less than 3% confirmed improvement, plus a short note in `docs/HASH_OPTIMIZATION_GOAL.md` explaining the remaining bottleneck and why risk is now higher than expected gain.

## Fixed Workload

The optimization target is the real mining hash workload:

- `t = 1` is fixed
- `p = 1` / `s = 1` is fixed
- `m = difficulty` / `diff` is variable and may change between benchmark or mining sessions
- salt, key, prefix, difficulty, matching, and result semantics must stay compatible with the current Hash API contract

The primary metric is warm steady-state CUDA attempts per second. The secondary metric is milliseconds per valid hash attempt, especially for generated-key mining batches.

## Non-Negotiable Correctness Rules

Do not optimize by changing the meaning of the work.

Never:

- replace `argon2id-xen` with another algorithm
- skip required Argon2 work
- approximate hashes
- fake successful matches
- weaken target matching
- change salt, key, prefix, difficulty, or attempt-index semantics without an explicit compatibility-preserving design

Every performance change must be validated on the path it changes before benchmark data can be trusted.

## Current Architecture Status

The Hash API extraction is already in place and should remain the center of the work.

Expected current shape:

- reusable Hash API code under `src/hashapi/`
- CPU/reference and CUDA backends behind a shared request/result contract
- CLI automation entrypoints: `hash-one`, `hash-batch`, and `hash-benchmark`
- miner integration using Hash API batch compute paths
- benchmark tooling with presets, scenarios, warm-up, repeats, JSON output, comparison, recommendations, batch scans, and variable-difficulty sequences
- timing metadata that separates validation, setup, input generation, compute, finalization, matching, and per-attempt costs

If the current structure blocks optimization, refactor the Hash API/backend boundary first. Do not drift into frontend, marketplace, wallet, lease, devfee, authentication, or broad platform work while this goal is active.

## Per-Iteration Evidence

Every completed iteration should leave enough evidence for the next agent to continue without guessing:

- current commit or dirty state
- benchmark scenario name
- backend, difficulty, batch size, seconds, warm-up count, repeat count, and XUNI setting
- before and after median warm throughput when comparing performance code
- min/max spread or a clear note that the run is smoke-only
- dominant timing field from benchmark metadata
- correctness commands that passed
- conclusion: accepted, rejected, or measurement-only

Keep raw reports in ignored local artifact directories. Commit concise public-safe summaries only when they explain a decision or prevent future repeated work.

## Resume Behavior

On every resume, context compaction, or new `/goal` run:

1. Treat `goal.md` as the entrypoint and `docs/HASH_OPTIMIZATION_GOAL.md` as the authoritative detailed plan.
2. Run `git status -sb`.
3. Read the latest benchmark/optimization commits.
4. Check whether there is a dirty experiment from a previous agent.
5. If the dirty experiment is known rejected and belongs to the current goal, revert only that experiment.
6. If the dirty change appears user-authored or unrelated, leave it alone.
7. Load or recreate the latest trustworthy baseline before editing performance-sensitive code.
8. Continue with the next smallest measurable step.

## Hardware Direction

Optimize on the current CUDA-capable local GPU first. Keep the design ready for RTX 3050-class and higher-end CUDA GPUs later.

Do not hard-code tuning to a private local device name. Prefer tuning decisions based on:

- difficulty
- batch size
- key mode
- same-difficulty versus variable-difficulty runs
- public CUDA device properties
- compute capability
- available memory
- measured stability

Preserve explicit user-supplied device and batch-size settings over automatic tuning.

## Autonomous Execution

Work without asking for approval for normal local development:

- inspect files, git state, diffs, and logs
- run tests
- run builds
- run CUDA smoke checks
- run local benchmark scripts
- create ignored benchmark artifacts under `.benchmarks/` or `benchmark-results/`
- edit scoped source, tests, scripts, build files, and documentation
- make small validated commits

Pause only for the stop conditions in this file. Do not pause just because a build, test, or benchmark takes time.

## Privacy And Public History

This is a public open-source repository. Keep docs and git history clean.

Never commit:

- local absolute paths
- usernames
- hostnames
- private machine identifiers
- secrets, tokens, cookies, private keys, wallet private data, or personal addresses
- raw benchmark reports containing command lines, binary paths, hardware identifiers, or local environment details
- local GPU model names when they identify a private machine

Use public-safe placeholders in docs and commit messages:

- `<miner-binary>`
- `<build-dir>`
- `<cuda-root>`
- `<vcpkg-toolchain>`
- `CUDA-capable local GPU`
- `RTX 3050-class GPU`
- `higher-end CUDA GPU`

Before every commit, inspect the staged diff for privacy leaks. If a leak appears in an unpushed local commit, fix the local history before continuing. If a leak has already been shared publicly, stop and ask before rewriting public history.

## Iteration Loop

Repeat this loop:

1. Run `git status -sb`.
2. Read this file and `docs/HASH_OPTIMIZATION_GOAL.md` after resume, compaction, or uncertainty.
3. Review recent benchmark and optimization commits.
4. Identify the latest usable baseline from ignored local benchmark output or run a fresh baseline.
5. Run focused correctness tests before editing performance-sensitive code.
6. Choose one measurable bottleneck from timing metadata.
7. Make the smallest useful source, benchmark harness, test, or documentation change.
8. Re-run correctness validation.
9. Benchmark before and after with identical settings.
10. Compare median warm throughput first, then min/max spread, per-attempt timings, and dominant timing stage.
11. Keep and commit correct useful improvements.
12. Revert only the current uncommitted experiment if it fails.
13. Document rejected experiments when the evidence prevents future repeated work.
14. Continue with the next bottleneck.

Prefer many small measured iterations over broad speculative rewrites.

## Bottleneck Order

Use timing evidence, but start with these likely targets:

- high `input_ms`: generated-key construction, salt/key preparation, first-block preparation
- high `keygen_ms`: random key generation, prefix handling, generated-key memory layout
- high `first_block_ms`: safe Argon2 first-block preparation and CPU parallelism
- high `setup_ms`: difficulty-derived setup, backend lifecycle, validation, device selection
- high `compute_ms`: CUDA allocation, transfers, launch geometry, memory behavior, occupancy, kernel timing
- high `finalize_ms`: hash finalization, base64 encoding, target matching, result collection, JSON work outside the hot path

Do not start risky CUDA kernel rewrites until timing data shows CPU-side preparation, setup, allocation, matching, and finalization are no longer dominant.

## Measurement Gates

Use machine-readable benchmark output as the source of truth.

Smoke checks:

- seconds: `1` to `3`
- warm-up: at least `1`
- repeat: at least `1` or `2`
- purpose: prove the binary works and catch obvious regressions
- do not use smoke-only data for committed performance claims

Serious comparisons:

- seconds: at least `10`
- warm-up: at least `1`
- repeat: at least `3`
- same binary type, backend, device index, difficulty, batch size, salt/key mode, XUNI setting, and seconds before and after
- rerun when the claimed gain is inside normal benchmark noise
- treat improvements as local evidence unless confirmed across more devices

Accept a performance code change only when:

- correctness checks pass
- before/after benchmark settings match
- median warm throughput improves materially or the change enables better future measurement
- spread is low enough to trust the conclusion, or the uncertainty is explicitly documented
- no private paths, hardware identifiers, or raw local reports are staged

Reject or revert an experiment when:

- correctness changes
- subprocesses become unstable
- benchmark output becomes malformed
- median warm throughput regresses without a compelling architecture reason
- the claimed improvement is indistinguishable from noise after confirmation

## Standard Validation Commands

Commands in this file use public-safe placeholders. Use concrete local paths only in the shell, not in committed files.

Focused Hash API tests:

```bash
python -m pytest tests/unit/test_hash_api_contract.py tests/unit/test_hash_api_service.py tests/unit/test_hash_api_benchmark.py tests/unit/test_hash_api_compare.py -q
```

Golden CUDA hash check:

```bash
<miner-binary> hash-one --salt aabbccddeeff0011 --key 0000000000000000000000000000000000000000000000000000000000000000 --backend cuda --device 0 --difficulty 8 --json
```

Expected `hash`:

```text
Rs/bYUkZR8dczsQh/KvLAyJGThm8HtjnIJVJEkldK+TQtBLdGf2tULquitejKRO7URrkbgieR7Sq42k5mNYVdw
```

Short main-target CUDA smoke:

```bash
python scripts/hash_api_benchmark.py --binary <miner-binary> --backend cuda --device 0 --preset warm-short --seconds 2 --warmup 1 --repeat 3 --no-xuni --output .benchmarks/warm-short-main-target.json
```

Variable-difficulty CUDA smoke:

```bash
python scripts/hash_api_benchmark.py --binary <miner-binary> --backend cuda --device 0 --preset difficulty-sequence --seconds 2 --warmup 1 --repeat 3 --no-xuni --output .benchmarks/difficulty-sequence-main-target.json
```

Stable main-target scan:

```bash
python scripts/hash_api_benchmark.py --binary <miner-binary> --backend cuda --device 0 --seconds 10 --warmup 1 --repeat 3 --no-xuni --scan-difficulty 1 --scan-difficulty 8 --scan-difficulty 64 --scan-batch-size 256 --scan-batch-size 512 --scan-batch-size 1024 --scan-batch-size 2048 --recommendations-only --output .benchmarks/batch-scan-stable-main-target.json
```

Before/after comparison:

```bash
python scripts/hash_api_compare.py .benchmarks/before.json .benchmarks/after.json --fail-on-regression --min-change-pct 1
```

## Immediate Queue

Start here unless `docs/HASH_OPTIMIZATION_GOAL.md` contains newer evidence:

1. Verify `git status -sb`.
2. If `src/argon2params.cpp` contains only the rejected direct hex parser experiment, revert it and do not commit it.
3. Confirm docs and recent commits contain no local paths or private machine details.
4. Run focused Hash API unit tests.
5. Build the available smoke CLI or full CUDA binary.
6. Run the golden CUDA hash check when a CUDA binary is available.
7. Run a short main-target CUDA benchmark.
8. Run or load a repeated d8/b2048 baseline because recent useful evidence used that scenario.
9. Inspect timing metadata and pick the next bottleneck.
10. Prefer input preparation and setup/measurement improvements before speculative finalization micro-optimizations.
11. Keep `docs/HASH_OPTIMIZATION_GOAL.md` updated with accepted and rejected experiments.

Known accepted and rejected experiments are documented in `docs/HASH_OPTIMIZATION_GOAL.md`. Do not retry rejected experiments unless the implementation shape has materially changed and the new attempt includes correctness checks.

## Benchmark Artifact Policy

Keep raw benchmark output ignored:

- `.benchmarks/`
- `benchmark-results/`

Commit only public-safe summaries when useful. A committed summary should include:

- scenario name
- backend
- difficulty
- batch size
- seconds
- warm-up count
- repeat count
- median before/after hashrate
- percentage change
- dominant timing field
- conclusion

Do not commit full raw JSON reports unless they are intentionally sanitized and useful to future contributors.

## Commit Discipline

Use English commit messages. Good prefixes:

- `perf(hash-api):`
- `perf(cuda):`
- `refactor(hash-api):`
- `refactor(cuda):`
- `test(hash-api):`
- `test(cuda):`
- `docs(hash-api):`

Before each commit:

1. Run relevant validation.
2. Review `git diff --stat`.
3. Stage only intended files.
4. Run `git diff --cached --check`.
5. Inspect the staged diff for private paths, usernames, hostnames, secrets, wallet data, raw reports, and local hardware identifiers.
6. Commit the smallest coherent slice.

## Stop Conditions

Stop and ask the user only if:

- a dirty user change conflicts with required edits
- a command requires credentials or unavailable proprietary software
- a design choice would permanently break the public Hash API contract
- an optimization requires changing hash semantics
- a CUDA change appears hardware-specific and risky without access to that hardware class
- tests reveal a pre-existing issue whose fix would significantly broaden scope
- public history rewrite is needed for commits that may already have been shared

Otherwise, keep moving through benchmark, optimize, validate, and commit cycles.

## Completion Rule

Do not mark this goal complete just because the current context is long, a benchmark is noisy, or one optimization is committed.

Completion requires one of the documented outcomes at the top of this file. Until then, continue iterating.
