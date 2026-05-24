# Codex Goal: Continuous Hash Throughput Optimization

## Goal Objective

Continuously optimize XenblocksMiner Hash API and CUDA hashing throughput for the real mining workload:

- `t = 1` is fixed.
- `p = 1` / `s = 1` is fixed.
- `m = difficulty` / `diff` is the only regularly changing hash parameter.

Minimize time per valid hash attempt and maximize warm steady-state attempts per second while preserving exact `argon2id-xen` semantics.

The aspirational target is at least a 1000% throughput improvement over the initial measured baseline. Keep iterating after that target only while low-risk, measurable improvements remain. Stop only when repeated benchmark and profiling evidence show a plateau or practical hardware limit.

Use `docs/HASH_OPTIMIZATION_GOAL.md` as the full operating brief. This file is the short `/goal` entrypoint that should survive context compaction and resume events.

## Run Mode

Operate autonomously. Do not ask for approval for routine local optimization work:

- inspect source files, docs, diffs, logs, and git status
- run tests, builds, local benchmark scripts, and CUDA smoke checks
- create ignored benchmark artifacts under `.benchmarks/`
- edit scoped source, tests, scripts, build files, and docs
- make small validated commits with English commit messages

Pause only for the blocker conditions in `Stop Conditions`.

## Non-Negotiable Constraints

Use English for all code, comments, tests, docs, benchmark names, API names, branch names, and commit messages.

Never commit:

- local absolute paths
- usernames or hostnames
- private machine identifiers
- secrets, tokens, cookies, private keys, wallet private data, or personal addresses
- raw benchmark reports containing binary paths, command lines, or local hardware details
- local GPU model names when they identify a private machine

Use public-safe placeholders in docs and commit messages:

- `<miner-binary>`
- `<build-dir>`
- `<cuda-root>`
- `<vcpkg-toolchain>`
- `CUDA-capable local GPU`
- `RTX 3050-class GPU`
- `higher-end CUDA GPU`

Before every commit, inspect the staged diff for privacy leaks and keep raw benchmark output ignored.

## Correctness Boundary

Optimization must never change accepted hash behavior.

Do not:

- replace the hash with a different algorithm
- skip required Argon2 work
- approximate results
- fake successful matches
- weaken target matching
- change key, salt, difficulty, or result semantics without an explicit design decision

Every performance change must include a correctness check that exercises the changed path before trusting benchmark data.

## Architecture Direction

Keep the hash engine cleanly separated from platform features.

Target architecture:

- Hash API code stays centered under `src/hashapi/`.
- CPU/reference and CUDA backends share the same request/result contract.
- `hash-one`, `hash-batch`, and `hash-benchmark` remain stable automation entrypoints.
- Benchmark output remains machine-readable and comparable across commits.
- GPU tuning parameters are explicit, measurable, and isolated from marketplace, wallet, frontend, lease, devfee, reporting, and other platform code.
- Runtime tuning depends on difficulty, batch shape, compute capability, and public device properties rather than private local hardware names.
- The miner may consume Hash API tuning results, but explicit user-supplied batch-size or device settings must continue to win.

If the current structure blocks serious optimization, refactor the Hash API/backend boundary first, then optimize the hot path.

## Current Status

The reusable Hash API extraction is already in place. Continue from that architecture instead of reworking the full platform first.

Current expected capabilities:

- `src/hashapi/` contains the reusable Hash API boundary.
- CPU/reference and CUDA backends are available behind a shared request/result contract.
- CLI entrypoints such as `hash-one`, `hash-batch`, and `hash-benchmark` are available for automation.
- Benchmark tooling supports scenarios, presets, warm-up runs, repeats, JSON output, comparison, and batch scans.
- Conservative CUDA batch tuning is wired into miner integration when no explicit batch size is provided.

Current optimization focus:

- Optimize the current local CUDA-capable GPU first.
- Keep the tuning design ready for RTX 3050-class and higher-end CUDA GPUs.
- Prefer benchmark-backed improvements in input preparation, setup reuse, allocation reuse, matching/finalization, batch sizing, and launch configuration before risky kernel rewrites.

## Continuous Iteration Loop

Repeat until the completion criteria are met:

1. Run `git status -sb`.
2. Read `docs/HASH_OPTIMIZATION_GOAL.md` after resume or context compaction.
3. Identify the latest benchmark-related commit and local baseline.
4. Run focused correctness tests before performance work.
5. Run a short CUDA smoke benchmark if a CUDA binary is available.
6. Pick one measurable bottleneck from timing metadata or benchmark evidence.
7. Make the smallest useful source, harness, test, or documentation change.
8. Re-run correctness validation.
9. Benchmark before and after with identical settings.
10. Compare median warm throughput first, then inspect min/max spread and timing breakdowns.
11. Commit only if the slice is correct, useful, and privacy-clean.
12. If an experiment fails, revert only the current uncommitted experiment, record useful public-safe evidence when it prevents repeated work, and continue.

Prefer many small, measurable iterations over broad speculative rewrites.

## Current Work Queue

Start or resume with this queue unless newer evidence in `docs/HASH_OPTIMIZATION_GOAL.md` supersedes it:

1. Finish and validate any existing dirty Hash API benchmark or documentation slice.
2. Run the focused Hash API unit tests.
3. Build the available smoke CLI or full CUDA binary.
4. Run a short CUDA smoke benchmark.
5. Run or load a repeated baseline for the next bottleneck.
6. Inspect timing fields and choose the next target:
   - high `input_ms`: reduce generated-key construction, salt/key preparation, or first-block preparation overhead
   - high `keygen_ms`: optimize key generation and prefix handling
   - high `first_block_ms`: improve safe Argon2 first-block preparation
   - high `setup_ms`: cache difficulty-derived or device-derived setup safely
   - high `compute_ms`: inspect CUDA allocation, transfers, launch geometry, occupancy, and memory behavior
   - high `finalize_ms`: reduce matching, encoding, result collection, or JSON work
7. Use main-target-only benchmarks when measuring normal mining throughput without secondary XUNI matching.
8. Consider batch-size selection and autotuning only with stable repeated benchmark evidence.
9. Keep miner integration aligned with Hash API tuning without breaking explicit user-supplied overrides.

## Measurement Rules

Use machine-readable benchmark output as the source of truth.

Smoke checks:

- seconds: `1` to `3`
- warm-up: at least `1`
- repeat: at least `1` or `2`
- purpose: catch obvious regressions and explore candidates
- do not use smoke-only data for committed performance claims

Serious comparisons:

- seconds: at least `10`
- warm-up: at least `1`
- repeat: at least `3`
- same binary type, backend, device index, difficulty, batch size, salt/key mode, and seconds before and after
- compare median warm throughput first
- rerun if the claimed gain is within normal benchmark noise

Standard commands, using local values only in the shell and never in committed docs:

```bash
python -m pytest tests/unit/test_hash_api_contract.py tests/unit/test_hash_api_service.py tests/unit/test_hash_api_benchmark.py tests/unit/test_hash_api_compare.py -q
```

```bash
python scripts/hash_api_benchmark.py --binary <miner-binary> --backend cuda --device 0 --preset warm-short --seconds 2 --warmup 1 --repeat 3 --no-xuni --output .benchmarks/warm-short-main-target.json
```

```bash
python scripts/hash_api_benchmark.py --binary <miner-binary> --backend cuda --device 0 --seconds 10 --warmup 1 --repeat 3 --no-xuni --scan-difficulty 1 --scan-difficulty 8 --scan-difficulty 64 --scan-batch-size 256 --scan-batch-size 512 --scan-batch-size 1024 --scan-batch-size 2048 --recommendations-only --output .benchmarks/batch-scan-stable-main-target.json
```

## Commit Discipline

Use small, coherent commits. Good prefixes include:

- `perf(hash-api):`
- `perf(cuda):`
- `refactor(hash-api):`
- `refactor(cuda):`
- `test(hash-api):`
- `test(cuda):`
- `docs(hash-api):`

Before each commit:

1. Run `git diff --stat`.
2. Review staged files.
3. Run relevant correctness validation.
4. Run at least one relevant benchmark for performance-affecting changes.
5. Confirm the staged diff contains no private paths, usernames, hostnames, secrets, raw benchmark reports, or local hardware identifiers.

Useful privacy check before committing:

```bash
git diff --cached --check
git diff --cached
```

Then manually inspect the staged diff.

## Stop Conditions

Stop and ask the user only if:

- a dirty user change conflicts with required edits
- a command requires credentials or unavailable proprietary software
- a design choice would permanently break the public Hash API contract
- an optimization requires changing hash semantics
- a CUDA change appears hardware-specific and risky without access to that hardware
- tests reveal a pre-existing issue whose fix would significantly broaden scope

Otherwise, keep moving through benchmark, optimize, validate, and commit cycles.

## Completion Criteria

This goal is complete only when one of these is true:

- throughput improves by at least 1000% over the initial measured baseline while preserving correctness and no obvious low-risk improvements remain
- repeated well-scoped attempts plateau and the remaining bottleneck is documented
- profiler evidence shows the implementation is near the practical hardware limit for the tested GPU class

Until then, continue iterating.
