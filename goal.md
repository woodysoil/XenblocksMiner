# Codex Goal: Continuous Hash Throughput Optimization

## Objective

Continuously optimize XenblocksMiner Hash API and CUDA hashing throughput for the real mining workload until the implementation reaches a documented plateau or a practical hardware limit.

Fixed workload:

- `t = 1`
- `p = 1` / `s = 1`
- `m = difficulty` / `diff`, which may change between benchmark or mining sessions

Primary target: minimize milliseconds per valid hash attempt and maximize warm steady-state attempts per second while preserving exact `argon2id-xen` semantics.

Aspirational target: improve throughput by at least 1000% over the initial measured baseline. This target does not permit weaker correctness, synthetic successes, skipped work, or algorithm changes.

Use `docs/HASH_OPTIMIZATION_GOAL.md` as the full operating brief. This file is the compact `/goal` entrypoint for long-running autonomous execution and resume after context compaction.

## Autonomous Run Mode

Operate without asking for approval for normal local work:

- inspect source files, docs, git status, diffs, and logs
- run tests, builds, local benchmark scripts, CUDA smoke checks, and comparison tools
- create ignored benchmark artifacts under `.benchmarks/` or `benchmark-results/`
- edit scoped source, tests, scripts, build files, and documentation
- make small validated commits with English commit messages

Pause only for the stop conditions near the end of this file. Do not pause just because a benchmark, build, or test takes time.

## Non-Negotiable Rules

Use English for code, comments, docs, tests, benchmark names, API names, branch names, and commit messages.

Never commit:

- local absolute paths
- usernames, hostnames, or private machine identifiers
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

Before every commit, inspect the staged diff for privacy leaks. Keep raw benchmark output ignored unless a report is intentionally sanitized for public use.

## Correctness Boundary

Optimization must preserve accepted hash behavior exactly.

Do not:

- replace the hash with a different algorithm
- skip required Argon2 work
- approximate results
- fake successful matches
- weaken target matching
- change key, salt, difficulty, or result semantics without an explicit design decision

Every performance change must include a correctness check that exercises the changed path before trusting benchmark data.

## Architecture Direction

Keep the hash engine clean and reusable:

- keep Hash API code centered under `src/hashapi/`
- keep CPU/reference and CUDA backends behind the same request/result contract
- keep `hash-one`, `hash-batch`, and `hash-benchmark` as stable automation entrypoints
- keep benchmark output machine-readable and comparable across commits
- isolate GPU tuning from marketplace, wallet, frontend, lease, devfee, reporting, and platform services
- base tuning on difficulty, batch shape, compute capability, and public device properties instead of private local hardware names
- preserve explicit user-supplied batch-size and device settings over automatic tuning

If the current structure blocks serious optimization, refactor the Hash API/backend boundary first, then optimize the hot path.

## Current State

The reusable Hash API extraction already exists. Continue from that architecture rather than reworking platform features first.

Expected capabilities:

- `src/hashapi/` contains the reusable Hash API boundary.
- CPU/reference and CUDA backends share a request/result contract.
- CLI automation entrypoints include `hash-one`, `hash-batch`, and `hash-benchmark`.
- Benchmark tooling supports scenarios, presets, warm-up runs, repeats, JSON output, comparison, recommendations, and batch scans.
- Conservative CUDA batch tuning is wired into miner integration when no explicit batch size is provided.

Current focus:

- optimize the current CUDA-capable local GPU first
- keep the design ready for RTX 3050-class and higher-end CUDA GPUs
- prefer benchmark-backed improvements in input preparation, setup reuse, allocation reuse, matching/finalization, batch sizing, and launch configuration before risky kernel rewrites
- specifically measure and optimize behavior when `m = difficulty` changes, not only repeated same-difficulty loops

## Continuous Iteration Loop

Repeat until the completion criteria are met:

1. Run `git status -sb`.
2. Read `docs/HASH_OPTIMIZATION_GOAL.md` after resume or context compaction.
3. Identify the latest benchmark-related commit and the current local baseline.
4. Run focused correctness tests before performance work.
5. Run a short CUDA smoke benchmark when a CUDA binary is available.
6. Pick one measurable bottleneck from timing metadata or benchmark evidence.
7. Make the smallest useful source, harness, test, or documentation change.
8. Re-run correctness validation.
9. Benchmark before and after with identical settings.
10. Compare median warm throughput first, then inspect min/max spread and timing breakdowns.
11. Commit only if the slice is correct, useful, and privacy-clean.
12. If an experiment fails, revert only the current uncommitted experiment, record useful public-safe evidence when it prevents repeated work, and continue.

Prefer many small, measurable iterations over broad speculative rewrites.

## Immediate Work Queue

Start or resume with this queue unless newer evidence in `docs/HASH_OPTIMIZATION_GOAL.md` supersedes it:

1. Verify the worktree is clean or identify unrelated dirty files.
2. Run the focused Hash API unit tests.
3. Build the available smoke CLI or full CUDA binary.
4. Run a short CUDA smoke benchmark.
5. Run or load a repeated baseline for the next bottleneck.
6. Add or improve benchmark coverage for variable `m = difficulty` behavior, such as same-difficulty versus alternating-difficulty warm loops.
7. Inspect timing fields and choose the next target:
   - high `input_ms`: reduce generated-key construction, salt/key preparation, or first-block preparation overhead
   - high `keygen_ms`: optimize key generation and prefix handling
   - high `first_block_ms`: improve safe Argon2 first-block preparation
   - high `setup_ms`: cache difficulty-derived or device-derived setup safely
   - high `compute_ms`: inspect CUDA allocation, transfers, launch geometry, occupancy, and memory behavior
   - high `finalize_ms`: reduce matching, encoding, result collection, or JSON work
8. Use main-target-only benchmarks when measuring normal mining throughput without secondary XUNI matching.
9. Change defaults or autotuning only with stable repeated benchmark evidence.
10. Keep miner integration aligned with Hash API tuning without breaking explicit user overrides.

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

Focused tests:

```bash
python -m pytest tests/unit/test_hash_api_contract.py tests/unit/test_hash_api_service.py tests/unit/test_hash_api_benchmark.py tests/unit/test_hash_api_compare.py -q
```

Short main-target CUDA benchmark:

```bash
python scripts/hash_api_benchmark.py --binary <miner-binary> --backend cuda --device 0 --preset warm-short --seconds 2 --warmup 1 --repeat 3 --no-xuni --output .benchmarks/warm-short-main-target.json
```

Stable main-target batch scan:

```bash
python scripts/hash_api_benchmark.py --binary <miner-binary> --backend cuda --device 0 --seconds 10 --warmup 1 --repeat 3 --no-xuni --scan-difficulty 1 --scan-difficulty 8 --scan-difficulty 64 --scan-batch-size 256 --scan-batch-size 512 --scan-batch-size 1024 --scan-batch-size 2048 --recommendations-only --output .benchmarks/batch-scan-stable-main-target.json
```

Before/after comparison:

```bash
python scripts/hash_api_compare.py .benchmarks/before.json .benchmarks/after.json --fail-on-regression --min-change-pct 1
```

Use local concrete paths only in shell commands. Never commit those paths.

## Benchmark Backlog

Work through this backlog before high-risk kernel rewrites:

1. Maintain a current local CUDA baseline under ignored benchmark output.
2. Run stable scans for common difficulty ranges.
3. Measure same-`m` warm loops versus alternating-`m` warm loops.
4. Reduce CPU-side generated-key and first-block preparation overhead where `input_ms` dominates.
5. Cache difficulty-derived setup only when correctness can be proven across salt, key mode, batch shape, backend state, and device state.
6. Reduce per-batch allocation and repeated normalization inside Hash API CUDA paths.
7. Measure CUDA allocation, copy, launch, and finalization overhead before rewriting kernel logic.
8. Extend batch-size tuning toward runtime autotuning after stable cross-difficulty data exists.
9. Tune launch parameters only after CPU-side overhead is under control.
10. Add profiler-backed CUDA kernel work only when benchmark timing shows compute is the dominant bottleneck.

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

Useful privacy check:

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
