# Long-Running Goal: Hash Throughput Optimization

## Active Objective

Continuously execute `docs/HASH_OPTIMIZATION_GOAL.md` as the authoritative operating brief.

Optimize XenblocksMiner Hash API CUDA hashing throughput for the fixed mining workload:

- `t = 1`
- `s = 1` / `p = 1`
- `m = diff` / `difficulty`, which may change between runs

Primary target: minimize time per valid hash attempt and maximize warm steady-state hashrate while preserving exact `argon2id-xen` semantics. The aspirational performance target is at least a 1000% throughput gain over the measured baseline, or a documented plateau backed by repeated benchmarks and profiling evidence.

## Autonomous Execution Contract

Keep working without asking for approval for routine local development tasks:

- inspect git status, diffs, logs, and source files
- run tests, builds, and local benchmark scripts
- create ignored benchmark artifacts under `.benchmarks/`
- make scoped source, test, benchmark, and documentation edits
- commit small validated slices with English commit messages

Stop only for the blocker conditions in `docs/HASH_OPTIMIZATION_GOAL.md`, such as a user-change conflict, a required credential, a required semantic hash change, or a risky public API break.

## Required Language And Privacy Rules

Use English for code, comments, docs, tests, benchmark scenario names, API names, branch names, and commit messages.

Never commit:

- local absolute paths
- usernames or hostnames
- private machine identifiers
- secrets, tokens, cookies, private keys, or wallet private data
- raw benchmark dumps containing binary paths or command lines
- local GPU model names when they identify a private machine

Use public-safe placeholders in docs and commit bodies:

- `<miner-binary>`
- `<build-dir>`
- `<cuda-root>`
- `<vcpkg-toolchain>`
- `CUDA-capable local GPU`
- `RTX 3050-class GPU`
- `higher-end CUDA GPU`

Before every commit, review the staged diff for privacy leaks and keep raw benchmark reports ignored.

## Core Architecture Direction

Keep the hash engine cleanly separated from platform features.

The long-term shape should make it easy for future agents and developers to optimize or replace hash backends without touching marketplace, wallet, frontend, lease, devfee, or reporting code.

Preserve these boundaries:

- Hash API source stays centered under `src/hashapi/`.
- CPU/reference and CUDA backends share the same request/result contract.
- `hash-one`, `hash-batch`, and `hash-benchmark` remain stable automation entrypoints.
- Benchmark output remains machine-readable and comparable across commits.
- GPU tuning parameters are explicit, measurable, and isolated from business logic.
- Runtime tuning should depend on difficulty, batch shape, and device capability rather than private local hardware names.

If the current structure blocks serious optimization, refactor the boundary first, then optimize the hot path.

## Continuous Iteration Loop

Repeat this loop until the definition of done in `docs/HASH_OPTIMIZATION_GOAL.md` is reached:

1. Run `git status -sb`.
2. Read `docs/HASH_OPTIMIZATION_GOAL.md` after resume or context compaction.
3. Identify the latest benchmark-related commit and local baseline.
4. Pick one measurable bottleneck from timing metadata or benchmark evidence.
5. Make the smallest useful code, harness, test, or documentation change.
6. Validate correctness before trusting any performance number.
7. Benchmark before and after with identical settings.
8. Compare median warm throughput first, then inspect min/max spread.
9. Commit only if the slice is correct, useful, and privacy-clean.
10. If an experiment fails, revert only the current uncommitted experiment, record useful evidence if needed, and continue.

Prefer many small, measurable iterations over broad speculative rewrites.

## Current First Actions

Start or resume with this sequence:

1. Verify the worktree and avoid unrelated dirty files.
2. Run focused Hash API tests.
3. Build the available smoke CLI or full CUDA binary.
4. Run a short CUDA smoke benchmark if a CUDA binary is available.
5. Run a longer repeated baseline for the next bottleneck.
6. Use timing fields to choose the next optimization:
   - high `input_ms`: reduce key generation or first-block preparation overhead
   - high `keygen_ms`: optimize generated-key construction and prefix handling
   - high `first_block_ms`: inspect safe Argon2 input preparation improvements
   - high `setup_ms`: reduce repeated setup or cache difficulty-derived state safely
   - high `compute_ms`: inspect CUDA allocation, transfer, launch, occupancy, and memory behavior
   - high `finalize_ms`: reduce matching, encoding, result collection, or JSON overhead

## Minimum Validation Ladder

Use the narrowest validation that proves the current change, then broaden before committing behavior that affects shared code.

Focused tests:

```bash
python -m pytest tests/unit/test_hash_api_contract.py tests/unit/test_hash_api_service.py tests/unit/test_hash_api_benchmark.py tests/unit/test_hash_api_compare.py -q
```

Smoke benchmark:

```bash
python scripts/hash_api_benchmark.py --binary <miner-binary> --backend cuda --device 0 --preset warm-short --seconds 2 --warmup 1 --repeat 3 --output .benchmarks/warm-short.json
```

Stable scan:

```bash
python scripts/hash_api_benchmark.py --binary <miner-binary> --backend cuda --device 0 --seconds 10 --warmup 1 --repeat 3 --scan-difficulty 1 --scan-difficulty 8 --scan-difficulty 64 --scan-batch-size 256 --scan-batch-size 512 --scan-batch-size 1024 --scan-batch-size 2048 --recommendations-only --output .benchmarks/batch-scan-stable.json
```

Treat short benchmark results as candidate signals only. Use longer repeated runs for committed performance claims.

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
5. Confirm the staged diff contains no private paths or machine details.

## Completion Criteria

This goal is complete only when one of these is true:

- throughput improves by at least 1000% over the initial measured baseline while preserving correctness
- repeated well-scoped attempts plateau and the remaining bottleneck is documented
- profiler evidence shows the implementation is near the practical hardware limit for the tested GPU class

Until then, continue iterating through benchmark, optimize, validate, and commit cycles.
