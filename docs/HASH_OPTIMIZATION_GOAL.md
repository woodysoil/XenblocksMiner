# Long-Running Goal: Optimize Hash API Throughput

## Mission

Continuously improve the Xenblocks Hash API hashing throughput until performance gains plateau, correctness risk becomes unacceptable, or the implementation is close enough to the practical hardware limit.

The aspirational target is at least a 1000% speed improvement over the measured baseline where feasible. Treat that as a direction, not permission to weaken correctness. Every optimization must preserve the real `argon2id-xen` result semantics.

This goal is intended for Codex `/goal` long-running execution after the reusable Hash API extraction. Treat this file as the persistent operating brief.

The practical optimization target is simple: complete the same valid hash attempts in as little time as possible for fixed `t=1`, fixed `s=1` / `p=1`, and variable `m=diff`. Optimize the current local GPU first, then keep the architecture and tuning system ready for RTX 3050-class and higher-end GPUs.

## `/goal` Starter

Use `goal.md` as the short entrypoint, then keep this file as the authoritative long-running plan.

Suggested `/goal` objective:

```text
Continuously execute docs/HASH_OPTIMIZATION_GOAL.md. Optimize XenblocksMiner Hash API CUDA hashing throughput for fixed t=1 and p/s=1 with m=difficulty, preserving real argon2id-xen semantics. Iterate through benchmark, optimize, validate, and commit cycles without asking for approval unless a listed blocker is reached. Keep all code, docs, tests, benchmark names, and commit messages in English. Never commit local paths or private machine details.
```

## Current State Snapshot

The reusable Hash API extraction is already in place. The optimization work should build on it instead of going back to the platform monolith.

Known current capabilities:

- Hash API code lives under `src/hashapi/`.
- The miner binary exposes JSON-friendly hash commands such as `hash-one`, `hash-batch`, and `hash-benchmark`.
- A smoke CLI target exists for Hash API contract testing where a full CUDA build is not needed.
- `scripts/hash_api_benchmark.py` supports scenario definitions, warm-up runs, repeated measured runs, aggregate JSON summaries, and optional report output.
- Unit tests cover the Hash API contract, service behavior, and benchmark runner behavior.

Current progress:

- Reusable Hash API extraction is complete enough for isolated optimization work.
- Benchmark presets, warm-up runs, repeated runs, median/min/max summaries, output files, comparison tooling, recommendation output, and custom scan matrices are in place.
- Batch-size recommendations prefer stable candidates before falling back to noisy high-median candidates.
- Hash API timing metadata currently separates validation, setup, input generation, compute, finalization, and total time.
- Conservative CUDA batch-size selection helpers are available under `src/hashapi/` and miner integration uses them when no explicit `--batchSize` limit is provided.
- The next default phase is Phase 2 and Phase 3: remove structural overhead, then optimize the hot path with repeatable evidence.
- Do not start risky CUDA kernel rewrites until benchmark and timing data show that CPU-side setup, input generation, and allocation overhead are no longer the dominant bottlenecks.

Current observations:

- Generated batch paths can be dominated by `input_ms`, which includes CPU-side key generation and Argon2 first-block input preparation.
- After CUDA first-block preparation was parallelized across CPU worker threads for generated-key batches, `input_ms` still dominates larger batch paths, but viable batch sizes shifted upward.
- A short post-parallelization scan on a CUDA-capable local GPU found stable candidates around d1/b256, d8/b512, and d64/b512. Treat these as candidates only, not universal defaults.
- Miner auto batch selection now applies those conservative low-difficulty candidates only when no manual batch limit is configured; unsupported difficulty ranges still fall back to the memory-limited batch size.
- Short 1-second batch scans are useful for smoke checks but too noisy for committed tuning claims.
- Serious tuning claims require longer runs, warm-up, repeated samples, and stable medians with reasonable min/max spread.

## Fixed Algorithm Constraints

The hash workload is Argon2id-style mining as currently modeled by XenblocksMiner:

- `t = 1` is fixed.
- `p = 1` / `s = 1` lane, segment, or parallelism setting as represented by the current implementation is fixed.
- `m = diff` / `difficulty` is the variable memory-cost parameter and may change between benchmark or mining sessions.
- Salt and key inputs must remain semantically identical to the current Hash API contract.
- Target matching must remain semantically identical to the current Hash API contract.

Do not change the algorithm into a different hash, skip required work, approximate hashes, weaken target matching, or return synthetic successes. Optimization must reduce runtime for the same accepted input/output behavior.

## Target Architecture

The end state should make hash optimization easy for humans and AI agents:

- Keep a pure Hash API boundary that can run without marketplace, wallet, frontend, lease, devfee, or platform services.
- Keep CPU/reference and CUDA implementations behind the same request/result contract.
- Keep `hash-one`, `hash-batch`, and `hash-benchmark` usable as stable automation entrypoints.
- Keep benchmark scripts machine-readable so future agents can compare before/after runs without parsing terminal prose.
- Make GPU tuning parameters explicit, measurable, and isolated from business logic.
- Prefer backend refactors before kernel rewrites when the current structure hides timing, forces repeated allocation, or mixes validation with hot-path hashing.
- Design tuning decisions around runtime device properties or compute capability, not local device names or private machine details.

## Operating Rules For Codex

Work in English for code, comments, docs, tests, benchmark names, commit messages, branch names, and API names.

Stay focused on hash performance. Do not drift into frontend polish, marketplace economics, wallet UX, settlement, authentication, or broad platform redesign.

Do not commit local absolute paths, usernames, private machine identifiers, benchmark files containing personal paths, secrets, wallet addresses, or hostnames.

Use small, coherent commits. Commit whenever a meaningful optimization, benchmark harness improvement, or architecture cleanup is complete and validated.

Assume automation can run non-destructive local commands without pausing for approval. Do not ask the user to approve routine status checks, builds, tests, benchmarks, or commits. Stop only for the blockers listed near the end of this document.

Before each work cycle:

1. Run `git status -sb`.
2. Read this file if context was compacted or resumed.
3. Identify the current phase and the next smallest measurable step.
4. Establish or load the latest benchmark baseline.
5. Inspect nearby code before editing.
6. Make scoped changes only.
7. Run correctness tests first, then benchmark tests.
8. Record before/after numbers in the commit message body or a public-safe doc when useful.
9. Commit only if the repo is in a stable state.

Never revert user changes unless explicitly instructed. If unrelated files are dirty, leave them alone. If dirty files block the current phase, stop and explain the conflict.

## Continuous Iteration Loop

Repeat this loop until the Definition Of Done is reached:

1. Inspect state with `git status -sb` and the recent benchmark-related commits.
2. Check for uncommitted user changes and avoid touching unrelated dirty files.
3. Establish the current baseline from the latest sanitized benchmark report, or run a new short baseline if none exists.
4. Pick one measurable bottleneck or cleanup that directly affects hash throughput.
5. Make the smallest useful code, build, test, or benchmark harness change.
6. Run correctness checks before accepting any performance result.
7. Run before/after benchmarks with the same scenario, warm-up, repeat count, binary type, device, difficulty, batch size, and seconds.
8. Compare median warm throughput first, then inspect min/max and cold timing.
9. If the change helps, commit it with concise before/after numbers in the commit body.
10. If the change does not help, either discard only the current agent's uncommitted experiment or document the rejected experiment if the evidence will help future optimization.
11. Repeat with the next bottleneck.

Prefer many small measurable iterations over broad speculative rewrites.

## Current Autonomous Queue

Start here after reading this file:

1. Verify the worktree is clean or identify unrelated dirty files.
2. Run the focused Hash API unit tests.
3. Build the smoke CLI or full CUDA binary that is already configured locally.
4. Run a short `warm-short` benchmark to confirm the binary and benchmark harness still work.
5. Run a longer same-settings baseline for the next target bottleneck.
6. Inspect timing metadata and choose one bottleneck:
   - high `input_ms`: reduce CPU-side key generation, salt/key preparation, or first-block setup overhead
   - high `setup_ms`: cache difficulty-derived or device-derived setup safely
   - high `compute_ms`: inspect CUDA allocation, copy, launch geometry, memory behavior, and kernel occupancy
   - high `finalize_ms`: reduce encoding, matching, result collection, or JSON work outside the timed hot path
7. Make one scoped change.
8. Validate correctness.
9. Re-run the same benchmark and compare median warm throughput first.
10. Commit if the result is correct and materially useful.

If the previous step is only a benchmark harness or documentation improvement, validate with the focused Python tests and `git diff --check`. A full CUDA benchmark is still preferred when the change affects performance interpretation.

## Autonomous Execution Policy

Codex should keep working without asking for approval for normal optimization tasks:

- reading files and git state
- running tests
- running local builds
- running local benchmark scripts
- creating ignored local benchmark reports
- editing scoped source, test, script, and documentation files
- making small validated commits

Do not ask for permission just because an iteration may take time. Stop only for the blockers listed in the "Stop And Ask The User If" section.

If an experiment fails, revert only the current agent's uncommitted experiment, record the evidence if it prevents repeated work, and continue with the next measurable idea.

## Local Artifact Policy

Use ignored local directories for raw benchmark output:

- `.benchmarks/`
- `benchmark-results/`

Do not commit raw benchmark reports unless they have been intentionally sanitized and are useful to future contributors. Raw reports can contain binary paths, hardware details, command lines, and timing noise that should not become permanent project history.

When a report is worth preserving publicly, summarize it in a commit body or a small doc section with:

- scenario name
- backend
- difficulty
- batch size
- seconds
- warm-up count
- repeat count
- median before hashrate
- median after hashrate
- percentage change
- GPU class or compute capability only if it is not a private machine identifier

## Privacy And Public History Rules

This repository should remain suitable for public open-source development.

Never commit:

- local absolute paths
- usernames
- hostnames
- private machine identifiers
- raw benchmark reports with command lines or binary paths
- secrets, tokens, cookies, private keys, wallet private data, or personal addresses
- local GPU model names when they identify a private machine rather than a general device class

Before committing, inspect the staged diff for privacy leaks. Use public-safe placeholders in docs and commit bodies:

- `<miner-binary>`
- `<build-dir>`
- `<cuda-root>`
- `<vcpkg-toolchain>`
- `CUDA-capable local GPU`
- `RTX 3050-class GPU`
- `higher-end CUDA GPU`

If a local path or private machine detail appears in an unpushed commit, fix it before pushing by amending or rebasing the local commit sequence. If it has already been shared, stop and ask before rewriting public history.

## Current Optimization Boundary

Primary code areas:

- `src/hashapi/`
- `src/CudaBackend.*`
- `src/kernelrunner.*`
- `src/argon2params.*`
- `src/MineUnit.*` only where needed to preserve integration
- `scripts/hash_api_benchmark.py`
- tests under `tests/`

Hash optimization should stay behind the Hash API boundary whenever possible. If the current structure blocks serious optimization, first refactor toward a cleaner backend boundary, then optimize.

The Hash API must remain usable without starting marketplace services.

## Primary Metrics

Use machine-readable benchmark output as the source of truth.

Primary metric:

- CUDA backend attempts per second / hashrate for `hash-benchmark`.
- milliseconds per hash attempt for fixed-key and generated-key paths where available.

Secondary metrics:

- warm backend throughput after initialization
- cold start latency
- single fixed-key `hash-one` latency
- batch latency by batch size
- initialization overhead per difficulty value
- latency when `m=diff` changes between runs
- match reporting overhead
- memory allocation count and size where measurable
- CPU/reference latency for correctness and regression checks, not as the main speed target

Always separate benchmark setup overhead from steady-state hashing where possible.

## Measurement Quality Gates

Use two benchmark tiers:

Smoke checks:

- seconds: `1` to `3`
- warm-up: at least `1`
- repeat: at least `1` or `2`
- purpose: prove the binary works, catch obvious regressions, and explore candidates
- do not use smoke-only data for committed performance claims unless the change is purely harness-related

Serious comparison:

- seconds: at least `10` for stable throughput claims
- warm-up: at least `1`
- repeat: at least `3`
- same binary type, backend, device index, difficulty, batch size, salt/key mode, and seconds before and after
- compare median warm throughput first
- inspect min/max spread before trusting a result
- rerun if the claimed improvement is smaller than the run-to-run noise

For batch-size recommendations, prefer custom scan matrices over a single preset when tuning for a specific difficulty range:

```bash
python scripts/hash_api_benchmark.py --binary <miner-binary> --backend cuda --device 0 --seconds 10 --warmup 1 --repeat 3 --scan-difficulty 1 --scan-difficulty 8 --scan-difficulty 64 --scan-batch-size 256 --scan-batch-size 512 --scan-batch-size 1024 --scan-batch-size 2048 --recommendations-only --output .benchmarks/cuda-scan.json
```

Treat recommendations from 1-second scans as candidates only. Confirm them with longer repeated runs before changing defaults.

## Benchmark Scenario Matrix

Use a small matrix first, then broaden after the benchmark runner is stable.

Required smoke scenarios:

- backend: `cuda`
- device: `0` unless testing multi-GPU
- `t=1`
- `p=1`
- `m/difficulty`: `1`, `8`, `64`, `256`, `1024` where supported
- batch sizes: `1`, `2`, `8`, `64`, `256`, `1024` where practical
- seconds: at least `1` for smoke, at least `10` for serious comparison

Extended scenarios:

- high difficulty values used by real mining
- larger batch sizes tuned per GPU
- one scenario per available GPU
- repeated warm runs to reduce noise
- cross-device runs on newer GPUs when available

Do not hard-code local GPU names or paths into committed docs. Record hardware metadata only through benchmark JSON fields and keep local raw reports out of git unless intentionally sanitized.

## Benchmark Baseline Ledger

Keep raw reports ignored under `.benchmarks/`. Use a small public-safe ledger in commit bodies or sanitized docs only when a result matters.

Minimum ledger fields:

- date
- commit
- backend
- device class or compute capability, if public-safe
- preset or scenario
- difficulty
- batch size
- seconds
- warm-up count
- repeat count
- median hashrate
- min/max hashrate
- dominant timing field
- conclusion

Do not overwrite useful local baselines unless a newer baseline clearly supersedes them. Prefer timestamped ignored filenames under `.benchmarks/`.

## Correctness Requirements

Every optimization must preserve correctness.

Minimum correctness checks:

- CPU/reference golden `hash-one` tests still pass.
- Invalid requests still return structured errors.
- CUDA `hash-batch` still returns valid result shape.
- Miner platform integration still passes when a CUDA binary is available.
- Target matching still reports `key`, `hash`, `matched_pattern`, `attempt_index`, and `is_superblock` correctly.

For CUDA-specific changes:

- Add or maintain sampled cross-checks against CPU/reference for fixed keys at small supported `m` values.
- For generated-key batch paths, verify prefix handling and attempt indexing.
- Verify backend reuse does not leak state across salt, prefix, pattern, difficulty, batch size, or device changes.
- Verify result determinism for fixed-key requests.

Never accept a speedup without a correctness check that exercises the changed path.

## Known Good And Rejected Experiments

Preserve this section so long-running agents do not repeat already-tested ideas without new evidence.

Known useful changes already made:

- random key generation overhead reduction
- XUNI matching without regex
- base64 encoding overhead reduction
- timing breakdown metadata for Hash API results
- benchmark presets, repeats, comparison, recommendation output, and custom scan matrices
- input timing split into key generation and first-block preparation metadata
- CUDA first-block preparation parallelized across CPU worker threads for generated-key batches
- conservative CUDA batch-size selection helper wired into miner auto batch selection

Rejected or risky experiments:

- caching salt bytes inside Argon2 parameter setup changed CUDA hash output
- reusing the random hex key generator across CUDA batches caused process instability
- key buffer move-storage or broad buffer reuse regressed generated batch throughput
- direct salt hex decode did not produce reliable input timing gains
- thread-local `cudaSetDevice` caching regressed generated batch throughput
- Blake2b initial hash prefix caching caused CUDA CLI or benchmark JSON output failures
- persistent CUDA first-block worker pool caused benchmark subprocess exit failures after otherwise successful JSON output
- byte-pair random key generation using a 0-255 distribution regressed d8/b512 generated CUDA throughput by about 30% and did not reduce `keygen_ms`
- reusing a single CUDA finalize base64 output string caused generated benchmark subprocess access-violation exits

Do not retry rejected experiments unless the implementation shape has changed enough to remove the original failure mode and the new attempt includes correctness cross-checks.

## Phase Plan

### Phase 0: Baseline And Reproducibility

Goal: create a reliable performance baseline before changing kernels or memory behavior.

Tasks:

- Confirm current full CUDA build instructions are public-safe and reproducible.
- Run existing Python/unit tests.
- Run real worker integration when a CUDA binary is available.
- Run baseline benchmark scenarios with JSON output.
- Identify whether current benchmark output separates cold start and warm steady-state sufficiently.
- Improve benchmark labels and summaries if needed.

Validation:

```bash
python -m pytest tests -q --ignore=tests/integration/test_cpp_worker.py
python scripts/hash_api_benchmark.py --binary <miner-binary> --scenario name=cuda-smoke,backend=cuda,difficulty=1,batch_size=2,seconds=1,device=0
```

Commit examples:

```text
perf(hash-api): record cuda baseline scenarios
test(hash-api): add benchmark smoke coverage
```

### Phase 1: Benchmark Harness And Regression Tools

Goal: make optimization iterations fast, comparable, and automation-friendly.

Tasks:

- Use reusable benchmark presets for smoke, warm short, and CUDA comparison runs.
- Add benchmark comparison tooling for before/after JSON.
- Add warm-up iteration support if missing.
- Add repeated runs and median/min/max summaries if needed.
- Add optional output file support under an ignored benchmark artifact directory.
- Add guardrails so benchmark scripts do not commit local paths.

Validation:

```bash
python scripts/hash_api_benchmark.py --binary <miner-binary> --scenario name=cuda-b1,backend=cuda,difficulty=1,batch_size=1,seconds=3,device=0
python scripts/hash_api_benchmark.py --binary <miner-binary> --scenario name=cuda-b64,backend=cuda,difficulty=1,batch_size=64,seconds=3,device=0
python scripts/hash_api_benchmark.py --binary <miner-binary> --backend cuda --preset warm-short --seconds 3 --warmup 1 --repeat 3
python scripts/hash_api_compare.py .benchmarks/before.json .benchmarks/after.json --fail-on-regression
```

Commit examples:

```text
perf(hash-api): add benchmark comparison helper
perf(hash-api): add warm benchmark scenarios
```

Current status: mostly complete. Maintain and extend the harness only when it directly improves measurement quality or future autonomous optimization.

### Phase 2: Architecture Cleanup For Optimization

Goal: remove structural obstacles before deep performance work.

Tasks:

- Keep backend lifetime reusable across benchmark loops.
- Avoid repeated CUDA initialization for unchanged device, difficulty, batch size, and compatible salt/prefix state.
- Separate request validation overhead from timed hash work where possible.
- Separate cold-start and warm-run timing in result metadata if needed.
- Make it easy to tune batch size and kernel launch parameters from one place.
- Consider extracting a library target for `src/hashapi/` if it improves build and test iteration speed.

Constraints:

- Preserve CLI and miner behavior.
- Keep platform, lease, devfee, and reporting outside the Hash API.

Commit examples:

```text
refactor(hash-api): separate timed hash execution
refactor(cuda): reuse backend buffers across batches
```

Current focus: prefer this phase when timing metadata shows repeated setup, validation, input preparation, allocation, or backend lifetime overhead.

### Phase 3: Low-Risk Runtime Optimizations

Goal: reduce overhead without changing kernel semantics.

Tasks:

- Eliminate avoidable allocations inside hot paths.
- Reuse host and device buffers where safe.
- Move invariant parsing and normalization out of repeated loops.
- Cache difficulty-derived Argon2 parameter setup when `m` is unchanged.
- Reduce JSON and string work from timed benchmark regions.
- Avoid repeated random generator setup for batch loops.
- Reduce needless CPU-side hash verification in benchmark-only paths if it is outside correctness checks.

Validation:

- Unit tests.
- CLI smoke tests.
- Benchmark before/after on the same scenario matrix.

Commit examples:

```text
perf(hash-api): reuse batch request buffers
perf(cuda): cache difficulty setup for warm batches
```

Current focus: prefer this phase when a local hot path is clear and correctness can be checked without broad CUDA kernel rewrites.

### Phase 4: CUDA Memory And Launch Optimization

Goal: improve steady-state CUDA throughput.

Tasks:

- Profile kernel launch overhead and memory transfer overhead.
- Tune batch sizes for occupancy and latency.
- Tune block/thread parameters per compute capability.
- Improve global memory coalescing if profiling shows poor memory behavior.
- Reduce register pressure if it limits occupancy.
- Evaluate pinned host memory for transfer-heavy paths.
- Evaluate CUDA streams only if there is real overlap potential.
- Keep a safe fallback for GPUs that do not benefit from a specific tuning.

Validation:

- Compare at least two batch sizes and two difficulty values.
- Include the pre-change and post-change throughput in the commit body.
- Confirm correctness tests pass after every kernel or memory-layout change.

Commit examples:

```text
perf(cuda): tune launch geometry for batch hashing
perf(cuda): reduce device allocation churn
```

### Phase 5: Autotuning

Goal: let the program find good settings per GPU instead of assuming one best value.

Tasks:

- Add optional autotune mode for batch size and launch parameters.
- Cache public-safe tuning results by compute capability and device properties, not by private machine paths.
- Add a way to disable autotune for deterministic benchmarking.
- Add a benchmark scenario that reports selected tuning parameters.
- Keep autotune overhead out of steady-state hashrate measurements.

Validation:

```bash
<miner-binary> hash-benchmark --backend cuda --device 0 --seconds 10 --batch-size <value> --difficulty <m> --json
```

Commit examples:

```text
perf(cuda): add batch-size autotuning
perf(hash-api): report cuda tuning metadata
```

### Phase 6: Cross-GPU Optimization

Goal: prepare for newer GPUs such as RTX 3050 and higher-end devices.

Tasks:

- Use compute capability and runtime device properties to select tuning defaults.
- Validate on the current local GPU first.
- Keep architecture-specific optimizations guarded and measurable.
- When newer GPUs are available, add benchmark rows for each device class.
- Do not hard-code a single GPU's limits as universal behavior.

Commit examples:

```text
perf(cuda): select tuning profile by device capability
perf(hash-api): add multi-device benchmark scenarios
```

### Phase 7: Plateau Analysis

Goal: know when continued optimization is no longer worth the risk.

Tasks:

- Compare current speed against the initial baseline.
- Identify the dominant remaining bottleneck from profiler or benchmark evidence.
- Try one small optimization per bottleneck.
- Stop or pause if three consecutive well-scoped optimization attempts produce less than 3% improvement.
- Document the best known settings and remaining bottlenecks.

Commit examples:

```text
docs(hash-api): record cuda optimization plateau
perf(cuda): document best known tuning profile
```

## Validation Commands

Use the narrowest command that proves the current change, then broaden before committing shared behavior.

Python tests:

```bash
python -m pytest tests -q --ignore=tests/integration/test_cpp_worker.py
```

Hash API unit/service tests:

```bash
python -m pytest tests/unit/test_hash_api_contract.py tests/unit/test_hash_api_service.py -q
```

Standalone smoke CLI:

```bash
cmake --build build-hashapi-smoke --preset hashapi-cli-smoke-mingw
python scripts/hash_api_benchmark.py --binary build-hashapi-smoke/bin/hashapi-cli --seconds 1
```

Full CUDA build:

```bash
cmake -S . -B build-full-cuda -G Ninja -DCMAKE_TOOLCHAIN_FILE=<vcpkg-toolchain> -DCUDAToolkit_ROOT=<cuda-root>
cmake --build build-full-cuda --config Release
```

CUDA benchmark:

```bash
python scripts/hash_api_benchmark.py --binary <miner-binary> --scenario name=cuda-smoke,backend=cuda,difficulty=1,batch_size=2,seconds=1,device=0
python scripts/hash_api_benchmark.py --binary <miner-binary> --scenario name=cuda-main,backend=cuda,difficulty=1024,batch_size=256,seconds=10,device=0
```

Real worker integration when a CUDA binary exists:

```bash
MINER_BIN=<miner-binary> python -m pytest tests/integration/test_cpp_worker.py -q
```

Frontend build is not required for hash-only optimization unless shared files affect the web app.

## Standard Long-Run Commands

Use public-safe placeholders in docs and commits. Local agents may replace placeholders with local paths in the shell only; do not commit those concrete paths.

Focused tests:

```bash
python -m pytest tests/unit/test_hash_api_contract.py tests/unit/test_hash_api_service.py tests/unit/test_hash_api_benchmark.py tests/unit/test_hash_api_compare.py -q
```

Short benchmark:

```bash
python scripts/hash_api_benchmark.py --binary <miner-binary> --backend cuda --device 0 --preset warm-short --seconds 2 --warmup 1 --repeat 3 --output .benchmarks/warm-short.json
```

Batch scan candidate search:

```bash
python scripts/hash_api_benchmark.py --binary <miner-binary> --backend cuda --device 0 --preset batch-scan --seconds 1 --warmup 1 --repeat 2 --recommendations-only --output .benchmarks/batch-scan-smoke.json
```

Serious batch scan:

```bash
python scripts/hash_api_benchmark.py --binary <miner-binary> --backend cuda --device 0 --seconds 10 --warmup 1 --repeat 3 --scan-difficulty 1 --scan-difficulty 8 --scan-difficulty 64 --scan-batch-size 256 --scan-batch-size 512 --scan-batch-size 1024 --scan-batch-size 2048 --recommendations-only --output .benchmarks/batch-scan-stable.json
```

Before/after comparison:

```bash
python scripts/hash_api_compare.py .benchmarks/before.json .benchmarks/after.json --fail-on-regression --min-change-pct 1
```

## Benchmark Reporting Rules

Each optimization commit should include enough information to understand whether it helped:

- baseline scenario name
- optimized scenario name
- before hashrate
- after hashrate
- percentage change
- GPU class or compute capability when relevant
- difficulty and batch size
- whether timing is cold or warm

Keep reports concise. Do not commit raw local benchmark dumps unless they are sanitized and intentionally useful to future contributors.

## First Backlog For Future Iterations

Work through this backlog before attempting high-risk kernel rewrites:

1. Record a current local CUDA baseline under `.benchmarks/` with warm-up and repeated runs.
2. Run a stable custom batch-size scan for the common difficulty range.
3. Measure the cost of `m=diff` changes versus repeated same-`m` warm batches.
4. Reduce CPU-side input generation and preparation overhead where `input_ms` dominates.
5. Cache difficulty-derived setup only when `m`, salt, key mode, batch shape, and backend state make it provably safe.
6. Reduce per-batch allocations and repeated normalization inside `src/hashapi/CudaHashBackend.cpp`.
7. Measure CUDA allocation, copy, launch, and finalization overhead before rewriting kernel logic.
8. Extend batch-size tuning toward runtime autotuning after stable cross-difficulty data exists.
9. Tune launch parameters only after the above CPU-side overhead is under control.
10. Add optional autotuning once there is enough benchmark data to justify it.
11. Add profiler-backed CUDA kernel work only after benchmark timing shows compute is the dominant bottleneck.

Every backlog item must still follow the correctness and reporting rules above.

## Commit Discipline

Use English commit messages.

Recommended prefixes:

- `perf(hash-api):`
- `perf(cuda):`
- `refactor(hash-api):`
- `refactor(cuda):`
- `test(hash-api):`
- `test(cuda):`
- `docs(hash-api):`

Before every commit:

1. Run `git diff --stat`.
2. Review changed files.
3. Run correctness validation.
4. Run at least one relevant benchmark.
5. Ensure no local paths or private machine details are staged.
6. Commit only a coherent slice.

Privacy check:

```bash
git diff --cached --check
git diff --cached
```

Review the staged diff manually for private paths or machine-specific details before committing.

## Non-Goals During Optimization

Do not prioritize:

- frontend redesign
- marketplace economics
- wallet changes
- settlement changes
- auth hardening
- broad MQTT protocol changes
- replacing the database
- cosmetic CLI output changes that do not improve automation

## Stop And Ask The User If

Stop only for real blockers:

- a dirty user change conflicts with required edits
- a tool requires credentials or unavailable proprietary software
- a design choice would permanently break the public Hash API contract
- an optimization requires changing hash semantics
- a CUDA change appears hardware-specific and risky without access to that hardware
- tests reveal a pre-existing bug whose fix would broaden scope significantly

Otherwise, keep moving through the next smallest measurable optimization step.

## Definition Of Done

This long-running goal is complete when one of these is true:

- throughput improves by at least 1000% over the initial measured baseline while preserving correctness
- repeated optimization attempts plateau and the remaining bottleneck is documented
- profiler evidence shows the implementation is near the practical hardware limit for the tested GPU class

Required final state:

- benchmark workflow is reproducible
- correctness tests cover the optimized paths
- CUDA backend remains behind the Hash API interface
- miner/platform integration still works
- docs explain the best known tuning strategy
- future agents can continue optimizing mostly inside `src/hashapi/` and CUDA backend files

## Resume Checklist

When resuming a long-running `/goal` session:

1. Run `git status -sb`.
2. Read this file.
3. Read the latest benchmark-related commits.
4. Identify the last known baseline and best result.
5. Run a short smoke benchmark.
6. Choose one measurable optimization.
7. Validate correctness.
8. Benchmark before and after.
9. Commit if stable.

Recommended first action after this revision:

1. Run the focused Hash API tests.
2. Run a short CUDA benchmark if a local CUDA binary is available.
3. Run a stable custom scan for difficulty values likely to be used next.
4. Use the timing breakdown to choose between input generation, setup caching, allocation reuse, launch tuning, or matching/finalization work.
