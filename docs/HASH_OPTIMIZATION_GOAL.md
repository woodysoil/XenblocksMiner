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

Active goal status:

- `/goal` is active for continuous Hash API and CUDA throughput optimization.
- The branch may be ahead of the remote with many local commits. Treat those commits as retained local work, not lost work, unless the user explicitly requests a squash, reorder, push, or public history rewrite.
- The current clean Release continuity baseline is the generated-key CUDA main-target scenario at difficulty `8`, batch size `2048`, warm-up `1`, repeat `3`, and no XUNI matching, with about `78.3k H/s` median and `3.8%` spread.
- The dominant measured bottleneck in that baseline is CPU-side input preparation, especially first-block preparation. Detailed timing showed `input_ms` at about `58-61%` of wall time and `first_block_ms` at about `54-57%`.
- This baseline is local evidence on a CUDA-capable GPU. Do not publish raw reports, local binary paths, hardware identifiers, or private machine details.

Known current capabilities:

- Hash API code lives under `src/hashapi/`.
- The miner binary exposes JSON-friendly hash commands such as `hash-one`, `hash-batch`, and `hash-benchmark`.
- A smoke CLI target exists for Hash API contract testing where a full CUDA build is not needed.
- `scripts/hash_api_benchmark.py` supports scenario definitions, warm-up runs, repeated measured runs, aggregate JSON summaries, and optional report output.
- Unit tests cover the Hash API contract, service behavior, and benchmark runner behavior.

Current progress:

- Reusable Hash API extraction is complete enough for isolated optimization work.
- The extracted automation surface is the command-line Hash API: `hash-one`, `hash-batch`, and `hash-benchmark`. Treat these as CLI entrypoints for reproducible optimization, not as frontend, websocket, marketplace, or hosted HTTP platform APIs.
- The "CLI API" wording means the command-line adapter around the extracted Hash API. It exposes hash execution and benchmark automation, but it is not a hosted HTTP API and it is not the full platform.
- Benchmark presets, warm-up runs, repeated runs, median/min/max summaries, output files, comparison tooling, recommendation output, and custom scan matrices are in place.
- Batch-size recommendations prefer stable candidates before falling back to noisy high-median candidates.
- Benchmark recommendations also include full candidate lists with min/max hashrate, spread, and per-attempt timing fields.
- Hash API timing metadata currently separates validation, setup, input generation, compute, finalization, and total time.
- CUDA timing metadata reports nested sub-measurements such as `kernel_ms`, `host_to_device_ms`, and `device_to_host_ms` inside `compute_ms`, plus `finalize_hash_ms`, `argon2_finalize_ms`, `base64_ms`, and `match_ms` inside `finalize_ms`, so future tuning can distinguish transfers, kernel time, hash finalization, encoding, and target matching from their parent stages.
- Benchmark `timing_analysis` includes `nested_stage_pct` so optimization agents can read nested diagnostics as percentages of their parent stage without treating them as additive wall time.
- Optional `--detailed-timings` also splits CUDA setup timing and first-block CPU timing for diagnosis. These detailed fields are nested diagnostic timing, not additive wall time.
- Hash API benchmark summaries include per-attempt timing fields for comparing cost per valid hash attempt.
- Hash API comparison tooling reports total timing deltas, per-attempt timing deltas, top-level and nested stage-percentage deltas, noisy improved/regressed/unchanged status, and variable-difficulty metadata for before/after runs.
- Hash API comparison tooling can match by config while ignoring only detailed-timing mode with `--ignore-detailed-timings`, which helps compare default and diagnostic reports for the same scenario without changing other matching fields.
- Hash API benchmark scenarios can measure variable `m = difficulty` sequences, including same-difficulty versus alternating-difficulty loops under one reusable backend lifecycle.
- Generated variable-difficulty sequence scenarios can enable detailed CUDA setup and first-block diagnostics with `--sequence-detailed-timings`.
- CUDA Hash API scenarios can cap first-block worker threads with `first_block_workers` / `--first-block-workers` for measured tuning while default `0` preserves automatic worker-count behavior. Benchmark scans can include this axis with `--scan-first-block-workers` and can enable detailed generated-scan diagnostics with `--scan-detailed-timings`.
- Hash API benchmark presets include an `isolation` matrix for comparing generated-key d8/b2048 throughput against fixed-key d8/b1 behavior before choosing between input-preparation, compute, and finalization work.
- Hash API benchmark summaries mark any nonzero benchmark subprocess exit as invalid even when stdout contains parseable JSON, so crashy optimization experiments cannot enter recommendations.
- Hash API benchmark recommendations expose `report_ok`, run counts, and invalid scenario names so automation can reject partial scan matrices before acting on surviving tuning candidates.
- Hash API benchmark reports can include public-safe build metadata via `--build-cache <build-dir>` so optimization agents can distinguish Release/Debug, CUDA architecture sets, generator, vcpkg triplet, and CUDA compiler version without committing local paths.
- Conservative CUDA batch-size selection helpers are available under `src/hashapi/` and miner integration uses them when no explicit `--batchSize` limit is provided.
- The next default phase is Phase 2 and Phase 3: remove structural overhead, then optimize the hot path with repeatable evidence.
- Do not start risky CUDA kernel rewrites until benchmark and timing data show that CPU-side setup, input generation, and allocation overhead are no longer the dominant bottlenecks.

Current observations:

- Generated batch paths can be dominated by `input_ms`, which includes CPU-side key generation and Argon2 first-block input preparation.
- After CUDA first-block preparation was parallelized across CPU worker threads for generated-key batches, `input_ms` still dominates larger batch paths, but viable batch sizes shifted upward.
- Repeated main-target-only scans on a CUDA-capable local GPU support d1/b512 and d8/b2048 as current conservative low-difficulty defaults. Treat this as local evidence only, not a universal hardware limit.
- Miner auto batch selection now applies conservative low-difficulty candidates only when no manual batch limit is configured; unsupported difficulty ranges still fall back to the memory-limited batch size.
- d64 batch-size scans have been noisy and should not be used to change defaults without stronger repeated evidence.
- Later 10-second d64 scans still conflicted between b1024 and b2048 stability versus median throughput, so keep the d64 default conservative until repeated evidence converges.
- A short d8 scan found b4096 as a fast candidate, but a 10-second d8/b4096 confirmation later had a benchmark subprocess access-violation exit and slower/noisier valid samples, so keep the d8 default at b2048 unless a future stable confirmation removes that instability.
- A newer d8 b1024/b2048 confirmation on the CUDA-capable local GPU found b1024 slightly faster by median and stable, while b2048 was slightly lower and just above the stability threshold. Treat b1024 as the current safer local A/B scenario, but do not change the miner default from b2048 until repeated evidence converges across longer runs or another GPU class.
- A later local d8 batch-window scan with b512, b1024, b2048, b3072, and b4096 kept b2048 as the best stable median at 55.2k H/s with 3.7% spread. b3072 was close but slightly lower, b1024 was lower, and b4096 had a higher max but was unstable, so keep the d8 default at b2048.
- A refreshed d8/b2048 generated-key CUDA baseline on the CUDA-capable local GPU reached 77.1k H/s median with 2.3% spread in default timing mode. The same scenario remained dominated by `input_ms` at about 58% of wall time, with `first_block_ms` about 54% and `finalize_ms` about 28%. Treat the default-timing run as the throughput continuity baseline; the paired detailed-timing run was useful for diagnosis but too noisy for a throughput claim.
- A later default-timing d8 batch-size refresh found stable b3072 and b4096 candidates, with b4096 at 71.5k H/s median and 8.0% spread, but the matrix report was invalid because b1024 had a subprocess access-violation exit. Do not change the d8 default from that partial matrix; rerun a clean targeted confirmation before acting on b3072 or b4096.
- A clean targeted d8 b2048/b4096 confirmation did not justify raising the default: b4096 was stable at 56.4k H/s median, while b2048 was noisy and the result remained below the 77.1k H/s continuity baseline. Keep d8 at b2048 unless a future clean confirmation beats the stable baseline.
- Detailed setup timing shows setup can matter in short runs, with CUDA activation usually the largest setup subfield. Direct activation caching was tested and rejected because benchmark subprocesses became unstable.
- Detailed first-block timing shows first-block digest work is a major CPU-side cost in generated-key batches. Because parallel first-block timing can sum worker-local CPU time, do not treat nested detailed fields as additive wall-clock components.
- An initial first-block worker-cap scan showed the new tuning surface is useful for exploration: d8/b1024 explicit caps were competitive with or faster than the noisy auto baseline, while d8/b2048 had noisy or unstable capped runs. Treat this as a candidate-search signal only; do not change defaults without longer stable confirmation.
- A longer d8/b1024 confirmation did not support changing the first-block worker default: auto had the highest median but was noisy, cap 8 was close but still above the stability threshold, and cap 4 regressed. Keep automatic worker count as the default and use explicit caps only for benchmark scans.
- A refreshed d8 first-block worker-cap scan with detailed timing again favored automatic worker selection. At b1024, auto reached 77.3k H/s median with 1.3% spread while caps 4, 8, 12, and 16 were slower or unstable. At b2048, auto was the only stable candidate and explicit caps regressed or exceeded the stability threshold. Do not change the default worker policy from this evidence.
- A detailed variable-difficulty d8/d64 sequence run showed alternating `m=diff` can spend about 20% of wall time in setup, with most detailed setup time in backend initialization, versus about 5% setup for same-difficulty d8 loops. This makes runner reuse across recent segment-block shapes a real architecture target, but it needs a memory-bounded design rather than a quick default change.
- H2D and D2H transfer timings are measurable but not currently the dominant d8/b2048 bottleneck. Pinned host staging was tested and rejected for the current implementation because the same-settings throughput comparison regressed, so revisit it only as part of a broader transfer-overlap or buffer-lifetime redesign.
- Short 1-second batch scans are useful for smoke checks but too noisy for committed tuning claims.
- Serious tuning claims require longer runs, warm-up, repeated samples, and stable medians with reasonable min/max spread.
- Local build configuration is part of benchmark quality. A stale local CUDA build cache was found using `CMAKE_BUILD_TYPE=Debug` and `CMAKE_CUDA_ARCHITECTURES=52`, so do not treat results from that build directory as a clean Release throughput baseline. Use a fresh Release CUDA preset such as `cuda-release-vcpkg-modern` or a matching architecture-specific preset before making new speed claims.
- A fresh Release CUDA build using the modern architecture preset preserved the golden d8 CUDA hash. New generated-key d8 baselines on the CUDA-capable local GPU remained noisy: b2048 reached 52.6k H/s median with 11.4% spread, while b1024 reached 63.7k H/s median with 16.7% spread. Both runs were `report_ok: true` but unstable by the 10% spread gate. They should be treated as current measurement evidence, not as default-tuning proof. Both remained dominated by `input_ms` at about 62% of wall time and `first_block_ms` at about 58%, so the next code experiments should continue to target generated input and first-block preparation before CUDA kernel rewrites.
- A rerun of the Release CUDA modern-architecture d8/b2048 baseline with benchmark build metadata produced a stable report: 78.3k H/s median, 3.8% spread, `report_ok: true`, `input_ms` about 58% of wall time, and `first_block_ms` about 54%. Treat this as the current clean Release continuity baseline for before/after comparisons until a newer stable report supersedes it.
- A matching detailed-timing d8/b2048 run stayed noisy but useful for diagnosis: `input_ms` remained about 61% of wall time and `first_block_ms` about 57%. Nested first-block CPU counters showed digest expansion dominates initial prehash by a wide margin, but those counters sum worker-local CPU time and are not additive wall time. Prefer digest/Blake2b or first-block scheduling experiments over keygen-only work unless newer timing contradicts this.
- A short d8/b2048 detailed worker-cap smoke using the first-block CPU-sum-to-wall ratio kept automatic first-block workers ahead of explicit caps: auto reached about `69.1k H/s`, while caps 4, 8, and 12 reached about `66.1k`, `64.4k`, and `66.3k H/s`. The auto first-block CPU-sum-to-wall ratio was about `5.2`, showing useful parallelism but not enough evidence to change worker defaults. Treat this as measurement-only smoke evidence, not a stable tuning claim.

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
2. Confirm docs and recent commits contain no local paths, usernames, hostnames, secrets, raw benchmark reports, or private hardware identifiers.
3. Confirm local progress with `git log --oneline`; if the branch is ahead of the remote, treat those commits as retained local work unless the user explicitly asks to squash, reorder, push, or rewrite them.
4. Run the focused Hash API unit tests.
5. Build the smoke CLI or full CUDA binary that is already configured locally.
6. Run the golden CUDA hash check when a CUDA binary is available.
7. Run a short main-target CUDA benchmark to confirm the binary and benchmark harness still work.
8. Run or load a repeated d8/b2048 baseline because recent accepted and rejected experiments used that scenario, then include d8/b1024 when the local GPU shows b2048 instability.
9. If no newer evidence supersedes this checkpoint, use `--detailed-timings` on d8/b2048 and d8/b1024 to confirm whether `input_ms` and first-block preparation still dominate before choosing the next experiment.
10. Inspect timing metadata and choose one bottleneck:
   - high `input_ms`: reduce CPU-side key generation, salt/key preparation, or first-block setup overhead
   - high `keygen_ms`: optimize random key generation, prefix handling, or generated-key memory layout
   - high `first_block_ms`: use `--detailed-timings` to split initial prehash and digest expansion, then improve safe Argon2 first-block preparation and CPU parallelism
   - high `setup_ms`: use `--detailed-timings` to split normalization, activation, device info, parameter construction, and backend initialization before caching difficulty-derived or device-derived setup safely
   - high `compute_ms`: inspect CUDA allocation, copy, launch geometry, memory behavior, and kernel occupancy
   - high `finalize_ms`: use `finalize_hash_ms`, `argon2_finalize_ms`, `base64_ms`, and `match_ms` to choose between hash finalization, encoding, matching, result collection, or JSON work outside the timed hot path
11. Prefer input preparation and setup/measurement improvements before speculative finalization micro-optimizations.
12. Make one scoped change.
13. Validate correctness.
14. Re-run the same benchmark and compare median warm throughput first.
15. Commit if the result is correct, materially useful, and privacy-clean.

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

Raw benchmark JSON must remain untracked. Sanitized summaries may be committed only when they have been reviewed for local paths, usernames, hostnames, hardware identifiers, secrets, wallet data, and personal addresses.

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
- generated variable-difficulty sequence matrices can enable detailed setup and first-block timing diagnostics with `--sequence-detailed-timings`
- CUDA first-block preparation parallelized across CPU worker threads for generated-key batches
- conservative CUDA batch-size selection helper wired into miner auto batch selection
- main-target-only benchmark mode for measuring normal mining without secondary XUNI matching
- per-attempt benchmark timing summaries and full recommendation candidate reporting
- d1 CUDA default batch size raised to 512 when no explicit user batch-size limit is configured
- d8 CUDA default batch size raised to 2048 when no explicit user batch-size limit is configured
- generated benchmark scan matrices can enable detailed setup and first-block timing diagnostics with `--scan-detailed-timings`
- little-endian `Blake2b` 64-bit load/store fast path reduced generated CUDA per-attempt cost in a d8/b2048 A/B benchmark
- `RandomHexKeyGenerator` now consumes multiple hex nibbles from each `mt19937` output instead of using per-character distribution calls; local d8/b2048 generated CUDA confirmation reduced median `keygen_ms` per attempt from about 0.00222 ms to about 0.000845 ms and reached 49.9k H/s with 5.15% spread
- Fixed-key CUDA requests now avoid constructing the generated-key random generator; isolation confirmation kept generated d8/b2048 stable at about 66.96k H/s median and improved fixed-key d8/b1 to about 4.41k H/s median with 0.8% spread
- `Blake2b::final` writes full 64-byte outputs directly into the destination buffer instead of staging through a temporary copy; local d8/b2048 generated CUDA confirmation stayed correct and reached 52.4k H/s median, with noisy but lower per-attempt first-block/finalize timings than the keygen baseline
- Argon2 initial hash setup now batches fixed 32-bit metadata into stack buffers for the no-secret/no-associated-data mining path, reducing local d8/b2048 generated CUDA `first_block_ms` per attempt from about 0.01148 ms to about 0.00820 ms and reaching 67.1k H/s median with 3.7% spread
- detailed CUDA transfer, first-block, and setup timing fields are available for diagnosis while the default non-detailed path avoids extra timing overhead
- benchmark comparison can classify noisy unchanged runs when the median change is below threshold but spread is too high to treat the result as stable
- benchmark timing analysis reports first-block detailed CPU-time sum and CPU-sum-to-wall ratio, so agents can distinguish digest-heavy work from first-block scheduling overhead without treating worker-local CPU counters as additive wall time

Rejected or risky experiments:

- caching salt bytes inside Argon2 parameter setup changed CUDA hash output
- reusing the random hex key generator across CUDA batches caused process instability
- key buffer move-storage or broad buffer reuse regressed generated batch throughput
- direct salt hex decode did not produce reliable input timing gains
- thread-local `cudaSetDevice` caching regressed generated batch throughput
- Blake2b initial hash prefix caching caused CUDA CLI or benchmark JSON output failures
- persistent CUDA first-block worker pool caused benchmark subprocess exit failures after otherwise successful JSON output
- fixed-buffer base64 finalization with string-view matching preserved the golden CUDA hash but regressed short generated CUDA sequence benchmarks, so it should not be retried without a broader finalization redesign
- byte-pair random key generation using a 0-255 distribution regressed d8/b512 generated CUDA throughput by about 30% and did not reduce `keygen_ms`
- reusing a single CUDA finalize base64 output string caused generated benchmark subprocess access-violation exits
- bypassing the lane XOR copy in `Argon2Params::finalize` for `lanes == 1` preserved the golden CUDA hash but did not improve fixed-key throughput and only produced a noisy generated-path improvement, so it should not be kept without stronger profiler-backed evidence
- reusing generated-key string storage with `fillRandomKey` slightly reduced keygen timing but regressed short generated CUDA sequence throughput and increased total input timing
- limiting first-block worker count by attempts per worker regressed short main-target CUDA throughput and did not produce a stable gain
- caching decoded salt bytes in `Argon2Params` preserved the golden CUDA hash but did not improve repeated generated-batch throughput
- parallel generated-key construction with one random generator per worker preserved correctness but regressed same-settings d8/b2048 throughput, so keygen parallelization should not be retried without a different design
- increasing CUDA finalize timing chunks from 64 to 256 preserved the golden CUDA hash but regressed a d8/b2048 generated CUDA run, so keep the smaller chunk unless a broader timing redesign changes the tradeoff
- pre-sizing the base64 output string and writing by index preserved the golden CUDA hash but did not improve d8/b2048 generated CUDA throughput, so keep the reserved `push_back` encoder unless a broader finalization redesign changes allocation behavior
- changing Argon2 `store32` to a little-endian `memcpy` fast path preserved the golden CUDA hash but produced noisy and then regressed d8/b2048 generated CUDA runs, so keep the explicit byte stores
- reusing per-chunk finalized hash strings with an output-parameter base64 encoder preserved the golden CUDA hash and lowered some finalize timing samples, but full d8/b2048 generated CUDA throughput remained noisy and regressed on confirmation
- parallelizing CUDA final hash materialization across CPU threads preserved the golden CUDA hash and lowered `finalize_hash_ms`, but benchmark subprocesses exited unstably during d8/b2048 generated CUDA runs, so keep finalization serial unless backend output-memory lifetime and thread-safety are redesigned
- moving final Argon2 digest materialization into a CUDA post-kernel produced the correct golden hash once, but repeated CUDA hash-one checks exited with access-violation status and single-hash kernel timing inflated to about 1.68s, so keep final digest materialization on the host until the device-side BLAKE2b/finalization design is rebuilt and stabilized
- default-constructing `Blake2b` without zero-initializing its state preserved the golden CUDA hash, but a d8/b2048 generated CUDA run regressed to 48.4k H/s median with 26.7% spread versus the latest accepted 52.4k H/s confirmation, so keep the explicit constructor initialization
- returning early from zero-length `Blake2b::update` calls preserved the golden CUDA hash, but a d8/b2048 generated CUDA run reached only 50.2k H/s median with 18.4% spread versus the latest accepted 52.4k H/s confirmation, so keep the simpler update path
- caching decoded salt bytes inside `Argon2Params` after the stack prehash optimization passed focused tests but made the CUDA golden hash command exit without JSON, so decoded salt caching remains rejected
- merging the whole no-secret/no-associated-data Argon2 initial hash input into one stack buffer preserved the golden CUDA hash, but confirmation only reached 67.7k H/s versus the accepted 67.1k H/s while slightly increasing `first_block_ms` per attempt, so it is too close to noise to keep
- merging `digestLong`'s 4-byte output length and 72-byte prehash seed into one update preserved the golden CUDA hash, but regressed d8/b2048 generated CUDA to 66.0k H/s median with 13.3% spread versus the accepted 67.1k H/s confirmation
- specializing `digestLong` for the 1024-byte Argon2 first-block output preserved the golden CUDA hash, but a d8/b2048 generated CUDA smoke regressed to 54.5k H/s median versus the current 58.3k H/s stable baseline and worsened per-attempt first-block/finalization timings
- replacing the 16 `Blake2b::compress` little-endian 64-bit loads with one 128-byte `memcpy` preserved the golden CUDA hash, but a d8/b2048 generated CUDA smoke regressed to 48.5k H/s median with about 57% spread and worse per-attempt first-block timing
- constructing generated random keys as a fixed-size string and writing hex nibbles by index preserved the golden CUDA hash, but a d8/b2048 generated CUDA smoke reached only 50.0k H/s median with 37% spread and higher per-attempt keygen timing than the accepted generator path
- caching the resolved CUDA device id inside `CudaHashBackend` preserved the golden CUDA hash, but a d8/b2048 generated CUDA smoke had warmup and repeated subprocess exits with code 3221226356, so keep per-batch device info lookup unless backend lifetime handling is redesigned
- replacing salt hex decoding's `substr` plus `std::stoi` path with direct nibble decoding preserved the golden CUDA hash, but a d8/b2048 generated CUDA comparison was unchanged at +0.76% median with an unstable 13.2% after-run spread, so keep the simpler decoder unless salt handling is redesigned more broadly
- caching CUDA activation inside a `CudaHashBackend` object for the current thread preserved the golden CUDA hash but reproduced benchmark subprocess access-violation exits with code 3221226356 in warmup/measured d8/b2048 generated runs, so keep per-batch `activate()` unless CUDA backend lifetime and shutdown ordering are redesigned
- caching two CUDA `KernelRunner` instances inside `CudaBackend` to reuse alternating difficulty shapes preserved printed JSON for golden and d8/d64 sequence commands, but the subprocess exited with access-violation status after output, so runner caching remains rejected until CUDA runner ownership and teardown ordering are redesigned
- replacing `KernelRunner` host staging buffers with `cudaMallocHost` pinned host allocations preserved the golden CUDA hash and reduced H2D/D2H timing in a generated CUDA d8/b2048 A/B run, but the same 10-second repeat-3 comparison regressed median throughput from 42.5k H/s to 35.7k H/s with noisy spreads, so keep ordinary host buffers unless a broader transfer-overlap design changes the tradeoff
- adding a `lanes == 1` fast path that manually emits the two first-block digests preserved the golden CUDA hash, but generated CUDA d8/b2048 normal-path comparison regressed median throughput from 34.2k H/s to 22.5k H/s with very high spread, so keep the generic lane loops unless a broader first-block redesign changes the compiler/runtime behavior
- setting an explicit first-block worker cap should remain a benchmark tuning option, not a default change, until longer confirmations show stable cross-scenario gains; the first d8/b1024 confirmation favored auto by median and found capped runs noisy or regressed
- making CUDA event transfer/kernel timings opt-in for default runs preserved the golden CUDA hash and kept default d8/b2048 throughput effectively unchanged at 76.7k H/s versus the 77.1k H/s baseline, but the paired detailed-timing scenario exited invalid with code 3221225477, so keep the existing always-available CUDA event timing until the event lifetime design is changed more broadly
- replacing the `Blake2b` rotate macro with an MSVC `_rotr64` intrinsic wrapper preserved focused tests and the CUDA golden hash, but d8/b2048 generated CUDA measured about `77.7k H/s` median with `11.9%` spread versus the stable `78.3k H/s` Release baseline, so keep the original rotate expression unless a broader compiler or Blake2b rewrite changes the evidence
- changing the `Blake2b` sigma table from `unsigned int` to `std::uint8_t` preserved focused tests and the CUDA golden hash, but d8/b2048 generated CUDA regressed to about `17.9k H/s` median with `59.4%` spread versus the stable `78.3k H/s` Release baseline, so keep the original `unsigned int` table layout

Measurement cautions:

- A later clean-source post-revert d8/b2048 confirmation was much slower at about `26.2k H/s` median with `16.5%` spread while the host CPU load was observed near saturation. Treat that run as a low-trust environment sample, not as a new baseline or a code regression.
- Benchmark reports now include public-safe environment metadata with aggregate CPU load samples around each benchmark subprocess and `benchmark_trust`. Do not accept CPU-side input/first-block throughput conclusions from reports marked `benchmark_trust: low` unless the result is only being used to diagnose environment noise.
- Measurement-only update: per-command environment sampling was validated with focused tests and a real CUDA smoke. A one-warmup, one-repeat CUDA smoke produced a sanitized summary with `sample_count: 4` and `benchmark_trust: normal`, confirming that automation can now detect mid-run CPU load spikes more reliably.

Do not retry rejected experiments unless the implementation shape has changed enough to remove the original failure mode and the new attempt includes correctness cross-checks.

## Next Autonomous Iteration

Start the next cycle from the latest clean commit and this decision tree:

1. If the worktree is dirty, identify whether it is a previous-agent experiment. Revert only rejected experiments owned by the current goal.
2. Run the focused Hash API tests before editing performance code.
3. Build or reuse the clean Release CUDA binary from the configured preset.
4. Run the CUDA golden hash check before trusting benchmark data.
5. Refresh or load the d8/b2048 generated-key CUDA baseline with build metadata.
6. If the baseline is unstable, rerun d8/b2048 and include d8/b1024 as a supplemental comparison before changing code.
7. Prefer a Track C experiment while `input_ms` and `first_block_ms` dominate.
8. Do not repeat rejected salt caching, decoded salt caching, activation caching, pinned host staging, runner caching, first-block lane fast paths, digestLong specializations, or `_rotr64` rotate changes without a materially different implementation shape.

Good next experiment shapes:

- Add measurement that separates first-block digest expansion from scheduling overhead more clearly without adding default benchmark overhead.
- Reduce generated first-block preparation allocations or copies if inspection finds a hot local allocation that is not already rejected.
- Improve first-block scheduling or worker chunking only behind an explicit benchmark knob until stable repeated evidence supports a default.
- Improve architecture boundaries that let CUDA backend state, difficulty-derived setup, or timing metadata be reused safely without activation or runner lifetime regressions.

## Phase Plan

### Phase 0: Baseline And Reproducibility

Goal: create a reliable performance baseline before changing kernels or memory behavior.

Tasks:

- Confirm current full CUDA build instructions are public-safe and reproducible.
- Use a Release CUDA build configured with an explicit modern architecture set or a public architecture-specific preset for benchmark claims.
- Avoid stale build directories whose cached `CMAKE_BUILD_TYPE` or `CMAKE_CUDA_ARCHITECTURES` differs from the intended benchmark target.
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
- Revisit pinned host memory only for transfer-heavy paths after a broader transfer-overlap or buffer-lifetime design exists.
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
cmake --build <hashapi-smoke-build-dir> --preset hashapi-cli-smoke-mingw
python scripts/hash_api_benchmark.py --binary <hashapi-cli> --seconds 1
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

Isolation benchmark:

```bash
python scripts/hash_api_benchmark.py --binary <miner-binary> --backend cuda --device 0 --preset isolation --seconds 4 --warmup 1 --repeat 3 --no-xuni --output .benchmarks/isolation.json
```

Variable-difficulty smoke:

```bash
python scripts/hash_api_benchmark.py --binary <miner-binary> --backend cuda --device 0 --preset difficulty-sequence --seconds 2 --warmup 1 --repeat 3 --no-xuni --output .benchmarks/difficulty-sequence-smoke.json
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

If two reports used different scenario names for the same backend, device, difficulty, batch size, seconds, warm-up, repeat, key mode, XUNI mode, and detailed-timing mode, compare by configuration instead of by name:

```bash
python scripts/hash_api_compare.py .benchmarks/before.json .benchmarks/after.json --match-by config --fail-on-regression --min-change-pct 1
```

Transfer-focused d8/b2048 checkpoint:

```bash
python scripts/hash_api_benchmark.py --binary <miner-binary> --backend cuda --device 0 --scenario name=cuda-transfer-before-d8-b2048,backend=cuda,difficulty=8,batch_size=2048,seconds=2,device=0,detailed_timings=true --warmup 1 --repeat 3 --no-xuni --output .benchmarks/transfer-before.json --sanitized-output .benchmarks/transfer-before-summary.json
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

1. Maintain current local CUDA baselines under `.benchmarks/` with warm-up and repeated runs, including d8/b2048 for continuity and d8/b1024 when b2048 is unstable locally.
2. Run stable custom batch-size scans for common difficulty ranges, keeping d8/b2048 for continuity and d8/b1024 for the current local unstable-baseline case.
3. Measure same-`m` warm loops versus alternating `m=diff` warm loops.
4. Reduce CPU-side generated-key and first-block preparation overhead where `input_ms` dominates.
5. Cache difficulty-derived setup only when `m`, salt, key mode, batch shape, backend state, and device state make it provably safe.
6. Reduce per-batch allocations and repeated normalization inside `src/hashapi/CudaHashBackend.cpp`.
7. Measure CUDA allocation, copy, launch, and finalization overhead before rewriting kernel logic.
8. Revisit pinned host memory only if profiling shows transfer cost dominates after input/setup overhead is reduced.
9. Extend batch-size tuning toward runtime autotuning after stable cross-difficulty data exists.
10. Tune launch parameters only after CPU-side overhead is under control.
11. Add optional autotuning once enough benchmark data justifies it.
12. Add profiler-backed CUDA kernel work only after benchmark timing shows compute is the dominant bottleneck.

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
5. Ensure no local paths, usernames, hostnames, secrets, raw benchmark reports, or private hardware identifiers are staged.
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
- public history rewrite is needed for commits that may already have been shared

Otherwise, keep moving through the next smallest measurable optimization step.

## Definition Of Done

This long-running goal is complete when one of these is true:

- throughput improves by at least 1000% over the initial measured baseline while preserving correctness and no obvious low-risk improvements remain
- repeated well-scoped optimization attempts plateau and the remaining bottleneck is documented with benchmark or profiler evidence
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
2. Build or reuse the local CUDA binary.
3. Run the golden CUDA hash check.
4. Run a short main-target CUDA benchmark.
5. Run or load repeated d8/b2048 and d8/b1024 baselines.
6. Use detailed timing breakdowns to choose between input generation, first-block preparation, setup caching, allocation reuse, launch tuning, or matching/finalization work.
7. Do not retry rejected pinned host staging, activation caching, salt decode, or first-block lane fast-path experiments unless the implementation shape has materially changed.
