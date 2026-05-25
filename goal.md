# Codex Goal: Continuous Hash Throughput Optimization

This file is the stable entrypoint for long-running `/goal` execution. The
detailed operating manual, experiment ledger, and latest evidence live in
`docs/HASH_OPTIMIZATION_GOAL.md`.

## Goal Command

Use this objective when starting, resuming, or recreating the goal:

```text
/goal Follow goal.md and docs/HASH_OPTIMIZATION_GOAL.md. Continuously optimize XenblocksMiner Hash API CUDA hashing throughput for the real mining workload where t=1 and s=1 are fixed, the current implementation is single-lane p=1, and only m=difficulty may change between sessions. Preserve exact argon2id-xen semantics, the public Hash API contract, and reproducible benchmark evidence. Keep iterating through inspect, validate, benchmark, optimize, document, privacy-check, and commit cycles until verified same-scenario warm throughput improves by at least 1000% over the recorded baseline, or until evidence-backed plateau/practical-limit criteria are met. Work autonomously without approval prompts except for the stop conditions in this file. Keep code, docs, tests, benchmark names, and commit messages in English. Never commit local paths, private machine details, raw benchmark reports, secrets, wallet/private data, local hardware identifiers, or local GPU model names.
```

If `/goal` is already active and points at this repository's hash optimization
work, do not recreate it. Treat the active goal as the runtime handle and this
file as the local control contract.

## Mission

Minimize end-to-end time per valid Hash API CUDA hash attempt.

The workload is narrow by design:

- `t = 1` is fixed.
- `s = 1` is fixed.
- `p = 1` / single-lane execution is fixed as represented by the current code.
- `m = difficulty` / `diff` is the only workload parameter expected to change.

The primary metric is warm steady-state generated-key CUDA attempts per second
for the same Hash API scenario. Secondary metrics are median milliseconds per
attempt, stage-level timing percentages, invalid subprocess rate, and portability
across CUDA-capable local GPUs, RTX 3050-class GPUs, and higher-end CUDA GPUs.

The aspirational target is a verified 1000% throughput improvement over the
selected same-scenario baseline. A 1000% improvement means `11x` the baseline,
not merely `10x` the baseline. If the target is not reachable on the current
GPU, continue until profiler or benchmark evidence supports a practical plateau.

## Current Truth

Read `docs/HASH_OPTIMIZATION_GOAL.md` before every new optimization cycle. The
short state below is only the resume snapshot.

- The reusable Hash API extraction is usable for isolated optimization.
- "CLI API" means the command-line Hash API entrypoints: `hash-one`,
  `hash-batch`, and `hash-benchmark`. It is not a hosted HTTP API, websocket API,
  frontend API, marketplace API, wallet flow, or full platform API.
- The extracted pieces include a request/result contract, CPU/reference and CUDA
  backend paths, JSON-friendly command output, golden-hash checks, repeatable
  benchmark scripts, comparison tooling, timing metadata, and CUDA tuning knobs.
- The miner-generated CUDA path now opts into the validated GPU first-block path
  and automatic first-block chunk selection where supported.
- Current local d8 generated CUDA GPU-first evidence favors automatic batch size
  `4096`, with sanitized confirmation around `196.86k H/s` median, normal
  benchmark trust, and zero invalid subprocesses.
- d1 and d64 tuning remain separate from d8. Do not generalize the local d8
  batch-size result to all GPU classes without confirmation.
- The most recent uncommitted `Blake2b` / `digestLong` one-shot prefix experiment
  was rejected because it preserved correctness but regressed d8 GPU-first
  throughput in a same-scenario smoke. Do not keep or retry that exact shape.
- The earlier `gpu_final_hashes` design was rejected because it produced
  access-violation subprocess exits. Do not reintroduce device-side final hash
  output unless output lifetime, synchronization, and repeated wrapper stability
  are materially redesigned.
- The current dominant post-GPU-first bottleneck is CPU finalization and remaining
  host-side overhead, especially `argon2_finalize_ms`, unless newer detailed
  timings show otherwise.
- Local commits ahead of the remote are retained progress. An `ahead N` status
  does not mean earlier local commits disappeared.

## Scope

Work only where it directly improves the hash path or makes future hash
optimization safer:

- `src/hashapi/`
- CUDA backend files and CUDA kernels
- Argon2 and Blake2b hot paths
- benchmark and comparison scripts under `scripts/`
- focused Hash API tests under `tests/unit/`
- narrowly related miner integration
- `goal.md` and `docs/HASH_OPTIMIZATION_GOAL.md`

Do not spend goal time on frontend, marketplace, wallet, settlement, auth,
database, UI, or unrelated platform work unless required to preserve Hash API
integration.

## Architecture Target

Keep the hash core pure enough that AI agents can optimize it repeatedly without
running the whole platform.

The target architecture should provide:

- a stable Hash API request/result contract independent of business logic
- CPU/reference and CUDA backends behind the same contract
- CLI automation through `hash-one`, `hash-batch`, and `hash-benchmark`
- machine-readable benchmark reports and comparison tools
- explicit CUDA tuning knobs for batch size, GPU first blocks, first-block chunk
  policy, difficulty sequences, detailed timings, and auto batch selection
- tuning decisions based on public device properties, compute capability, memory
  limits, and measured evidence rather than local machine assumptions
- clean fallbacks for RTX 3050-class and higher-end CUDA GPUs

If the current structure blocks serious optimization, refactor toward this
boundary first, then continue performance work.

## Progress Accounting

Measure progress against named same-scenario baselines, not against memory.

Baseline selection:

- use the earliest trustworthy machine-readable CUDA benchmark for the same
  scenario after Hash API extraction
- if no trustworthy report exists, create a new baseline and document it
- keep raw reports ignored under `.benchmarks/` or `benchmark-results/`

Best result selection:

- same backend, key mode, device index, difficulty or difficulty sequence, batch
  size policy, GPU-first setting, first-block policy, XUNI mode, binary type,
  warm-up count, repeat count, and seconds
- normal benchmark trust and no invalid subprocesses
- stable spread for serious claims, or explicitly marked smoke-only evidence

Improvement formula:

```text
(best_median_hps - baseline_median_hps) / baseline_median_hps * 100
```

Do not claim a large gain from one noisy run. Confirm speed claims with repeated
same-scenario runs, correctness checks, report-quality checks, and public-safe
documentation.

## Autonomous Loop

Normal optimization cycles must not ask for approval. Run:

```text
inspect -> validate -> benchmark -> change one thing -> validate -> benchmark -> decide -> document -> privacy-check -> commit -> repeat
```

Each cycle must have one hypothesis and one changed implementation shape.
Measurement-only cycles must improve benchmark quality, correctness confidence,
or future optimization choice.

Default resume sequence:

1. Read `goal.md`.
2. Read `docs/HASH_OPTIMIZATION_GOAL.md`.
3. Run `git status -sb`.
4. Inspect recent benchmark and optimization commits.
5. Classify dirty files before editing.
6. If dirty work is a rejected experiment owned by this goal, revert only that
   experiment or document the rejection.
7. Run focused correctness tests when code changed or validation is stale.
8. Build or reuse a clean Release CUDA binary.
9. Run CUDA golden-hash checks before trusting CUDA benchmarks.
10. Refresh or load the current d8 generated-key CUDA GPU-first baseline.
11. Refresh or load the preferred variable-`m` sequence baseline.
12. Pick exactly one measurable bottleneck.
13. Make the smallest useful source, test, benchmark, or docs change.
14. Re-run focused validation.
15. Run same-scenario before/after benchmarks for performance claims.
16. Update `docs/HASH_OPTIMIZATION_GOAL.md` with accepted or rejected evidence.
17. Stage only intended files.
18. Run whitespace and privacy checks.
19. Commit a coherent English slice.
20. Continue to the next cycle unless a stop condition is reached.

End each turn in one of these states:

- accepted: useful change, correctness passed, benchmark evidence reviewed,
  privacy checks passed, and a small commit was made
- rejected: current uncommitted experiment was reverted or documented as rejected
  evidence
- measurement-only: benchmark, timing, test, or documentation infrastructure was
  improved and committed
- blocked: a stop condition was reached with a concrete blocker

## Next-Step Selector

Choose the next step in this order:

1. If the worktree is dirty, identify whether it is previous-agent work, a
   rejected experiment, a user change, or an unrelated local artifact.
2. If correctness validation is stale, run focused tests and golden CUDA checks.
3. If the binary changed, rebuild before benchmarking.
4. If no trustworthy current baseline exists, run a short smoke and then a stable
   d8 generated-key CUDA GPU-first baseline.
5. If benchmark results are noisy or invalid, improve measurement quality or rerun
   a narrower scenario before changing performance code.
6. If `argon2_finalize_ms` or `finalize_ms` dominates after GPU-first first-block
   preparation, isolate finalization, result collection, base64, matching, and
   output ownership before changing parallelism or device finalization.
7. If `input_ms` or `first_block_ms` again dominates, target generated input,
   salt/key materialization, and Argon2 first-block preparation.
8. If setup dominates variable `m=diff` sequences, target safe backend lifecycle
   reuse or difficulty-derived setup reuse.
9. If CUDA compute, transfer, or launch timing dominates, target memory layout,
   launch geometry, occupancy, streams, and transfer overlap with profiler-backed
   evidence.
10. If stable manual settings repeatedly beat defaults, add conservative autotuning
    based on public device properties and measured stability.
11. If same-difficulty gains flatten, validate variable `m=diff` sequences before
    choosing another fixed-`m` change.

Avoid broad rewrites unless smaller measured changes are blocked by the current
structure.

## Standard Validation

Use concrete local paths only in shell commands. Never commit those paths.

Focused Hash API tests:

```bash
python -m pytest tests/unit/test_hash_api_contract.py tests/unit/test_hash_api_service.py tests/unit/test_hash_api_benchmark.py tests/unit/test_hash_api_compare.py -q
```

Golden CUDA hash check:

```bash
<miner-binary> hash-one --backend cuda --salt aabbccddeeff0011 --key 0000000000000000000000000000000000000000000000000000000000000000 --difficulty 8 --device 0 --no-xuni --json
```

Expected `hash`:

```text
Rs/bYUkZR8dczsQh/KvLAyJGThm8HtjnIJVJEkldK+TQtBLdGf2tULquitejKRO7URrkbgieR7Sq42k5mNYVdw
```

Also run the same golden check with `--gpu-first-blocks` after CUDA backend or
GPU-first changes.

Short d8 GPU-first smoke:

```bash
python scripts/hash_api_benchmark.py --binary <miner-binary> --backend cuda --device 0 --scenario name=cuda-d8-auto-batch-gfb-smoke,backend=cuda,difficulty=8,batch_size=0,auto_batch_size=true,gpu_first_blocks=true,first_block_dynamic_chunk_auto=true,seconds=2,warmup=1,repeat=2 --no-xuni --output .benchmarks/d8-auto-batch-gfb-smoke.json
```

Preferred variable-`m` smoke:

```bash
python scripts/hash_api_benchmark.py --binary <miner-binary> --backend cuda --device 0 --difficulty-sequence 1,8,64 --sequence-auto-batch-size --sequence-first-block-dynamic-chunk-auto --gpu-first-blocks --seconds 2 --warmup 1 --repeat 2 --no-xuni --output .benchmarks/difficulty-sequence-gfb-smoke.json
```

Stable comparison rules:

- use at least `8` to `10` seconds, warm-up `1`, repeat `3` for serious claims
- keep backend, device index, difficulty or sequence, batch policy, GPU-first
  setting, first-block policy, XUNI mode, key mode, binary type, and detailed
  timing mode matched
- compare median warm throughput first, then spread, invalid subprocesses,
  `report_ok`, `report_quality_ok`, `benchmark_trust`, and per-attempt timings
- treat smoke-only data as operational validation unless the doc says otherwise

Comparison command:

```bash
python scripts/hash_api_compare.py .benchmarks/before.json .benchmarks/after.json --match-by config --fail-on-regression --fail-on-report-quality --min-change-pct 1
```

## Experiment Acceptance

Accept a performance change only when all are true:

- changed path has correctness coverage
- golden CUDA hash still matches when CUDA is affected
- benchmark report has no invalid subprocesses
- same-scenario median improves by at least `1%` beyond noise, unless the change is
  a measurement-only or architecture-only prerequisite
- staged diff contains no local paths or private details
- docs or commit body record public-safe before/after evidence when useful

Reject an experiment when it:

- changes hash output or request/result semantics
- causes invalid subprocesses, no-JSON exits, hangs, or unstable teardown
- regresses same-scenario throughput beyond noise
- wins only a nested metric while end-to-end throughput regresses
- depends on private local hardware identifiers or paths

Rejected experiments should be reverted if uncommitted. Document the rejection
only when it prevents future repeated work.

## Privacy And Public History

This is a public open-source repository. Keep tracked files and git history clean.

Never commit:

- local absolute paths
- usernames
- hostnames
- private machine identifiers
- raw benchmark reports with command lines or binary paths
- secrets, tokens, cookies, private key material, wallet credentials, or personal
  addresses
- local GPU model names or other local hardware identifiers

Use public-safe placeholders:

- `<miner-binary>`
- `<build-dir>`
- `<cuda-root>`
- `<vcpkg-toolchain>`
- `CUDA-capable local GPU`
- `RTX 3050-class GPU`
- `higher-end CUDA GPU`

Before every commit:

```bash
git diff --cached --check
git diff --cached
```

Review the staged diff for private paths, usernames, hostnames, secrets, raw
benchmark reports, and hardware identifiers. If a leak appears in an unpushed
local commit, fix local history before continuing. If a leak may already have
been shared publicly, stop and ask before rewriting public history.

## Commit Discipline

Use English commit messages. Preferred prefixes:

- `perf(hash-api):`
- `perf(cuda):`
- `refactor(hash-api):`
- `refactor(cuda):`
- `test(hash-api):`
- `test(cuda):`
- `docs(hash-api):`
- `docs(goal):`

Commit only coherent slices:

- measurement-only tooling or docs after focused checks and privacy review
- performance code after correctness checks and same-scenario comparison
- rejected-experiment documentation only when it prevents repeated work

Do not bundle unrelated refactors with benchmark claims.

## Stop Conditions

Stop and ask the user only if:

- a dirty user change conflicts with required edits
- a command requires credentials or unavailable proprietary software
- a design choice would permanently break the public Hash API contract
- an optimization requires changing hash semantics
- a CUDA change appears hardware-specific and risky without access to that
  hardware class
- tests reveal a pre-existing issue whose fix would significantly broaden scope
- public history rewrite is needed for commits that may already have been shared

Otherwise, keep moving through the next smallest measurable optimization step.

## Completion Rule

Do not mark the goal complete because one iteration finished, a benchmark was
noisy, a context window became long, or the next step is uncertain.

The goal is complete only when one of these is proven:

- throughput improves by at least 1000% over the selected same-scenario baseline,
  correctness is preserved, and no obvious low-risk improvements remain
- repeated well-scoped attempts plateau and the remaining bottleneck is documented
  with benchmark or profiler evidence
- profiler evidence shows the implementation is near the practical hardware limit
  for the tested GPU class

Plateau evidence requires at least three consecutive well-scoped attempts against
the current dominant bottleneck with less than `3%` confirmed improvement, plus a
public-safe note in `docs/HASH_OPTIMIZATION_GOAL.md` explaining the remaining
bottleneck and risk/reward tradeoff.

## Immediate Queue

Start here unless `docs/HASH_OPTIMIZATION_GOAL.md` contains newer evidence:

1. Confirm the worktree is clean or classify dirty files.
2. Run focused Hash API tests.
3. Build or reuse the clean Release CUDA binary.
4. Run CUDA golden hashes with and without `--gpu-first-blocks`.
5. Run a short d8 auto-batch GPU-first smoke.
6. Run or load a stable d8 auto-batch GPU-first baseline.
7. Run or load a preferred variable-`m` GPU-first sequence baseline.
8. Use detailed timing to confirm whether finalization, input/first-block work,
   setup, or CUDA compute dominates now.
9. Pick one bottleneck and one implementation shape.
10. Validate, benchmark, document, privacy-check, commit, and continue.

The next agent should make measurable progress, not restate this plan.
