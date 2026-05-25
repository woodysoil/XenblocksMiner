# Codex Goal: Continuous Hash Throughput Optimization

This file is the stable entrypoint for long-running `/goal` execution. The detailed
operating manual and experiment ledger live in `docs/HASH_OPTIMIZATION_GOAL.md`.

## Goal Command

Use this command when starting or recreating the goal:

```text
/goal Follow goal.md and docs/HASH_OPTIMIZATION_GOAL.md. Continuously optimize XenblocksMiner Hash API CUDA hashing throughput for the real mining workload where t=1 and s=1 are fixed, the current implementation runs single-lane p=1, and only m=difficulty may change between sessions. Keep iterating until the best verified warm steady-state rate is at least 1000% over the recorded same-scenario baseline, or until evidence-backed plateau/practical-limit criteria are met. Preserve exact argon2id-xen semantics and the public Hash API contract. Work autonomously through inspect, benchmark, optimize, validate, document, and commit cycles without asking for approval except for the listed stop conditions. Keep all code, docs, tests, benchmark names, and commit messages in English. Never commit local paths, private machine details, raw benchmark reports, secrets, wallet/private data, local hardware identifiers, or local GPU model names.
```

If `get_goal` already shows an active objective that points at this file and
`docs/HASH_OPTIMIZATION_GOAL.md`, do not recreate it. Treat the active goal as
the runtime handle and this file as the stronger local contract.

## Mission

Continuously reduce the time required to complete each valid Hash API CUDA hash
attempt.

The workload is intentionally narrow:

- `t = 1` is fixed.
- `s = 1` is fixed.
- `p = 1` / single-lane execution is fixed as represented by the current code.
- `m = difficulty` / `diff` is the only expected workload parameter that may
  change between benchmark or mining sessions.

Optimize the current CUDA-capable local GPU first, but keep the architecture
portable to RTX 3050-class and higher-end CUDA GPUs. Use public device
properties, compute capability, memory limits, runtime tuning, and benchmark
evidence instead of local machine assumptions.

The aspirational target is a verified 1000% throughput improvement over the
selected same-scenario baseline. If that target is not reachable on the current
GPU, continue until profiler or benchmark evidence supports a practical plateau.

## Scope

Focus on the reusable hash core and the automation surface that lets future AI
iterations run without the full platform:

- `src/hashapi/`
- CUDA backend files and Argon2/Blake2b hot paths
- benchmark scripts under `scripts/`
- focused Hash API tests under `tests/unit/`
- narrowly related miner integration
- `goal.md` and `docs/HASH_OPTIMIZATION_GOAL.md`

Do not spend goal time on frontend, marketplace, wallet, settlement, auth,
database, UI, or unrelated platform work unless a change is required to preserve
the Hash API integration.

## Current State

The Hash API extraction is already usable for isolated optimization. The current
automation surface is the command-line adapter:

- `hash-one`
- `hash-batch`
- `hash-benchmark`

"CLI API" means these command-line Hash API entrypoints. It is not a hosted HTTP
API, websocket API, frontend API, marketplace API, wallet flow, or full platform
API.

The extracted pieces provide:

- a request/result Hash API contract
- CPU/reference and CUDA backend paths
- JSON-friendly command output
- golden-hash checks
- repeatable benchmark scripts
- before/after comparison tooling
- timing metadata for setup, input generation, compute, finalization, and nested
  CUDA/finalization stages
- tuning knobs for batch size, variable difficulty sequences, first-block worker
  diagnostics, and first-block dynamic chunk policy

The current trusted local evidence still points to CPU-side generated input and
first-block preparation as the dominant generated-key CUDA bottleneck. Keygen-only
micro-optimizations are not the priority unless newer detailed timing shows key
generation has become dominant.

Latest important resume facts:

- The active `/goal` is already running for Hash API/CUDA throughput work.
- The branch can be ahead of the remote during autonomous work. Ahead commits are
  retained local progress unless the user explicitly asks to squash, push, reorder,
  or rewrite history.
- Many earlier optimization commits are still visible in the normal git log; an
  `ahead 1` status means only the final local commit is currently ahead of the
  tracked remote, not that earlier work disappeared.
- Current miner-equivalent d8 tuning evidence favors generated CUDA batches around
  b3072 with automatic first-block dynamic chunk selection, while d8/b2048 remains
  an important continuity scenario for historical comparisons.
- Current preferred variable-`m` evidence uses `difficulty_sequence=1,8,64`,
  automatic sequence batch sizing, automatic sequence first-block dynamic chunk
  selection, warm-up, repeated samples, and no-XUNI.
- The most recent accepted/rejected experiments and exact benchmark numbers are in
  `docs/HASH_OPTIMIZATION_GOAL.md`; read that file before starting a new experiment.

## Progress Accounting

Measure progress against a named baseline, not against terminal memory.

- Baseline: the earliest trustworthy machine-readable CUDA benchmark for the
  selected scenario after the Hash API extraction, or a newly documented baseline
  if no trustworthy report exists.
- Best result: the highest confirmed median warm throughput for the same scenario,
  key mode, backend, difficulty, batch size, XUNI mode, and correctness surface.
- Improvement formula:

```text
(best_median_hps - baseline_median_hps) / baseline_median_hps * 100
```

A 1000% improvement means the confirmed best median throughput is at least `11x`
the selected baseline median throughput. Do not confuse this with reaching
`1000% of baseline`, which would be `10x`.

Do not claim a large gain from a single noisy run. Confirm performance claims with
stable repeated runs, matching settings, correctness checks, and public-safe
documentation.

## Completion Rule

Do not mark the goal complete because one iteration finished, a benchmark was
noisy, a context window became long, or the next step is uncertain.

The goal is complete only when one of these outcomes is proven:

- throughput improves by at least 1000% over the selected same-scenario baseline,
  correctness is preserved, and no obvious low-risk improvements remain
- repeated well-scoped attempts plateau and the remaining bottleneck is documented
  with benchmark or profiler evidence
- profiler evidence shows the implementation is near the practical hardware limit
  for the tested GPU class

Plateau evidence requires at least three consecutive well-scoped attempts against
the current dominant bottleneck with less than 3% confirmed improvement, plus a
public-safe note in `docs/HASH_OPTIMIZATION_GOAL.md` explaining the remaining
bottleneck and risk/reward tradeoff.

## Autonomous Runtime Protocol

Normal optimization cycles must not ask for approval. The agent may autonomously:

- inspect files, diffs, git logs, and status
- run builds, tests, CUDA smoke checks, and benchmark scripts
- create ignored benchmark artifacts under `.benchmarks/` or `benchmark-results/`
- edit scoped source, tests, scripts, build files, and docs
- revert only the current uncommitted failed experiment
- make small validated English commits

Stop only for the stop conditions in this file.

Default continuation turn:

1. Read `goal.md`.
2. Read `docs/HASH_OPTIMIZATION_GOAL.md` after resume, compaction, or uncertainty.
3. Run `git status -sb`.
4. Inspect recent benchmark/optimization commits.
5. Classify any dirty files before editing.
6. If useful dirty work exists, finish its validation before starting new work.
7. Select exactly one measurable bottleneck or architecture cleanup.
8. Run correctness checks before trusting speed results.
9. Make the smallest useful change.
10. Re-run focused validation.
11. Benchmark before/after with matching settings for performance claims.
12. Update `docs/HASH_OPTIMIZATION_GOAL.md` when the result changes future choices.
13. Stage only intended files.
14. Run whitespace and privacy checks.
15. Commit a coherent English slice.
16. Continue to the next measurable step.

Each turn should end in one of these states:

- accepted: correctness passed, the change is useful, privacy checks passed, and a
  small commit was made
- rejected: the current uncommitted experiment was reverted or documented as
  rejected evidence
- measurement-only: benchmark, timing, test, or documentation infrastructure
  improved and was committed without a speed claim
- blocked: an explicit stop condition was reached and the remaining blocker is
  concrete

## Next-Step Selector

At the start of each autonomous cycle, choose the next step by this order:

1. If the worktree is dirty, identify whether it is previous-agent work, a rejected
   experiment, a user change, or an unrelated local artifact.
2. If correctness validation is stale or the binary changed, run focused tests and
   the CUDA golden hash check first.
3. If no trustworthy current baseline exists, run or load the d8 generated-key CUDA
   continuity baseline.
4. If benchmark results are noisy, improve measurement quality or rerun a narrower
   scenario before changing performance code.
5. If `input_ms` or `first_block_ms` dominates, target generated input,
   salt/key materialization, and Argon2 first-block preparation.
6. If setup or lifecycle cost dominates, target Hash API/CUDA backend lifetime,
   difficulty-derived setup, validation, device selection, or allocation churn.
7. If CUDA compute, transfer, or launch timing dominates, target CUDA memory,
   launch geometry, occupancy, streams, or transfer overlap with profiler-backed
   evidence.
8. If finalization dominates, isolate argon2 finalization, base64, matching, and
   result collection before changing ownership or threading.
9. If stable manual settings repeatedly beat defaults, add conservative autotuning
   based on public device properties and measured stability.
10. If same-difficulty gains flatten, validate variable `m=diff` sequences before
    choosing another fixed-`m` change.

Avoid broad rewrites unless smaller measured changes are blocked by the current
structure.

## Standard Validation

Use concrete local paths only in the shell, never in committed docs or commit
messages.

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

Preferred variable-`m` smoke:

```bash
python scripts/hash_api_benchmark.py --binary <miner-binary> --backend cuda --device 0 --difficulty-sequence 1,8,64 --sequence-auto-batch-size --sequence-first-block-dynamic-chunk-auto --seconds 2 --warmup 1 --repeat 3 --no-xuni --output .benchmarks/difficulty-sequence-smoke.json
```

Stable comparison runs:

- use the same binary type, backend, device index, difficulty, batch size, key mode,
  XUNI mode, warm-up count, repeat count, seconds, and detailed-timing mode
- use at least `10` seconds, warm-up `1`, repeat `3` for serious throughput claims
- compare median warm throughput first, then spread and per-attempt timings
- treat smoke-only data as operational validation, not a committed performance claim

Before/after comparison:

```bash
python scripts/hash_api_compare.py .benchmarks/before.json .benchmarks/after.json --fail-on-regression --fail-on-report-quality --min-change-pct 1
```

## Benchmark Artifact Policy

Raw benchmark output must stay ignored:

- `.benchmarks/`
- `benchmark-results/`

Commit only public-safe summaries when they change future decisions. A useful
summary records:

- scenario
- backend
- difficulty
- batch size
- seconds
- warm-up count
- repeat count
- median before/after throughput
- percentage change
- dominant timing field
- conclusion

Do not commit raw reports that include command lines, binary paths, local hardware
identifiers, environment details, hostnames, or private machine data.

## Privacy And Public History

This is a public open-source repository. Keep tracked files and git history clean.

Never commit:

- local absolute paths
- usernames
- hostnames
- private machine identifiers
- secrets, tokens, cookies, private key material, wallet credentials, or personal
  addresses
- raw benchmark reports containing command lines, binary paths, hardware
  identifiers, or local environment details
- local GPU model names or other local hardware identifiers

Use public-safe placeholders in docs and commit messages:

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

Review the staged diff manually or with a local regex scan for the forbidden items
listed above. If a leak appears in an unpushed local commit, fix local history
before continuing. If a leak may already have been shared publicly, stop and ask
before rewriting public history.

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
- performance code after correctness checks and a same-scenario comparison
- rejected-experiment documentation only when it prevents repeated work

Do not bundle unrelated refactors with benchmark claims.

## Stop Conditions

Stop and ask the user only if:

- a dirty user change conflicts with required edits
- a command requires credentials or unavailable proprietary software
- a design choice would permanently break the public Hash API contract
- an optimization requires changing hash semantics
- a CUDA change appears hardware-specific and risky without access to that hardware
  class
- tests reveal a pre-existing issue whose fix would significantly broaden scope
- public history rewrite is needed for commits that may already have been shared

Otherwise, keep moving through benchmark, optimize, validate, document, and commit
cycles.

## Immediate Queue

Start here unless `docs/HASH_OPTIMIZATION_GOAL.md` contains newer evidence:

1. Run `git status -sb`.
2. Confirm recent commits and privacy state.
3. Run focused Hash API unit tests.
4. Build or reuse the clean Release CUDA binary.
5. Run the golden CUDA hash check.
6. Run a short main-target CUDA benchmark.
7. Refresh or load the d8 generated-key CUDA continuity baseline.
8. Refresh or load the preferred variable-`m` sequence baseline.
9. Use detailed timings to choose one Track C or Track D step.
10. Prefer first-block digest/preparation structure, generated input
    materialization, setup/lifecycle cleanup, or carefully isolated finalization
    diagnostics over more keygen-only work.
11. Do not retry rejected salt caching, decoded salt caching, activation caching,
    pinned host staging, runner-cache shapes, first-block lane fast paths,
    digestLong specializations, `_rotr64`, fixed-64-byte base64 fast path, initial
    hash prefix cache, indexed key-generation fill, or broad finalization
    parallelism unless the implementation shape materially changes.
12. Keep `docs/HASH_OPTIMIZATION_GOAL.md` updated with accepted and rejected
    evidence.

The next agent should make measurable progress instead of restating this plan.
