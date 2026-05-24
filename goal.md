# Codex Goal: Continuous Hash Throughput Optimization

## Goal Command

Use this exact command when starting or recreating the long-running goal:

```text
/goal Follow goal.md and docs/HASH_OPTIMIZATION_GOAL.md. Continuously optimize XenblocksMiner Hash API CUDA hashing throughput for the real mining workload where t=1 and s=1/p=1 are fixed and only m=difficulty may change between sessions. Keep iterating until the best verified warm steady-state rate is at least 1000% over the recorded baseline, or until evidence-backed plateau/practical-limit criteria are met. Verify progress with machine-readable Hash API benchmarks, golden hash checks, focused tests, privacy-clean staged diffs, and small validated commits. Preserve exact argon2id-xen semantics and the public Hash API contract. Work autonomously through inspect, benchmark, optimize, validate, document, and commit cycles without asking for approval except for the listed stop conditions. Keep all code, docs, tests, benchmark names, and commit messages in English. Never commit local paths, private machine details, raw benchmark reports, secrets, private wallet data, or local hardware identifiers.
```

This file is the compact persistent goal contract. `docs/HASH_OPTIMIZATION_GOAL.md` is the detailed operating manual, phase plan, and experiment ledger. A running `/goal` agent should read this file first, then use the detailed document for current evidence, rejected experiments, and the next measurable step.

The goal is thread-scoped and evidence-based. Do not mark it complete because one iteration finished, the context is long, a benchmark is noisy, or the next step is uncertain. Completion requires concrete evidence from tests, benchmark output, changed files, privacy checks, and the documented completion rule.

## Goal Runtime Protocol

This goal is intended to run for many continuation turns without human approval prompts. Each turn should make one measurable step and leave a useful checkpoint for the next turn.

Default turn shape:

1. Read `goal.md`, then `docs/HASH_OPTIMIZATION_GOAL.md` if context is stale or compacted.
2. Run `git status -sb` and inspect the latest benchmark or optimization commits.
3. Classify any dirty files before editing: previous-agent in-progress work, rejected experiment, user change, or unrelated local artifact.
4. Finish validation for useful dirty work before starting a new experiment.
5. Select one bottleneck or architecture cleanup from current benchmark evidence.
6. Make the smallest code, test, script, or doc change that advances that step.
7. Run correctness checks before trusting any speed result.
8. Run a smoke benchmark for instrumentation changes and a repeated comparison for performance claims.
9. Stage only intended files, run whitespace and privacy checks, then commit a coherent English slice.
10. Update `docs/HASH_OPTIMIZATION_GOAL.md` when a result changes future decisions.

The agent may run local builds, tests, CUDA smoke checks, benchmark scripts, ignored benchmark artifact writes, scoped edits, and small commits without asking. It should stop only for the stop conditions in this file.

Each turn must end in one of these states:

- accepted: correctness passed, the change is useful, privacy checks passed, and a small commit was made
- rejected: the current uncommitted experiment was reverted or documented as rejected evidence
- measurement-only: benchmark, timing, test, or documentation infrastructure improved and was committed without a speed claim
- blocked: an explicit stop condition was reached and the remaining blocker is concrete

## Long-Run State Model

Treat this file as the stable entrypoint and `docs/HASH_OPTIMIZATION_GOAL.md` as the detailed state ledger. A long-running `/goal` agent should keep enough state in committed English text that another agent can resume without private terminal history.

Every accepted or rejected performance iteration should update the detailed goal document when the result changes future decisions. The update should answer:

- what scenario was measured
- which bottleneck was targeted
- what changed
- which correctness checks passed
- what the before/after median throughput was, if this was a performance code change
- whether the result is accepted, rejected, or measurement-only
- what the next measurable experiment should be

Do not store private raw reports in git. Store raw JSON and stdout only under ignored local benchmark directories, then summarize public-safe findings in docs or commit bodies when the finding matters.

## Active Goal Objective

Run an autonomous, long-lived optimization loop for the extracted Hash API and CUDA backend. The practical target is to reduce time per real hash attempt as far as possible for the fixed workload:

- `t = 1`
- `s = 1` / `p = 1`
- `m = diff` / `difficulty`, which may change between sessions or benchmark sequences

The aspirational target is a verified 1000% throughput increase over the selected baseline. If that target is not reachable on the current hardware, keep iterating until the remaining bottleneck is supported by benchmark or profiler evidence and the risk of further changes is higher than the expected gain.

The current local CUDA-capable GPU is the first test platform. The resulting architecture must stay portable enough for future RTX 3050-class and higher-end CUDA GPUs by using public device properties, compute capability, explicit tuning parameters, and runtime measurements instead of private device names or local machine assumptions.

The execution target is not a single benchmark tweak. The goal is to keep improving the reusable hash core so an AI agent can repeatedly measure, change, validate, and commit without depending on frontend, marketplace, wallet, network, lease, or platform services.

If the current structure makes that loop awkward, first refactor toward a cleaner Hash API boundary, then continue performance work. Treat architecture cleanup as part of the performance goal when it removes repeated setup, hidden state, noisy timing, unsafe lifetime ownership, or platform coupling from the hash hot path.

## Long-Running Goal Contract

This goal follows the strongest Codex Goal shape and is designed for unattended `/goal` execution:

- Outcome: maximize real Hash API CUDA hashing throughput, with an aspirational 1000% improvement target, then continue until plateau or practical hardware-limit evidence is documented.
- Verification surface: focused Hash API tests, CUDA golden hash checks, machine-readable benchmark JSON, before/after comparison output, and committed diffs.
- Constraints: preserve exact `argon2id-xen` semantics, Hash API compatibility, mining result fields, target matching, privacy rules, and public-repo hygiene.
- Boundaries: focus on `src/hashapi/`, CUDA backend files, Argon2/Blake2b hot paths, benchmark scripts, tests, and narrowly related integration code.
- Iteration policy: choose the next experiment from measured timing bottlenecks, validate correctness before trusting speed, commit useful stable slices, then immediately continue to the next measurable bottleneck.
- Blocked stop condition: stop only when a listed stop condition is reached, no defensible optimization path remains, required tooling is unavailable, or public history rewrite decisions need the user.
- State policy: keep the goal files, commit messages, and benchmark summaries useful to future automation while keeping local paths, usernames, hostnames, private hardware identifiers, secrets, and raw reports out of tracked history.

The agent should not ask for approval during normal local optimization cycles. Builds, tests, CUDA smoke checks, benchmark runs, ignored local artifacts, scoped source edits, scoped documentation edits, and small validated commits are expected parts of the loop.

The agent should continue across compaction and resume events by reading this file first, then `docs/HASH_OPTIMIZATION_GOAL.md`, then the latest commits and dirty diff. If a previous agent left a correct dirty experiment in progress, finish its validation before starting a new experiment.

## Current Immediate State

This section should be refreshed whenever it would prevent duplicated work after a resume.

- The branch can be ahead of the remote during autonomous work. Ahead commits are retained local progress unless the user explicitly asks to squash, reorder, push, or rewrite history.
- The Hash API extraction is already usable for isolated optimization. The current stable automation surface is the CLI adapter: `hash-one`, `hash-batch`, and `hash-benchmark`.
- "CLI API" means those command-line Hash API entrypoints. It is not the frontend, websocket layer, marketplace API, wallet flow, or hosted HTTP API.
- Current trusted Release continuity evidence for generated-key CUDA d8/b2048 is about `79.2k H/s` median with normal benchmark trust. Older `78.3k H/s` evidence is still useful continuity context but should not override newer trusted evidence.
- Current timing evidence still points at CPU-side generated input and first-block preparation as the dominant bottleneck for the d8/b2048 generated-key path.
- The first-block scheduling metadata slice is complete. Hash API JSON and benchmark summaries now expose `first_block_worker_count` and `first_block_chunk_size`; commit `a19d069` validated it with focused tests, a clean Release CUDA rebuild, the golden CUDA hash, and a short CUDA smoke.
- If a future struct-layout change touches the full miner binary, prefer a clean Release CUDA rebuild before trusting CLI results, because stale object files can corrupt JSON fields.
- Do not repeat the rejected digest length-prefix static fast path unless the implementation shape materially changes. It preserved correctness but regressed the d8/b2048 generated CUDA confirmation against the refreshed trusted baseline.

## Strong Goal Shape

Treat the active `/goal` as a compact contract, not a vague "keep improving" prompt. Every continuation turn should preserve these six parts:

- Outcome: reduce time per valid Hash API CUDA hash attempt as far as correctness allows.
- Verification surface: focused unit tests, CUDA golden hash checks, machine-readable benchmark JSON, sanitized summaries, comparison reports, and staged diffs.
- Constraints: preserve `argon2id-xen` semantics, Hash API request/result compatibility, target matching, generated-key attempt indexing, and public-repo privacy.
- Boundaries: stay inside the reusable hash core, CUDA backend, benchmark tooling, tests, docs, and narrowly related miner integration.
- Iteration policy: pick the next smallest measurable bottleneck from timing evidence, validate it, commit it if useful, document or revert it if rejected, then continue.
- Blocked policy: stop only for the explicit stop conditions in this file; otherwise choose the next safe local action without asking for approval.

Long instructions belong in this file and `docs/HASH_OPTIMIZATION_GOAL.md`. The `/goal` command should point at these files instead of trying to carry every rule inline.

## Continuation Turn Rules

At the start of every automatic continuation turn:

1. Read `goal.md` and `docs/HASH_OPTIMIZATION_GOAL.md` if context is stale, compacted, or uncertain.
2. Run `git status -sb` and inspect recent commits before editing.
3. If the worktree is dirty, classify the dirty state first: previous-agent accepted work, previous-agent rejected experiment, user change, or unrelated local artifact.
4. Finish validation for useful dirty work before starting a new experiment.
5. Select exactly one next measurable step from the current bottleneck evidence.
6. Run correctness checks before trusting performance numbers.
7. Commit small validated slices with English messages.
8. Update the detailed goal document when a result changes future decisions.

Do not use a continuation turn only to restate the plan. Make measurable progress unless a stop condition is reached.

Each turn should end as one of:

- accepted: correctness passed, the change is useful, privacy checks passed, and a small commit was made
- rejected: the current uncommitted experiment was reverted or left documented as rejected evidence
- measurement-only: tooling, timing, docs, or benchmark reliability improved and was committed
- blocked: an explicit stop condition was reached and the remaining blocker is concrete

## Outcome

Optimize the extracted Hash API and CUDA hashing path until one of these is true:

- throughput improves by at least 1000% over the initial measured baseline while correctness is preserved and no obvious low-risk improvements remain
- repeated well-scoped attempts plateau and the remaining bottleneck is documented with benchmark or profiler evidence
- profiler evidence shows the implementation is near the practical hardware limit for the tested GPU class

Until one of those outcomes is proven, keep iterating.

The target improvement is measured on the same benchmark scenario, not across unrelated difficulty, batch-size, key-mode, or hardware changes. A 1000% claim needs a recorded baseline, a confirmed best result, and the exact comparison formula from the progress accounting section.

A 1000% improvement means the confirmed best median throughput is at least `11x` the selected baseline median throughput by the formula below. Do not confuse this with reaching `1000% of baseline`, which would be only `10x`.

## Progress Accounting

Maintain progress against a named baseline rather than against memory or terminal prose.

- Baseline: the earliest trustworthy machine-readable CUDA benchmark for the selected scenario after the Hash API extraction, or a new documented baseline if no trustworthy report exists.
- Best result: the highest confirmed median warm throughput for the same scenario and correctness surface.
- Improvement: `(best_median_hps - baseline_median_hps) / baseline_median_hps * 100`.
- Main scenario for continuity: generated-key CUDA, main-target-only, difficulty `8`, batch size `2048`, warm-up `1`, repeat `3`, with the same seconds value for before/after comparisons.
- Supplemental scenarios: difficulty `1`, `64`, `256`, and `1024`, variable-difficulty sequences, and batch-size scans when the main scenario no longer explains the bottleneck.

For example, a confirmed best of `79.2k H/s` over a `78.3k H/s` baseline is about `1.1%` improvement, not a meaningful step toward the 1000% target. A 1000% improvement from `78.3k H/s` would require about `861.3k H/s` on the same scenario and correctness surface.

Do not claim the 1000% target from a single noisy run. Confirm large claims with repeated runs or a stable scan, and keep the raw report ignored unless a sanitized summary is intentionally committed.

Plateau evidence requires at least three consecutive well-scoped optimization attempts against the current dominant bottleneck with less than 3% confirmed improvement, plus a short note in `docs/HASH_OPTIMIZATION_GOAL.md` explaining the remaining bottleneck and why risk is now higher than expected gain.

A benchmark can become the active baseline or best result only when:

- `report_ok` is true
- `benchmark_trust` is `normal` when environment metadata is present
- build metadata is public-safe and matches the intended Release CUDA configuration
- the scenario matches the comparison target
- spread is at or below the configured stability threshold, or the uncertainty is explicitly documented as measurement-only
- correctness checks passed on the affected path

Reports marked `benchmark_trust: low` are useful for diagnosing local environment noise, but they must not replace the baseline or justify performance claims.

## Fixed Workload

The optimization target is the real mining hash workload:

- `t = 1` is fixed
- `s = 1` and `p = 1` are fixed as represented by the current implementation
- `m = difficulty` / `diff` is the only workload parameter expected to vary between benchmark or mining sessions
- salt, key, prefix, difficulty, matching, and result semantics must stay compatible with the current Hash API contract

The primary metric is warm steady-state CUDA attempts per second. The secondary metric is milliseconds per valid hash attempt, especially for generated-key mining batches.

Optimize same-difficulty warm loops first because they are the easiest to compare. Then confirm that the architecture also handles variable `m=diff` sequences without repeated setup or allocation costs dominating the run.

When choosing between alternatives, prefer designs that keep `m=diff` explicit and cheap to retune. Do not bake in one local difficulty, one local batch size, one local GPU name, or one private build path.

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

## Hash Core Extraction Target

The long-term shape should be a small, reusable hash engine with adapters around it:

- Core contract: typed request/result structures for salt, key mode, prefix, difficulty, batch size, target matching, backend choice, device selection, and timing metadata.
- CPU adapter: correctness/reference backend that can run without CUDA.
- CUDA adapter: optimized backend that owns GPU state, buffers, launch choices, and device-derived tuning.
- CLI adapter: `hash-one`, `hash-batch`, and `hash-benchmark` as stable automation entrypoints.
- Miner adapter: existing miner integration should call the Hash API instead of duplicating hashing logic.
- Future adapters: library, local service, or other programs may be added later without changing hash semantics.

Keep dependencies pointing inward. The hash core may know about hashing requests, validation, backends, tuning metadata, and result matching. It should not know about frontend screens, user accounts, marketplace flows, wallet UX, platform reporting, settlement, MQTT business logic, or remote lease policy.

When an optimization requires wider ownership changes, prefer this order:

1. Make the Hash API contract explicit.
2. Move platform-specific concerns out of the timed hash path.
3. Separate cold setup from warm steady-state execution.
4. Isolate backend state and buffer lifetime.
5. Add or improve benchmarks for the new boundary.
6. Optimize the now-measurable hot path.

## Current Checkpoint

This checkpoint exists so a long-running `/goal` session can resume without reinterpreting the project direction.

- The Hash API extraction is complete enough for isolated optimization work.
- "CLI API" means the command-line Hash API automation surface: `hash-one`, `hash-batch`, and `hash-benchmark`.
- That CLI surface is the current stable driver for AI optimization loops, benchmarks, correctness checks, and future embedding work.
- It is not the marketplace, wallet, frontend, websocket, or hosted HTTP platform API.
- Local commits ahead of the remote branch are retained local progress. Do not assume they were lost; verify with `git status -sb` and `git log`.
- The branch can stay ahead of the remote during autonomous work. Keep improving and committing locally unless the user explicitly requests squash, reorder, push, or public history rewrite.
- Current stable Release continuity evidence for generated-key CUDA d8/b2048 is about `78.3k H/s` median with `3.8%` spread. Treat it as local benchmark evidence, not as a public hardware claim.
- Current timing evidence points to CPU-side input and first-block preparation as the dominant bottleneck, so prefer first-block/input-preparation work before risky CUDA kernel rewrites unless newer measurements contradict it.
- Benchmark reports include public-safe environment trust metadata sampled before and after each run. Prefer `benchmark_trust: normal` reports for CPU-side input and first-block conclusions.
- The current benchmark harness can scan difficulty, batch size, and CUDA first-block worker caps. First-block worker caps are tuning parameters for measurement, not a default behavior change unless stable repeated evidence supports it.
- Recent timing evidence shows generated CUDA d8/b2048 work is usually CPU-side dominated by `input_ms`, especially first-block preparation, while `setup_ms` and transfer timings are still useful secondary targets.
- Recent rejected experiments include CUDA activation caching, pinned host staging buffers, runner caching, a lanes==1 first-block fast path, `_rotr64` Blake2b rotate replacement, and several salt/key/finalization micro-optimizations; read `docs/HASH_OPTIMIZATION_GOAL.md` before retrying any similar idea.
- If there is no newer evidence, the next default work is to refresh d8/b2048 and d8/b1024 CUDA baselines, inspect detailed `input_ms` and first-block timing, then choose the smallest input/setup/backend-boundary improvement that preserves the Hash API contract.

## Architecture Direction

If the current layout makes serious optimization difficult, improve the structure before chasing micro-optimizations. Acceptable structural work includes:

- keeping hot hashing paths callable without marketplace, wallet, frontend, lease, devfee, or network services
- moving difficulty-derived setup, backend state, buffer ownership, and timing metadata behind clear Hash API or backend contracts
- making CUDA tuning knobs explicit and easy to benchmark
- separating cold setup, warm steady-state hashing, input preparation, kernel execution, transfer time, finalization, and result matching
- keeping CPU/reference behavior available for correctness checks
- adding benchmark or comparison tooling that makes future AI iterations harder to misread

Do not introduce a new platform layer while optimizing hash speed. The preferred future shape is a small, reusable hash core with stable CLI and test entrypoints, so external programs or future agents can optimize or embed it without understanding the full miner.

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

## Dirty Work Policy

When the worktree is dirty at the start of a turn, do not assume changes are lost or bad.

- If the dirty files match the active goal and tests already passed in the previous checkpoint, finish the remaining validation and commit them before starting a new experiment.
- If the dirty files are a rejected experiment from the current agent, revert only those files and document the rejection when it prevents repeated work.
- If the dirty files appear user-authored, unrelated, or ambiguous, leave them untouched unless they block the current step.
- If the same file contains both useful prior work and new required edits, inspect the diff carefully and preserve the prior work.
- Never use `git reset --hard`, broad checkout, or history rewrite as a cleanup shortcut.

Current dirty measurement-only work should be validated as a coherent slice before new optimization code starts. A representative validation sequence is focused Hash API tests, Release CUDA rebuild, golden CUDA hash, short CUDA benchmark smoke, `git diff --check`, staged privacy scan, then commit.

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

If a dirty measurement-only change is already present, finish its validation and commit it before starting a new optimization experiment. Measurement improvements are useful when they make later performance decisions more reliable, even if they do not directly raise hashrate.

If the active workspace is ahead of the remote branch, treat those commits as retained local progress unless the user explicitly asks to squash, reorder, or push them. Do not infer that commits were lost just because they are not present on the remote.

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

## Optimization Tracks

Use these tracks as the long-running work queue. Work on the earliest track that is currently blocking reliable speed gains, but switch tracks when benchmark evidence points elsewhere.

Track A: measurement and reproducibility

- Keep `hash-benchmark` output machine-readable and comparable.
- Keep warm-up, repeat count, spread, per-attempt timing, invalid-run detection, and before/after comparison reliable.
- Add timing fields only when they change decisions or reduce ambiguity.
- Reject partial or unstable benchmark matrices before changing defaults.

Track B: pure Hash API architecture

- Keep hash code callable without frontend, marketplace, wallet, lease, devfee, network services, or platform startup.
- Keep CPU/reference and CUDA backends behind one request/result contract.
- Move repeated validation, setup, difficulty normalization, device selection, allocation, and backend lifetime decisions into explicit backend or tuning components.
- Keep CLI commands stable so future agents can iterate through scripts instead of manual UI flows.

Track C: generated input and first-block preparation

- Prioritize this track while `input_ms`, `keygen_ms`, or `first_block_ms` dominate.
- Optimize generated-key construction, salt/key materialization, Argon2 first-block preparation, CPU parallelism, and data layout.
- Preserve fixed `t=1`, fixed `s/p=1`, and variable `m=diff` semantics.
- Do not retry rejected keygen, salt caching, or first-block fast-path experiments unless the implementation shape materially changed.

Track D: CUDA warm execution

- Prioritize this track when `compute_ms`, allocation, transfer, launch, or kernel timing dominates.
- Reduce allocation churn, tune batch size, inspect transfer cost, tune launch geometry, and measure occupancy or memory behavior before rewriting kernels.
- Treat pinned memory, streams, runner caching, and device finalization as redesign topics, not quick retries, because earlier isolated attempts were unstable or slower.

Track E: autotuning by public device properties

- Add tuning only after stable cross-scenario evidence exists.
- Base automatic choices on difficulty, batch size, key mode, compute capability, available memory, and measured stability.
- Preserve explicit user settings over autotune defaults.
- Keep autotune overhead out of steady-state benchmark measurements.

Track F: cross-GPU readiness

- Validate locally first, then keep the tuning model ready for RTX 3050-class and higher-end CUDA GPUs.
- Avoid local GPU names, local build paths, private machine identifiers, or one-device assumptions in committed code or docs.
- Keep architecture-specific choices guarded by public compute capability or runtime properties.
- Document best-known settings as local evidence unless confirmed across more devices.

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

When a command takes a long time, let it run to completion and continue from the result. Do not ask the user to approve routine rebuilds, CUDA checks, or repeated benchmarks.

If a local command fails because a dependency or CUDA build is unavailable, record the blocker in English, fall back to the narrowest available validation, and continue with measurement/tooling/code work that does not require the unavailable dependency. Stop only when no useful next step remains.

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

Each loop should end in one of three states:

- accepted: correctness passed, benchmark evidence is useful, and a small commit was made
- rejected: correctness or benchmark evidence failed, the current uncommitted experiment was reverted, and the rejection was documented only if it prevents repeated work
- measurement-only: no speed claim was made, but benchmark, timing, test, or documentation infrastructure improved and was committed

## Autonomous Work Queue

Use this queue as the default order when no newer evidence is available:

1. Confirm the worktree state, active goal, latest optimization commits, and privacy status.
2. Run focused Hash API tests and a CUDA golden hash check before trusting performance data.
3. Refresh the main generated-key CUDA baseline for d8/b2048 with warm-up and repeated samples.
4. Inspect per-attempt timing and choose the largest credible bottleneck.
5. If `input_ms` dominates, work on generated-key preparation, salt/key materialization, and Argon2 first-block setup.
6. If `setup_ms` dominates, reduce repeated validation, difficulty setup, device resolution, allocation, or backend lifecycle costs.
7. If `compute_ms` dominates, inspect CUDA allocation churn, transfer cost, launch geometry, memory behavior, occupancy, and kernel timing.
8. If `finalize_ms` dominates, use the nested finalization timings before changing hash finalization, base64 encoding, matching, or result collection.
9. When single-scenario gains flatten, run variable-`m=diff` and batch-scan scenarios to avoid overfitting one local setting.
10. Use first-block worker-cap scans as a diagnostic axis when `first_block_ms` or `input_ms` dominates, but keep automatic worker behavior as the default unless longer repeated evidence is stable.
11. After stable cross-scenario evidence exists, add or improve autotuning based on public CUDA device properties and measured stability.
12. Keep accepted and rejected experiments documented so future long-running agents do not repeat failed work.

A structural cleanup can be the next iteration if it directly enables one of these work items or improves the reliability of future measurements.

## Next Iteration Selector

At the start of each autonomous cycle, choose exactly one next step by this rule:

1. If correctness validation is stale or the binary changed, run focused tests and the CUDA golden hash check first.
2. If no trustworthy current baseline exists, run or load the d8/b2048 generated-key CUDA baseline.
3. If benchmark results are noisy, improve measurement quality or rerun a narrower scenario before editing performance code.
4. If `input_ms` or `first_block_ms` dominates, select a Track C experiment.
5. If setup or lifecycle cost dominates, select a Track B experiment.
6. If CUDA compute, allocation, transfer, or launch dominates, select a Track D experiment.
7. If stable manual settings repeatedly beat defaults, select a Track E autotuning experiment.
8. If the same optimization no longer transfers across difficulty values, add a variable-`m=diff` scenario before choosing another code change.
9. If three consecutive well-scoped attempts against the same bottleneck fail to improve confirmed throughput by at least 3%, document plateau evidence and move to the next bottleneck.

Do not choose broad rewrites unless the selector shows that smaller measurable changes are blocked by the current structure.

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

For small changes that mainly affect fixed-key single-hash latency, use the isolation preset before changing the generated-key path. Keep generated-key d8/b2048 throughput as the continuity scenario until a newer documented scenario supersedes it.

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

First-block worker-cap diagnostic scan:

```bash
python scripts/hash_api_benchmark.py --binary <miner-binary> --backend cuda --device 0 --seconds 4 --warmup 1 --repeat 3 --no-xuni --scan-difficulty 8 --scan-batch-size 1024 --scan-batch-size 2048 --scan-first-block-workers 0 --scan-first-block-workers 4 --scan-first-block-workers 8 --output .benchmarks/first-block-worker-scan.json --sanitized-output .benchmarks/first-block-worker-scan-summary.json
```

Before/after comparison:

```bash
python scripts/hash_api_compare.py .benchmarks/before.json .benchmarks/after.json --fail-on-regression --min-change-pct 1
```

## Immediate Queue

Start here unless `docs/HASH_OPTIMIZATION_GOAL.md` contains newer evidence:

1. Verify `git status -sb`.
2. Confirm docs and recent commits contain no local paths or private machine details.
3. Confirm local commits are still present with `git log`; do not treat ahead-of-remote commits as lost work.
4. Run focused Hash API unit tests.
5. Build the available smoke CLI or full CUDA binary.
6. Run the golden CUDA hash check when a CUDA binary is available.
7. Run a short main-target CUDA benchmark.
8. Run or load repeated d8/b2048 and d8/b1024 baselines because recent useful evidence used both scenarios.
9. Use detailed timings to confirm the current dominant bottleneck, especially `input_ms`, key generation, first-block preparation, setup, transfers, compute, and finalization.
10. Do not retry rejected pinned host staging, CUDA activation caching, salt decode, or first-block lane fast-path experiments unless the implementation shape has materially changed.
11. Prefer input preparation and setup/measurement improvements before speculative finalization micro-optimizations.
12. Keep `docs/HASH_OPTIMIZATION_GOAL.md` updated with accepted and rejected experiments.

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

Use a staged-diff privacy scan before each commit. A non-match exit code from the scan is acceptable:

```bash
git diff --cached --check
git diff --cached | rg -n "[A-Za-z]:[/\\\\]|[/]Users[/]|Users[\\\\]|<private-user>|[h]ostname=|[H]OSTNAME|[S]ECRET|[P]RIVATE KEY|[B]EGIN .*KEY|wallet [p]rivate"
```

Do not commit local GPU model names, local absolute paths, or raw benchmark report contents even when they look harmless.

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
