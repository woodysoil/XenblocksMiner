# Long-Running Goal: Optimize Hash API Throughput

## Mission

Continuously improve the Xenblocks Hash API hashing throughput until performance gains plateau, correctness risk becomes unacceptable, or the implementation is close enough to the practical hardware limit.

The aspirational target is at least a 1000% speed improvement over the measured baseline where feasible. A 1000% throughput improvement means `11x` the baseline, not just `10x`. Treat that as a direction, not permission to weaken correctness. Every optimization must preserve the real `argon2id-xen` result semantics.

This goal is intended for Codex `/goal` long-running execution after the reusable Hash API extraction. Treat this file as the persistent operating brief.

The practical optimization target is simple: complete the same valid hash attempts in as little time as possible for fixed `t=1`, fixed `s=1`, current single-lane `p=1`, and variable `m=diff`. Real mining difficulty is expected to be in the thousands to tens of thousands because this is a memory-hard hash path. Use tiny difficulties such as d1, d8, and d64 only as smoke, correctness, and harness diagnostics unless a specific low-d regression needs to be isolated. Optimize the current CUDA-capable local GPU first, then keep the architecture and tuning system ready for RTX 3050-class and higher-end GPUs.

Treat this as a latency-first, throughput-measured goal. Median warm attempts
per second is the main repeatable metric, but every accepted change should also
reduce or explain median time per attempt for the same real hash path.

## `/goal` Starter

Use `goal.md` as the short entrypoint, then keep this file as the authoritative long-running plan.

Suggested `/goal` objective:

```text
Continuously execute goal.md and docs/HASH_OPTIMIZATION_GOAL.md. Optimize XenblocksMiner Hash API CUDA hashing throughput for fixed t=1, fixed s=1, current single-lane p=1, and only m=difficulty changing between sessions, preserving real argon2id-xen semantics. Primary performance benchmarks must target realistic memory-cost difficulty values in the thousands to tens of thousands; use tiny difficulty values only as correctness, smoke, and benchmark-harness checks. Iterate through inspect, benchmark, optimize, validate, document, and commit cycles without asking for approval unless a listed blocker is reached. Keep all code, docs, tests, benchmark names, and commit messages in English. Never commit local paths, raw benchmark reports, local hardware identifiers, or private machine details.
```

## Current Control Summary

Read this section first after every resume. The experiment ledger below preserves
older evidence; when an older "current state" bullet conflicts with this summary,
this summary is the newer control point.

The goal is active and should keep running autonomous optimization cycles. Do not
ask for approval for routine inspection, tests, builds, benchmarks, source edits,
documentation edits, privacy checks, or small validated commits.

Routine long-run work is authorized by the goal contract. Agents may run
non-destructive local commands, create ignored benchmark artifacts, edit
hash-path code, validate, benchmark, document, privacy-check, and make coherent
English commits without pausing. Ask only for the explicit stop conditions near
the end of this file.

Current confirmed architecture state:

- Hash API extraction is complete enough for isolated optimization.
- The current automation surface is the CLI Hash API: `hash-one`, `hash-batch`,
  and `hash-benchmark`. This is not a hosted HTTP API, websocket API, frontend
  API, marketplace API, wallet flow, or full platform API.
- Miner-generated CUDA requests use the validated GPU first-block path and the
  automatic first-block chunk policy where supported.
- The real optimization workload is high-memory `m=diff` in the thousands to
  tens of thousands. Existing d1, d8, and d64 reports are useful for smoke,
  continuity, and regression triage, but they are not representative completion
  evidence for the mining workload.
- Benchmark automation supports fixed difficulty, variable `m=diff` sequences,
  automatic batch sizing, GPU first blocks, first-block dynamic chunk policy,
  detailed timings, report-quality checks, and before/after comparison.
- The Hash API boundary should remain independent of frontend, marketplace,
  wallet, settlement, auth, lease, and devfee code unless a narrow integration
  fix is required.

Latest public-safe performance checkpoint:

- A stable high-difficulty baseline in the thousands-to-tens-of-thousands range
  is now established on the current CUDA-capable local GPU. For generated-key
  CUDA d4096 with GPU first blocks, auto batch selection, automatic first-block
  dynamic chunking, no XUNI, warm-up `1`, repeat `3`, and normal benchmark
  trust, the current local candidate is batch size `797` at about `10.74k H/s`
  median with `5.49%` spread and zero invalid subprocesses.
- The corresponding high-difficulty variable-`m` sequence
  `difficulty_sequence=4096,8192,16384` with the same GPU-first policy reached
  about `4.42k H/s` median with `9.54%` spread and zero invalid subprocesses.
  This is a valid high-memory workload baseline, but it is slower because the
  sequence includes larger `m` values.
- A short fixed-difficulty auto-batch scan now covers d8192 and d16384 with both
  default first-block input and GPU first blocks. With GPU first blocks enabled,
  d8192 selected batch size `398` and reached about `5.32k H/s` median with
  `3.03%` spread; d16384 selected batch size `199` and reached about `2.76k H/s`
  median with `1.18%` spread. Both runs had normal benchmark trust and zero
  invalid subprocesses. Treat this as short high-difficulty direction evidence,
  not as a final tuning plateau.
- For generated-key CUDA d8 with GPU first blocks, auto batch selection, automatic
  first-block dynamic chunking, no XUNI, warm-up `1`, repeat `3`, and normal
  benchmark trust, the best current local candidate is batch size `4096` at about
  `196.86k H/s` median with `1.95%` spread and zero invalid subprocesses. This
  is low-difficulty evidence only.
- A later pre-launch-shape refresh of the same d8/b4096 GPU-first scenario
  measured about `191.87k H/s` median with `2.05%` spread and zero invalid
  subprocesses. Treat this as the current launch-geometry baseline for the next
  kernel-runner experiment, while keeping the earlier `196.86k H/s` checkpoint
  as the best accepted public-safe d8/b4096 result.
- The same local scan saw batch size `3072` at about `191.19k H/s` median with
  `3.15%` spread. Keep b3072 as a nearby fallback candidate, not the current d8
  winner.
- d1 and d64 keep their separate tuning evidence. Do not generalize any
  low-difficulty result to realistic high-memory difficulty values or GPU
  classes without confirmation.
- The current post-GPU-first bottleneck should be rechecked with detailed timing,
  but recent evidence points toward CUDA compute/kernel work on realistic
  high-difficulty runs. On the current d4096 baseline, `compute_ms` is about
  `92%` of wall time, while `first_block_ms` is about `1.9%` and
  `finalize_ms` is about `4.4%`.
- Current GPU-first timing also exposes a measurable first-block launch component
  around the first-block device path, but the realistic high-difficulty baseline
  does not support launch-geometry as the dominant bottleneck. The recent
  one-parameter `threadsPerBlock` test was a useful measurement slice, but it
  did not improve the high-d baseline and was reverted.
- The latest attempted main-kernel `__launch_bounds__(THREADS_PER_LANE, 4)`
  shape preserved focused tests, the Release CUDA build, and CUDA golden hashes,
  but a d4096 GPU-first smoke measured only about `5.86k H/s` median versus the
  established d4096 high-difficulty baseline region around `10.74k H/s`. Treat
  this as a severe high-difficulty regression. The uncommitted source experiment
  was reverted, and focused tests plus CUDA golden hashes passed again after the
  revert.
- A later source-lane-only address-block selection experiment preserved focused
  tests, the Release CUDA build, and CUDA golden hashes, but did not beat the
  d4096 high-difficulty baseline. The short d4096 GPU-first smoke reached about
  `10.16k H/s` median with `2.85%` spread versus the established baseline around
  `10.74k H/s`, and the sm75 main-kernel register count increased from `52` to
  `56`. The source experiment was reverted and the Release CUDA binary was
  rebuilt back to the baseline resource shape.
- Nsight Compute profiling is not currently usable without enabling NVIDIA GPU
  performance counter permissions on the local system. A permission-limited
  d4096 GPU-first profiler attempt still executed the hash command correctly but
  did not collect counter data. Do not loop on `ncu` in autonomous runs when it
  reports `ERR_NVGPUCTRPERM`; use offline resource summaries and benchmark
  timings until permissions are available.
- Current offline CUDA resource evidence for the restored Release binary shows
  the main `argon2_kernel_oneshot` kernel has no stack or local memory usage,
  `1024` bytes of shared memory, and `52` registers for sm75 while sm80/sm86/sm89
  use `40` registers. The device first-block kernel remains register-heavy at
  `255` registers with stack use, but high-difficulty benchmark timing shows it
  is not the dominant bottleneck after GPU first blocks are enabled.
- Accepted main-kernel loop split: `argon2_kernel_oneshot` now separates the
  Argon2id indexed-address slices from the data-dependent slices and starts the
  first slice at offset `2`, removing the hot-loop `skip` branch and the
  per-block `slice < ARGON2_SYNC_POINTS / 2` branch without changing memory
  layout or hash semantics. Validation passed focused Hash API tests, a Release
  CUDA rebuild, CUDA golden hashes with and without GPU first blocks, and
  public-safe resource summarization. The main kernel resource shape changed to
  sm75 `56` registers, sm80 `32` registers, sm86/sm89 `40` registers, `1024`
  bytes shared memory, and no local memory. High-difficulty GPU-first evidence
  stayed stable at d4096 with about `10.76k H/s` median and `2.54%` spread,
  improved d8192 from about `5.32k H/s` to about `5.60k H/s` (`+5.16%`),
  improved d16384 from about `2.76k H/s` to about `2.83k H/s` (`+2.84%`), and
  improved the `4096,8192,16384` variable-`m` sequence from about `4.42k H/s`
  to about `4.72k H/s` (`+6.86%`). Keep the change, but note that d4096 itself
  is effectively flat and compute/kernel time remains above `93%`, so the next
  cycle should continue with high-difficulty main-kernel or memory-behavior
  candidates rather than CPU-side input work.
- Measurement-only d32768 checkpoint after the loop split: a short
  high-difficulty generated-key CUDA auto-batch scan with GPU first blocks,
  warm-up `1`, repeat `2`, no XUNI, and normal report quality selected batch
  size `99` and reached about `1.36k H/s` median with `4.76%` spread. Compute
  time was about `96.37%` of wall time. Treat this as current tens-of-thousands
  difficulty coverage, not as a separate accepted tuning change.
- Accepted indexed-address word selection change: the Argon2id indexed-address
  step now reads the selected address word through a warp-uniform switch instead
  of the constant-time mask-select helper used for thread-varying reads. This is
  valid for the indexed-address path because the selected `block_th` word is
  derived from `addr_index / THREADS_PER_LANE` and is uniform across the warp
  for a given step, while the lane-specific source word is still selected by the
  existing shuffle. Validation passed focused Hash API tests, a Release CUDA
  rebuild, CUDA golden hashes with and without GPU first blocks, and public-safe
  resource summarization. Main-kernel resource use improved on sm75 from `56`
  to `53` registers while sm80 stayed at `32`, sm86/sm89 stayed at `40`, shared
  memory stayed at `1024` bytes, and local memory stayed at zero. Normal-trust
  high-difficulty GPU-first confirmations were non-regressive to slightly
  positive against the loop-split baseline: d4096 reached about `10.77k H/s`
  with `2.59%` spread, d8192 about `5.62k H/s` with `0.91%` spread, d16384
  about `2.85k H/s` with `0.24%` spread, d32768 about `1.40k H/s` with `2.22%`
  spread, and the `4096,8192,16384` variable-`m` sequence about `4.74k H/s`
  with `0.89%` spread. Keep the change as a resource-pressure and portability
  improvement, but continue searching because compute/kernel time still
  dominates at roughly `93%` to `98%` on realistic high-memory difficulties.
- Measurement-only batch-scan caution: a broad d8192/d16384 adjacent batch-size
  probe timed out before writing a report, and a smaller d8192 b320/b398/b480
  probe completed under low benchmark trust with high CPU load and unstable
  spreads. Do not use those local batch-scan artifacts for tuning decisions.
  Future high-difficulty batch tuning should run smaller targeted matrices under
  normal benchmark trust and compare against the accepted auto-batch rows before
  changing `HashApiTuning`.
- Measurement-only d8192 batch-window refresh: a narrow GPU-first b160/b192/b256/
  b320/b398 run completed with zero invalid subprocesses, but benchmark trust
  was low because CPU load was high and `report_quality_ok=false`. The best
  stable median in that run was b192 at about `4.94k H/s`, below the accepted
  d8192 GPU-first confirmation region around `5.62k H/s`, while b398 was
  unstable. Do not change d8192 tuning from this artifact; rerun under normal
  benchmark trust before accepting any high-difficulty batch default.
- Measurement-only sm75 high-difficulty batch-window refresh: a d4096
  b384/b512/b640/b797 diagnostic run was rejected as a report because the
  environment reached low benchmark trust and the larger rows were unstable.
  A later normal-trust focused confirmation accepted b384 and b512 as stable
  diagnostic rows, with b512 around `10.26k H/s` and `6.98%` spread, but this
  stayed below the accepted d4096 best around `10.77k H/s`. A d8192
  b192/b256/b398 diagnostic again lost final report quality to high CPU load;
  b256 was the best stable diagnostic row at about `5.21k H/s`, still below
  the accepted d8192 best around `5.62k H/s`, while b398 was unstable. Do not
  change high-difficulty batch defaults from these artifacts; treat smaller
  fixed batches as stability diagnostics, not throughput wins.
- Measurement-only high-difficulty batch-window retry: a broad d4096/d8192
  matrix over auto and adjacent fixed batches timed out before writing a
  report, so it produced no tuning evidence. Smaller normal-trust retries were
  stable and had zero invalid subprocesses, but still did not beat the accepted
  auto-batch rows. For d4096, auto/b797 with GPU first blocks reached about
  `10.39k H/s` and b640 reached about `9.79k H/s`, both below the accepted
  d4096 best around `10.77k H/s`. For d8192, auto/b398 with GPU first blocks
  reached about `5.42k H/s`, below the accepted d8192 best around `5.62k H/s`.
  Keep the current high-difficulty memory-limited auto-batch behavior and do
  not add fixed high-difficulty tuning caps from these scans.

Current rejected experiment checkpoint:

- The uncommitted `Blake2b::updateWithUint32Prefix` / `Argon2Params::digestLong`
  one-shot length-prefix experiment preserved focused tests, the Blake2b copy
  self-test, Release CUDA build, and CUDA golden hashes, but regressed the d8
  auto-batch GPU-first smoke from about `196.81k H/s` median to about
  `179.40k H/s` median. The source experiment was reverted. Do not retry this
  exact one-shot prefix update shape.
- The fixed char-buffer base64 plus `std::string_view` matching experiment
  reduced nested base64 timing in short detailed diagnostics, but did not produce
  a stable end-to-end win. The final clean d8/b4096 generated CUDA GPU-first
  confirmation regressed from about `196.86k H/s` to about `192.78k H/s`, and a
  rerun stayed unstable at about `195.35k H/s` with high spread. The source
  experiment was reverted. Do not retry this allocation-only finalization shape
  unless result ownership, output materialization, and benchmark stability are
  redesigned together.
- The final-hash Blake2b prefix-state cache experiment was rejected by the fast
  correctness gate. It reused a prepared `digestLong` output-length prefix state
  for `Argon2Params::finalize`, but the finalization diagnostic and both CUDA
  golden hash commands exited without JSON. The source experiment was reverted,
  and the diagnostic plus CUDA golden checks passed again after rebuilding. Do
  not retry this exact final-prefix cache shape.
- The direct final-digest helper experiment was rejected by the hardened
  finalization diagnostic. It added a `Blake2b` helper for the exact
  32-bit-length-prefix plus 1024-byte final block case and preserved the expected
  sample hash/checksum, but diagnostic throughput regressed from about
  `276-280k` finalizes per second to about `236-238k`. The source experiment was
  reverted before full CUDA benchmarking. Do not retry this exact helper shape.
- The rejected `gpu_final_hashes` experiment must stay rejected unless final hash
  output ownership, synchronization, and subprocess teardown stability are
  materially redesigned and repeated wrapper benchmarks pass with zero invalid
  subprocesses.
- The host-owned parallel finalization snapshot experiment was rejected. It
  copied CUDA output blocks into host-owned memory before CPU-threaded
  `Argon2Params::finalize` and base64 encoding, but stable acceptance failed:
  the parallel-match sub-shape returned structured `bad allocation`, and the
  safer chunked finalize/base64-only sub-shape later produced no-JSON
  access-violation subprocess exits in the d8/b4096 GPU-first confirmation. The
  source experiment was reverted. Do not retry host-owned parallel finalization
  snapshots unless thread lifetime, per-thread output ownership, subprocess
  teardown stability, and repeated wrapper zero-invalid checks are redesigned
  together.
- The one-parameter first-block launch-geometry experiment (`threadsPerBlock`
  changed from `128` to `256`) preserved the CUDA golden hashes and produced a
  valid d4096 high-difficulty smoke, but the clean d4096 comparison did not
  improve the baseline and the source experiment was reverted. Do not retry
  this exact launch-geometry shape without a new hypothesis and a better
  high-difficulty target signal.
- The main Argon2 kernel launch-bounds experiment
  (`__launch_bounds__(THREADS_PER_LANE, 4)`) is rejected. It preserved focused
  Hash API tests, the Release CUDA build, and CUDA golden hashes, but the short
  d4096 GPU-first smoke regressed to about `5.86k H/s` median versus the
  established high-difficulty baseline region around `10.74k H/s`. The source
  experiment was reverted, and focused tests plus CUDA golden hashes passed
  after the revert. Do not retry this exact launch-bounds shape without profiler
  evidence that changes the occupancy/register-pressure hypothesis.
- The source-lane-only address-block selection experiment is rejected. It changed
  address-block field selection so only the warp source lane selected the field
  before `shuffle`, preserving correctness but failing the high-difficulty
  acceptance gate. The d4096 GPU-first smoke was stable at about `10.16k H/s`
  median, below the established d4096 baseline around `10.74k H/s`, and it
  increased sm75 main-kernel registers from `52` to `56`. The source experiment
  was reverted; do not retry this shape unless a profiler shows the branch and
  register tradeoff changes on a different architecture.
- The main-kernel reference-area accumulator experiment is rejected. It moved
  `slice * segment_blocks + offset - 1` out of `compute_ref_pos` and passed the
  pre-change focused Hash API tests and CUDA golden hashes on the clean baseline,
  but the Release CUDA rebuild did not complete within the autonomous build
  timeout and left the CUDA compiler subprocess stalled before producing a new
  binary. The source experiment was reverted with no benchmark claim. Do not
  retry this ref-area accumulator shape unless compile-time behavior is first
  isolated in a smaller CUDA build or paired with evidence that the generated
  resource shape is worth the compile risk.

Next cycle:

1. Confirm the worktree is clean or classify dirty files.
2. Run focused Hash API tests if validation is stale.
3. Build or reuse a clean Release CUDA binary.
4. Run CUDA golden hashes with and without `--gpu-first-blocks`.
5. Run a short low-difficulty auto-batch GPU-first smoke only as a safety gate.
6. Use `argon2-finalize-benchmark` as a fast host-side finalization diagnostic
   before trying materially different final digest implementations.
7. Establish or refresh a stable high-difficulty auto-batch GPU-first baseline,
   starting with d4096 and then expanding toward tens-of-thousands difficulty
   as memory permits.
8. Establish or refresh a realistic variable-`m` GPU-first sequence baseline,
   for example d4096,d8192,d16384, and add d32768 only when auto batch sizing
   and run time remain practical.
9. Use detailed timing to pick exactly one bottleneck.
10. Do not retry the rejected `threadsPerBlock=256` first-block launch shape,
   the rejected `__launch_bounds__(THREADS_PER_LANE, 4)` main-kernel shape, or
   the rejected source-lane-only address-block selection shape unless profiler
   data produces a materially different hypothesis.
11. Prefer CUDA kernel efficiency, memory-access behavior, realistic
   high-difficulty batch selection, setup/lifecycle cleanup for variable
   `m=diff`, or other measured CUDA backend bottlenecks over another keygen-only,
   one-shot prefix, host-owned parallel finalization snapshot, or
   allocation-only base64/matching micro-optimization.
12. Validate, benchmark, document, privacy-check, commit, and continue.

## Current State Snapshot

The reusable Hash API extraction is already in place. The optimization work should build on it instead of going back to the platform monolith.

Active goal status:

- `/goal` is active for continuous Hash API and CUDA throughput optimization.
- The branch may be ahead of the remote with many local commits. Treat those commits as retained local work, not lost work, unless the user explicitly requests a squash, reorder, push, or public history rewrite.
- Real mining difficulty is expected to be in the thousands to tens of
  thousands. The low-difficulty evidence below remains useful for smoke,
  correctness, harness, and regression continuity, but future optimization
  acceptance should be based on realistic high-memory `m=diff` scenarios unless
  the change is explicitly measurement-only.
- The current trusted Release continuity evidence for static/default generated-key CUDA low-difficulty behavior is the difficulty `8`, batch size `2048`, warm-up `1`, repeat `3`, no-XUNI scenario, with about `79.2k H/s` median from the latest normal-trust refresh. Older `78.3k H/s` evidence remains useful continuity context but should not override newer trusted evidence.
- The current accepted optimization evidence supports the automatic first-block dynamic chunk policy for covered generated-key CUDA d1, d8, and d64 batches. The latest longer d8 same-settings confirmation reached about `87.16k H/s` median at d8/b2048, about `+8.1%` over the matching static run, with normal benchmark trust. A later d64/b2048 confirmation improved from `75.63k H/s` static to `79.51k H/s` with chunk `16`, about `+5.1%`.
- The dominant measured bottleneck after the auto policy is still CPU-side input preparation, especially first-block preparation. Latest detailed post-auto timing showed `input_ms` at about `57-59%` of wall time and `first_block_ms` at about `53-54%`. A current clean d8/b2048 generated CUDA auto refresh after the d64 policy commit stayed stable at about `86.95k H/s` median with `3.17%` spread, and a short detailed run showed `input_ms` about `58.94%`, `first_block_ms` about `54.39%`, `finalize_ms` about `29.58%`, and `keygen_ms` about `3.97%`.
- This baseline is local evidence on a CUDA-capable GPU. Do not publish raw reports, local binary paths, hardware identifiers, or private machine details.
- Commit `a19d069` completed the measurement-only first-block scheduling metadata slice. Hash API JSON, benchmark summaries, comparison output, docs, and contract tests now expose `first_block_worker_count` and `first_block_chunk_size`.
- The latest stable code slice extends request-level `first_block_dynamic_chunk_auto` to select chunk `16` for covered d1 generated-key CUDA batches, chunk `32` for d8/b1024 batches, chunk `16` for d8 batches with at least 2048 attempts, and chunk `16` for d64 batches up to b2048. Larger d64 batches fall back to static first-block scheduling. Miner-generated CUDA batches opt into the policy for covered scenarios while preserving explicit static and manual dynamic CLI behavior.
- A new explicit `gpu_first_blocks` experiment moves Argon2 first-block generation onto the CUDA device for supported `t=1`, single-lane requests. It is opt-in and not a default policy yet. Correctness passed the CUDA golden hash for both default and opt-in paths, focused Hash API tests, and a Release CUDA rebuild. Initial d8/b2048 generated CUDA evidence is strong but still needs broader confirmation before automatic enablement.
- Rejected `gpu_final_hashes` experiment: moving the final Argon2 `digestLong(last block -> final hash)` work onto CUDA and copying back fixed-size final hashes preserved the golden CUDA hash in early checks and produced much faster valid samples, but repeated benchmark wrapper runs produced access-violation subprocess exits with no JSON. The uncommitted source experiment was reverted. Do not retry this exact device-side final hash output path unless the finalization buffer lifetime and synchronization design materially changes and repeated wrapper benchmarks pass with zero invalid subprocesses.
- Accepted single-lane finalization cleanup: `Argon2Params::finalize` now skips the temporary 1 KiB lane-XOR copy when `lanes == 1` and hashes the final block directly. This preserves the multi-lane path and matches the fixed single-lane goal workload. Validation passed focused Hash API tests, a Release CUDA rebuild, and CUDA golden hashes for default and `gpu_first_blocks`. A short d8/b2048 generated CUDA `gpu_first_blocks` smoke improved from about `152.03k H/s` median with `12.26%` spread to about `171.93k H/s` median with `3.97%` spread, with normal benchmark trust and zero invalid subprocesses. Treat this as accepted local smoke evidence, not a cross-GPU final plateau.
- Miner-generated CUDA requests now opt into `gpu_first_blocks` together with existing automatic first-block chunk selection. This keeps the CLI flag available for explicit benchmarking while making the miner use the fastest validated generated-key path by default. Same-binary short d8/b2048 evidence after the single-lane finalization cleanup showed generated CUDA without `gpu_first_blocks` at about `79.11k H/s` median with `7.77%` spread, while generated CUDA with `gpu_first_blocks` reached about `171.93k H/s` median with `3.97%` spread, normal benchmark trust, and zero invalid subprocesses. Validation passed focused Hash API tests, a Release CUDA rebuild, and CUDA golden hashes for default and `gpu_first_blocks`.
- Accepted d8 GPU-first batch-size update: after miner-generated CUDA requests moved to `gpu_first_blocks`, a clean targeted d8 generated CUDA scan with auto first-block dynamic chunks compared b3072 and b4096 for seconds `8`, warm-up `1`, repeat `3`, no-XUNI, normal benchmark trust, and zero invalid subprocesses. b3072 reached about `191.19k H/s` median with `3.15%` spread; b4096 reached about `196.86k H/s` median with `1.95%` spread and became the recommended stable candidate. The CUDA batch-size helper now recommends b4096 for difficulty up to d8 while keeping d1 at b2048 and d64 at b3072. This is local low-difficulty GPU-first evidence; confirm on realistic high-memory difficulty values and future GPU classes before treating it as a mining-workload policy.
- Measurement-only finalization diagnostic: `argon2-finalize-benchmark` now builds
  with the hash diagnostics targets and times deterministic host
  `Argon2Params::finalize` loops without CUDA, local paths, hardware metadata, or
  platform code. This gives future agents a fast correctness and timing guard for
  materially different final digest implementations before full CUDA wrapper
  benchmarks. Validation passed focused Hash API tests, a Release CUDA rebuild,
  the diagnostic executable, the Blake2b copy self-test, and CUDA golden hashes
  with and without `gpu_first_blocks`. A local diagnostic run over 4096 blocks and
  128 repeats completed about `303.74k` finalizes per second at about `3292 ns`
  per finalize; treat this as host-side diagnostic evidence only, not a miner
  throughput claim.
- Measurement-only finalization diagnostic hardening: the default
  `argon2-finalize-benchmark` shape now verifies a fixed expected sample hash and
  checksum in addition to deterministic repeated output. This catches
  wrong-but-repeatable final digest rewrites before CUDA golden or wrapper
  benchmarks. Validation passed focused Hash API tests plus default and small
  non-default diagnostic runs.

Known current capabilities:

- Hash API code lives under `src/hashapi/`.
- The miner binary exposes JSON-friendly hash commands such as `hash-one`, `hash-batch`, and `hash-benchmark`.
- A smoke CLI target exists for Hash API contract testing where a full CUDA build is not needed.
- `scripts/hash_api_benchmark.py` supports scenario definitions, warm-up runs, repeated measured runs, aggregate JSON summaries, and optional report output.
- `scripts/cuda_resource_summary.py` summarizes `cuobjdump --dump-resource-usage`
  output into public-safe JSON containing only architecture, kernel aliases, and
  resource counts. It can also compare two public-safe summaries and fail on
  register, stack, or local-memory regressions. Use it before and after CUDA
  kernel experiments to catch resource-pressure regressions before spending time
  on expensive high-difficulty benchmarks, without committing local paths, GPU
  names, or raw tool output.
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
- Detailed first-block diagnostics include `first_block_max_worker_ms`, which records the slowest worker-local wall time during parallel first-block preparation and helps separate scheduling/imbalance from aggregate digest CPU time.
- Detailed first-block scheduling diagnostics include worker thread launch time, worker start skew, and worker finish span fields, which help separate thread creation/start latency, worker-local digest work, and post-worker join overhead.
- Benchmark timing analysis derives `first_block_worker_wall_to_wall` and `first_block_scheduling_overhead_ms` from detailed first-block worker timing, so agents can see how much first-block wall time remains outside the slowest worker-local loop.
- Benchmark timing analysis derives `first_block_finish_wall_to_wall` and `first_block_post_worker_overhead_ms` from worker finish timing, so agents can distinguish last-worker completion from post-worker join/accounting overhead.
- Hash API benchmark summaries include per-attempt timing fields for comparing cost per valid hash attempt.
- Hash API comparison tooling reports total timing deltas, per-attempt timing deltas, top-level and nested stage-percentage deltas, noisy improved/regressed/unchanged status, and variable-difficulty metadata for before/after runs.
- Hash API comparison tooling can match by config while ignoring only detailed-timing mode with `--ignore-detailed-timings`, which helps compare default and diagnostic reports for the same scenario without changing other matching fields.
- Hash API comparison tooling can filter reports with `--min-difficulty` so
  high-memory optimization cycles can ignore low-difficulty smoke rows. For
  fixed-difficulty rows, the row difficulty must be at least the threshold; for
  variable-`m` rows, every difficulty in the sequence must meet the threshold.
- Hash API benchmark scenarios can measure variable `m = difficulty` sequences, including same-difficulty versus alternating-difficulty loops under one reusable backend lifecycle.
- Hash API benchmark scenarios can pair variable `m = difficulty` values with variable batch sizes in one reusable backend lifecycle, for example pairing `difficulty_sequence=1,8,64` with `batch_size_sequence=2048,3072,3072`.
- Mixed difficulty/batch-size sequence runs report public-safe metadata such as `batch_size_sequence`, `batch_size_mode`, `batch_size_changes`, `batch_size_min`, and `batch_size_max`, and are excluded from fixed-shape batch-size recommendations.
- Generated variable-difficulty sequence scenarios can enable detailed CUDA setup and first-block diagnostics with `--sequence-detailed-timings`.
- Fixed-difficulty scan scenarios treat `--scan-batch-size 0` as auto-batch
  selection and name those rows `bauto`. This lets high-difficulty scans use the
  same automatic memory-bounded batch selection as ordinary generated-key runs
  instead of producing invalid `batch_size=0` benchmark rows.
- CUDA Hash API scenarios can cap first-block worker threads with `first_block_workers` / `--first-block-workers` for measured tuning while default `0` preserves automatic worker-count behavior. Benchmark scans can include this axis with `--scan-first-block-workers`, can enable backend-selected first-block dynamic chunks with `--scan-first-block-dynamic-chunk-auto`, and can enable detailed generated-scan diagnostics with `--scan-detailed-timings`.
- CUDA Hash API scenarios can set `first_block_dynamic_chunk_size` / `--first-block-dynamic-chunk-size` to benchmark dynamic first-block chunk distribution. The default `0` keeps static chunking and current miner behavior; nonzero values are explicit experiments for worker start skew and load-balance diagnostics.
- CUDA Hash API scenarios can opt into `first_block_dynamic_chunk_auto` / `--first-block-dynamic-chunk-auto` to benchmark backend-selected dynamic chunk sizing without changing explicit static `0` semantics. The conservative policy targets generated-key CUDA batches with at least 1024 attempts: d1 selects chunk `16`, d8 selects chunk `32` at b1024 and chunk `16` at b2048 or larger, d64 selects chunk `16` through b2048, and larger d64 batches remain static. Miner-generated CUDA batches now opt into this request-level policy for covered scenarios.
- Hash API benchmark presets include an `isolation` matrix for comparing generated-key d8/b2048 throughput against fixed-key d8/b1 behavior before choosing between input-preparation, compute, and finalization work.
- Hash API benchmark summaries mark any nonzero benchmark subprocess exit as invalid even when stdout contains parseable JSON, so crashy optimization experiments cannot enter recommendations.
- Hash API benchmark recommendations expose `report_ok`, run counts, and invalid scenario names so automation can reject partial scan matrices before acting on surviving tuning candidates.
- Hash API benchmark and comparison quality gates require every measured run to have warm evidence and stable spread evidence before `report_quality_ok` / `--fail-on-report-quality` can pass. Reports with cold rows, low-trust environment samples, invalid subprocesses, or unstable measured rows are diagnostics only and must not support speedup claims.
- Hash API benchmark preflight supports `--preflight-skip-retries` for long-running
  goal automation. This retries a subprocess only when the report-quality
  preflight gate skips launch because the environment is temporarily low-trust;
  it does not retry real hash failures and does not relax final
  `report_quality_ok` requirements.
- Hash API benchmark recommendations include `report_quality_failure_reasons`
  so long-running agents can distinguish invalid subprocesses, low benchmark
  trust, high CPU load, missing warm evidence, and unstable measured rows before
  deciding whether to rerun, reduce the matrix, or reject an optimization.
- Hash API benchmark reports can include public-safe build metadata via `--build-cache <build-dir>` so optimization agents can distinguish Release/Debug, CUDA architecture sets, generator, vcpkg triplet, and CUDA compiler version without committing local paths.
- Conservative CUDA batch-size selection helpers are available under `src/hashapi/` and miner integration uses them when no explicit `--batchSize` limit is provided.
- The next default phase is Phase 2 and Phase 3: remove structural overhead, then optimize the hot path with repeatable evidence.
- Do not start risky CUDA kernel rewrites until benchmark and timing data show that CPU-side setup, input generation, and allocation overhead are no longer the dominant bottlenecks.

Current observations:

- Generated batch paths can be dominated by `input_ms`, which includes CPU-side key generation and Argon2 first-block input preparation.
- After CUDA first-block preparation was parallelized across CPU worker threads for generated-key batches, `input_ms` still dominates larger batch paths, but viable batch sizes shifted upward.
- Repeated main-target-only scans on a CUDA-capable local GPU support d1/b2048, d8/b3072, and d64/b3072 as current conservative low-difficulty defaults. Treat this as local evidence only, not a universal hardware limit.
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
- A later trusted generated-key CUDA d8/b2048 refresh superseded the 78.3k H/s continuity number with about `79.2k H/s` median and normal benchmark trust. Use the newer value for rejected/accepted comparison until a future stable report supersedes it.
- A short d8/b2048 detailed worker-cap smoke using the first-block CPU-sum-to-wall ratio kept automatic first-block workers ahead of explicit caps: auto reached about `69.1k H/s`, while caps 4, 8, and 12 reached about `66.1k`, `64.4k`, and `66.3k H/s`. The auto first-block CPU-sum-to-wall ratio was about `5.2`, showing useful parallelism but not enough evidence to change worker defaults. Treat this as measurement-only smoke evidence, not a stable tuning claim.
- After first-block scheduling metadata was added and a clean Release CUDA rebuild was used, a generated-key d8/b2048 default-timing refresh produced `61.99k H/s` median with `3.24%` spread, `benchmark_trust: normal`, `report_ok: true`, automatic first-block worker count `8`, and chunk size `256`. Treat this as the current clean-binary local baseline, but do not let it replace the higher trusted `79.2k H/s` best result for progress accounting without another stable confirmation.
- A matching post-metadata d8/b2048 detailed run produced `64.68k H/s` median with `4.65%` spread. `input_ms` stayed dominant at about `61%` of wall time, `first_block_ms` at about `57%`, and first-block CPU-sum-to-wall was about `4.32`, confirming digest-heavy first-block preparation with partial CPU parallelism.
- A post-metadata detailed first-block worker-cap scan on d8/b2048 found auto workers still the best stable default: auto reached `58.88k H/s` median with `9.29%` spread, cap 4 was stable but slower at `58.07k H/s`, cap 2 was slow and unstable at `41.55k H/s`, and caps 6 and 8 had higher/noisy medians but exceeded the stability threshold. Keep automatic first-block worker selection as the default.
- A refreshed d8 detailed worker-wall diagnostic run with build metadata and normal benchmark trust kept `input_ms` dominant. d8/b2048 reached `69.93k H/s` median with `2.28%` spread, about `61.1%` input time, about `57.0%` first-block time, worker-wall ratio about `0.74`, and first-block scheduling overhead about `591 ms` per 4-second sample. d8/b1024 reached `66.94k H/s` median with `5.56%` spread, about `61.7%` input time, worker-wall ratio about `0.67`, and first-block scheduling overhead about `774 ms`. Treat this as diagnostic evidence that thread start/scheduling overhead is material, but keep digest-heavy first-block work as the dominant Track C target until the new thread-start fields are confirmed on b2048/b1024.
- A later d8 start-skew diagnostic with normal benchmark trust kept `input_ms` dominant and confirmed large worker-start offsets. d8/b1024 was stable at `77.30k H/s` median with `2.34%` spread, about `60.8%` input time, about `56.7%` first-block time, worker-wall ratio about `0.76`, thread launch about `4.4%` of first-block wall time, latest worker start about `39.7%`, and worker start span about `36.3%`. d8/b2048 reached `84.03k H/s` median but was unstable with `13.39%` spread, so use it only as diagnostic evidence. This suggests scheduler experiments should compare worker finish span and join overhead before changing defaults.
- A follow-up d8 finish-overhead diagnostic with normal benchmark trust showed post-worker join/accounting overhead is not the main first-block problem. d8/b2048 was stable at `80.06k H/s` median with `4.60%` spread, about `60.7%` input time, about `56.6%` first-block time, worker-wall ratio about `0.82`, finish-wall ratio about `0.99`, scheduling overhead about `404 ms` per 4-second sample, and post-worker overhead about `22 ms`. d8/b1024 was stable at `74.25k H/s` median with `2.01%` spread, about `61.5%` input time, about `57.6%` first-block time, worker-wall ratio about `0.75`, finish-wall ratio about `0.98`, scheduling overhead about `576 ms`, and post-worker overhead about `36 ms`. This points away from post-join cleanup and toward worker start skew, worker-local digest cost, or a safer first-block architecture experiment.
- A short d8 dynamic first-block chunk scan with normal benchmark trust found candidate scheduler gains but is not enough for defaults. At b1024, static chunking reached `73.67k H/s`, dynamic chunk `16` reached `79.07k H/s`, chunk `32` reached `80.40k H/s`, and chunk `64` reached `78.46k H/s`, all stable. At b2048, static chunking reached `81.75k H/s`; dynamic chunks `16`, `32`, and `64` reached `87.64k`, `85.21k`, and `86.78k H/s`, all stable in the short matrix. This justified a longer confirmation but not a default change.
- A longer d8 dynamic chunk confirmation with normal benchmark trust confirmed b1024 chunk `32` as a strong candidate: static b1024 reached `57.81k H/s` with `5.41%` spread, while chunk `32` reached `71.59k H/s` with `2.63%` spread and lower input share. The same confirmation kept b2048 unresolved: static b2048 reached `68.75k H/s` with `2.93%` spread, while chunk `16` reached `78.85k H/s` but had `11.26%` spread, above the stability gate. Do not change the b2048 default from this noisy candidate; confirm b2048 chunk `32` and chunk `64`, or rerun chunk `16` longer, before making a default policy change.
- A later 8-second repeat-3 b2048-only dynamic chunk confirmation with normal benchmark trust resolved the earlier b2048 uncertainty enough to justify a default-policy experiment, but not a broad default change by itself. Static b2048 reached `80.52k H/s` with `4.47%` spread and about `61.0%` input time. Dynamic chunk `16` reached `87.50k H/s` with `0.98%` spread and about `57.5%` input time. Dynamic chunk `32` reached `86.07k H/s` with `2.16%` spread and about `57.6%` input time. Dynamic chunk `64` reached `80.20k H/s` with `11.50%` spread and remains unstable. Treat chunk `16` as the best current b2048 candidate and chunk `32` as the safer cross-scenario candidate. Preserve the explicit static `0` behavior until an automatic policy has its own clear override semantics.
- The first opt-in automatic dynamic chunk policy selects chunk `32` only for generated-key d8 CUDA batches with at least 1024 attempts. A short 3-second repeat-2 smoke with normal benchmark trust validated the implementation and summary fields: b1024 static reached `67.61k H/s` with `4.99%` spread while auto reached `77.81k H/s` with `2.41%` spread; b2048 static reached `78.19k H/s` with `2.46%` spread while auto reached `81.96k H/s` with `1.91%` spread. Treat this as implementation validation and local evidence for longer confirmation, not as a miner-default change.
- A longer 8-second repeat-3 opt-in auto confirmation with normal benchmark trust and stable spreads strengthened the policy evidence. b1024 static reached `76.35k H/s` with `2.04%` spread, while auto chunk `32` reached `81.65k H/s` with `2.20%` spread, about `+6.9%`. b2048 static reached `80.61k H/s` with `3.70%` spread, while auto chunk `32` reached `87.16k H/s` with `3.44%` spread, about `+8.1%`. `input_ms` remained dominant but dropped from about `61.2%` to `58.6%` at b1024 and from about `61.0%` to `57.5%` at b2048. This is enough local evidence to consider wiring opt-in auto policy into the miner-generated CUDA path for covered scenarios, while keeping manual/static override semantics intact.
- A d1/d64 dynamic chunk diagnostic scan with normal benchmark trust found chunk `16` stable and promising for both difficulty values. At d1/b1024, static reached `71.71k H/s` and chunk `16` reached `78.21k H/s`; at d1/b2048, static reached `68.88k H/s` and chunk `16` reached `89.33k H/s`. At d64/b1024, static reached `70.74k H/s` and chunk `16` reached `75.94k H/s`; at d64/b2048, static reached `75.13k H/s` and chunk `16` reached `81.65k H/s`. A rebuilt auto confirmation accepted d1 only: d1/b2048 auto selected chunk `16` and reached `85.86k H/s` with `3.38%` spread, while d8 still selected chunk `32` in a policy smoke. The d64 rebuilt confirmation was inconsistent, so d64 remains static under auto policy until a longer same-settings confirmation is stable and positive.

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
- Keep generated-key indexing, salt/key materialization, Argon2 setup, CUDA
  execution, finalization, encoding, and matching separable in both code and
  timing metadata.
- Keep variable `m = diff` sequences cheap to benchmark through one reusable
  automation surface.
- Prefer narrow architecture cleanup when it unlocks repeated AI optimization,
  but avoid platform rewrites that do not shorten the hash path.

The future program architecture should make it practical to take this Hash API
and tune it on another CUDA machine, including RTX 3050-class or higher-end
GPUs, without copying frontend, wallet, marketplace, or deployment state.

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
- secrets, tokens, cookies, private key material, wallet credentials, or personal addresses
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
- run serious benchmark reports with `--preflight-report-quality` and
  `--fail-on-report-quality` so high CPU load or low benchmark trust returns a
  non-zero exit code, while still writing a diagnostic report when possible
- use `--preflight-only` for a low-cost environment gate before expensive
  high-difficulty runs; it writes an empty quality report and never launches
  benchmark subprocesses
- when `--preflight-wait-seconds` is set, the benchmark runner waits before the
  report and before each warm-up or measured subprocess; if the wait expires,
  that subprocess is skipped instead of starting a known low-trust high-difficulty
  run
- if the environment is noisy but long-run automation should keep searching for
  a stable launch window, use a longer report-level `--preflight-wait-seconds`
  together with a shorter `--subprocess-preflight-wait-seconds` so each
  warm-up/repeat still has a quality gate without letting one report spend most
  of its timeout budget waiting before every subprocess
- a positive preflight wait now requires two consecutive normal-trust samples by
  default before launching, and each subprocess is checked one more time
  immediately before launch; override with `--preflight-stable-samples` only for
  short diagnostics, not serious high-difficulty claims
- rerun if the claimed improvement is smaller than the run-to-run noise

For batch-size recommendations, prefer custom scan matrices over a single preset when tuning for a specific difficulty range:

```bash
python scripts/hash_api_benchmark.py --binary <miner-binary> --backend cuda --device 0 --seconds 10 --warmup 1 --repeat 3 --scan-difficulty 1 --scan-difficulty 8 --scan-difficulty 64 --scan-batch-size 256 --scan-batch-size 512 --scan-batch-size 1024 --scan-batch-size 2048 --scan-batch-size 3072 --scan-first-block-dynamic-chunk-auto --recommendations-only --preflight-report-quality --preflight-wait-seconds 60 --preflight-wait-interval 5 --fail-on-report-quality --output .benchmarks/cuda-scan.json
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
- an earlier per-chunk finalized hash string reuse attempt produced noisy/regressed confirmation; this rejection is superseded by the later accepted scoped base64 buffer reuse with a 10-second repeat-3 d8/b2048 confirmation
- parallelizing CUDA final hash materialization across CPU threads preserved the golden CUDA hash and lowered `finalize_hash_ms`, but benchmark subprocesses exited unstably during d8/b2048 generated CUDA runs; a later per-batch local-vector parallel finalization/base64 attempt also produced an access-violation subprocess exit during d8/b3072 detailed benchmarking, so keep finalization serial unless backend output-memory lifetime and thread-safety are redesigned
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
- replacing the per-call `digestLong` length-prefix `store32` with static little-endian constants for the 64-byte hash and 1024-byte block output lengths preserved focused tests and the CUDA golden hash, but a d8/b2048 generated CUDA confirmation reached only about `77.5k H/s` median with `3.2%` spread versus the refreshed trusted `79.2k H/s` baseline, so keep the simple dynamic `store32` path
- running the final first-block chunk on the caller thread while creating one fewer background worker preserved focused tests, a Release CUDA rebuild, and the CUDA golden hash, and a short d8/b2048 generated CUDA run improved median throughput from about `56.7k H/s` to `59.4k H/s`; the longer 10-second repeat-3 confirmation regressed from about `58.8k H/s` to `58.1k H/s` despite stable spreads, so keep the existing all-background-worker scheduling unless a broader scheduler redesign changes the tradeoff
- moving the 64-slot CUDA finalization hash buffers and base64 strings from per-batch local scratch into reusable `CudaHashBackend` members preserved focused tests, rebuilt successfully, and preserved the CUDA golden hash, but the repeated d8/b2048 generated CUDA auto benchmark hung past the command timeout without writing a report and left a benchmark subprocess running. The experiment was reverted; do not retry backend-lifetime finalization scratch reuse unless the backend lifetime and thread-safety design changes materially.

Measurement cautions:

- A later clean-source post-revert d8/b2048 confirmation was much slower at about `26.2k H/s` median with `16.5%` spread while the host CPU load was observed near saturation. Treat that run as a low-trust environment sample, not as a new baseline or a code regression.
- Benchmark reports now include public-safe environment metadata with aggregate CPU load samples around each benchmark subprocess and `benchmark_trust`. Do not accept CPU-side input/first-block throughput conclusions from reports marked `benchmark_trust: low` unless the result is only being used to diagnose environment noise.
- Measurement-only update: per-command environment sampling was validated with focused tests and a real CUDA smoke. A one-warmup, one-repeat CUDA smoke produced a sanitized summary with `sample_count: 4` and `benchmark_trust: normal`, confirming that automation can now detect mid-run CPU load spikes more reliably.
- Measurement-only update: first-block scheduling metadata is now exposed as public-safe integers in Hash API JSON and benchmark summaries. The fields are `first_block_worker_count` and `first_block_chunk_size`. This does not change default scheduling behavior; it only makes automatic worker count and chunking visible for future scans.
- Measurement-only update: comparison text output now includes `first_block_workers`, `first_block_worker_count`, and `first_block_chunk_size`, so before/after CSV-style reports expose both the requested cap and the selected first-block scheduling shape without requiring JSON output.
- Measurement-only update: benchmark summaries and recommendation entries now include `first_block_workers`, `first_block_worker_count`, and `first_block_chunk_size`, so batch-size and worker-cap scans preserve both the requested cap and selected first-block schedule in selected and candidate rows.
- Measurement-only update: detailed first-block timing now includes `first_block_max_worker_ms`, allowing agents to compare the slowest worker-local wall time against `first_block_ms` and aggregate first-block CPU time without adding default benchmark overhead.
- Measurement-only update: benchmark timing analysis now derives first-block worker-wall ratio and scheduling-overhead fields from existing detailed timings. This does not add new CUDA timing overhead.
- Measurement-only update: detailed first-block timing now includes thread launch duration, latest worker start offset, worker start span, latest worker finish offset, and worker finish span. These fields are detailed-only diagnostics for scheduler experiments and do not add default benchmark timing overhead.
- Measurement-only update: worker finish-span diagnostics now expose whether first-block wall time is explained by the last worker to finish or by post-worker join/accounting overhead, reducing the risk of misreading start skew as a scheduler optimization target.
- Measurement-only update: benchmark timing analysis now derives first-block finish-wall ratio and post-worker overhead fields from existing detailed worker finish timing. This does not add new CUDA timing overhead.
- Measurement-only update: first-block dynamic chunk sizing is now an explicit benchmark-only scheduling knob. It preserves default static chunking at `0` and lets future scans test whether smaller dynamic work chunks absorb worker start skew.
- Measurement-only smoke: a short d8/b256 generated CUDA run confirmed the dynamic chunk knob executes and reports `first_block_dynamic_chunk_size` correctly with normal benchmark trust. Static chunking reached about `50.6k H/s`; dynamic chunk size `64` reached about `45.6k H/s`. Treat this as a functionality smoke only, not a tuning claim, because it used one short sample.
- Measurement-only update: dynamic first-block chunk scans are now useful tuning evidence. Current local evidence supports b1024 chunk `32`, b2048 chunk `16`, and b2048 chunk `32` as candidates for an explicit automatic-policy experiment. Keep chunk `64` diagnostic-only because the longer b2048 confirmation was unstable.
- Measurement-only update: the automatic dynamic chunk policy is now an opt-in benchmark surface. Manual `first_block_dynamic_chunk_size` takes precedence over auto policy, and auto-disabled `0` remains forced static chunking.
- Miner integration update: generated CUDA mining batches now opt into the automatic dynamic chunk policy. The backend applies the policy only to covered generated-key d1, d8, and d64 batches with at least 1024 attempts, so unsupported difficulties and small batches keep static chunking.
- Post-integration validation: focused Hash API tests, a Release CUDA build, the CUDA golden hash, and a short miner-equivalent d8/b2048 auto smoke passed. The smoke reached `82.79k H/s` median with `3.96%` spread, selected dynamic chunk `32`, and kept normal benchmark trust. Treat this as integration validation; use longer confirmations for future default-policy claims.
- Post-auto detailed timing evidence: a short repeat-2 detailed run with normal benchmark trust kept `input_ms` as the dominant stage after auto chunking. d8/b1024 auto reached `82.50k H/s` median with `0.89%` spread, `input_ms` about `58.55%`, `first_block_ms` about `54.21%`, first-block scheduling overhead about `212 ms` per 4-second sample, and post-worker overhead about `44 ms`. d8/b2048 auto reached `89.54k H/s` median with `0.48%` spread, `input_ms` about `57.35%`, `first_block_ms` about `52.75%`, first-block scheduling overhead about `114 ms`, and post-worker overhead about `26 ms`. Treat this as current diagnostic evidence that the auto policy reduced scheduler overhead, but first-block digest/input work and finalization remain larger next targets.
- Auto policy extension: generated-key CUDA d1 batches with at least 1024 attempts now select chunk `16` under `first_block_dynamic_chunk_auto`, while d8 kept chunk `32` at that point and unsupported difficulties kept forced static behavior. Validation passed focused Hash API tests, a Release CUDA rebuild, the CUDA golden hash, a d1/d64 policy smoke, and a d8 policy smoke. Later d64 and d8/b3072 evidence extended this policy further.
- A clean Release CUDA rebuild was needed after adding fields to `HashApiResult`; an incremental rebuild produced corrupted aggregate JSON fields before the clean rebuild. For future `HashApiResult` layout changes, prefer clean rebuild validation before trusting CLI benchmark output.
- Measurement-only update: Hash API JSON, benchmark summaries, recommendations, and comparison output now expose first-block selected chunk `_min` and `_max` fields across aggregated benchmark loops. This prevents variable-`m=diff` runs from being misread when the final batch's selected chunk differs from earlier batches. Validation passed focused Hash API tests, a clean Release CUDA rebuild, the CUDA golden hash, and a short d1/d8/d64 auto sequence smoke; the smoke showed dynamic chunk range `0..32` and selected chunk range `16..256`.
- Accepted finalization optimization: CUDA finalization now keeps reusable per-chunk base64 string buffers alive across finalize chunks instead of destroying and reallocating them for every hash. A same-scenario generated CUDA d8/b2048 auto comparison with seconds `10`, warm-up `1`, repeat `3`, and normal report quality improved median throughput from `82.52k H/s` to `88.05k H/s` (`+6.71%`), reduced spread from `5.05%` to `2.20%`, and reduced `finalize_ms` by about `285 ms`. Correctness validation passed focused Hash API tests, a Release CUDA rebuild, and the CUDA golden hash. `input_ms` remains the dominant stage, so the next optimization should return to generated input and first-block preparation unless newer evidence points elsewhere.
- Rejected finalization experiment: adding a host-side `lanes == 1` fast path in `Argon2Params::finalize` preserved the CUDA golden hash, but the same d8/b2048 auto comparison after base64 buffer reuse improved only from `88.05k H/s` to `88.87k H/s` (`+0.93%`), below the `1%` acceptance threshold and marked unchanged by comparison tooling. Do not retry this exact host finalize-copy shortcut unless the finalization design changes materially.
- Auto policy extension: generated-key CUDA d64/b2048 with manual dynamic chunk `16` was reconfirmed after the finalization buffer optimization. A seconds `10`, warm-up `1`, repeat `3` comparison improved median throughput from `75.63k H/s` static to `79.51k H/s` with chunk `16` (`+5.1%`), with both reports stable and normal-quality. After wiring the policy, a direct d64/b2048 auto confirmation selected chunk `16` and reached `84.65k H/s` with `1.63%` spread. The automatic first-block dynamic chunk policy now includes d64 chunk `16` through b2048; unsupported higher difficulties and larger d64 batches remain static until stable evidence exists.
- Measurement-only refresh: after the d64 auto policy commit, a clean Release d8/b2048 generated CUDA auto run with seconds `10`, warm-up `1`, repeat `3`, and normal benchmark trust reached `86.95k H/s` median with `3.17%` spread. The matching detailed seconds `4`, warm-up `1`, repeat `3` run reached `88.91k H/s` median with `7.48%` spread and confirmed `input_ms` remains dominant at about `58.94%`; `first_block_ms` was about `54.39%`, `finalize_ms` about `29.58%`, and `keygen_ms` only about `3.97%`. First-block worker-local digest CPU time is still much larger than first-block wall time because it sums across workers, while post-worker overhead was only about `28 ms` per 4-second sample. Do not spend the next iteration on keygen-only changes or post-worker cleanup unless newer measurements contradict this.
- Rejected finalization lifetime experiment: backend-member scratch storage for the 64-slot finalization buffers preserved correctness through focused tests, a Release rebuild, and the CUDA golden hash, but the repeated d8/b2048 generated CUDA auto benchmark hung past the command timeout and produced no benchmark report. The lingering subprocesses were stopped and the source experiment was reverted. Treat this as instability evidence, not a performance result.
- Rejected finalization parallelism experiment: per-batch local vectors for all finalized buffers and hashes plus CPU-threaded `Argon2Params::finalize` and base64 encoding preserved the CUDA golden hash, but a generated CUDA d8/b3072 auto detailed run with seconds `4`, warm-up `1`, and repeat `3` had one subprocess access-violation exit, returned `report_ok=false`, and produced no valid performance claim. The source experiment was reverted. Do not retry this local-vector parallel finalization/base64 shape without a broader output-memory lifetime and thread-safety redesign.
- Measurement-only caution: post-revert d8/b3072 worker-cap scanning and a d8 b2048/b3072 confirmation both ran under low-trust high-CPU-load conditions and produced unstable candidates. Do not use those local reports to change worker caps or batch defaults; rerun under normal benchmark trust before making tuning decisions.
- Measurement-only d8 batch confirmation: a follow-up normal-trust d8 b2048/b3072 generated CUDA auto scan with seconds `8`, warm-up `1`, repeat `3`, and `report_quality_ok=true` kept b3072 slightly ahead at about `88.03k H/s` median with `9.98%` spread, versus b2048 at about `86.90k H/s` median with `1.92%` spread. Keep the d8 b3072 default, but treat the near-threshold b3072 spread as a reason to reconfirm before future default changes.
- Measurement-only d8/b3072 chunk scan: a normal-trust explicit dynamic-chunk scan of static, `8`, `16`, and `32` found chunk `8` as the best stable candidate at about `81.87k H/s`, while chunk `16` and `32` had higher or similar medians but exceeded the stability threshold. This does not beat the latest d8/b3072 auto confirmation, so keep the current auto chunk `16` policy and do not switch to chunk `8` without a stronger same-scenario confirmation.
- Accepted d8 batch-size tuning update: after rebuilding away from the rejected finalization scratch experiment, a miner-equivalent d8 auto batch-size window with seconds `10`, warm-up `1`, repeat `3`, and normal report quality found b3072 as the best stable candidate at about `86.94k H/s` with `2.57%` spread. A targeted d8 b2048/b3072 confirmation then reached about `84.33k H/s` at b2048 versus `92.32k H/s` at b3072 with `1.18%` b3072 spread, so the conservative d8 miner default is now b3072. Keep d8/b2048 as the continuity comparison scenario for historical progress accounting.
- Accepted d1 batch-size tuning update: a miner-equivalent d1 auto scan with seconds `10`, warm-up `1`, repeat `3`, and normal report quality reached about `70.34k H/s` at b512, `89.16k H/s` at b2048, and `93.58k H/s` at b3072. A follow-up b3072/b4096 confirmation made b3072 unstable while b4096 was stable at about `89.15k H/s`, effectively tied with the stable b2048 result. The conservative d1 miner default is now b2048; keep b3072 and b4096 as candidate sizes for future confirmations rather than defaults.
- Accepted d64 batch-size tuning update: a miner-equivalent d64 auto scan with seconds `10`, warm-up `1`, repeat `3`, and normal report quality reached about `67.83k H/s` at b512, `85.94k H/s` at b2048, and `87.86k H/s` at b3072. A follow-up b3072/b4096 confirmation kept b3072 best and stable at about `87.61k H/s`, while b4096 was lower at about `84.77k H/s`. The conservative d64 miner default is now b3072.
- Accepted d8 auto chunk policy refinement: a d8/b3072 dynamic chunk scan with seconds `8`, warm-up `1`, repeat `3`, and normal report quality found chunk `16` as the best stable candidate at about `95.56k H/s`, compared with about `94.62k H/s` for chunk `32` and `90.62k H/s` for static chunking. After the policy change, a direct d8/b3072 auto confirmation selected chunk `16` and reached about `95.01k H/s` with `1.59%` spread. A short threshold smoke confirmed d8/b1024 still selects chunk `32` while d8/b2048 selects chunk `16`.
- Accepted d64 auto chunk policy refinement: a d64/b3072 dynamic chunk scan with seconds `8`, warm-up `1`, repeat `3`, and normal report quality found static first-block scheduling best at about `80.48k H/s`, compared with about `79.28k H/s` for chunk `16`. After the policy change, a threshold smoke confirmed d64/b2048 still selects chunk `16` while d64/b3072 falls back to static scheduling. The d64 auto policy now keeps chunk `16` for b1024/b2048 evidence and falls back to static scheduling for b3072 and larger generated-key CUDA batches.
- Measurement-only variable-shape benchmark support: focused Hash API contract/benchmark/compare tests passed, the Release CUDA build completed, the CUDA golden hash stayed unchanged, and a short `difficulty_sequence=1,8,64` plus `batch_size_sequence=2048,3072,3072` smoke returned `report_ok: true` with `batch_size_min=2048` and `batch_size_max=3072`. Treat this as tooling validation, not a performance claim.
- Measurement-only recommendation quality update: benchmark recommendations now carry `benchmark_trust`, `high_cpu_load`, `environment_available`, `environment_sample_count`, and `report_quality_ok`, so `--recommendations-only` output exposes low-trust/high-load runs without requiring the full raw report.
- Measurement-only variable-shape baseline: a normal-trust `difficulty_sequence=1,8,64` plus `batch_size_sequence=2048,3072,3072` generated CUDA auto run with seconds `8`, warm-up `1`, repeat `3`, and `report_quality_ok=true` reached about `68.06k H/s` median but had `17.68%` spread, so it is diagnostic only. Mixed `m=diff` remained dominated by `input_ms` at about `49.70%`, while `setup_ms` rose to about `15.38%`; future variable-`m` work should reduce setup/lifecycle cost only after a narrower stable confirmation.
- Measurement-only d8/d64 sequence checkpoint: a normal-trust `difficulty_sequence=8,64` plus fixed b3072 generated CUDA auto run with seconds `8`, warm-up `1`, repeat `3`, and `report_quality_ok=true` was stable at about `67.08k H/s` median with `5.33%` spread. Default timing showed `input_ms` about `50.61%` and `setup_ms` about `13.74%`; repeated detailed-timing attempts for the same sequence had access-violation subprocess exits, so do not rely on partial detailed setup subfields until variable-sequence detailed mode is stabilized.
- Accepted variable-sequence stability fix: `KernelRunner` teardown now synchronizes its CUDA stream before destroying events, stream, and device memory. This preserved the CUDA golden hash, passed a direct d8/d64 detailed sequence stress of 5 consecutive runs, made the wrapper d8/d64 detailed sequence return `report_ok=true`, and a same-scenario default-timing comparison improved from about `67.08k H/s` to `68.27k H/s` (`+1.77%`) with normal report quality. The now-valid detailed run shows variable-`m` setup is mostly backend reinitialization, with `setup_backend_init_cpu_ms` about `78%` of setup and activation about `22%`.
- Post-fix mainline safety check: a normal-trust d8/b3072 generated CUDA auto run after the teardown sync reached about `87.17k H/s` median with `8.02%` spread. Against the prior normal-trust d8/b3072 confirmation at about `88.03k H/s`, comparison marked it unchanged at `-0.98%`, so the teardown synchronization did not show a mainline same-difficulty regression.
- Rejected runner-cache retry: a bounded two-shape `KernelRunner` cache in `CudaBackend` targeted repeated backend initialization for alternating d8/d64 b3072 generated CUDA runs after the teardown sync. The CUDA golden hash still matched, but a direct d8/d64 detailed sequence stress had four access-violation exits in five attempts, so the experiment was reverted. Do not retry this pointer-based runner cache shape; any future lifecycle reuse must redesign ownership, stream/event synchronization, and memory lifetime rather than retaining recent `KernelRunner` instances inside `CudaBackend`.
- Accepted variable-sequence runner reuse: `KernelRunner` now keeps a single allocation-sized segment-block capacity and can reconfigure to a smaller or equal `segmentBlocks` value when type, version, passes, lanes, and batch size match. This avoids the rejected multi-runner ownership shape while reducing repeated CUDA stream/event/memory setup after a sequence expands to a larger `m=diff` shape. Validation passed focused Hash API tests, a Release CUDA rebuild, the CUDA golden hash, a d8/d64 detailed direct stress of 5 consecutive runs, a d8/d64 default direct stress of 5 consecutive runs, and clean wrapper repeated reports after rerun. The d8/d64 b3072 generated CUDA auto default comparison improved from about `68.27k H/s` to `73.64k H/s` (`+7.86%`) with normal report quality; the detailed comparison improved from about `69.82k H/s` to `72.42k H/s` (`+3.72%`) and reduced `setup_backend_init_cpu_ms` materially. A same-difficulty d8/b3072 generated CUDA auto safety check was unchanged at about `+0.59%`, so this did not show a mainline fixed-`m` regression.
- Measurement-only post-runner-reuse fixed-`m` timing: a d8/b3072 generated CUDA auto detailed run with seconds `4`, warm-up `1`, repeat `3`, and normal report quality reached about `88.21k H/s` median with `6.01%` spread. `input_ms` remained dominant at about `58.29%`, `first_block_ms` was about `53.49%`, `finalize_ms` about `30.54%`, and setup was only about `4.19%`. The next fixed-`m` optimization should still target first-block/input preparation before setup or CUDA kernel rewrites.
- Measurement-only d8/b3072 worker-cap scan after runner reuse: a detailed generated CUDA auto scan with seconds `4`, warm-up `1`, repeat `3`, and normal report quality compared automatic workers with explicit caps `4`, `6`, and `8`. Automatic worker selection stayed best and stable at about `90.36k H/s` median with `5.29%` spread. Cap `6` was stable but lower at about `86.99k H/s`, cap `4` regressed to about `78.07k H/s`, and explicit cap `8` was unstable. Do not change the d8/b3072 worker policy from this evidence; keep using explicit caps only as diagnostics.
- Rejected local finalization string array: replacing the per-batch local `std::vector<std::string>` for finalization hashes with a fixed local `std::array<std::string, 64>` preserved focused tests, the Release CUDA build, and the CUDA golden hash, but the d8/b3072 generated CUDA auto comparison was unchanged at about `+0.05%` (`87.68k H/s` to `87.72k H/s`). The source experiment was reverted; keep the current vector shape unless a broader finalization allocation redesign changes the evidence.
- Measurement-only post-runner-reuse variable-shape refresh: a clean rerun of `difficulty_sequence=1,8,64` with `batch_size_sequence=2048,3072,3072`, seconds `8`, warm-up `1`, repeat `3`, generated CUDA auto, and normal report quality reached about `65.67k H/s` median with `3.67%` spread. A first wrapper attempt had one access-violation subprocess exit, but five direct same-command runs exited cleanly and the wrapper rerun was clean. Against the older diagnostic baseline at about `68.06k H/s` with `17.68%` spread, comparison reported `-3.51%` but the before-run was too noisy for a strong regression claim. Treat the current stable mixed-shape evidence as about `65.67k H/s`, with `input_ms` about `49.96%`, setup about `15.44%`, and finalization about `23.94%`.
- Rejected runner batch-capacity reuse: extending single-runner reuse from equal batch sizes to smaller-or-equal batch sizes targeted mixed `batch_size_sequence=2048,3072,3072` setup overhead, but direct variable-shape stress failed with access-violation exits in four of five attempts and an abnormal exit in the remaining attempt. The source experiment was reverted. Keep runner reuse limited to equal batch sizes unless the kernel runner memory layout and batch-size reconfiguration rules are redesigned and stress-tested.
- Superseded stale-binary warning: earlier fixed-b3072 and fixed-b2048 d1/d8/d64 sequence failures were measured before rebuilding after a reverted batch-capacity experiment, so those rejection records were invalid. After a clean Release CUDA rebuild and golden hash check, two-difficulty direct sequence probes all exited cleanly, and wrapper repeats for fixed b2048 and fixed b3072 were both clean with normal report quality.
- Accepted fixed-batch three-difficulty evidence: `difficulty_sequence=1,8,64` with fixed b2048 reached about `83.21k H/s` median with `3.51%` spread, while fixed b3072 reached about `82.13k H/s` median with `4.28%` spread. Both avoid the mixed-shape setup churn seen in `batch_size_sequence=2048,3072,3072`, where the latest stable evidence was about `65.67k H/s` and setup was about `15.44%`. Treat fixed b2048 as the current best local d1/d8/d64 sequence shape, with fixed b3072 close but slightly lower; confirm on future GPU classes before making this universal.
- Tuning helper update: Hash API tuning now exposes sequence-aware CUDA batch-size helpers. For variable-`m` difficulty sets, the helper picks the most restrictive known tuned batch size across the sequence and applies the memory limit for the maximum difficulty, preserving explicit user limits. This makes the fixed b2048 d1/d8/d64 evidence reusable by future CLI, miner, service, or autotuning integration without hard-coding local benchmark command lines.
- Measurement infrastructure update: the Hash API CLI now exposes CUDA automatic batch-size selection through `--auto-batch-size`, and `scripts/hash_api_benchmark.py` exposes generated variable-difficulty scenarios through `--sequence-auto-batch-size`. This wires the sequence-aware tuning helper into the long-running benchmark loop so future agents can measure variable `m=diff` sequences without manually hard-coding the fixed sequence batch size.
- Measurement-only sequence-auto comparison: a normal-trust wrapper run with `difficulty_sequence=1,8,64`, seconds `8`, warm-up `1`, repeat `3`, and detailed sequence timings showed `--sequence-auto-batch-size` selecting fixed b2048 and completing cleanly at about `75.70k H/s` median with `7.64%` spread. The matching manual fixed b2048 scenario completed at about `72.88k H/s` median with `7.26%` spread. The mixed `batch_size_sequence=2048,3072,3072` scenario had one access-violation subprocess exit and left the whole report at `report_ok=false`; its two successful samples were about `68.88k H/s` median with setup still about `15.75%`. Treat this as measurement-infrastructure confirmation, not a new performance-best claim, and keep mixed-shape sequence runs classified as unstable unless the runner reconfiguration design changes.
- Accepted sequence auto plus first-block auto evidence: after adding `--sequence-first-block-dynamic-chunk-auto`, a normal-trust single-scenario wrapper run for `difficulty_sequence=1,8,64`, `--sequence-auto-batch-size`, seconds `8`, warm-up `1`, repeat `3`, and detailed timings completed with `report_ok=true`. It selected fixed b2048, selected first-block dynamic chunk `16`, and reached about `85.46k H/s` median with `2.69%` spread. Against the earlier sequence-auto static-first-block measurement at about `75.70k H/s`, this is about `+12.89%` for the same fixed-batch variable-`m` sequence shape. Treat `--sequence-auto-batch-size --sequence-first-block-dynamic-chunk-auto` as the current preferred variable-`m` benchmark shape.
- Non-detailed preferred sequence baseline: the same `difficulty_sequence=1,8,64`, `--sequence-auto-batch-size`, `--sequence-first-block-dynamic-chunk-auto`, seconds `8`, warm-up `1`, repeat `3` scenario without detailed timings completed with `report_ok=true` at about `84.83k H/s` median and `7.45%` spread. This is close to the detailed `85.46k H/s` run, so detailed timing overhead does not appear to distort the current preferred variable-`m` sequence conclusion. The default timing breakdown still showed `input_ms` dominant at about `56.76%`, with first-block wall time about `52.12%`, finalization about `29.49%`, and keygen about `4.00%`.
- Measurement-only sequence chunk scan: a fixed b2048 `difficulty_sequence=1,8,64` manual chunk scan with seconds `8`, warm-up `1`, repeat `3`, and normal environment trust left the wrapper report at `report_ok=false` because chunk `8` and chunk `32` each had one access-violation subprocess exit. Static first-block scheduling completed but was unstable at about `75.66k H/s` median with `11.35%` spread. Manual chunk `16` was the only clean stable dynamic candidate, reaching about `82.26k H/s` median with `2.63%` spread. Keep the current auto-selected chunk `16` for this variable-`m` shape; do not promote chunk `8` or `32` without a clean repeat.
- Rejected finalize chunk-size experiment: increasing the local host finalization chunk from `64` to `128` preserved the CUDA golden hash and built cleanly, but the preferred non-detailed `difficulty_sequence=1,8,64`, `--sequence-auto-batch-size`, `--sequence-first-block-dynamic-chunk-auto`, seconds `8`, warm-up `1`, repeat `3` benchmark regressed to about `80.38k H/s` median with `11.56%` spread versus the current about `84.83k H/s` baseline. `finalize_ms` also rose to about `30.04%` of total time. The source experiment was reverted; keep the 64-item finalization chunk unless a broader finalization layout redesign changes the evidence.
- Rejected fixed-64-byte base64 fast path: a specialized encoder for the fixed 64-byte final hash preserved focused Hash API tests, a Release CUDA build, and the CUDA golden hash, and it reduced nested `base64_ms` by roughly half. End-to-end evidence did not support keeping it: the preferred `difficulty_sequence=1,8,64`, `--sequence-auto-batch-size`, `--sequence-first-block-dynamic-chunk-auto`, seconds `8`, warm-up `1`, repeat `3` benchmark regressed to about `75.03k H/s` median with `15.72%` spread, and a d8/b2048 generated CUDA auto confirmation had one subprocess access-violation exit with `report_ok=false`. The source experiment was reverted; do not retry this exact fixed-length encoder unless the finalization design also addresses the throughput instability and subprocess crash.
- Measurement-only post-base64-revert smoke: after reverting the fixed-64-byte base64 fast path, the CUDA golden hash still matched and five direct one-second `difficulty_sequence=1,8,64` generated CUDA auto subprocess runs exited cleanly. A wrapper smoke for the same preferred sequence shape with seconds `4`, warm-up `1`, repeat `3` had three stable measured samples around `91.46k H/s` median, but the warm-up subprocess exited with an access violation, leaving `report_ok=false`; do not treat this as a performance-best claim. Treat it as evidence that occasional sequence subprocess teardown instability can still invalidate otherwise stable samples, so future accepted sequence claims need clean wrapper reports or separate direct stress confirmation.
- Rejected manual salt hex decoder: replacing `hex_to_bytes` internals with a direct nibble decoder avoided `substr` and `std::stoi` temporaries, preserved focused Hash API tests, rebuilt cleanly, and preserved the CUDA golden hash. The d8/b3072 generated CUDA auto confirmation was stable and normal-quality at about `94.08k H/s`, but it did not beat the current d8/b3072 auto evidence around `95.0k-95.6k H/s`; `input_ms` and first-block timing remained dominant. The source experiment was reverted; do not retry salt decode micro-optimizations without a broader first-block prehash redesign or a same-scenario before/after showing material gain.
- Measurement-only d8/b3072 worker-cap refresh: a detailed d8/b3072 generated CUDA auto refresh with seconds `4`, warm-up `1`, repeat `3`, and normal report quality reached about `82.75k H/s` median and again showed `input_ms` dominant at about `58.24%`, first-block wall time about `53.29%`, and first-block digest CPU sum far above wall time. A worker-cap scan for caps `0`, `4`, `6`, and `8` left `report_ok=false` because cap `4` had a subprocess access-violation exit; among valid rows cap `8` was highest at about `89.48k H/s`, cap `6` reached about `85.80k H/s`, and auto cap `0` reached about `82.87k H/s`. Do not change worker defaults from this invalid/noisy scan, especially because explicit cap `8` and auto both selected 8 workers; rerun a clean targeted auto-vs-cap8 confirmation before treating worker caps as a tuning win.
- Measurement-only auto-vs-cap8 confirmation: a clean d8/b3072 generated CUDA auto-vs-explicit-cap8 confirmation with seconds `8`, warm-up `1`, repeat `3`, and normal report quality showed no material worker-cap difference. Auto workers reached about `89.97k H/s` median with `4.54%` spread, while explicit cap `8` reached about `90.01k H/s` median with `0.39%` spread; both selected 8 workers and first-block chunk `16`. The about `+0.05%` median delta is below the acceptance threshold and confirms the earlier cap8 scan difference was measurement/order noise, not a policy signal. Keep automatic worker selection unchanged.
- Rejected initial-hash prefix cache: caching a prepared BLAKE2b state for the common 64-byte password initial-hash prefix passed focused Hash API tests and compiled in a Release CUDA build, but the CUDA golden hash subprocess exited without JSON after the build. The source experiment was reverted and the binary rebuilt. Do not retry initial-hash state caching unless the design includes a smaller standalone correctness harness for copying BLAKE2b state and preserves the golden CUDA command before any throughput benchmark.
- Rejected indexed key-generation fill: changing `RandomHexKeyGenerator::nextRandomKey` from `reserve`/`push_back` to pre-sized indexed writes preserved key-prefix and focused Hash API tests, a clean Release CUDA rebuild, and the CUDA golden hash. The d8/b3072 generated CUDA auto confirmation was not acceptable: median throughput dropped to about `79.27k H/s` with `10.72%` spread, below current d8/b3072 evidence, even though `keygen_ms` fell to about `2.92%` of wall time. The source experiment was reverted; do not spend more cycles on keygen-only micro-optimizations unless newer timing shows key generation is a dominant stage.
- Measurement-only compare fix: config matching now uses requested `first_block_dynamic_chunk_size` and `first_block_dynamic_chunk_auto` from the scenario instead of selected summary values, so auto variable-sequence reports can be compared even when selected chunk ranges differ between runs.
- Measurement-only input residual diagnostics: benchmark `timing_analysis` now derives `input_explained_ms`, `input_residual_ms`, `input_explained_to_input`, and `input_residual_pct` from `input_ms`, `keygen_ms`, and `first_block_ms`. This helps future agents distinguish true generated-input materialization overhead from first-block work before changing the hot path. Validation passed focused benchmark/compare unit tests, and short CUDA detailed smokes showed the fields in JSON summaries. A d8/b3072 generated CUDA auto smoke with seconds `2`, warm-up `1`, repeat `2`, and normal report quality showed `input_residual_pct` about `1.08%`, `input_ms` about `55.98%`, first-block wall time about `51.57%`, and first-block worker wall/wall about `96.48%`. Treat this as measurement-only evidence that the next hot-path step should focus on first-block digest/preparation rather than generic input bookkeeping.
- Measurement-only compare diagnostics: `scripts/hash_api_compare.py` now carries numeric `timing_analysis` scalar fields through normalization and reports `analysis_metric_deltas`, so before/after reports can directly compare input residual and first-block scheduling ratios. Validation passed focused benchmark/compare unit tests and a self-compare of the ignored d8/b3072 residual smoke report.
- Measurement-only Blake2b copy self-test: CMake now builds `blake2b-copy-selftest`, a small diagnostic executable that verifies copying a Blake2b state after prefix absorption matches one-shot hashing and that copied states remain independent for different suffixes. This does not change the hash hot path or retry the rejected initial-hash prefix cache. It creates the standalone correctness guard required before any materially different prefix-state caching experiment. Validation passed the target build, the self-test executable, and the focused Hash API contract test.
- Rejected initial-hash prefix cache retry: after adding the standalone Blake2b copy self-test, a narrower retry cached the common 64-byte password Argon2 initial-hash prefix inside `Argon2Params`. It preserved the focused Hash API tests, the Blake2b copy self-test, the Release CUDA build, and the CUDA golden hash, but the same d8/b3072 generated CUDA auto detailed smoke regressed from about `86.18k H/s` to about `75.64k H/s` (`-12.23%`) with normal report quality. The source experiment was reverted. Do not retry initial-hash prefix-state caching around the 64-byte generated password path unless the first-block prehash design changes more broadly and the same-scenario comparison shows material improvement.
- Accepted opt-in GPU first-block experiment: the Hash API now exposes `gpu_first_blocks` / `--gpu-first-blocks` as an explicit CUDA-only experiment that prepares the first two Argon2 blocks on the device for supported `t=1`, single-lane requests. Default host-prepared first blocks remain unchanged. The implementation includes device-side BLAKE2b/Argon2 prehash helpers, a CUDA first-block kernel, separated `gpu_first_block_ms` timing, Python benchmark/compare propagation, service payload forwarding, validation rules, docs, and contract tests. Validation passed focused Hash API tests, a Release CUDA rebuild, and the CUDA golden hash with and without `--gpu-first-blocks`. A short d8/b2048 generated CUDA no-XUNI comparison with seconds `2`, warm-up `1`, repeat `3`, and normal report quality improved median throughput from about `56.08k H/s` to about `116.98k H/s` (`+108.59%`), with both rows stable. A longer seconds `10`, warm-up `1`, repeat `3` check improved from about `66.91k H/s` to about `143.63k H/s` (`+114.68%`) with normal trust, but both spreads were slightly above the `10%` stability gate (`12.70%` default, `11.04%` opt-in), so do not enable this path by default yet. Next confirmations should rerun d8/b2048 and d8/b3072, then test the preferred variable-`m` sequence before adding auto policy.
- Measurement-only GPU first-block follow-up: short generated CUDA confirmations with seconds `4`, warm-up `1`, repeat `3`, and normal report quality continued to favor the opt-in device first-block path. At d8/b3072, the default row was too noisy to trust as a stable baseline (`47.34k H/s` median, `64.04%` spread), while `gpu_first_blocks=true` reached about `163.51k H/s` median with `13.54%` spread (`+245.43%` versus that noisy median). For fixed-b2048 `difficulty_sequence=1,8,64` with first-block dynamic chunk auto, the default row reached about `62.11k H/s` with `22.31%` spread, while `gpu_first_blocks=true` reached about `160.13k H/s` with `2.12%` spread (`+157.81%`). These are not default-policy proof because some default comparisons were noisy, but they strongly support longer clean confirmations and a benchmark-harness path for scanning `gpu_first_blocks` across fixed and variable `m=diff` scenarios.
- Measurement-only benchmark harness update: `scripts/hash_api_benchmark.py` now supports `--gpu-first-blocks` to enable the explicit device first-block path across generated scenarios and `--scan-gpu-first-blocks` to emit paired default/GPU-first variants in generated fixed-difficulty scans. Validation passed the focused benchmark unit tests. Use this to rerun clean d8/b2048, d8/b3072, and preferred variable-`m` sequence confirmations before considering an automatic policy.
- Accepted auto-batch stability fix: CUDA automatic batch-size selection now queries free device memory through a lightweight CUDA runtime helper instead of constructing a temporary full `CudaBackend`. This preserves the reusable Hash API backend lifecycle for real work and avoids the no-JSON subprocess crash seen in direct `--auto-batch-size` probes. Validation passed focused Hash API tests, a Release CUDA rebuild, direct fixed-d8 auto-batch JSON smoke, direct `difficulty_sequence=1,8,64` auto-batch JSON smoke, and CUDA golden hash checks with and without `--gpu-first-blocks`.
- Measurement-only variable-`m` GPU first-block confirmation: after the auto-batch fix, clean normal-trust wrapper runs for the preferred `difficulty_sequence=1,8,64`, `--sequence-auto-batch-size`, `--sequence-first-block-dynamic-chunk-auto`, seconds `8`, warm-up `1`, repeat `3`, and no XUNI completed for both default and explicit GPU first-block paths. The default path selected fixed b2048 and reached about `76.52k H/s` median with `11.24%` spread, slightly above the stability gate. The `gpu_first_blocks=true` path selected the same fixed b2048, reached about `117.34k H/s` median with `6.34%` spread, and improved median throughput by about `+53.34%` versus that same-run default median. Treat this as strong support for continuing GPU-first policy work, but do not enable it by default until fixed-difficulty d8/b2048 and d8/b3072 confirmations are also clean enough.
- Measurement-only fixed-d8 GPU first-block confirmation: a normal-trust generated CUDA no-XUNI scan over d8/b2048 and d8/b3072 with seconds `8`, warm-up `1`, repeat `3`, and paired default/GPU-first variants produced strong fixed-`m` evidence but not full auto-policy proof. The b2048 rows were clean and stable: default reached about `77.60k H/s` with `2.87%` spread, while `gpu_first_blocks=true` reached about `179.67k H/s` with `1.60%` spread, about `+131.55%` versus the same-run default row. The b3072 default row was also clean and stable at about `82.16k H/s` with `1.15%` spread, and the b3072 GPU-first measured samples were stable at about `186.09k H/s` with `5.84%` spread, but its warm-up subprocess exited without JSON, leaving the combined report invalid. A follow-up serialized b3072 GPU-first repeat completed with normal report quality and no invalid subprocesses at about `180.20k H/s`, but spread was `10.36%`, just above the `10%` stability gate. Keep GPU-first as a high-confidence opt-in optimization for d8/b2048 and variable-`m` b2048; require one more clean b3072 confirmation before enabling any automatic policy for larger d8 batches.
- Rejected host-owned parallel finalization snapshot: copying CUDA output blocks
  into local host memory and CPU-threading finalization/base64 preserved the
  CUDA golden hashes and produced promising clean smokes, including d8/b4096
  GPU-first samples above the previous local checkpoint and a preferred
  variable-`m` GPU-first sequence at about `194.77k H/s` with `3.57%` spread.
  It was not stable enough to keep. The parallel-match sub-shape returned
  structured `bad allocation`, and the safer chunked finalize/base64-only
  sub-shape later produced no-JSON access-violation exits during a d8/b4096
  GPU-first repeat-3 confirmation even though two measured samples were valid.
  Focused Hash API tests, the Release CUDA build, Blake2b self-test,
  finalization diagnostic, and CUDA golden hashes were used during the cycle,
  but wrapper stability failed the acceptance gate. The source experiment was
  reverted; do not retry host-owned parallel finalization snapshots without a
  materially different ownership and subprocess-teardown design.
- Rejected main-kernel register scratch experiment: replacing the main
  `argon2_kernel_oneshot` shared-memory scratch block with an extra register
  `block_th` scratch targeted high-difficulty compute time, but the CUDA
  Release build did not complete within the autonomous build timeout and left
  the compiler subprocess stalled. Focused Hash API tests had passed before the
  build, but no binary or benchmark evidence was produced. The source
  experiment was reverted. Do not retry this all-register core scratch shape
  unless the implementation is split into a smaller compile unit or otherwise
  proves acceptable compile time and resource usage before benchmarking.
- Rejected main-kernel 64-bit shared scratch experiment: replacing the split
  `block_l` shared scratch arrays (`lo` / `hi`) with a single `uint64_t`
  shared array preserved the focused Hash API contract tests, but the Release
  CUDA build did not complete within the autonomous build timeout and left the
  compiler subprocess stalled before a new binary or resource summary was
  produced. The source experiment was reverted. Do not retry this shared
  scratch storage shape unless compile time is isolated first in a smaller CUDA
  build and the resource summary can be produced before benchmarking.
- Measurement update after stable-evidence gates: the current local environment
  intermittently reached low benchmark trust during d4096 refresh attempts, so
  new high-difficulty runs were treated as diagnostics or skipped by preflight
  and were not accepted as speed evidence. Existing trusted warm, stable,
  normal-trust trend points still show best high-difficulty continuity around
  d4096 `10.77k H/s`, d8192 `5.62k H/s`, d16384 `2.85k H/s`, d32768
  `1.40k H/s`, and the `4096,8192,16384` variable-`m` sequence `4.74k H/s`.
  Continue from these trusted points until a fresh normal-trust refresh
  supersedes them.
- Rejected indexed-address bit-ops micro-optimization: replacing the
  `offset % ARGON2_QWORDS_IN_BLOCK`, `addr_index % THREADS_PER_LANE`, and
  `addr_index / THREADS_PER_LANE` expressions with equivalent power-of-two bit
  operations preserved focused tests, rebuilt successfully, preserved CUDA
  golden hashes with and without GPU first blocks, and left the public-safe
  resource shape unchanged. A short high-difficulty GPU-first smoke did not
  support keeping it: d4096 reached about `10.59k H/s` and d8192 about
  `5.48k H/s`, both below the latest accepted loop-split confirmation region.
  The source experiment was reverted. Do not retry this exact index bit-op
  rewrite unless a compiler/resource change creates a materially different
  code-generation hypothesis.
- Rejected slice-template main-kernel experiment: templating the indexed and
  dependent step helpers by fixed slice id and manually expanding the dependent
  slice loop preserved focused tests, rebuilt successfully, and preserved CUDA
  golden hashes with and without GPU first blocks, but it increased sm75 main
  kernel register use from `56` to `57` and short high-difficulty GPU-first
  smoke results were below the accepted loop-split region. d4096 reached about
  `10.53k H/s` and d8192 about `5.41k H/s`, both with normal report quality.
  The source experiment was reverted. Do not retry this slice-templating shape
  unless profiler/resource evidence shows the extra specialization reduces
  instruction cost without increasing register pressure on the target GPU class.
- Rejected indexed-range chunking experiment: grouping Argon2id indexed slices
  into address-word spans preserved focused tests, rebuilt successfully,
  preserved CUDA golden hashes with and without GPU first blocks, but increased
  sm75 main-kernel register use from `53` to `54`. A normal-trust
  high-difficulty GPU-first smoke regressed against the accepted uniform word
  selection baseline: d4096 reached about `10.22k H/s` and d8192 about
  `5.32k H/s`. The source experiment was reverted. Do not retry this
  address-word span helper or reference-pointer loop shape unless profiling
  shows a materially different control-flow or register-pressure hypothesis.
- Rejected ref-area accumulator experiment: moving `slice * segment_blocks +
  offset - 1` out of `compute_ref_pos` into the caller preserved focused tests,
  the CUDA golden hashes, and the public-safe resource summary shape on the
  restored sm75 Release binary, but the trusted d4096 GPU-first benchmark did
  not improve enough to keep the change. The first normal-trust d4096 run fell
  to about `10.34k H/s` with `14.57%` spread and the longer confirmation fell
  to about `7.59k H/s` with `34.99%` spread, both below the accepted
  high-difficulty region and not stable enough for `report_quality_ok`. The
  source experiment was reverted. Do not retry this exact ref-area accumulator
  shape unless a materially different control-flow or compiler hypothesis can
  be defended with a new resource change.
- Rejected main-kernel restrict alias hint experiment: adding targeted
  `__restrict__` qualifiers to main-kernel load/store, scratch, and step-helper
  pointer parameters preserved focused tests, rebuilt successfully, preserved
  CUDA golden hashes with and without GPU first blocks, and left the sm75
  resource summary unchanged at `53` registers, `1024` bytes shared memory, and
  no stack/local memory. It did not produce a credible high-difficulty win:
  d4096 GPU-first reached about `7.78k H/s` under low benchmark trust, then a
  normal-trust confirmation reached only about `5.95k H/s` with `61.00%`
  spread. The source experiment was reverted. Do not retry generic pointer
  alias hints on the main kernel unless the generated resource or instruction
  shape changes.
- Rejected force-inline helper experiment: adding `__forceinline__` to the main
  Argon2 CUDA helper chain preserved focused tests, rebuilt successfully,
  preserved CUDA golden hashes with and without GPU first blocks, and left the
  public-safe resource shape unchanged at sm75 `53`, sm80 `32`, sm86/sm89 `40`,
  `1024` bytes shared memory, and no local memory. A normal-trust
  high-difficulty GPU-first smoke did not support keeping it: d4096 reached
  about `10.61k H/s` and d8192 about `5.51k H/s`, both below the latest
  accepted uniform word selection confirmation region. The source experiment
  was reverted. Do not retry blanket helper force-inlining unless a targeted
  helper or architecture-specific compiler change creates a different resource
  or instruction-count hypothesis.
- Rejected low-32 reference shuffle experiment: replacing full 64-bit
  reference-index shuffles with low-32-bit shuffles preserved focused tests,
  rebuilt successfully, and preserved CUDA golden hashes with and without GPU
  first blocks, but increased sm75 main-kernel register use from `53` to `57`
  while leaving sm80/sm86/sm89 unchanged. The source experiment was reverted
  before high-difficulty benchmarking because the register-pressure regression
  matched previously rejected shapes. Do not retry low-half-only reference
  shuffles unless profiler or architecture-specific code generation evidence
  shows the register cost no longer applies.
- Rejected GPU-first contiguous key upload experiment: adding a byte-buffer
  overload for `prepareInputBlocksOnDevice` and filling generated 64-byte keys
  into contiguous host storage preserved focused Hash API tests, rebuilt
  successfully, preserved CUDA golden hashes with and without GPU first blocks,
  and left the public-safe sm75 resource summary unchanged. It did not produce
  acceptable high-difficulty evidence: the strict d4096 GPU-first auto-batch
  benchmark returned `report_quality_ok=false` with invalid subprocess exits,
  and the only valid sample was about `8.04k H/s`, below the trusted d4096
  baseline region around `10.77k H/s`. The source experiment was reverted. Do
  not retry this extra contiguous-key staging path unless first-block input
  timing becomes dominant again or the key ownership model changes enough to
  avoid the additional lifetime/instability risk.
- Rejected main-kernel multi-warp block experiment: launching four independent
  32-thread Argon2 jobs per CUDA block targeted higher warp occupancy and
  memory-latency hiding on high-difficulty workloads. It preserved focused Hash
  API tests, rebuilt successfully, and preserved CUDA golden hashes with and
  without GPU first blocks, but the sm75 main-kernel register count increased
  from `53` to `57`. Normal-trust high-difficulty GPU-first evidence did not
  justify that resource cost: d4096 reached about `10.73k H/s` and d8192 about
  `5.44k H/s`, both below the accepted best regions around `10.77k H/s` and
  `5.62k H/s`. The source experiment was reverted. Do not retry the same
  four-warp-per-block launch shape unless a different architecture or occupancy
  analysis shows that the extra register pressure no longer applies.

Do not retry rejected experiments unless the implementation shape has changed enough to remove the original failure mode and the new attempt includes correctness cross-checks.

## Next Autonomous Iteration

Start the next cycle from the latest clean commit and this decision tree:

1. If the worktree is dirty, identify whether it is a previous-agent experiment. Revert only rejected experiments owned by the current goal.
2. If the worktree contains another measurement-only tooling slice, finish its tests, documentation, CUDA smoke, privacy scan, and commit before editing performance code.
3. Run the focused Hash API tests before editing performance code.
4. Build or reuse the clean Release CUDA binary from the configured preset.
5. Run CUDA golden hash checks with and without `--gpu-first-blocks` before trusting benchmark data.
6. Refresh or load a realistic high-difficulty generated-key CUDA GPU-first baseline with auto batch sizing, no XUNI, warm-up `1`, repeat `3`, and zero invalid subprocesses. The current d4096 baseline is already available; extend toward d8192, d16384, and d32768 as memory and run time permit.
7. Refresh or load the realistic variable-`m` GPU-first sequence baseline, for example `difficulty_sequence=4096,8192,16384`.
8. Use detailed timing to decide whether CUDA compute/kernel efficiency, batch sizing, setup/lifecycle, transfers, or matching is the current bottleneck.
9. If continuing directly from the latest checkpoint, do not revisit the rejected `threadsPerBlock=256` first-block launch-geometry shape unless a new high-difficulty hypothesis appears.
10. Do not revisit the rejected `__launch_bounds__(THREADS_PER_LANE, 4)` main-kernel shape or the rejected source-lane-only address-block selection shape unless profiling supports a materially different occupancy/register-pressure hypothesis.
11. Do not retry indexed-address bit operations, slice templating, or mask-select
    replacement shapes unless the new design differs materially from the
    accepted uniform word selection and the rejected variants above.
12. Do not retry indexed-range chunking of the Argon2id address-word loop unless
    the implementation avoids the extra sm75 register pressure and has profiler
    evidence that the control-flow savings are real.
13. Do not retry blanket `__forceinline__` on the main Argon2 helper chain
    without a narrower target and evidence that code generation changes.
14. Do not retry low-32-bit-only reference shuffles unless resource summaries no
    longer show the sm75 register-pressure regression.
15. If compute dominates, prefer CUDA kernel-side, memory-layout, or measurement/tooling work over another rejected finalization parallelism snapshot.
16. If Nsight Compute reports GPU performance counter permission errors, do not
    keep retrying it in autonomous runs. Capture `scripts/cuda_resource_summary.py`
    output under ignored benchmark storage and use benchmark timing plus resource
    deltas for the current cycle.
17. Do not repeat rejected salt caching, decoded salt caching, activation caching, pinned host staging, runner caching, first-block lane fast paths, digestLong specializations, `_rotr64` rotate changes, source-lane address selection, fixed-64-byte base64, final-prefix cache, direct final-digest helper, `gpu_final_hashes`, or host-owned parallel finalization snapshots without a materially different implementation shape.

Good next experiment shapes:

- Tune CUDA first-block launch geometry one parameter at a time only when the current high-difficulty baseline shows launch overhead is material.
- Profile or instrument the main Argon2 CUDA kernel only with public-safe output,
  then test one kernel-side hypothesis at a time against d4096 and the
  d4096,d8192,d16384 sequence.
- When profiler counters are unavailable, compare public-safe CUDA resource
  summaries before and after each kernel experiment, then reject changes that
  raise registers, stack, or local memory without stable high-difficulty
  throughput gains.
- Inspect finalization ownership and materialization before attempting any new parallel or device-side final hash path.
- Reduce setup/lifecycle overhead for variable `m=diff` sequences if detailed timings show repeated difficulty changes are costing wall time.
- Reduce generated input preparation overhead only if newer timing shows host input work has become dominant again.
- Treat old CPU first-block scheduling work as secondary now that GPU first blocks are the miner-generated default.
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
python scripts/hash_api_compare.py .benchmarks/before.json .benchmarks/after.json --fail-on-regression --fail-on-report-quality
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

Realistic high-difficulty smoke:

```bash
python scripts/hash_api_benchmark.py --binary <miner-binary> --backend cuda --device 0 --scenario name=cuda-d4096-auto-batch-gfb-smoke,backend=cuda,difficulty=4096,batch_size=0,auto_batch_size=true,gpu_first_blocks=true,first_block_dynamic_chunk_auto=true,seconds=2,warmup=1,repeat=2 --no-xuni --output .benchmarks/d4096-auto-batch-gfb-smoke.json
```

Realistic variable-difficulty smoke:

```bash
python scripts/hash_api_benchmark.py --binary <miner-binary> --backend cuda --device 0 --difficulty-sequence 4096,8192,16384 --sequence-auto-batch-size --sequence-first-block-dynamic-chunk-auto --gpu-first-blocks --seconds 2 --warmup 1 --repeat 2 --no-xuni --output .benchmarks/difficulty-sequence-high-gfb-smoke.json
```

Local high-difficulty trend page:

```bash
python scripts/hash_benchmark_trends.py --input-dir .benchmarks --output .benchmarks/hash-trends/index.html --min-difficulty 4096
```

Optional public-safe trusted summary JSON:

```bash
python scripts/hash_benchmark_trends.py --input-dir .benchmarks --output .benchmarks/hash-trends/index.html --summary-output .benchmarks/hash-trends/trusted-summary.json --min-difficulty 4096
```

Long-running local trend server:

```bash
python scripts/hash_benchmark_trends.py --serve --input-dir .benchmarks --output .benchmarks/hash-trends/index.html --min-difficulty 4096 --host localhost --port 8766
```

Open `http://localhost:8766/` during optimization to watch the performance
curve, trusted gain counters, report quality, spread, batch size, and scenario
history update as new ignored benchmark reports are written.

To launch the same local server and open the trend page in the default browser:

```bash
python scripts/hash_benchmark_trends.py --serve --open-browser --input-dir .benchmarks --output .benchmarks/hash-trends/index.html --min-difficulty 4096 --host localhost --port 8766
```

The default server mode does not open a browser window, which keeps autonomous
long-running optimization sessions non-interactive. Use `--open-browser` only
when a local operator wants to inspect the curve directly.

The trend chart defaults to warm, stable, quality-approved points and draws
separate series per difficulty or difficulty sequence, so cold-start,
single-repeat, low-trust, and unrelated `m=diff` workloads do not get merged
into one misleading curve. The default trusted trend requires at least one
warm-up run and at least two repeats. The `Diagnostics` quality mode can show
invalid, cold, low-trust, or unstable report points for troubleshooting only;
do not use those points as performance evidence.
The benchmark report quality gate follows the same rule: reports with measured
runs must contain warm evidence (`warmup >= 1` and `repeat >= 2`) before
`report_quality_ok` can be true. Preflight-only empty reports remain allowed
because they do not claim throughput.
The comparison gate also checks `report_quality_ok`, `warm_evidence_run_count`,
and `cold_scenarios`, so `hash_api_compare.py --fail-on-report-quality` rejects
cold or single-repeat reports before they can be treated as before/after
performance evidence.
Use the chart's metric selector to switch between median H/s, median
milliseconds per attempt, compute percentage, kernel percentage, and spread.
Median H/s remains the primary repeatable throughput metric, while
milliseconds per attempt is the latency-first view for checking whether each
valid hash is actually getting shorter.
The trusted gain counters use the latest visible point's own difficulty or
sequence as the comparison group, rather than comparing unrelated `m` values.
The page also includes a per-difficulty trusted summary table for latest rate,
best rate, and best gain so All-view scans remain comparable across `m` groups.
The same table shows the `11x` target H/s, best target progress percentage, and
remaining multiplier so long-running sessions can see at a glance how far each
realistic high-difficulty group remains from the aspirational 1000% goal.
Use `--summary-output` when an agent needs the same trusted first/latest/best
gain data as machine-readable JSON without scraping the HTML page.
The summary JSON also reports the `11x` target implied by a `1000%`
throughput-improvement goal, including target H/s, latest/best target progress
percentage, and remaining multiplier for each difficulty group.
The local trend server caches the benchmark directory signature and avoids
reparsing unchanged JSON reports on every page refresh, keeping the monitoring
view lightweight during long runs.

The generated trend page is an ignored local artifact. It embeds only
public-safe benchmark summary fields and excludes raw command lines, local
binary paths, host metadata, salts, and hardware names.

Batch scan candidate search:

```bash
python scripts/hash_api_benchmark.py --binary <miner-binary> --backend cuda --device 0 --preset batch-scan --seconds 1 --warmup 1 --repeat 2 --recommendations-only --output .benchmarks/batch-scan-smoke.json
```

Serious batch scan:

```bash
python scripts/hash_api_benchmark.py --binary <miner-binary> --backend cuda --device 0 --seconds 8 --warmup 1 --repeat 3 --scan-difficulty 1024 --scan-difficulty 4096 --scan-difficulty 8192 --scan-difficulty 16384 --scan-batch-size 0 --scan-gpu-first-blocks --scan-first-block-dynamic-chunk-auto --recommendations-only --preflight-report-quality --preflight-wait-seconds 60 --preflight-wait-interval 5 --fail-on-report-quality --output .benchmarks/high-diff-batch-scan-stable.json
```

Before/after comparison:

```bash
python scripts/hash_api_compare.py .benchmarks/before.json .benchmarks/after.json --fail-on-regression --fail-on-report-quality --min-change-pct 1
```

CUDA resource summary:

```bash
python scripts/cuda_resource_summary.py --binary <miner-binary> --output .benchmarks/resource-after.json
```

CUDA resource comparison:

```bash
python scripts/cuda_resource_summary.py --compare-before .benchmarks/resource-before.json --compare-after .benchmarks/resource-after.json --fail-on-regression --output .benchmarks/resource-compare.json
```

Use resource comparison as a fast gate for CUDA kernel experiments. If a change
raises registers, stack, or local memory, reject or redesign it unless a
normal-trust high-difficulty benchmark proves a stable same-scenario throughput
gain that justifies the extra resource pressure. The generated summaries and
comparisons are ignored local artifacts; keep only public-safe conclusions in
tracked docs.

High-difficulty-only comparison:

```bash
python scripts/hash_api_compare.py .benchmarks/before.json .benchmarks/after.json --match-by config --min-difficulty 4096 --fail-on-report-quality --min-change-pct 1
```

If two reports used different scenario names for the same backend, device, difficulty, batch size, seconds, warm-up, repeat, key mode, XUNI mode, and detailed-timing mode, compare by configuration instead of by name:

```bash
python scripts/hash_api_compare.py .benchmarks/before.json .benchmarks/after.json --match-by config --fail-on-regression --fail-on-report-quality --min-change-pct 1
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

1. Establish and maintain realistic high-difficulty CUDA baselines under `.benchmarks/` with warm-up and repeated runs, starting at d4096 and extending toward d8192, d16384, and d32768 as memory and run time permit.
2. Preserve d8/b2048, d8/b3072, and d8/b4096 only as historical continuity or fallback scenarios when they answer a concrete comparison question.
3. Run stable high-difficulty auto-batch scans and realistic variable `m=diff` sequences before changing conservative defaults.
4. Tune CUDA first-block launch geometry with one parameter per cycle when device first-block timing is material.
5. Isolate finalization, result ownership, base64, matching, and output materialization before trying new parallel or device-side finalization designs.
6. Measure same-`m` warm loops versus alternating `m=diff` warm loops.
7. Cache difficulty-derived setup only when `m`, salt, key mode, batch shape, backend state, and device state make it provably safe.
8. Reduce per-batch allocations and repeated normalization inside `src/hashapi/CudaHashBackend.cpp`.
9. Revisit transfer overlap, streams, or pinned memory only if profiling shows transfer cost dominates after current GPU-first and finalization work.
10. Extend batch-size and launch-shape tuning toward runtime autotuning after stable cross-difficulty data exists.
11. Add optional autotuning once enough benchmark data justifies it.
12. Add profiler-backed CUDA kernel work when benchmark timing shows compute or launch overhead is a dominant bottleneck.
13. Keep public-safe CUDA resource summaries for accepted or rejected kernel
    experiments so future agents can avoid repeating register-pressure regressions.

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
5. Run focused Hash API tests if validation is stale.
6. Build or reuse the clean Release CUDA binary.
7. Run CUDA golden hash checks with and without `--gpu-first-blocks`.
8. Run a short low-difficulty GPU-first smoke.
9. Refresh or load the current realistic high-difficulty GPU-first baseline and the realistic variable-`m` sequence.
10. Choose one measurable optimization.
11. Validate correctness.
12. Benchmark before and after.
13. Privacy-check the staged diff.
14. Commit if stable.

Recommended first action after this revision:

1. Run the focused Hash API tests.
2. Build or reuse the local CUDA binary.
3. Run golden CUDA hash checks with and without `--gpu-first-blocks`.
4. Run or load repeated d4096 auto-batch GPU-first baseline evidence, then extend to d8192 and d16384 if practical.
5. Run or load a realistic variable-`m` GPU-first sequence such as d4096,d8192,d16384.
6. If no newer evidence exists, test CUDA first-block launch geometry with a single scoped change only when the high-d baseline shows launch overhead is material.
7. Use detailed timing breakdowns to choose between finalization, launch geometry, setup caching, allocation reuse, matching, or variable-`m` lifecycle work.
8. Do not retry rejected pinned host staging, activation caching, salt decode, first-block lane fast paths, fixed-length base64, final-prefix cache, direct final-digest helper, `gpu_final_hashes`, or host-owned parallel finalization snapshots unless the implementation shape has materially changed.
