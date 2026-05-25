# Hash API Contract

The Hash API is the reusable hashing boundary for XenblocksMiner. It is designed to be usable by the miner, local benchmark tools, future services, and optimization agents without depending on marketplace, wallet, MQTT, settlement, or frontend code.

## Scope

The Hash API owns:

- request and result models
- input validation
- salt and hex normalization
- key prefix handling
- CPU/reference hashing
- CUDA batch hashing once implemented
- CLI and benchmark output

The Hash API does not own:

- leases
- provider or renter state
- wallet authentication
- marketplace pricing
- settlement
- MQTT transport
- React dashboard state

## C++ Types

The initial C++ contract lives in `src/hashapi/`.

Core files:

- `HashApiTypes.h`
- `HashApiValidation.h`
- `CpuHashBackend.h`
- `CudaHashBackend.h`
- `HashApiJson.h`
- `HashApiCli.h`

### Request

`HashApiRequest` fields:

- `request_id`: optional caller-provided correlation ID.
- `algorithm`: currently `argon2id-xen`.
- `backend`: `cpu`, `reference`, or `cuda`.
- `salt_hex`: even-length hex salt, with optional `0x` prefix accepted.
- `key`: optional fixed 64-hex key for `hash-one`.
- `key_prefix`: optional hex prefix for generated keys.
- `target_pattern`: output substring to search for, default `XEN11`.
- `difficulty`: Argon2 memory cost / mining difficulty. Must be greater than zero. The libargon2 CPU/reference backend requires at least 8 because lower memory costs are rejected by libargon2.
- `batch_size`: number of generated-key attempts for batch paths.
- `device_id`: non-negative device identifier.
- `allow_xuni`: enables secondary `XUNI\d` match detection.
- `first_block_workers`: optional CUDA first-block worker-thread cap. `0` keeps automatic worker-count behavior.
- `first_block_dynamic_chunk_size`: optional CUDA first-block dynamic scheduling chunk size. `0` keeps the default static chunking behavior; nonzero values are for benchmark-only scheduler experiments.
- `first_block_dynamic_chunk_auto`: optional CUDA first-block dynamic scheduling policy. `false` preserves explicit static or manual chunk behavior; `true` lets the CUDA backend choose a benchmark-informed dynamic chunk size for supported generated-key scenarios.
- `gpu_first_blocks`: optional CUDA-only experiment flag. `false` preserves the default host-prepared first blocks. `true` asks the CUDA backend to generate the Argon2 initial two blocks on the device for supported `t=1`, single-lane generated-key or fixed-key requests.

### Result

`HashApiResult` fields:

- `request_id`
- `ok`
- `error`
- `algorithm`
- `backend`
- `device_id`
- `batch_size`
- `attempts`
- `first_block_dynamic_chunk_size`
- `first_block_dynamic_chunk_auto`
- `first_block_worker_count`
- `first_block_chunk_size`
- `first_block_dynamic_chunk_size_min`
- `first_block_dynamic_chunk_size_max`
- `first_block_chunk_size_min`
- `first_block_chunk_size_max`
- `gpu_first_blocks`
- `elapsed_ms`
- `hashrate`
- `timings`
- `hash`
- `matches`

`hash` is populated for fixed-key `hash-one` requests.

`first_block_worker_count`, `first_block_chunk_size`, `first_block_dynamic_chunk_size`, and `first_block_dynamic_chunk_auto` describe the CUDA first-block scheduling shape selected for the request, including automatic worker selection when `first_block_workers` is `0`. Static first-block chunking remains the default when `first_block_dynamic_chunk_size` is `0` and auto policy is disabled. Nonzero dynamic chunk sizes are an explicit tuning surface for measuring worker start skew and load-balance effects, while auto policy is an opt-in way to benchmark conservative backend-selected chunks without changing forced-static requests. The `_min` and `_max` first-block chunk fields expose the selected scheduling range across aggregated benchmark loops; fixed-difficulty runs report identical min/max values, while variable-difficulty runs can reveal mixed dynamic and static policy selections. `gpu_first_blocks` records whether the request used the explicit device-side first-block path. That path is CUDA-only, experimental, and opt-in so existing miner behavior and CPU/reference behavior stay unchanged. `timings` is a machine-readable millisecond breakdown for optimization. Current additive stage fields are `validation_ms`, `setup_ms`, `input_ms`, `keygen_ms`, `first_block_ms`, `compute_ms`, `finalize_ms`, and `total_ms`. CUDA results also report nested sub-measurements: `kernel_ms`, `host_to_device_ms`, `gpu_first_block_ms`, and `device_to_host_ms` inside `compute_ms`, plus `finalize_hash_ms`, `argon2_finalize_ms`, `base64_ms`, and `match_ms` inside `finalize_ms`. `gpu_first_block_ms` is nonzero only for the explicit device-side first-block path and separates that device kernel from host-to-device transfer timing. When `--detailed-timings` is enabled, CUDA results also report diagnostic setup counters `setup_normalize_cpu_ms`, `setup_activate_cpu_ms`, `setup_device_info_cpu_ms`, `setup_params_cpu_ms`, and `setup_backend_init_cpu_ms`, plus first-block diagnostics `first_block_initial_hash_cpu_ms`, `first_block_digest_cpu_ms`, `first_block_max_worker_ms`, `first_block_thread_launch_ms`, `first_block_max_worker_start_ms`, `first_block_worker_start_span_ms`, `first_block_max_worker_finish_ms`, and `first_block_worker_finish_span_ms`. First-block CPU-time counters can exceed `first_block_ms` on parallel first-block preparation because they sum worker-local CPU time, not wall time; `first_block_max_worker_ms` is the slowest worker-local wall time observed for the batch, while the worker start and finish fields help separate thread launch latency, worker-loop work, and post-worker join overhead. The default path leaves detailed fields at `0.0` to avoid extra hot-path timing overhead. Unsupported or irrelevant stages are reported as `0.0`.

Each match includes:

- `key`
- `hash`
- `matched_pattern`
- `attempt_index`
- `is_superblock`

## Validation Rules

`validateRequest()` enforces:

- supported algorithm: `argon2id-xen`
- supported backend: `cpu`, `reference`, `cuda`
- `salt_hex` required, even-length, hex-only, at least 16 hex characters
- `key_prefix` hex-only and at most 64 hex characters
- `key` must be exactly 64 hex characters when provided
- `key` must start with `key_prefix` when both are provided
- `target_pattern` required and at most 128 characters
- `difficulty` greater than zero
- `batch_size` greater than zero
- CPU/reference `batch_size` no greater than 10000
- `device_id` non-negative
- `gpu_first_blocks` requires `backend=cuda`

## CLI

The CLI commands are intentionally separate from the existing miner run mode. CPU/reference commands work in the standalone CLI build. CUDA commands are available in the full miner build, where `--backend cuda --device <id>` creates a CUDA backend behind the same `IHashBackend` contract.

```bash
xenblocksMiner hash-help
xenblocksMiner hash-one --salt <hex> --key <64-hex> --backend cpu --difficulty 1024 --json
xenblocksMiner hash-batch --salt <hex> --backend cuda --device 0 --prefix <hex> --pattern XEN11 --batch-size 10 --difficulty 1024 --json
xenblocksMiner hash-benchmark --salt <hex> --backend cuda --device 0 --prefix <hex> --seconds 30 --batch-size 10 --difficulty 1024 --json
xenblocksMiner hash-benchmark --salt <hex> --backend cuda --device 0 --seconds 30 --batch-size 512 --difficulty-sequence 1,8,1,8 --json
xenblocksMiner hash-benchmark --salt <hex> --backend cuda --device 0 --seconds 30 --difficulty-sequence 1,8,64 --batch-size-sequence 2048,3072,3072 --json
xenblocksMiner hash-benchmark --salt <hex> --backend cuda --device 0 --seconds 30 --batch-size 2048 --difficulty 8 --first-block-workers 4 --json
xenblocksMiner hash-benchmark --salt <hex> --backend cuda --device 0 --seconds 30 --batch-size 2048 --difficulty 8 --first-block-dynamic-chunk-size 64 --json
```

The standalone CLI target uses the same commands through `hashapi-cli`:

```bash
hashapi-cli hash-one --salt <hex> --key <64-hex> --difficulty 1024 --json
```

For dependency-light CLI smoke tests, the build system also provides `XENBLOCKS_HASHAPI_STUB_BACKEND=ON`. This deterministic stub backend verifies CLI parsing, validation, and JSON output, but it is not a mining backend and must not be used for correctness or performance measurements.

Use low difficulty values for local CPU smoke tests. Real mining difficulty can be much more expensive on CPU.

## JSON Output

All Hash API CLI commands support `--json`.

Example success shape:

```json
{
  "request_id": "",
  "ok": true,
  "error": "",
  "algorithm": "argon2id-xen",
  "backend": "cpu",
  "device_id": 0,
  "batch_size": 1,
  "attempts": 1,
  "first_block_dynamic_chunk_size": 0,
  "first_block_dynamic_chunk_auto": false,
  "first_block_worker_count": 0,
  "first_block_chunk_size": 0,
  "elapsed_ms": 12.3,
  "hashrate": 81.3,
  "timings": {
    "validation_ms": 0.1,
    "setup_ms": 0.2,
    "setup_normalize_cpu_ms": 0.0,
    "setup_activate_cpu_ms": 0.0,
    "setup_device_info_cpu_ms": 0.0,
    "setup_params_cpu_ms": 0.0,
    "setup_backend_init_cpu_ms": 0.0,
    "input_ms": 0.0,
    "keygen_ms": 0.0,
    "first_block_ms": 0.0,
    "first_block_initial_hash_cpu_ms": 0.0,
    "first_block_digest_cpu_ms": 0.0,
    "first_block_max_worker_ms": 0.0,
    "first_block_thread_launch_ms": 0.0,
    "first_block_max_worker_start_ms": 0.0,
    "first_block_worker_start_span_ms": 0.0,
    "first_block_max_worker_finish_ms": 0.0,
    "first_block_worker_finish_span_ms": 0.0,
    "compute_ms": 12.0,
    "kernel_ms": 0.0,
    "host_to_device_ms": 0.0,
    "gpu_first_block_ms": 0.0,
    "device_to_host_ms": 0.0,
    "finalize_ms": 0.0,
    "finalize_hash_ms": 0.0,
    "argon2_finalize_ms": 0.0,
    "base64_ms": 0.0,
    "match_ms": 0.0,
    "total_ms": 12.4
  },
  "hash": "$argon2id$...",
  "matches": []
}
```

Example failure shape:

```json
{
  "request_id": "",
  "ok": false,
  "error": "salt_hex is required",
  "algorithm": "argon2id-xen",
  "backend": "cpu",
  "device_id": 0,
  "batch_size": 1,
  "attempts": 0,
  "first_block_dynamic_chunk_size": 0,
  "first_block_dynamic_chunk_auto": false,
  "first_block_worker_count": 0,
  "first_block_chunk_size": 0,
  "elapsed_ms": 0.0,
  "hashrate": 0.0,
  "timings": {
    "validation_ms": 0.1,
    "setup_ms": 0.0,
    "setup_normalize_cpu_ms": 0.0,
    "setup_activate_cpu_ms": 0.0,
    "setup_device_info_cpu_ms": 0.0,
    "setup_params_cpu_ms": 0.0,
    "setup_backend_init_cpu_ms": 0.0,
    "input_ms": 0.0,
    "keygen_ms": 0.0,
    "first_block_ms": 0.0,
    "first_block_initial_hash_cpu_ms": 0.0,
    "first_block_digest_cpu_ms": 0.0,
    "first_block_max_worker_ms": 0.0,
    "first_block_thread_launch_ms": 0.0,
    "first_block_max_worker_start_ms": 0.0,
    "first_block_worker_start_span_ms": 0.0,
    "first_block_max_worker_finish_ms": 0.0,
    "first_block_worker_finish_span_ms": 0.0,
    "compute_ms": 0.0,
    "kernel_ms": 0.0,
    "host_to_device_ms": 0.0,
    "gpu_first_block_ms": 0.0,
    "device_to_host_ms": 0.0,
    "finalize_ms": 0.0,
    "finalize_hash_ms": 0.0,
    "argon2_finalize_ms": 0.0,
    "base64_ms": 0.0,
    "match_ms": 0.0,
    "total_ms": 0.1
  },
  "hash": "",
  "matches": []
}
```

## Benchmark Runner

`scripts/hash_api_benchmark.py` runs repeatable `hash-benchmark --json` scenarios and emits an aggregate JSON report for optimization agents.

```bash
python scripts/hash_api_benchmark.py --binary <hashapi-cli> --seconds 3
python scripts/hash_api_benchmark.py --binary <miner-binary> --backend cuda --device 0 --seconds 10
python scripts/hash_api_benchmark.py --binary <miner-binary> --build-cache <build-dir> --backend cuda --device 0 --seconds 10 --sanitized-output .benchmarks/cuda-summary.json
python scripts/hash_api_benchmark.py --binary <miner-binary> --scenario name=cuda-small,backend=cuda,difficulty=1024,batch_size=64,seconds=10,device=0
python scripts/hash_api_benchmark.py --binary <miner-binary> --scenario name=cuda-fixed,backend=cuda,difficulty=8,batch_size=1,key=0000000000000000000000000000000000000000000000000000000000000000,seconds=10,device=0
python scripts/hash_api_benchmark.py --binary <miner-binary> --backend cuda --device 0 --difficulty-sequence 1,8,64 --sequence-auto-batch-size --sequence-first-block-dynamic-chunk-auto --gpu-first-blocks --seconds 8 --warmup 1 --repeat 3
python scripts/hash_api_benchmark.py --binary <miner-binary> --backend cuda --device 0 --scan-difficulty 8 --scan-batch-size 2048 --scan-batch-size 3072 --scan-gpu-first-blocks --seconds 4 --warmup 1 --repeat 3
```

The report schema is `xenblocks.hashapi.benchmark.v1`. Each run records the scenario, command, process exit code, host metadata, CUDA/NVIDIA probe output when available, public-safe environment metadata, optional CMake build metadata from `--build-cache`, wall-clock duration, a comparable summary, and the parsed Hash API result. Build metadata intentionally keeps only public-safe fields such as build type, generator, CUDA architecture list, vcpkg triplet, CUDA compiler basename, and CUDA compiler version; it omits compiler and build-directory paths. Environment metadata samples aggregate CPU load before and after each benchmark subprocess, reports the maximum as `cpu_load_pct`, keeps `start_cpu_load_pct`, `end_cpu_load_pct`, and `sample_count`, and marks `benchmark_trust` as `low` when any sample is highly loaded, so automation can avoid accepting distorted CPU-side timing results. Run summaries include median/min/max hashrate, `hashrate_spread_pct`, `ms_per_attempt`, median timing breakdowns, per-attempt timing breakdowns, and `timing_analysis` fields that identify the dominant measured stage. `timing_analysis.stage_pct` reports top-level stage percentages of `total_ms`; `timing_analysis.nested_stage_pct` reports nested diagnostic fields as percentages of their parent stage, such as transfer timings relative to `compute_ms` and first-block detail timings relative to `first_block_ms`. Nested percentages can exceed 100% when the underlying diagnostic counter sums worker-local CPU time. `timing_analysis.first_block_cpu_sum_ms` and `timing_analysis.first_block_cpu_sum_to_wall` report the first-block detailed CPU-time sum and its ratio to first-block wall time, while `first_block_max_worker_ms`, `timing_analysis.first_block_worker_wall_to_wall`, and `timing_analysis.first_block_scheduling_overhead_ms` report the slowest worker-local wall time and the remaining first-block wall time outside that worker. `timing_analysis.first_block_finish_wall_to_wall` and `timing_analysis.first_block_post_worker_overhead_ms` compare the latest worker finish offset with first-block wall time, which helps separate last-worker completion from post-worker join/accounting overhead. These fields help distinguish digest work, parallelism, and scheduling overhead during parallel first-block preparation. Summaries also report `difficulty_mode`, `difficulty_sequence`, and `difficulty_changes` when a scenario measures variable `m = difficulty` behavior, plus `batch_size_mode`, `batch_size_sequence`, `batch_size_changes`, `batch_size_min`, and `batch_size_max` when a scenario measures variable batch sizes in the same backend lifecycle. `key_mode` distinguishes generated-key and fixed-key measurements. Reports include `recommendations.batch_size_by_difficulty`, which selects the best stable median hashrate per backend, device, and fixed difficulty from generated-key scenarios in that report. Sequence and fixed-key scenarios are excluded from fixed-difficulty recommendations so alternating-difficulty, variable-batch, and fixed-key isolation runs do not distort generated-mining batch-size defaults. If no candidate is stable, recommendations fall back to the best successful median hashrate and mark `selection_reason` as `no_stable_candidate`. Recommendation rows include requested first-block worker cap, selected first-block worker count, first-block chunk size, batch-size min/max, spread percentage, dominant timing stage, `selection_reason`, and a `stable` flag based on the report's `stable_spread_pct` threshold. `recommendations.candidates_by_difficulty` keeps the full candidate list with min/max hashrate, spread, first-block schedule, and `ms_per_attempt` so tuning agents can inspect noisy alternatives. Recommendation output also includes `report_ok`, run counts, and `invalid_scenarios`; do not use a tuning recommendation as a default change when the report is partial or invalid.

Raw benchmark reports are intended for ignored local artifact directories because they may include local binary paths, command lines, hardware probe output, salts, prefixes, and raw run details. Use `--sanitized-output <path>` when a run should also produce a public-safe summary. The sanitized report uses schema `xenblocks.hashapi.benchmark-summary.v1` and keeps only scenario metadata, aggregate summaries, and recommendations while omitting local paths, host and hardware details, commands, raw iterations, salts, prefixes, and raw results.

Reusable presets include `smoke`, `warm-short`, `cuda-compare`, `batch-scan`, `difficulty-sequence`, and `isolation`. Use `batch-scan` before hard-coding batch assumptions on a new GPU; it compares medium and large batch sizes for low difficulties while keeping raw reports under ignored local benchmark directories.

Use `--difficulty-sequence` with `--sequence-batch-size` to measure the cost of `m = difficulty` changes while the benchmark CLI reuses one backend lifecycle. Use `--difficulty-sequence` with `--sequence-auto-batch-size` to let the CUDA Hash API tuning helper select one fixed batch size for the whole difficulty sequence. Use `--difficulty-sequence` with `--batch-size-sequence` when the benchmark should pair each difficulty shape with its miner-equivalent batch size, such as `1,8,64` with `2048,3072,3072`. The two sequences must have the same length unless one side has length `1`. Add `--sequence-detailed-timings` when generated sequence scenarios should include detailed CUDA setup and first-block diagnostics, add `--sequence-first-block-dynamic-chunk-auto` when sequence scenarios should use the backend-selected first-block dynamic chunk policy, and add `--gpu-first-blocks` when generated scenarios should use the explicit CUDA device-side first-block path. For example, compare a same-difficulty sequence such as `1,1,1,1` against an alternating sequence such as `1,8,1,8` with the same batch size, seconds, warm-up, repeat count, backend, and device. The `difficulty-sequence` preset provides a small reusable matrix for this measurement. Manual `--scenario` entries are comma-separated, so use `difficulty_sequence=1|8|1|8` and `batch_size_sequence=2048|3072|3072` inside a manual scenario.

Use `key=<64-hex>` inside a manual scenario to benchmark the fixed-key path repeatedly. This is useful for isolating CUDA compute and finalization from generated-key preparation overhead.

Use `--preset isolation` to run a generated-key d8/b2048 scenario next to a fixed-key d8/b1 scenario. This is the quickest standard split between generated-key/first-block preparation and fixed-key CUDA compute/finalization behavior.

Use `--recommendations-only` when an automation step only needs the selected tuning recommendations on stdout while still optionally writing the full report with `--output`.

For larger GPUs or deeper tuning, use repeated `--scan-difficulty`, `--scan-batch-size`, `--scan-first-block-workers`, and `--scan-first-block-dynamic-chunk-size` options to generate a custom matrix without editing the script. Add `--scan-first-block-dynamic-chunk-auto` when the scan should benchmark the backend-selected dynamic chunk policy, add `--scan-gpu-first-blocks` when the scan should emit both default and explicit GPU first-block variants, and add `--scan-detailed-timings` when the generated scan should include detailed CUDA setup and first-block diagnostic counters. Add `first_block_dynamic_chunk_auto=true` or `gpu_first_blocks=true` in a manual `--scenario` for one-off backend-selected dynamic chunk or GPU first-block scenarios.

Use `scripts/hash_api_compare.py` for before/after reports. It compares median hashrate, reports total timing, per-attempt timing, top-level stage-percentage deltas, and nested stage-percentage deltas, preserves variable-difficulty metadata, variable-batch metadata, first-block worker cap, first-block dynamic chunk size, first-block dynamic chunk auto policy, selected worker count, and first-block chunk size, and marks improved, regressed, and unchanged scenarios as noisy when either run's spread exceeds the configured threshold. Comparison output includes report-quality metadata from `report_ok`, invalid run counts, and `benchmark_trust`; add `--fail-on-report-quality` when automation should reject partial or low-trust reports before accepting a performance conclusion. By default, comparison matches runs by scenario name. Use `--match-by config` when two reports describe the same comparable benchmark settings but use different scenario names. Config matching includes backend, device, difficulty mode and sequence, key mode, batch size mode and sequence, seconds, warm-up, repeat, XUNI mode, detailed-timing mode, first-block worker cap, first-block dynamic chunk size, and first-block dynamic chunk auto policy. Add `--ignore-detailed-timings` with config matching only when comparing default-timing and detailed-timing reports for the same scenario.

Use `--no-xuni` with `scripts/hash_api_benchmark.py` when benchmarking the normal main-target path without secondary XUNI matching.

## CUDA Batch Tuning

`src/hashapi/HashApiTuning.*` contains conservative batch-size helpers shared by miner integration and future autotuning work. The helper separates memory-limited safety from benchmark-informed defaults:

- explicit miner `--batchSize` values remain an upper limit and are not overridden by tuning defaults
- no explicit limit uses benchmark-informed defaults only for difficulty ranges with stable local evidence
- unsupported difficulty ranges fall back to the memory-limited batch size

Current conservative defaults are `2048` attempts through difficulty `1`, `3072` attempts through difficulty `8`, and `3072` attempts through difficulty `64`. Treat these as starting points for future autotuning, not universal hardware limits.

For variable-`m` sequences, prefer a fixed batch size selected for the whole difficulty set instead of pairing each difficulty with its single-difficulty default. The sequence helper uses the most restrictive tuned default across the requested difficulties and applies the memory limit for the maximum difficulty. The C++ CLI exposes this through `--auto-batch-size`, and the benchmark wrapper exposes generated sequence scenarios through `--sequence-auto-batch-size`. Current local evidence for `difficulty_sequence=1,8,64` favors fixed `2048` over `2048,3072,3072` because it avoids repeated backend setup churn.

## Local Hash Service

The optional local service is a separate FastAPI app under `server/hash_api/`. It is not registered on the marketplace platform server and does not depend on marketplace routers, MQTT, leases, wallets, settlement, or SQLite.

Run it against a Hash API CLI binary:

```bash
python -m server.hash_api.server --binary <hashapi-cli> --host 127.0.0.1 --port 8765
```

Endpoints:

- `GET /hash/v1/health`
- `GET /hash/v1/backends`
- `POST /hash/v1/validate`
- `POST /hash/v1/hash-one`
- `POST /hash/v1/batch`
- `POST /hash/v1/benchmark`

The service validates requests before spawning the CLI, runs commands with a configurable timeout, and limits subprocess concurrency. Use `XENBLOCKS_HASH_API_BINARY`, `XENBLOCKS_HASH_API_TIMEOUT`, and `XENBLOCKS_HASH_API_CONCURRENCY` when constructing the app from environment variables.

## Implementation Status

Implemented:

- C++ request/result structs
- validation helpers
- CPU/reference backend using existing `Argon2idHasher`
- CUDA backend adapter using `ComputeBackend`
- conservative CUDA batch-size tuning helpers
- shared match detection and base64 helpers
- miner batch flow consuming `HashApiResult` from the CUDA backend
- JSON serialization
- CLI command dispatcher
- dependency-light CLI smoke backend
- benchmark runner script
- optional standalone local HTTP hash service

Planned:

- broader miner/platform compatibility verification through the Hash API
