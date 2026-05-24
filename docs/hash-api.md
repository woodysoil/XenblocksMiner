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
- `elapsed_ms`
- `hashrate`
- `timings`
- `hash`
- `matches`

`hash` is populated for fixed-key `hash-one` requests.

`timings` is a machine-readable millisecond breakdown for optimization. Current additive stage fields are `validation_ms`, `setup_ms`, `input_ms`, `keygen_ms`, `first_block_ms`, `compute_ms`, `finalize_ms`, and `total_ms`. CUDA results also report nested sub-measurements: `kernel_ms`, `host_to_device_ms`, and `device_to_host_ms` inside `compute_ms`, plus `finalize_hash_ms`, `argon2_finalize_ms`, `base64_ms`, and `match_ms` inside `finalize_ms`. When `--detailed-timings` is enabled, CUDA results also report diagnostic CPU-time counters `first_block_initial_hash_cpu_ms` and `first_block_digest_cpu_ms`; these can exceed `first_block_ms` on parallel first-block preparation because they sum worker-local CPU time, not wall time. The default path leaves those fields at `0.0` to avoid extra hot-path timing overhead. Unsupported or irrelevant stages are reported as `0.0`.

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

## CLI

The CLI commands are intentionally separate from the existing miner run mode. CPU/reference commands work in the standalone CLI build. CUDA commands are available in the full miner build, where `--backend cuda --device <id>` creates a CUDA backend behind the same `IHashBackend` contract.

```bash
xenblocksMiner hash-help
xenblocksMiner hash-one --salt <hex> --key <64-hex> --backend cpu --difficulty 1024 --json
xenblocksMiner hash-batch --salt <hex> --backend cuda --device 0 --prefix <hex> --pattern XEN11 --batch-size 10 --difficulty 1024 --json
xenblocksMiner hash-benchmark --salt <hex> --backend cuda --device 0 --prefix <hex> --seconds 30 --batch-size 10 --difficulty 1024 --json
xenblocksMiner hash-benchmark --salt <hex> --backend cuda --device 0 --seconds 30 --batch-size 512 --difficulty-sequence 1,8,1,8 --json
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
  "elapsed_ms": 12.3,
  "hashrate": 81.3,
  "timings": {
    "validation_ms": 0.1,
    "setup_ms": 0.2,
    "input_ms": 0.0,
    "keygen_ms": 0.0,
    "first_block_ms": 0.0,
    "first_block_initial_hash_cpu_ms": 0.0,
    "first_block_digest_cpu_ms": 0.0,
    "compute_ms": 12.0,
    "kernel_ms": 0.0,
    "host_to_device_ms": 0.0,
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
  "elapsed_ms": 0.0,
  "hashrate": 0.0,
  "timings": {
    "validation_ms": 0.1,
    "setup_ms": 0.0,
    "input_ms": 0.0,
    "keygen_ms": 0.0,
    "first_block_ms": 0.0,
    "first_block_initial_hash_cpu_ms": 0.0,
    "first_block_digest_cpu_ms": 0.0,
    "compute_ms": 0.0,
    "kernel_ms": 0.0,
    "host_to_device_ms": 0.0,
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
python scripts/hash_api_benchmark.py --binary <miner-binary> --scenario name=cuda-small,backend=cuda,difficulty=1024,batch_size=64,seconds=10,device=0
python scripts/hash_api_benchmark.py --binary <miner-binary> --scenario name=cuda-fixed,backend=cuda,difficulty=8,batch_size=1,key=0000000000000000000000000000000000000000000000000000000000000000,seconds=10,device=0
```

The report schema is `xenblocks.hashapi.benchmark.v1`. Each run records the scenario, command, process exit code, host metadata, CUDA/NVIDIA probe output when available, wall-clock duration, a comparable summary, and the parsed Hash API result. Run summaries include median/min/max hashrate, `hashrate_spread_pct`, `ms_per_attempt`, median timing breakdowns, per-attempt timing breakdowns, and `timing_analysis` fields that identify the dominant measured stage. Summaries also report `difficulty_mode`, `difficulty_sequence`, and `difficulty_changes` when a scenario measures variable `m = difficulty` behavior, plus `key_mode` to distinguish generated-key and fixed-key measurements. Reports include `recommendations.batch_size_by_difficulty`, which selects the best stable median hashrate per backend, device, and fixed difficulty from generated-key scenarios in that report. Sequence and fixed-key scenarios are excluded from fixed-difficulty recommendations so alternating-difficulty measurements and fixed-key isolation runs do not distort generated-mining batch-size defaults. If no candidate is stable, recommendations fall back to the best successful median hashrate and mark `selection_reason` as `no_stable_candidate`. Recommendation rows include the spread percentage, dominant timing stage, `selection_reason`, and a `stable` flag based on the report's `stable_spread_pct` threshold. `recommendations.candidates_by_difficulty` keeps the full candidate list with min/max hashrate, spread, and `ms_per_attempt` so tuning agents can inspect noisy alternatives.

Raw benchmark reports are intended for ignored local artifact directories because they may include local binary paths, command lines, hardware probe output, salts, prefixes, and raw run details. Use `--sanitized-output <path>` when a run should also produce a public-safe summary. The sanitized report uses schema `xenblocks.hashapi.benchmark-summary.v1` and keeps only scenario metadata, aggregate summaries, and recommendations while omitting local paths, host and hardware details, commands, raw iterations, salts, prefixes, and raw results.

Reusable presets include `smoke`, `warm-short`, `cuda-compare`, `batch-scan`, `difficulty-sequence`, and `isolation`. Use `batch-scan` before hard-coding batch assumptions on a new GPU; it compares medium and large batch sizes for low difficulties while keeping raw reports under ignored local benchmark directories.

Use `--difficulty-sequence` with `--sequence-batch-size` to measure the cost of `m = difficulty` changes while the benchmark CLI reuses one backend lifecycle. For example, compare a same-difficulty sequence such as `1,1,1,1` against an alternating sequence such as `1,8,1,8` with the same batch size, seconds, warm-up, repeat count, backend, and device. The `difficulty-sequence` preset provides a small reusable matrix for this measurement. Manual `--scenario` entries are comma-separated, so use `difficulty_sequence=1|8|1|8` inside a manual scenario.

Use `key=<64-hex>` inside a manual scenario to benchmark the fixed-key path repeatedly. This is useful for isolating CUDA compute and finalization from generated-key preparation overhead.

Use `--preset isolation` to run a generated-key d8/b2048 scenario next to a fixed-key d8/b1 scenario. This is the quickest standard split between generated-key/first-block preparation and fixed-key CUDA compute/finalization behavior.

Use `--recommendations-only` when an automation step only needs the selected tuning recommendations on stdout while still optionally writing the full report with `--output`.

For larger GPUs or deeper tuning, use repeated `--scan-difficulty` and `--scan-batch-size` options to generate a custom matrix without editing the script.

Use `scripts/hash_api_compare.py` for before/after reports. It compares median hashrate, reports total timing and per-attempt timing deltas, preserves variable-difficulty metadata, and marks improved, regressed, and unchanged scenarios as noisy when either run's spread exceeds the configured threshold.

Use `--no-xuni` with `scripts/hash_api_benchmark.py` when benchmarking the normal main-target path without secondary XUNI matching.

## CUDA Batch Tuning

`src/hashapi/HashApiTuning.*` contains conservative batch-size helpers shared by miner integration and future autotuning work. The helper separates memory-limited safety from benchmark-informed defaults:

- explicit miner `--batchSize` values remain an upper limit and are not overridden by tuning defaults
- no explicit limit uses benchmark-informed defaults only for difficulty ranges with stable local evidence
- unsupported difficulty ranges fall back to the memory-limited batch size

Current conservative defaults are `512` attempts through difficulty `1`, `2048` attempts through difficulty `8`, and `512` attempts through difficulty `64`. Treat these as starting points for future autotuning, not universal hardware limits.

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
