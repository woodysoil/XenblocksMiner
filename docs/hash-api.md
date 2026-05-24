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

`timings` is a machine-readable millisecond breakdown for optimization. Current fields are `validation_ms`, `setup_ms`, `input_ms`, `compute_ms`, `finalize_ms`, and `total_ms`. Unsupported or irrelevant stages are reported as `0.0`.

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
    "compute_ms": 12.0,
    "finalize_ms": 0.0,
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
    "compute_ms": 0.0,
    "finalize_ms": 0.0,
    "total_ms": 0.1
  },
  "hash": "",
  "matches": []
}
```

## Benchmark Runner

`scripts/hash_api_benchmark.py` runs repeatable `hash-benchmark --json` scenarios and emits an aggregate JSON report for optimization agents.

```bash
python scripts/hash_api_benchmark.py --binary build-hashapi-smoke/bin/hashapi-cli.exe --seconds 3
python scripts/hash_api_benchmark.py --binary build/bin/xenblocksMiner.exe --backend cuda --device 0 --seconds 10
python scripts/hash_api_benchmark.py --binary build/bin/xenblocksMiner.exe --scenario name=cuda-small,backend=cuda,difficulty=1024,batch_size=64,seconds=10,device=0
```

The report schema is `xenblocks.hashapi.benchmark.v1`. Each run records the scenario, command, process exit code, host metadata, CUDA/NVIDIA probe output when available, wall-clock duration, a comparable summary, and the parsed Hash API result. Reports also include `recommendations.batch_size_by_difficulty`, which selects the best successful median hashrate per backend, device, and difficulty from the scenarios in that report.

Reusable presets include `smoke`, `warm-short`, `cuda-compare`, and `batch-scan`. Use `batch-scan` before hard-coding batch assumptions on a new GPU; it compares medium and large batch sizes for low difficulties while keeping raw reports under ignored local benchmark directories.

Use `--recommendations-only` when an automation step only needs the selected tuning recommendations on stdout while still optionally writing the full report with `--output`.

For larger GPUs or deeper tuning, use repeated `--scan-difficulty` and `--scan-batch-size` options to generate a custom matrix without editing the script.

## Local Hash Service

The optional local service is a separate FastAPI app under `server/hash_api/`. It is not registered on the marketplace platform server and does not depend on marketplace routers, MQTT, leases, wallets, settlement, or SQLite.

Run it against a Hash API CLI binary:

```bash
python -m server.hash_api.server --binary build-hashapi-smoke/bin/hashapi-cli.exe --host 127.0.0.1 --port 8765
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
- shared match detection and base64 helpers
- miner batch flow consuming `HashApiResult` from the CUDA backend
- JSON serialization
- CLI command dispatcher
- dependency-light CLI smoke backend
- benchmark runner script
- optional standalone local HTTP hash service

Planned:

- broader miner/platform compatibility verification through the Hash API
