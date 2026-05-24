# Development Baseline

This document records the local commands used to validate work while extracting the reusable Hash API.

## Python Platform Tests

Install runtime and development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

If the active Python environment cannot read user site-packages, install into the repository-local ignored dependency directory:

```bash
python -m pip install -r requirements-dev.txt --target .deps/python
```

Run the CI-style platform test suite without the real C++ worker integration test:

```bash
python -m pytest tests -q --ignore=tests/integration/test_cpp_worker.py
```

In restricted local environments, use:

```powershell
$env:PYTHONPATH="$PWD\.deps\python"
$env:PYTHONDONTWRITEBYTECODE="1"
python -m pytest tests -q --ignore=tests/integration/test_cpp_worker.py -p no:cacheprovider
```

## Frontend Build

```bash
cd web
npm ci
npm run build
```

## C++ Build

Initialize the vcpkg submodule from a clean clone:

```bash
git submodule update --init --recursive
```

Configure and build:

```bash
cmake -S . -B build --preset ninja-multi-vcpkg
cmake --build build --preset ninja-vcpkg-release
```

Expected local prerequisites:

- CMake 3.18 or newer
- Ninja
- CUDA toolkit
- vcpkg bootstrap network access
- C++17 compiler

The C++ build may fail before compilation if vcpkg cannot download `vcpkg.exe`, Ninja is unavailable, or CUDA compiler discovery is not configured.

Windows full-build example using Visual Studio, Ninja, vcpkg, and CUDA:

```cmd
cmd /V:ON /c "call ""<vs-dev-shell>"" && set ""PATH=<cuda-root>\bin;<ninja-dir>;!PATH!"" && cmake -S . -B <build-dir> -G Ninja -DCMAKE_TOOLCHAIN_FILE=<vcpkg-toolchain> -DVCPKG_INSTALLED_DIR=<vcpkg-installed-dir> -DCMAKE_CUDA_COMPILER=""<cuda-root>/bin/nvcc.exe"" -DCUDAToolkit_ROOT=""<cuda-root>"""
cmd /V:ON /c "call ""<vs-dev-shell>"" && set ""PATH=<cuda-root>\bin;<ninja-dir>;!PATH!"" && cmake --build <build-dir> --config Release"
```

Use a CUDA toolkit version compatible with the selected host compiler.

## Standalone Hash API CLI Build

The Hash API CLI can be configured without enabling the full CUDA miner:

```bash
cmake -S . -B <hashapi-build-dir> --preset hashapi-cli-mingw
cmake --build <hashapi-build-dir> --preset hashapi-cli-mingw
```

This target still needs a C++17 compiler, `libargon2`, and optionally `nlohmann_json`. It does not require CUDA, MQTT, Boost, Crow, OpenSSL, or the marketplace server.

For CLI/JSON smoke testing on machines without `libargon2`, build the deterministic stub backend:

```bash
cmake -S . -B <hashapi-smoke-build-dir> --preset hashapi-cli-smoke-mingw
cmake --build <hashapi-smoke-build-dir> --preset hashapi-cli-smoke-mingw
<hashapi-cli> hash-one --salt aabbccddeeff0011 --key 0000000000000000000000000000000000000000000000000000000000000000 --difficulty 1 --json
```

The stub backend is only for local CLI smoke tests. It is not a mining or correctness backend.

## Real Worker Integration Test

Run only when a CUDA-capable binary exists:

```bash
python -m pytest tests/integration/test_cpp_worker.py -q
```

## Current Hash API Smoke Coverage

The contract-level Python smoke tests live in:

```text
tests/unit/test_hash_api_contract.py
```

They check that the C++ Hash API contract files, validation messages, CLI docs, and boundary documentation remain present. They do not replace a full C++ compile or runtime test.

The broader suite currently covers the standalone local Hash API service as well:

```text
tests/unit/test_hash_api_service.py
```

## Hash API Validation Snapshot

As of the current Hash API extraction commits:

- CPU/reference backend and validation are implemented under `src/hashapi/`.
- CUDA batch backend is exposed through `hashapi::CudaHashBackend`.
- `MineUnit` routes batch mining through `HashApiResult` while keeping lease, devfee, and submission handling outside the Hash API.
- `hash-one`, `hash-batch`, and `hash-benchmark` support JSON output.
- `scripts/hash_api_benchmark.py` emits aggregate benchmark JSON with host and CUDA/NVIDIA probe metadata.
- `server/hash_api/` provides an optional standalone FastAPI service, separate from the marketplace server.

Validated locally:

```powershell
$env:PYTHONPATH="$PWD\.deps\python"
$env:PYTHONDONTWRITEBYTECODE="1"
python -m pytest tests -q --ignore=tests/integration/test_cpp_worker.py -p no:cacheprovider
cmake --build <hashapi-smoke-build-dir> --preset hashapi-cli-smoke-mingw
python scripts/hash_api_benchmark.py --binary <hashapi-cli> --seconds 1
```

Full CUDA build and real worker validation now also pass locally with:

```powershell
$env:PATH="<build-dir>\bin;<cuda-root>\bin;$env:PATH"
<miner-binary> hash-batch --salt aabbccddeeff0011 --backend cuda --batch-size 2 --difficulty 1 --json
python scripts\hash_api_benchmark.py --binary <miner-binary> --scenario name=cuda-smoke,backend=cuda,difficulty=1,batch_size=2,seconds=1,device=0

$env:PYTHONPATH="$PWD\.deps\python"
$env:PYTHONDONTWRITEBYTECODE="1"
$env:MINER_BIN="<miner-binary>"
python -m pytest tests\integration\test_cpp_worker.py -q -p no:cacheprovider
```

The real libargon2 CPU/reference backend requires difficulty at least 8. The deterministic smoke stub still accepts lower values so CLI parsing and benchmark plumbing can be tested cheaply.
