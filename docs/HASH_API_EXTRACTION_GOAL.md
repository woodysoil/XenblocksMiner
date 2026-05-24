# Long-Running Goal: Extract a Reusable Hash API

## Mission

Extract the XenblocksMiner hashing and mining core into a clean, reusable Hash API before expanding the marketplace, platform, wallet, or frontend features.

This goal is intended for Codex `/goal` long-running execution. Treat it as the persistent operating brief. Continue iterating through the phases below until the Definition of Done is met or a real blocker requires user input.

## Operating Rules For Codex

Work in English for code, comments, docs, tests, commit messages, branch names, and API names.

Stay focused on Hash API extraction. Do not drift into frontend polish, marketplace economics, wallet UX, or broad platform redesign unless a phase explicitly requires a narrow integration change.

Use small, coherent commits. Commit whenever a meaningful step is complete and validated. Prefer many stable commits over one large unreviewable change.

Before each work cycle:

1. Run `git status -sb`.
2. Read this file if context was compacted or resumed.
3. Identify the current phase and the next smallest valuable step.
4. Inspect nearby code before editing.
5. Make scoped changes only.
6. Run the relevant validation commands.
7. Commit if the repo is in a stable state.
8. Update docs/tests when behavior or contracts change.

Never revert user changes unless explicitly instructed. If unrelated files are dirty, leave them alone. If dirty files block the current phase, stop and explain the conflict.

## Current Project Shape

The repository has three major surfaces:

- `src/`: C++17/CUDA miner, current hashing and worker logic.
- `server/`: Python FastAPI mock platform, MQTT broker, SQLite storage, marketplace, auth, settlement, and monitoring.
- `web/`: React/Vite dashboard.

The existing hash/mining logic is coupled to runtime globals, device selection, callbacks, platform mode, and submission/reporting flows. The long-term task is to separate reusable hashing primitives from those concerns.

## Target Outcome

The final architecture should provide:

- a narrow C++ Hash API contract
- a CPU/reference backend usable without CUDA
- a CUDA backend behind the same interface
- structured request/result models
- validation helpers for salt, prefix, difficulty, pattern, and batch size
- JSON-capable CLI commands for automation
- optional local HTTP/service access that is separate from the marketplace API
- benchmark output suitable for AI-driven optimization loops
- compatibility with the existing miner and platform mode

The Hash API must be usable without starting the marketplace platform.

## Hard Boundaries

The Hash API must not depend on:

- FastAPI routers
- MQTT topics
- leases
- marketplace pricing
- provider/renter roles
- wallets
- JWT or API keys
- SQLite schemas
- React dashboard state
- settlement logic

The Hash API may depend on:

- algorithm parameters
- difficulty
- salt/address normalization
- key generation policy
- prefix validation
- batch sizing
- backend/device selection
- output matching
- benchmark metrics
- cancellation/progress hooks

## Suggested Module Boundaries

Prefer adding new code in focused modules before migrating existing callers.

Recommended paths:

- `src/hashapi/` for C++ API contracts and implementations.
- `docs/hash-api.md` for the public contract and examples.
- `tests/` additions for validation behavior and CLI/service behavior.
- `scripts/` for benchmark helpers if needed.

Only add `server/hash_api/` or HTTP endpoints after the C++ contract and CLI are stable.

## API Contract Direction

Use structured inputs and outputs. Do not require consumers to parse console output.

Core request fields should cover:

- `request_id`
- `algorithm`
- `salt_hex`
- `key_prefix`
- `target_pattern`
- `difficulty`
- `batch_size`
- `device_id`
- `backend`
- `allow_xuni`

Core result fields should cover:

- `request_id`
- `ok`
- `error`
- `backend`
- `device_id`
- `batch_size`
- `attempts`
- `elapsed_ms`
- `hashrate`
- `matches`

Each match should include:

- `key`
- `hash`
- `matched_pattern`
- `attempt_index`
- `is_superblock`
- optional implementation metadata

Initial C++ interface shape:

```cpp
struct HashApiRequest {
    std::string request_id;
    std::string algorithm = "argon2id-xen";
    std::string salt_hex;
    std::string key_prefix;
    std::string target_pattern = "XEN11";
    uint32_t difficulty = 42069;
    size_t batch_size = 0;
    int device_id = 0;
    bool allow_xuni = true;
};

struct HashApiMatch {
    std::string key;
    std::string hash;
    std::string matched_pattern;
    size_t attempt_index = 0;
    bool is_superblock = false;
};

struct HashApiResult {
    std::string request_id;
    bool ok = false;
    std::string error;
    std::string backend;
    int device_id = 0;
    size_t batch_size = 0;
    size_t attempts = 0;
    double elapsed_ms = 0;
    double hashrate = 0;
    std::vector<HashApiMatch> matches;
};

class IHashBackend {
public:
    virtual ~IHashBackend() = default;
    virtual HashApiResult runBatch(const HashApiRequest& request) = 0;
};
```

Exact names may change if the codebase suggests better local names. Preserve the capabilities.

## Phase Plan

### Phase 0: Baseline And Hygiene

Goal: make repeated AI iterations reliable.

Tasks:

- Add or update a dev/test dependency file for test-only Python packages such as `pytest`, `pytest-asyncio`, and `httpx`.
- Document baseline commands for Python, frontend, and C++.
- Confirm the `vcpkg` submodule setup is documented and reproducible.
- Keep existing platform behavior unchanged.

Validation:

```bash
python -m pytest tests -q --ignore=tests/integration/test_cpp_worker.py
cd web && npm run build
```

Commit examples:

```text
docs: document hash api baseline
test: add python development requirements
```

### Phase 1: Define Hash API Types And Validation

Goal: establish the stable contract before moving implementation.

Tasks:

- Add C++ request/result structs.
- Add a backend interface.
- Add validation helpers for salt, key prefix, target pattern, difficulty, batch size, and backend selection.
- Add focused tests for validation and result shaping.
- Add `docs/hash-api.md` with the public contract.

Constraints:

- Do not change miner runtime behavior yet.
- Do not introduce platform or frontend dependencies.

Validation:

```bash
python -m pytest tests -q --ignore=tests/integration/test_cpp_worker.py
```

Also run C++ configure/build when the local environment can support it.

Commit examples:

```text
feat(hash-api): define request and result types
test(hash-api): cover request validation
docs: add hash api contract
```

### Phase 2: Add CPU Reference Backend

Goal: provide a deterministic implementation that works without CUDA.

Tasks:

- Wrap existing Argon2id behavior in a CPU/reference backend.
- Add a `hash-one` path that does not require a GPU.
- Add golden tests for fixed salt, key, and difficulty.
- Return structured errors for invalid input.

Constraints:

- CPU reference correctness is more important than speed.
- Keep the backend independent from MQTT, FastAPI, and global miner state.

Validation:

```bash
python -m pytest tests -q --ignore=tests/integration/test_cpp_worker.py
```

Run any added C++/CLI tests if available.

Commit examples:

```text
feat(hash-api): add cpu reference backend
test(hash-api): add golden hash cases
```

### Phase 3: Adapt CUDA Batch Backend

Goal: expose the existing high-performance CUDA path through the same Hash API.

Tasks:

- Wrap or move the batch logic currently centered around `MineUnit::batchCompute`.
- Keep `CudaBackend`, `KernelRunner`, `Argon2Params`, and device handling as implementation details.
- Return `HashApiResult` instead of relying only on callbacks and globals.
- Preserve existing self-mining behavior by making current miner flow call the new API.

Constraints:

- Keep compatibility first. Do not rewrite CUDA kernels unless required.
- Avoid broad global-state cleanup unless it directly blocks the interface.

Validation:

```bash
cmake -S . -B build --preset ninja-multi-vcpkg
cmake --build build --preset ninja-vcpkg-release
python -m pytest tests -q --ignore=tests/integration/test_cpp_worker.py
```

If a CUDA binary exists:

```bash
python -m pytest tests/integration/test_cpp_worker.py -q
```

Commit examples:

```text
feat(hash-api): add cuda batch backend
refactor(miner): route batch mining through hash api
```

### Phase 4: Add CLI Automation Surface

Goal: let agents and external tools use the Hash API without linking directly.

Tasks:

- Add commands equivalent to:
  - `hash-one`
  - `hash-batch`
  - `hash-benchmark`
- Add `--json` output.
- Add clear exit codes.
- Keep existing miner CLI behavior backward compatible.
- Add docs and examples.

Validation:

Run CLI smoke tests for success and failure cases. Confirm JSON parses.

Commit examples:

```text
feat(cli): expose hash api commands
test(cli): cover hash command validation
```

### Phase 5: Optional Local Hash Service

Goal: allow other programs to use hashing through a local service.

Tasks:

- Decide whether the service should be C++, Python, or a separate process.
- Keep it separate from marketplace endpoints.
- Add narrow endpoints:
  - `GET /hash/v1/health`
  - `GET /hash/v1/backends`
  - `POST /hash/v1/validate`
  - `POST /hash/v1/hash-one`
  - `POST /hash/v1/batch`
  - `POST /hash/v1/benchmark`
- Add concurrency limits, timeouts, and structured errors.

Validation:

Run HTTP smoke tests. Confirm invalid and long-running requests fail cleanly.

Commit examples:

```text
feat(hash-api): add local service endpoints
test(hash-api): cover local service errors
```

### Phase 6: Reconnect Platform Mode

Goal: keep the marketplace worker flow working while using the cleaner Hash API.

Tasks:

- Refactor `PlatformManager`, `MiningCoordinator`, and `MineUnit` only as needed.
- Keep lease assignment and block reporting outside the Hash API.
- Keep MQTT protocol stable unless a versioned change is explicitly required.
- Improve telemetry by using structured hash result metrics.

Validation:

```bash
python -m pytest tests -q --ignore=tests/integration/test_cpp_worker.py
```

When possible, run the real C++ worker integration test.

Commit examples:

```text
refactor(platform): integrate worker mining with hash api
test(platform): preserve worker lease flow
```

### Phase 7: Benchmark And Optimization Loop

Goal: prepare for continuous AI-driven performance work.

Tasks:

- Add reproducible benchmark scenarios.
- Emit stable JSON benchmark output.
- Record hardware, CUDA version, driver version, backend, difficulty, batch size, memory use, elapsed time, and hashrate.
- Add a benchmark runner script.
- Add regression thresholds only when results are stable enough.
- Keep optimization commits small and include before/after numbers.

Validation:

Benchmark output must be machine-readable and comparable across commits.

Commit examples:

```text
perf(hash-api): add benchmark runner
perf(hash-api): optimize cuda batch setup
```

## Validation Commands

Use the narrowest command that proves the current change. Broaden validation before commits that affect shared behavior.

Python platform tests:

```bash
python -m pytest tests -q --ignore=tests/integration/test_cpp_worker.py
```

Frontend build:

```bash
cd web
npm run build
```

C++ build:

```bash
cmake -S . -B build --preset ninja-multi-vcpkg
cmake --build build --preset ninja-vcpkg-release
```

Real worker integration test:

```bash
python -m pytest tests/integration/test_cpp_worker.py -q
```

If a command cannot run because of local environment limitations, record the reason in the final response and prefer adding tests or docs that can run locally.

## Commit Discipline

Use English commit messages.

Recommended prefixes:

- `docs:`
- `test:`
- `feat(hash-api):`
- `refactor(hash-api):`
- `refactor(miner):`
- `refactor(platform):`
- `feat(cli):`
- `perf(hash-api):`
- `fix(hash-api):`

Before every commit:

1. Run `git diff --stat`.
2. Review changed files.
3. Run relevant validation.
4. Ensure no unrelated generated files are staged.
5. Commit only a coherent slice.

## Non-Goals Until Hash API Is Stable

Do not prioritize:

- frontend redesign
- dashboard visual polish
- marketplace economics expansion
- wallet UX changes
- settlement model changes
- replacing SQLite
- production auth hardening
- broad MQTT protocol rewrites
- CUDA kernel rewrites not required by the API boundary

These can resume after the Hash API is stable and integrated.

## Stop And Ask The User If

Stop only for real blockers:

- a dirty user change conflicts with required edits
- a build requires credentials or unavailable proprietary tooling
- the C++ build requires a local CUDA/MSVC setup that cannot be discovered or installed safely
- a design choice would permanently break existing CLI or platform behavior
- tests reveal a pre-existing bug whose fix would broaden scope significantly

Otherwise, keep moving through the next smallest phase task.

## Definition Of Done

This long-running goal is complete when:

- hashing primitives are callable without starting the marketplace platform
- CPU/reference hash path works without CUDA
- CUDA batch path is behind the same interface
- existing self-mining behavior still works
- platform mode still registers, receives tasks, reports heartbeats, and reports blocks
- CLI exposes JSON-capable hash and benchmark commands
- optional local service, if implemented, is separate from marketplace APIs
- benchmark data is stable and machine-readable
- tests cover validation, CPU reference behavior, CLI/service behavior, and platform compatibility
- docs explain C++ API usage, CLI usage, local service usage, and benchmark workflow
- future optimization agents can work mostly inside the Hash API boundary

## Resume Checklist

When resuming a long-running `/goal` session:

1. Read the latest commit log.
2. Run `git status -sb`.
3. Identify the current phase from completed files and commits.
4. Continue from the next incomplete task.
5. Validate and commit before moving to the next phase.
