# Scripts

## run_mock_server.sh

Convenience wrapper to start the mock platform server for development or
manual testing. Automatically installs Python dependencies if missing.

### Usage

```bash
# Start with default ports (MQTT 1883, API 8080)
./scripts/run_mock_server.sh

# Override ports
./scripts/run_mock_server.sh --mqtt-port 11883 --api-port 18080

# Custom database path
./scripts/run_mock_server.sh --db-path /tmp/my-marketplace.db
```

All arguments are forwarded to `python3 -m server.server`. See the server
documentation for the full list of options.

---

## test_cpp_integration.sh

End-to-end integration test that launches the mock platform server and the
real C++ miner binary, then verifies registration, lease assignment, mining,
block discovery, release, and settlement.

### Prerequisites

1. **CUDA GPU** - The C++ miner requires an NVIDIA GPU with CUDA support.
   `nvidia-smi` must be available and working.

2. **Built C++ binary** - Build the miner first:
   ```bash
   mkdir -p build && cd build && cmake .. && make -j$(nproc)
   ```
   The script auto-detects the binary in common build directories:
   - `build/bin/xenblocksMiner`
   - `cmake-build-release/bin/xenblocksMiner`
   - `cmake-build-debug/bin/xenblocksMiner`
   - `out/build/Release/bin/xenblocksMiner`

   Or set `MINER_BIN=/path/to/xenblocksMiner` explicitly.

3. **Python dependencies** - Install the mock server requirements:
   ```bash
   pip install -r server/requirements.txt
   ```

### Usage

```bash
# Auto-detect binary
./scripts/test_cpp_integration.sh

# Specify binary explicitly
MINER_BIN=/path/to/xenblocksMiner ./scripts/test_cpp_integration.sh
```

### Environment Variables

| Variable         | Default | Description                              |
|------------------|---------|------------------------------------------|
| `MINER_BIN`      | (auto)  | Path to compiled miner binary            |
| `MQTT_PORT`      | 21883   | MQTT broker port                         |
| `API_PORT`       | 28080   | REST API port                            |
| `MINING_DURATION` | 30     | How long to mine in seconds              |
| `SKIP_CLEANUP`   | 0       | Set to 1 to keep processes running after |

The script exits 0 (success) if no CUDA GPU or binary is found, so it is
safe to include in CI pipelines that may lack GPU hardware.

---

## demo.sh

Interactive demo of the hashpower marketplace using simulated Python workers
(no CUDA GPU required). Starts the mock server, launches simulated workers,
rents hashpower, watches blocks, and prints settlement.

### Prerequisites

- **Python 3** with mock server dependencies installed:
  ```bash
  pip install -r server/requirements.txt
  ```

### Usage

```bash
./scripts/demo.sh
```

The demo uses hardcoded ports (MQTT=11883, API=18080) and runs 2 simulated
workers with a block interval of 3 seconds for a 15-second mining session.
It walks through the full lifecycle:

1. Start mock platform server
2. Launch simulated workers
3. Verify worker registration
4. Rent hashpower via REST API
5. Wait and watch blocks being mined
6. Stop lease and trigger settlement
7. Print final summary (lease detail, account balances, settlements)

---

## Pytest Version

There is also a pytest harness at `tests/integration/test_cpp_worker.py`
that covers the same scenarios. It uses `pytest.mark.skipif` to skip
gracefully when CUDA or the binary is unavailable.

```bash
# Run with auto-detection
python3 -m pytest tests/integration/test_cpp_worker.py -v --tb=short

# Specify binary
MINER_BIN=/path/to/xenblocksMiner python3 -m pytest tests/integration/test_cpp_worker.py -v
```

The pytest version uses different ports (MQTT=31883, API=38080) to avoid
conflicts with the bash script or demo.
