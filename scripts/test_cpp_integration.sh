#!/usr/bin/env bash
# test_cpp_integration.sh - Integration test: real C++ miner ↔ mock platform server
#
# Starts the mock platform server, launches the actual C++ miner binary in
# platform mode, rents hashpower via REST API, waits for blocks, stops the
# lease, and verifies settlement.
#
# Prerequisites:
#   - Built C++ binary (build/bin/xenblocksMiner or specify via MINER_BIN)
#   - Python 3 with fastapi, uvicorn, gmqtt, aiosqlite installed
#   - CUDA GPU available (the miner needs it)
#
# Usage:
#   ./scripts/test_cpp_integration.sh                    # auto-detect binary
#   MINER_BIN=/path/to/xenblocksMiner ./scripts/test_cpp_integration.sh
#
# Environment variables:
#   MINER_BIN        Path to the compiled miner binary (default: auto-detect)
#   MQTT_PORT        MQTT broker port (default: 21883)
#   API_PORT         REST API port (default: 28080)
#   MINER_PORT       Miner Crow HTTP port (not directly controllable, uses 42069)
#   MINING_DURATION  How long to mine in seconds (default: 30)
#   SKIP_CLEANUP     Set to 1 to keep processes running after test

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration (use unique ports to avoid conflicts with demo.sh)
MQTT_PORT="${MQTT_PORT:-21883}"
API_PORT="${API_PORT:-28080}"
MINING_DURATION="${MINING_DURATION:-30}"
WORKER_ID="test-cpp-worker-001"
CONSUMER_ID="consumer-1"
CONSUMER_ADDR="0xaabbccddee1234567890abcdef1234567890abcd"
MINER_ADDR="0x0000000000000000000000000000000000000000"
DB_PATH="/tmp/test-cpp-integration.db"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()  { echo -e "${CYAN}[test]${NC} $*"; }
pass() { echo -e "${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

FAILURES=0
PASSES=0

assert_eq() {
    local desc="$1" expected="$2" actual="$3"
    if [ "$expected" = "$actual" ]; then
        pass "$desc"
        PASSES=$((PASSES + 1))
    else
        fail "$desc (expected='$expected', actual='$actual')"
        FAILURES=$((FAILURES + 1))
    fi
}

assert_ge() {
    local desc="$1" min="$2" actual="$3"
    if [ "$actual" -ge "$min" ] 2>/dev/null; then
        pass "$desc (got $actual >= $min)"
        PASSES=$((PASSES + 1))
    else
        fail "$desc (expected >= $min, got '$actual')"
        FAILURES=$((FAILURES + 1))
    fi
}

assert_not_empty() {
    local desc="$1" value="$2"
    if [ -n "$value" ]; then
        pass "$desc"
        PASSES=$((PASSES + 1))
    else
        fail "$desc (value was empty)"
        FAILURES=$((FAILURES + 1))
    fi
}

# ── Find binary ──────────────────────────────────────────────────────────

find_miner_binary() {
    if [ -n "${MINER_BIN:-}" ] && [ -x "$MINER_BIN" ]; then
        echo "$MINER_BIN"
        return 0
    fi
    # Check common build locations
    for candidate in \
        "$PROJECT_ROOT/build/bin/xenblocksMiner" \
        "$PROJECT_ROOT/cmake-build-release/bin/xenblocksMiner" \
        "$PROJECT_ROOT/cmake-build-debug/bin/xenblocksMiner" \
        "$PROJECT_ROOT/out/build/Release/bin/xenblocksMiner" \
        "$PROJECT_ROOT/xenblocksMiner"; do
        if [ -x "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

# ── REST API helper ──────────────────────────────────────────────────────

api() {
    local method="$1" path="$2"
    shift 2
    if [ "$method" = "POST" ]; then
        python3 -c "
import http.client, json, sys
conn = http.client.HTTPConnection('localhost', $API_PORT, timeout=10)
conn.request('$method', '$path', json.dumps($1), {'Content-Type': 'application/json'})
r = conn.getresponse()
data = r.read().decode()
conn.close()
print(data)
"
    else
        python3 -c "
import http.client, json
conn = http.client.HTTPConnection('localhost', $API_PORT, timeout=10)
conn.request('$method', '$path')
r = conn.getresponse()
data = r.read().decode()
conn.close()
print(data)
"
    fi
}

jq_py() {
    # Poor man's jq using Python
    python3 -c "import json,sys; data=json.loads(sys.stdin.read()); print($1)"
}

# ── Cleanup ──────────────────────────────────────────────────────────────

cleanup() {
    if [ "${SKIP_CLEANUP:-0}" = "1" ]; then
        warn "SKIP_CLEANUP=1, leaving processes running:"
        [ -n "${MINER_PID:-}" ] && warn "  Miner PID=$MINER_PID"
        [ -n "${SERVER_PID:-}" ] && warn "  Server PID=$SERVER_PID"
        return
    fi
    log "Cleaning up..."
    [ -n "${MINER_PID:-}" ] && kill "$MINER_PID" 2>/dev/null && wait "$MINER_PID" 2>/dev/null || true
    [ -n "${SERVER_PID:-}" ] && kill "$SERVER_PID" 2>/dev/null && wait "$SERVER_PID" 2>/dev/null || true
    rm -f "$DB_PATH" "${DB_PATH}-wal" "${DB_PATH}-shm"
    log "Cleanup done."
}
trap cleanup EXIT

# ── Preflight checks ────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}   C++ Worker Integration Test${NC}"
echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
echo ""

# Check for CUDA
if ! command -v nvidia-smi &>/dev/null; then
    warn "nvidia-smi not found. CUDA GPU required for C++ miner."
    warn "Skipping test."
    exit 0
fi

if ! nvidia-smi &>/dev/null; then
    warn "nvidia-smi failed. No CUDA GPU available."
    warn "Skipping test."
    exit 0
fi

# Find miner binary
if ! MINER_BIN=$(find_miner_binary); then
    warn "C++ miner binary not found."
    warn "Build with: mkdir build && cd build && cmake .. && make"
    warn "Or set MINER_BIN=/path/to/xenblocksMiner"
    warn "Skipping test."
    exit 0
fi
log "Miner binary: $MINER_BIN"

# Check Python deps
if ! python3 -c "import fastapi, uvicorn" 2>/dev/null; then
    warn "Python dependencies not installed."
    warn "Run: pip install -r server/requirements.txt"
    warn "Skipping test."
    exit 0
fi

cd "$PROJECT_ROOT"

# ══════════════════════════════════════════════════════════════════════════
# Test Execution
# ══════════════════════════════════════════════════════════════════════════

# ------------------------------------------------------------------
# Step 1: Start mock platform server
# ------------------------------------------------------------------
log "Step 1: Starting mock platform server (MQTT=$MQTT_PORT, API=$API_PORT)..."
rm -f "$DB_PATH" "${DB_PATH}-wal" "${DB_PATH}-shm"
python3 -m server.server \
    --mqtt-port "$MQTT_PORT" \
    --api-port "$API_PORT" \
    --db-path "$DB_PATH" > /tmp/test-cpp-server.log 2>&1 &
SERVER_PID=$!
sleep 2

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    fail "Server failed to start"
    cat /tmp/test-cpp-server.log
    exit 1
fi
pass "Server started (PID=$SERVER_PID)"

# Verify server is responsive
SERVER_ROOT=$(api GET /)
assert_not_empty "Server responds to GET /" "$SERVER_ROOT"

# ------------------------------------------------------------------
# Step 2: Launch C++ miner in platform mode
# ------------------------------------------------------------------
log "Step 2: Launching C++ miner in platform mode..."
"$MINER_BIN" --execute \
    --platform-mode \
    --mqtt-broker "tcp://localhost:$MQTT_PORT" \
    --worker-id "$WORKER_ID" \
    --minerAddr "$MINER_ADDR" \
    --totalDevFee 0 \
    --testFixedDiff 1 \
    --donotupload \
    > /tmp/test-cpp-miner.log 2>&1 &
MINER_PID=$!
sleep 3

if ! kill -0 "$MINER_PID" 2>/dev/null; then
    fail "Miner failed to start"
    echo "--- Miner log ---"
    cat /tmp/test-cpp-miner.log
    exit 1
fi
pass "Miner started (PID=$MINER_PID)"

# ------------------------------------------------------------------
# Step 3: Verify worker registration
# ------------------------------------------------------------------
log "Step 3: Verifying worker registration..."

# Poll for worker to appear (max 15 seconds)
REGISTERED=false
for i in $(seq 1 15); do
    WORKERS=$(api GET /api/workers)
    WORKER_COUNT=$(echo "$WORKERS" | jq_py "len(data)")
    if [ "$WORKER_COUNT" -ge 1 ]; then
        REGISTERED=true
        break
    fi
    sleep 1
done

if [ "$REGISTERED" = true ]; then
    pass "Worker registered (count=$WORKER_COUNT)"
else
    fail "Worker did not register within 15 seconds"
    echo "--- Miner log (last 30 lines) ---"
    tail -30 /tmp/test-cpp-miner.log
    exit 1
fi

# Verify worker fields
WORKER_DATA=$(echo "$WORKERS" | python3 -c "
import json, sys
workers = json.loads(sys.stdin.read())
w = next((w for w in workers if w.get('worker_id') == '$WORKER_ID'), None)
if w:
    print(json.dumps(w))
else:
    print('{}')
")

if [ "$WORKER_DATA" != "{}" ]; then
    W_STATE=$(echo "$WORKER_DATA" | jq_py "data.get('state', '')")
    W_GPU=$(echo "$WORKER_DATA" | jq_py "data.get('gpu_count', 0)")
    assert_eq "Worker state is AVAILABLE" "AVAILABLE" "$W_STATE"
    assert_ge "Worker has at least 1 GPU" 1 "$W_GPU"
else
    fail "Worker $WORKER_ID not found in workers list"
fi

# ------------------------------------------------------------------
# Step 4: Verify miner's local Crow HTTP endpoints
# ------------------------------------------------------------------
log "Step 4: Checking miner's local HTTP endpoints..."

MINER_STATS=$(python3 -c "
import http.client, json
try:
    conn = http.client.HTTPConnection('localhost', 42069, timeout=5)
    conn.request('GET', '/platform/status')
    r = conn.getresponse()
    print(r.read().decode())
    conn.close()
except Exception as e:
    print('{}')
" 2>/dev/null || echo '{}')

if [ "$MINER_STATS" != "{}" ]; then
    PLAT_MODE=$(echo "$MINER_STATS" | jq_py "data.get('platform_mode', False)")
    assert_eq "Miner reports platform_mode=True" "True" "$PLAT_MODE"
else
    warn "Could not reach miner Crow endpoint at :42069/platform/status"
fi

# ------------------------------------------------------------------
# Step 5: Rent hashpower via REST API
# ------------------------------------------------------------------
log "Step 5: Renting hashpower (${MINING_DURATION}s)..."
LEASE_RESULT=$(api POST /api/rent "{\"consumer_id\": \"$CONSUMER_ID\", \"consumer_address\": \"$CONSUMER_ADDR\", \"duration_sec\": $MINING_DURATION, \"worker_id\": \"$WORKER_ID\"}")

LEASE_ID=$(echo "$LEASE_RESULT" | jq_py "data.get('lease_id', '')")
PREFIX=$(echo "$LEASE_RESULT" | jq_py "data.get('prefix', '')")
assert_not_empty "Lease ID assigned" "$LEASE_ID"
assert_not_empty "Prefix assigned" "$PREFIX"
log "  Lease: $LEASE_ID"
log "  Prefix: $PREFIX"

# ------------------------------------------------------------------
# Step 6: Wait for mining and check for blocks
# ------------------------------------------------------------------
log "Step 6: Waiting for blocks (${MINING_DURATION}s)..."

ELAPSED=0
LAST_BLOCK_COUNT=0
while [ "$ELAPSED" -lt "$MINING_DURATION" ]; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    BLOCKS=$(api GET "/api/blocks?lease_id=$LEASE_ID" 2>/dev/null || echo "[]")
    BLOCK_COUNT=$(echo "$BLOCKS" | jq_py "len(data)" 2>/dev/null || echo "0")
    echo -e "  ${CYAN}[${ELAPSED}s]${NC} Blocks found: $BLOCK_COUNT"
    LAST_BLOCK_COUNT="$BLOCK_COUNT"
done

# Check worker state during mining
WORKER_STATUS=$(api GET /api/workers)
MINING_STATE=$(echo "$WORKER_STATUS" | python3 -c "
import json, sys
workers = json.loads(sys.stdin.read())
w = next((w for w in workers if w.get('worker_id') == '$WORKER_ID'), {})
print(w.get('state', 'UNKNOWN'))
")
log "  Worker state during mining: $MINING_STATE"

# ------------------------------------------------------------------
# Step 7: Stop lease and verify settlement
# ------------------------------------------------------------------
log "Step 7: Stopping lease and verifying settlement..."
STOP_RESULT=$(api POST /api/stop "{\"lease_id\": \"$LEASE_ID\"}")
STOP_STATE=$(echo "$STOP_RESULT" | jq_py "data.get('state', '')")
STOP_BLOCKS=$(echo "$STOP_RESULT" | jq_py "data.get('blocks_found', 0)")
assert_eq "Lease state after stop" "completed" "$STOP_STATE"

# Give the miner a moment to process the release
sleep 3

# ------------------------------------------------------------------
# Step 8: Verify worker returns to AVAILABLE
# ------------------------------------------------------------------
log "Step 8: Checking worker returns to AVAILABLE..."

RECOVERED=false
for i in $(seq 1 10); do
    WORKERS=$(api GET /api/workers)
    STATE=$(echo "$WORKERS" | python3 -c "
import json, sys
workers = json.loads(sys.stdin.read())
w = next((w for w in workers if w.get('worker_id') == '$WORKER_ID'), {})
print(w.get('state', 'UNKNOWN'))
")
    if [ "$STATE" = "AVAILABLE" ]; then
        RECOVERED=true
        break
    fi
    sleep 1
done

if [ "$RECOVERED" = true ]; then
    pass "Worker returned to AVAILABLE after lease"
else
    warn "Worker state is '$STATE' (expected AVAILABLE)"
    FAILURES=$((FAILURES + 1))
fi

# ------------------------------------------------------------------
# Step 9: Verify prefix correctness on any found blocks
# ------------------------------------------------------------------
log "Step 9: Verifying block prefix correctness..."

BLOCKS=$(api GET "/api/blocks?lease_id=$LEASE_ID" 2>/dev/null || echo "[]")
BLOCK_COUNT=$(echo "$BLOCKS" | jq_py "len(data)")

if [ "$BLOCK_COUNT" -gt 0 ]; then
    # Check that all blocks have valid prefix
    PREFIX_OK=$(echo "$BLOCKS" | python3 -c "
import json, sys
blocks = json.loads(sys.stdin.read())
prefix = '$PREFIX'.lower()
all_ok = all(b.get('key', '')[:16].lower() == prefix for b in blocks)
print(all_ok)
")
    assert_eq "All block keys start with assigned prefix" "True" "$PREFIX_OK"
    pass "Found $BLOCK_COUNT blocks with correct prefix"
else
    warn "No blocks found during test (difficulty may be too high)"
    warn "This is expected with --testFixedDiff 1 on some GPUs"
fi

# ------------------------------------------------------------------
# Step 10: Check settlement
# ------------------------------------------------------------------
log "Step 10: Checking settlement..."

SETTLEMENTS=$(api GET /api/settlements)
SETTLEMENT_COUNT=$(echo "$SETTLEMENTS" | jq_py "len(data)")
assert_ge "At least 1 settlement" 1 "$SETTLEMENT_COUNT"

# ══════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
if [ "$FAILURES" -eq 0 ]; then
    echo -e "${GREEN}  All $PASSES checks passed!${NC}"
else
    echo -e "${RED}  $FAILURES checks failed, $PASSES passed${NC}"
fi
echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
echo ""

exit "$FAILURES"
