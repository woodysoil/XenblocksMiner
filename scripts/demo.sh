#!/usr/bin/env bash
# demo.sh - End-to-end demo of the hashpower marketplace
#
# Starts the mock platform server, launches simulated workers, rents hashpower
# via REST API, waits for blocks, stops the lease, and prints settlement.
#
# Usage: ./scripts/demo.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MQTT_PORT=11883
API_PORT=18080
NUM_WORKERS=2
BLOCK_INTERVAL=3
MINING_DURATION=15
DB_PATH="/tmp/demo-marketplace.db"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

log()  { echo -e "${CYAN}[demo]${NC} $*"; }
ok()   { echo -e "${GREEN}[demo]${NC} $*"; }
warn() { echo -e "${YELLOW}[demo]${NC} $*"; }
err()  { echo -e "${RED}[demo]${NC} $*"; }

cleanup() {
    log "Cleaning up..."
    [ -n "${SIM_PID:-}" ] && kill "$SIM_PID" 2>/dev/null && wait "$SIM_PID" 2>/dev/null || true
    [ -n "${SERVER_PID:-}" ] && kill "$SERVER_PID" 2>/dev/null && wait "$SERVER_PID" 2>/dev/null || true
    rm -f "$DB_PATH" "${DB_PATH}-wal" "${DB_PATH}-shm"
    log "Done."
}
trap cleanup EXIT

# Helper to make REST API calls
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

pretty() {
    python3 -c "import json,sys; print(json.dumps(json.loads(sys.stdin.read()), indent=2))"
}

cd "$PROJECT_ROOT"

echo ""
echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}     XenMiner Hashpower Marketplace - End-to-End Demo${NC}"
echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
echo ""

# ------------------------------------------------------------------
# Step 1: Start mock server
# ------------------------------------------------------------------
log "Step 1: Starting mock platform server (MQTT=$MQTT_PORT, API=$API_PORT)..."
rm -f "$DB_PATH" "${DB_PATH}-wal" "${DB_PATH}-shm"
python3 -m server.server --mqtt-port "$MQTT_PORT" --api-port "$API_PORT" --db-path "$DB_PATH" > /tmp/demo-server.log 2>&1 &
SERVER_PID=$!
sleep 2

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    err "Server failed to start. Log:"
    cat /tmp/demo-server.log
    exit 1
fi
ok "Server running (PID=$SERVER_PID)"

# ------------------------------------------------------------------
# Step 2: Start simulated workers
# ------------------------------------------------------------------
log "Step 2: Starting $NUM_WORKERS simulated workers (block every ~${BLOCK_INTERVAL}s)..."
python3 -m server.simulator \
    --workers "$NUM_WORKERS" \
    --mqtt-port "$MQTT_PORT" \
    --block-interval "$BLOCK_INTERVAL" \
    --gpu-count 2 \
    --api-port "$API_PORT" > /tmp/demo-simulator.log 2>&1 &
SIM_PID=$!
sleep 3

if ! kill -0 "$SIM_PID" 2>/dev/null; then
    err "Simulator failed to start. Log:"
    cat /tmp/demo-simulator.log
    exit 1
fi
ok "Simulator running (PID=$SIM_PID)"

# ------------------------------------------------------------------
# Step 3: Check workers
# ------------------------------------------------------------------
log "Step 3: Checking registered workers..."
WORKERS=$(api GET /api/workers)
echo "$WORKERS" | pretty
echo ""

WORKER_COUNT=$(echo "$WORKERS" | python3 -c "import json,sys; print(len(json.loads(sys.stdin.read())))")
if [ "$WORKER_COUNT" -lt 1 ]; then
    err "No workers registered!"
    exit 1
fi
ok "$WORKER_COUNT workers registered"

# ------------------------------------------------------------------
# Step 4: Rent hashpower
# ------------------------------------------------------------------
log "Step 4: Consumer renting hashpower for ${MINING_DURATION}s..."
LEASE_RESULT=$(api POST /api/rent "{\"consumer_id\": \"consumer-1\", \"consumer_address\": \"0xaabbccddee1234567890abcdef1234567890abcd\", \"duration_sec\": $MINING_DURATION}")
echo "$LEASE_RESULT" | pretty
echo ""

LEASE_ID=$(echo "$LEASE_RESULT" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['lease_id'])")
PREFIX=$(echo "$LEASE_RESULT" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['prefix'])")
WORKER_ID=$(echo "$LEASE_RESULT" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['worker_id'])")
ok "Lease created: $LEASE_ID"
ok "  Worker: $WORKER_ID"
ok "  Prefix: $PREFIX"
echo ""

# ------------------------------------------------------------------
# Step 5: Wait and watch blocks
# ------------------------------------------------------------------
log "Step 5: Mining in progress... watching for blocks (${MINING_DURATION}s)..."
echo ""
ELAPSED=0
while [ "$ELAPSED" -lt "$MINING_DURATION" ]; do
    sleep 3
    ELAPSED=$((ELAPSED + 3))
    BLOCKS=$(api GET "/api/blocks?lease_id=$LEASE_ID")
    BLOCK_COUNT=$(echo "$BLOCKS" | python3 -c "import json,sys; print(len(json.loads(sys.stdin.read())))")
    STATUS=$(api GET /api/status)
    echo -e "  ${CYAN}[${ELAPSED}s]${NC} Blocks found: $BLOCK_COUNT"
done
echo ""

# ------------------------------------------------------------------
# Step 6: Stop lease
# ------------------------------------------------------------------
log "Step 6: Stopping lease and settling..."
STOP_RESULT=$(api POST /api/stop "{\"lease_id\": \"$LEASE_ID\"}")
echo "$STOP_RESULT" | pretty
echo ""

# ------------------------------------------------------------------
# Step 7: Final summary
# ------------------------------------------------------------------
log "Step 7: Final summary"
echo ""

echo -e "${BOLD}--- Lease Detail ---${NC}"
api GET "/api/leases/$LEASE_ID" | pretty
echo ""

echo -e "${BOLD}--- Account Balances ---${NC}"
echo -e "  Consumer (consumer-1):"
api GET /api/accounts/consumer-1/balance | pretty
echo ""
echo -e "  Provider ($WORKER_ID):"
api GET "/api/accounts/$WORKER_ID/balance" | pretty
echo ""
echo -e "  Platform Treasury:"
api GET /api/accounts/platform-treasury/balance | pretty
echo ""

echo -e "${BOLD}--- All Settlements ---${NC}"
api GET /api/settlements | pretty
echo ""

FINAL_BLOCKS=$(api GET "/api/blocks?lease_id=$LEASE_ID")
FINAL_COUNT=$(echo "$FINAL_BLOCKS" | python3 -c "import json,sys; print(len(json.loads(sys.stdin.read())))")

echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Demo complete!${NC}"
echo -e "  Lease:  $LEASE_ID"
echo -e "  Worker: $WORKER_ID"
echo -e "  Prefix: $PREFIX"
echo -e "  Blocks: $FINAL_COUNT"
echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
echo ""
