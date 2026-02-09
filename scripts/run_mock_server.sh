#!/usr/bin/env bash
# run_mock_server.sh - Start the mock platform server for development/testing
# Usage: ./scripts/run_mock_server.sh [--mqtt-port 1883] [--api-port 8080]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Starting mock platform server..."
echo "Project root: $PROJECT_ROOT"

# Install dependencies if needed
if ! python3 -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "Installing Python dependencies..."
    pip install -r "$PROJECT_ROOT/server/requirements.txt"
fi

cd "$PROJECT_ROOT"
exec python3 -m server.server "$@"
