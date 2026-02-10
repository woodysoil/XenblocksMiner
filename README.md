# XenBlocks Mining Platform

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-18-61dafb?logo=react&logoColor=white)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7-3178c6?logo=typescript&logoColor=white)](https://typescriptlang.org)
[![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-4-06b6d4?logo=tailwindcss&logoColor=white)](https://tailwindcss.com)

## Overview

XenBlocks is a hashpower marketplace and mining monitoring platform for the XEN ecosystem. It combines a real-time mining fleet dashboard with a decentralized marketplace where providers list GPU capacity and renters lease hashpower on-demand. The platform features an embedded MQTT broker for worker telemetry, WebSocket-driven live updates, and JWT-authenticated wallet-based accounts.

<!-- Screenshots coming soon -->

## Architecture

```
Browser
  |
  v
Vite Dev Server (:5173)
  |
  v
React SPA (TanStack Query + React Router)
  |
  +--> REST API (:8080/api/...)  ---+
  |                                 |
  +--> WebSocket (:8080/ws)  -------+--> Python Backend (FastAPI + uvicorn)
                                           |
                                    +------+------+
                                    |             |
                                 SQLite      MQTT Broker (:1883)
                              (aiosqlite)        |
                                           Mining Workers
```

## Tech Stack

| Backend | Frontend |
|---|---|
| Python 3.10+ | React 18 |
| FastAPI + uvicorn | TypeScript 5.7 |
| SQLite (aiosqlite) | Vite 6 |
| Embedded MQTT broker | TailwindCSS v4 |
| JWT auth (PyJWT) | TanStack Query v5 |
| Pydantic v2 | Lightweight Charts |
| eth-account | Recharts |
|  | Sonner (toast notifications) |
|  | Radix UI (dialogs) |
|  | React Router v7 |

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm or yarn

### Backend

```bash
# Install Python dependencies
pip install -r server/requirements.txt

# Start the platform server (MQTT broker + REST API)
./scripts/run_mock_server.sh

# Or run directly:
python -m server.server --mqtt-port 1883 --api-port 8080
```

The server starts an embedded MQTT broker on port 1883 and the REST API on port 8080.

### Frontend

```bash
cd web

# Install dependencies
npm install

# Start dev server
npm run dev
```

The dashboard opens at `http://localhost:5173` and proxies API requests to the backend.

### Generate Test Data

```bash
# Spawn a simulated mining fleet (connects to the MQTT broker)
python scripts/mock_fleet.py --workers 10 --broker localhost --port 1883

# Populate the database with historical test data
python scripts/generate_test_data.py
```

## Project Structure

```
XenblocksMiner/
├── server/                  # Python backend
│   ├── server.py            # Entry point (MQTT + API orchestration)
│   ├── broker.py            # Embedded async MQTT broker
│   ├── storage.py           # SQLite persistence layer
│   ├── watcher.py           # Block watcher / telemetry ingestion
│   ├── monitoring.py        # Fleet monitoring service
│   ├── matcher.py           # Hashpower order matching engine
│   ├── settlement.py        # Lease settlement engine
│   ├── pricing.py           # Dynamic pricing engine
│   ├── reputation.py        # Provider reputation scoring
│   ├── account.py           # Wallet-based account management
│   ├── auth.py              # JWT authentication
│   ├── ws.py                # WebSocket connection manager
│   ├── routers/             # FastAPI route modules
│   │   ├── overview.py
│   │   ├── monitoring.py
│   │   ├── marketplace.py
│   │   ├── provider.py
│   │   ├── rental.py
│   │   ├── account.py
│   │   ├── wallet.py
│   │   ├── admin.py
│   │   └── ws.py
│   └── requirements.txt
├── web/                     # React frontend
│   ├── src/
│   │   ├── App.tsx          # Router + providers
│   │   ├── api.ts           # API client with JWT handling
│   │   ├── types.ts         # Shared TypeScript interfaces
│   │   ├── pages/           # Route-level page components
│   │   │   ├── Overview.tsx
│   │   │   ├── Monitoring.tsx
│   │   │   ├── Marketplace.tsx
│   │   │   ├── Provider.tsx
│   │   │   ├── Renter.tsx
│   │   │   └── Account.tsx
│   │   ├── design/          # Design system (tokens, reusable components)
│   │   │   ├── tokens.ts    # Color, spacing, and chart theme tokens
│   │   │   ├── MetricCard.tsx
│   │   │   ├── ChartCard.tsx
│   │   │   ├── DataTable.tsx
│   │   │   ├── StatusBadge.tsx
│   │   │   ├── GpuBadge.tsx
│   │   │   └── ...
│   │   ├── hooks/           # Custom React hooks
│   │   ├── utils/           # Formatting helpers
│   │   ├── context/         # React context providers
│   │   ├── components/      # Layout, pagination
│   │   └── lib/             # Query client configuration
│   └── package.json
├── scripts/                 # Development & testing utilities
│   ├── mock_fleet.py        # Simulated mining fleet (MQTT workers)
│   ├── generate_test_data.py
│   ├── run_mock_server.sh
│   ├── demo.sh
│   └── test_cpp_integration.sh
├── src/                     # C++ miner core (CUDA)
├── doc/                     # Build instructions, API docs
└── proto/                   # Protocol definitions
```

## Pages

| Page | Route | Description |
|---|---|---|
| **Overview** | `/` | Fleet summary dashboard -- worker counts, total hashrate, block production, and recent activity table. |
| **Monitoring** | `/monitoring` | Real-time fleet monitoring with hashrate charts, per-worker status table, and block history. Uses WebSocket for live updates. |
| **Marketplace** | `/marketplace` | Browse available hashpower listings from providers. Filter by GPU type, price, and availability. |
| **Provider** | `/provider` | Provider management console -- list/delist GPU capacity, monitor active leases, view earnings charts via Lightweight Charts. |
| **Renter** | `/renter` | Renter dashboard for browsing, leasing, and managing active hashpower rentals. Requires wallet connection. |
| **Account** | `/account` | Wallet-based account management -- connect wallet, view balances, and manage authentication. |

## Development

### Mock Fleet

The fleet simulator spawns N virtual mining workers that connect via MQTT and produce realistic telemetry:

```bash
python scripts/mock_fleet.py \
  --workers 20 \
  --broker localhost \
  --port 1883 \
  --block-interval 60
```

Workers send registration, heartbeat (every 30s), and block-found messages. They randomly go offline/online to simulate real fleet behavior. GPU profiles range from RTX 3060 Ti to H100.

### Test Data

```bash
python scripts/generate_test_data.py
```

Populates the SQLite database with historical worker data, blocks, and marketplace activity.

### Build for Production

```bash
cd web
npm run build    # Outputs to web/dist/
```

For C++ miner build instructions, see [doc/BUILD_INSTRUCTIONS.md](./doc/BUILD_INSTRUCTIONS.md).

## Precompiled Binaries

Precompiled miner binaries for supported platforms are available in the [Releases](https://github.com/woodysoil/XenblocksMiner/releases) section.

## License

MIT
