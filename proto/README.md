# XenblocksMiner MQTT Protocol Specification

## Overview

This directory contains the formal MQTT message protocol schemas for the
XenblocksMiner hashpower marketplace. All messages are JSON-encoded and
published/subscribed via MQTT with QoS 1.

## Topic Structure

All topics follow the pattern:

```
xenminer/{worker_id}/{message_type}
```

Where `worker_id` is the machine-unique identifier (`machineId` global) and
`message_type` is one of the predefined suffixes.

### Worker -> Platform (published by worker)

| Topic Suffix | Description |
|---|---|
| `register` | Worker registration with GPU capabilities |
| `heartbeat` | Periodic stats (every 30 seconds) |
| `status` | State change notifications |
| `block` | Block-found report during a lease |

### Platform -> Worker (subscribed by worker)

| Topic Suffix | Description |
|---|---|
| `task` | Lease assignment commands (`register_ack`, `assign_task`, `release`) |
| `control` | Operational commands (`pause`, `resume`, `shutdown`) |

## Message Dispatch

Platform-to-worker messages arrive on two topics and are dispatched by
different fields:

- **`task` topic**: dispatched by the `command` field -- `register_ack`,
  `assign_task`, or `release`.
- **`control` topic**: dispatched by the `action` field -- `pause`, `resume`,
  or `shutdown`.

On the C++ side, if the `command` field does not match `register_ack`,
`assign_task`, or `release`, the message is forwarded to the control handler
which reads the `action` field instead.

## Schema Files

- `worker_to_platform.json` - JSON Schema for all worker-published messages
- `platform_to_worker.json` - JSON Schema for all platform-published messages
- `examples/` - Example payloads for each message type:
  - `register.json` - Worker registration
  - `heartbeat.json` - Periodic heartbeat
  - `status.json` - Status update with active lease
  - `status_offline.json` - Status update going offline
  - `block_found.json` - Block discovery report
  - `register_ack_accepted.json` - Registration accepted
  - `register_ack_rejected.json` - Registration rejected
  - `assign_task.json` - Lease assignment
  - `release.json` - Lease release
  - `control_pause.json` - Pause command
  - `control_resume.json` - Resume command
  - `control_shutdown.json` - Shutdown command

Note: The example files include `_description`, `_topic`, and `_source`
metadata fields for documentation purposes. These fields are not part of the
wire protocol and would not pass strict `additionalProperties: false` schema
validation.

## Worker States

The worker progresses through a 6-state machine, plus a virtual `offline`
state used in status messages when disconnecting:

```
IDLE -> AVAILABLE -> LEASED -> MINING -> COMPLETED -> AVAILABLE
                                    \-> ERROR -> IDLE -> AVAILABLE
```

| State | Description |
|---|---|
| `IDLE` | Not connected to platform |
| `AVAILABLE` | Registered and waiting for lease assignment |
| `LEASED` | Lease assigned, preparing to mine |
| `MINING` | Actively mining for a consumer |
| `COMPLETED` | Lease completed, transitioning back |
| `ERROR` | Error state, will attempt recovery |
| `offline` | Sent in status messages when the worker is shutting down (not a state machine state) |

## Key Constants

| Constant | Value | Description |
|---|---|---|
| `PLATFORM_PREFIX_LENGTH` | 16 | Required length of key prefix (hex chars) |
| `QOS` | 1 | MQTT Quality of Service level |
| `HEARTBEAT_INTERVAL_SEC` | 30 | Heartbeat publish interval |
| `WATCHDOG_INTERVAL_SEC` | 5 | Lease expiry check interval |
| `KEEPALIVE_INTERVAL_SEC` | 60 | MQTT keep-alive interval |
