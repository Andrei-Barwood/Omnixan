# Persistent Memory Module

**Status: ✅ IMPLEMENTED**

Production-ready persistent memory (PMEM) implementation providing byte-addressable persistent storage with ACID transactions, crash recovery, and high-performance data persistence.

## Features

- **ACID Transactions**
  - BEGIN, COMMIT, ABORT semantics
  - Read/write isolation
  - Automatic rollback on abort

- **Durability**
  - Write-ahead logging (WAL)
  - Synchronous/asynchronous persistence
  - Crash recovery (REDO/UNDO)

- **Performance**
  - Byte-addressable storage
  - Memory-mapped access patterns
  - Low-latency writes
  - Automatic checkpointing

## Quick Start

```python
from omnixan.edge_computing_network.persistent_memory_module.module import (
    PersistentMemoryModule,
    PMEMConfig,
    PersistenceMode
)

# Configure
config = PMEMConfig(
    storage_path="/var/omnixan/pmem",
    size=1024 * 1024 * 100,  # 100 MB
    persistence_mode=PersistenceMode.SYNC,
    enable_wal=True
)

module = PersistentMemoryModule(config)
await module.initialize()

# Simple operations (auto-commit)
await module.put("user:1", b'{"name": "Alice"}')
value = await module.get("user:1")

# Transaction
tx = await module.begin_transaction()
try:
    await module.put("account:1", b'{"balance": 1000}', tx)
    await module.put("account:2", b'{"balance": 2000}', tx)
    await module.commit(tx)
except Exception:
    await module.abort(tx)

await module.shutdown()
```

## Persistence Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `SYNC` | Immediate flush to disk | Critical data, low tolerance for loss |
| `ASYNC` | Periodic background flush | High throughput, some loss acceptable |
| `PERIODIC` | Timed flush intervals | Balanced approach |

## Transaction API

```python
# Begin transaction
tx_id = await module.begin_transaction()

# Read/Write within transaction
await module.put("key1", b"value1", tx_id)
await module.put("key2", b"value2", tx_id)
value = await module.get("key1", tx_id)  # See uncommitted data

# Commit or abort
await module.commit(tx_id)  # Make changes permanent
# OR
await module.abort(tx_id)   # Rollback changes
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│           PersistentMemoryModule                    │
├─────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────┐   │
│  │            Transaction Manager               │   │
│  │  • Active transactions                       │   │
│  │  • Read/Write sets                          │   │
│  │  • Conflict detection                       │   │
│  └────────────────────┬────────────────────────┘   │
│                       ↓                             │
│  ┌─────────────────────────────────────────────┐   │
│  │           Write-Ahead Log (WAL)              │   │
│  │  • Log entries with LSN                      │   │
│  │  • Checksums for integrity                  │   │
│  │  • Truncation on checkpoint                 │   │
│  └────────────────────┬────────────────────────┘   │
│                       ↓                             │
│  ┌─────────────────────────────────────────────┐   │
│  │            Persistent Store                  │   │
│  │  • Key-value index                          │   │
│  │  • Data file (memory-mapped)                │   │
│  │  • Space reclamation                        │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
                        ↓
              ┌─────────────────┐
              │  Disk Storage   │
              │  /var/pmem/     │
              │  ├── data.pmem  │
              │  ├── index.json │
              │  └── wal.log    │
              └─────────────────┘
```

## Recovery Process

On startup, the module performs crash recovery:

1. **Load WAL** - Read all log entries
2. **Identify transactions** - Find committed/active transactions
3. **REDO committed** - Replay committed writes
4. **UNDO active** - Rollback uncommitted changes
5. **Checkpoint** - Create recovery point

```
Recovery Timeline:
─────────────────────────────────────────────────────
BEGIN T1 → WRITE → WRITE → COMMIT T1 → BEGIN T2 → WRITE → CRASH
                                                           ↑
                                        T1: REDO (committed)
                                        T2: UNDO (active, uncommitted)
```

## Configuration

```python
PMEMConfig(
    storage_path="/var/omnixan/pmem",
    size=104857600,                    # 100 MB
    persistence_mode="sync",           # sync/async/periodic
    recovery_mode="checkpoint",        # full/checkpoint/none
    enable_wal=True,
    checkpoint_interval=60.0,          # seconds
    flush_interval=1.0,                # for async mode
    max_log_size=10485760              # 10 MB WAL limit
)
```

## Metrics

```python
{
    "total_size": 104857600,
    "num_writes": 1500,
    "num_reads": 3000,
    "num_transactions": 200,
    "committed_transactions": 195,
    "aborted_transactions": 5,
    "active_transactions": 0,
    "avg_write_latency_us": 45.2,
    "recovery_time_ms": 125.0
}
```

## Integration

Part of OMNIXAN Edge Computing Network for persistent data storage at edge nodes with strong durability guarantees.
