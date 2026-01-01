# Memory Pooling Module

**Status: ✅ IMPLEMENTED**

High-performance shared memory pool with efficient allocation, deallocation, and defragmentation for edge computing environments.

## Features

- **Multiple Allocation Strategies**
  - First-Fit: Use first block that fits
  - Best-Fit: Use smallest suitable block
  - Worst-Fit: Use largest block
  - Buddy System: Power-of-2 allocation
  - Slab Allocator: Fixed-size object pools

- **Memory Management**
  - Automatic defragmentation
  - Block coalescing
  - Memory alignment
  - Multi-tenant isolation

- **Performance**
  - Thread-safe operations
  - Async-compatible API
  - Low fragmentation
  - O(1) deallocation

## Quick Start

```python
from omnixan.edge_computing_network.memory_pooling_module.module import (
    MemoryPoolingModule,
    MemoryPoolConfig,
    AllocationStrategy
)

# Configure pool
config = MemoryPoolConfig(
    pool_size=1024 * 1024 * 100,  # 100 MB
    allocation_strategy=AllocationStrategy.BEST_FIT,
    enable_defragmentation=True,
    defrag_threshold=0.3
)

# Initialize module
module = MemoryPoolingModule(config)
await module.initialize()

# Allocate memory
result = await module.allocate("default", 4096, owner="tenant1")
if result.success:
    # Write data
    module.write("default", result.block_id, b"Hello, Pool!")
    
    # Read data
    data = module.read("default", result.block_id, 12)
    print(data)  # b'Hello, Pool!'

# Deallocate
await module.deallocate("default", result.block_id)

# Get metrics
metrics = module.get_metrics("default")
print(f"Utilization: {metrics['utilization']:.2%}")
print(f"Fragmentation: {metrics['fragmentation_ratio']:.2%}")

await module.shutdown()
```

## Allocation Strategies

| Strategy | Best For | Trade-off |
|----------|----------|-----------|
| First-Fit | Fast allocation | Higher fragmentation |
| Best-Fit | Low fragmentation | Slower allocation |
| Worst-Fit | Large allocations | Moderate fragmentation |
| Buddy | Power-of-2 sizes | Internal fragmentation |
| Slab | Fixed-size objects | Limited flexibility |

## Module Interface

### Configuration

```python
MemoryPoolConfig(
    pool_size=1073741824,      # Total pool size (1GB)
    min_block_size=64,         # Minimum block size
    max_block_size=104857600,  # Maximum allocation (100MB)
    allocation_strategy="best_fit",
    enable_defragmentation=True,
    defrag_threshold=0.3,
    alignment=8
)
```

### Operations

- `create_pool(name, config)` - Create named memory pool
- `allocate(pool, size, owner, pinned)` - Allocate memory block
- `deallocate(pool, block_id)` - Free memory block
- `write(pool, block_id, data, offset)` - Write to block
- `read(pool, block_id, size, offset)` - Read from block
- `defragment(pool)` - Compact memory

## Architecture

```
┌─────────────────────────────────────────────────┐
│              MemoryPoolingModule                │
├─────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐              │
│  │   Pool A    │  │   Pool B    │  ...         │
│  ├─────────────┤  ├─────────────┤              │
│  │ [Allocated] │  │ [Free     ]│              │
│  │ [Free     ]│  │ [Allocated]│              │
│  │ [Allocated] │  │ [Free     ]│              │
│  └─────────────┘  └─────────────┘              │
│         ↓                 ↓                     │
│  ┌─────────────────────────────────────────┐   │
│  │         Allocation Strategy              │   │
│  │   (Best-Fit / First-Fit / Buddy / ...)   │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

## Multi-Tenant Support

```python
# Allocate for different tenants
result1 = await module.allocate("default", 1024, owner="tenant_a")
result2 = await module.allocate("default", 2048, owner="tenant_b")

# Track allocations per tenant
# Module tracks tenant_allocations[owner] -> Set[block_ids]
```

## Metrics

```python
{
    "total_size": 1073741824,
    "used_size": 52428800,
    "free_size": 1021313024,
    "utilization": 0.049,
    "num_allocations": 150,
    "num_deallocations": 100,
    "num_free_blocks": 25,
    "num_allocated_blocks": 50,
    "fragmentation_ratio": 0.15,
    "peak_usage": 104857600,
    "allocation_failures": 0
}
```

## Integration

Part of OMNIXAN Edge Computing Network for shared memory management across edge nodes.
