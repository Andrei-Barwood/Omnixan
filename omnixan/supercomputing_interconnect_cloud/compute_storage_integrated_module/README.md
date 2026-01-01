# Compute-Storage Integrated Module

**Status: ✅ IMPLEMENTED**

Production-ready compute-storage integration providing unified access, intelligent data placement, and optimized data pipelines.

## Features

- **Storage Tiers**
  - L1 Cache (GPU)
  - HBM (High Bandwidth Memory)
  - DRAM (System Memory)
  - NVMe SSD
  - Object Storage

- **Data Placement**
  - Compute-local placement
  - Storage-local placement
  - Hot/Cold tiering
  - Balanced placement

- **Data Pipelines**
  - Transform operations
  - Aggregations
  - Multi-source processing

## Quick Start

```python
from omnixan.supercomputing_interconnect_cloud.compute_storage_integrated_module.module import (
    ComputeStorageIntegratedModule,
    CSIConfig,
    StorageTier,
    DataPlacement,
    OperationType
)
import numpy as np

# Initialize
config = CSIConfig(
    cache_size_mb=256,
    placement_strategy=DataPlacement.BALANCED,
    enable_tiering=True
)

module = ComputeStorageIntegratedModule(config)
await module.initialize()

# Store data
data = np.random.randn(1000, 256).astype(np.float32)
obj = await module.store("dataset1", data, StorageTier.DRAM)

# Load data (with caching)
loaded = await module.load(obj.object_id)

# Create pipeline
pipeline = await module.create_pipeline(
    stages=[
        (OperationType.TRANSFORM, {"func": "normalize"}),
        (OperationType.AGGREGATE, {"agg": "mean", "axis": 0})
    ],
    source_objects=[obj.object_id],
    target_tier=StorageTier.DRAM
)

result = await module.run_pipeline(pipeline.pipeline_id)

await module.shutdown()
```

## Storage Tiers

| Tier | Latency | Bandwidth | Capacity |
|------|---------|-----------|----------|
| L1_CACHE | ~1ns | ~TB/s | KB |
| HBM | ~100ns | 1-3 TB/s | 16-80 GB |
| DRAM | ~50ns | 100+ GB/s | 100+ GB |
| NVME | ~10µs | 3-7 GB/s | TB |
| OBJECT | ~1ms | GB/s | PB |

## Data Placement Strategies

| Strategy | Description |
|----------|-------------|
| COMPUTE_LOCAL | Place near compute resources |
| STORAGE_LOCAL | Place on storage tier |
| BALANCED | Consider both compute and storage |
| HOT_COLD | Tier based on access patterns |
| REPLICATED | Replicate across nodes |

## Pipeline Operations

```python
# Transform
(OperationType.TRANSFORM, {"func": "normalize"})
(OperationType.TRANSFORM, {"func": "sqrt"})
(OperationType.TRANSFORM, {"func": "log"})

# Aggregate
(OperationType.AGGREGATE, {"agg": "sum", "axis": 0})
(OperationType.AGGREGATE, {"agg": "mean", "axis": None})
(OperationType.AGGREGATE, {"agg": "max"})
```

## Automatic Tiering

```python
# Hot data (frequent access) promoted to faster tiers
config = CSIConfig(
    enable_tiering=True,
    hot_data_threshold=10  # Access count
)

# Data accessed >10 times → promoted to HBM
```

## Metrics

```python
{
    "total_objects": 50,
    "total_bytes_stored": 1073741824,
    "total_reads": 1000,
    "total_writes": 50,
    "cache_hit_rate": 0.85,
    "avg_read_latency_ms": 0.05,
    "avg_write_latency_ms": 0.5,
    "data_movement_bytes": 536870912,
    "cache_stats": {
        "size_bytes": 268435456,
        "utilization": 0.75,
        "entries": 25
    }
}
```

## Integration

Part of OMNIXAN Supercomputing Interconnect Cloud for unified compute-storage access.
