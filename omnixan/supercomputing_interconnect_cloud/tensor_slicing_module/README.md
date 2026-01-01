# Tensor Slicing Module

**Status: ✅ IMPLEMENTED**

Production-ready tensor slicing and partitioning for distributed tensor operations, model parallelism, and efficient memory management.

## Features

- **Slicing Strategies**
  - Row slicing (data parallelism)
  - Column slicing (model parallelism)
  - Block slicing (2D partitioning)
  - Batch slicing
  - Custom patterns

- **Parallelism Support**
  - Data parallelism
  - Model parallelism
  - Pipeline parallelism
  - Expert parallelism (MoE)

- **Gather Operations**
  - Concatenate
  - Sum reduction
  - Mean reduction
  - Max/Min reduction

## Quick Start

```python
from omnixan.supercomputing_interconnect_cloud.tensor_slicing_module.module import (
    TensorSlicingModule,
    SlicingConfig,
    SlicingStrategy,
    ReduceOperation
)
import numpy as np

# Initialize
config = SlicingConfig(
    num_devices=4,
    default_strategy=SlicingStrategy.ROW,
    balance_workload=True
)

module = TensorSlicingModule(config)
await module.initialize()

# Slice tensor
tensor = np.random.randn(1000, 512).astype(np.float32)
partition = await module.slice_tensor(
    tensor,
    strategy=SlicingStrategy.ROW,
    num_slices=4
)

# Access slices
for i, spec in enumerate(partition.slices):
    slice_data = module.get_slice(partition.partition_id, i)
    print(f"Slice {i}: shape={slice_data.shape}")

# Gather back
reconstructed = await module.gather_partition(
    partition.partition_id,
    reduce_op=ReduceOperation.CONCAT
)

await module.shutdown()
```

## Slicing Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| ROW | Split along first axis | Data parallelism |
| COLUMN | Split along second axis | Model parallelism |
| BLOCK | 2D grid partitioning | Matrix operations |
| BATCH | Split batch dimension | Batch processing |

## Example: Model Parallelism

```python
# Split weights across GPUs
weights = np.random.randn(4096, 4096).astype(np.float32)

partition = await module.slice_tensor(
    weights,
    strategy=SlicingStrategy.COLUMN,
    num_slices=4
)

# Each GPU gets a column slice
# GPU 0: weights[:, 0:1024]
# GPU 1: weights[:, 1024:2048]
# ...
```

## Overlap for Halo Exchange

```python
config = SlicingConfig(
    num_devices=4,
    enable_overlap=True,
    overlap_size=2  # 2-element halo
)

# Slices will overlap by 2 elements
partition = await module.slice_tensor(
    tensor,
    strategy=SlicingStrategy.ROW,
    overlap=2
)
```

## Metrics

```python
{
    "total_slices_created": 100,
    "total_bytes_sliced": 1073741824,
    "total_gather_operations": 25,
    "avg_slice_time_ms": 0.5,
    "avg_gather_time_ms": 1.2,
    "active_partitions": 10,
    "num_devices": 4
}
```

## Integration

Part of OMNIXAN Supercomputing Interconnect Cloud for distributed tensor operations.
