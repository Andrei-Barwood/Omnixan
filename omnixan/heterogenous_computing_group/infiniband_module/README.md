# InfiniBand Module

**Status: ✅ IMPLEMENTED**

Production-ready InfiniBand networking module for high-performance computing with low-latency interconnect and RDMA support.

## Features

- **IB Speeds**: SDR to XDR (2.5 - 200 Gbps per lane)
- **Link Widths**: 1x, 4x, 8x, 12x
- **Queue Pair Types**: RC, UC, UD, XRC
- **RDMA Operations**: Read, Write, Atomics

## Quick Start

```python
from omnixan.heterogenous_computing_group.infiniband_module.module import (
    InfiniBandModule, IBConfig, IBSpeed
)

module = InfiniBandModule(IBConfig(default_speed=IBSpeed.HDR))
await module.initialize()

# Create queue pairs
qp1 = await module.create_queue_pair()
qp2 = await module.create_queue_pair()

# Connect
await module.connect_queue_pair(qp1.qp_id, qp2.local_port.lid, qp2.qp_num)

# Register memory and do RDMA
mr = await module.register_memory_region(4096)
await module.rdma_write(qp1.qp_id, mr.mr_id, 0x1000, 0x2000, data)

await module.shutdown()
```

## Speed Rates

| Speed | Per Lane | 4x Link |
|-------|----------|---------|
| HDR | 50 Gbps | 200 Gbps |
| NDR | 100 Gbps | 400 Gbps |
| XDR | 200 Gbps | 800 Gbps |

## Metrics

```python
{
    "rdma_write_ops": 1000,
    "rdma_read_ops": 500,
    "avg_latency_us": 0.8,
    "peak_bandwidth_gbps": 180.5
}
```
