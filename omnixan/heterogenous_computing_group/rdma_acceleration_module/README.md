# RDMA Acceleration Module

**Status: ✅ IMPLEMENTED**

Production-ready RDMA acceleration for zero-copy transfers and kernel-bypass networking.

## Features

- **Transports**: InfiniBand, RoCE v1/v2, iWARP
- **Operations**: RDMA Read, Write, Send, Atomics
- **Zero-Copy**: Direct memory access
- **Memory Registration**: Protected regions

## Quick Start

```python
from omnixan.heterogenous_computing_group.rdma_acceleration_module.module import (
    RDMAAccelerationModule, RDMAConfig, RDMATransport
)

module = RDMAAccelerationModule(RDMAConfig(transport=RDMATransport.ROCE_V2))
await module.initialize()

# Create endpoint and connect
ep = module.connection_manager.create_endpoint("192.168.1.1", 4791, RDMATransport.ROCE_V2)
conn = await module.connection_manager.connect(ep.endpoint_id, "192.168.1.2", 4791)

# Register memory
buffer = await module.register_memory(data, [MemoryAccessFlags.REMOTE_WRITE])

# RDMA write
result = await module.rdma_write(conn.connection_id, buffer.buffer_id, remote_addr, remote_rkey)

await module.shutdown()
```

## Transport Comparison

| Transport | Network | Latency |
|-----------|---------|---------|
| InfiniBand | IB fabric | ~1µs |
| RoCE v2 | Ethernet | ~2µs |
| iWARP | TCP/IP | ~10µs |

## Metrics

```python
{
    "rdma_write_ops": 5000,
    "rdma_read_ops": 3000,
    "avg_latency_us": 1.2,
    "peak_bandwidth_gbps": 95.5
}
```
