# Fog Computing Module

**Status: ✅ IMPLEMENTED**

Production-ready fog computing implementation providing distributed computation between edge devices and cloud with intelligent task offloading and resource management.

## Features

- **Fog Node Management**
  - Edge, Fog, Cloud, Gateway node types
  - Resource tracking (CPU, memory, bandwidth)
  - Health monitoring

- **Task Offloading**
  - Latency-first strategy
  - Energy-first strategy
  - Cost-first strategy
  - Deadline-aware scheduling
  - Balanced optimization

- **Priority Scheduling**
  - Critical, High, Normal, Low, Background
  - Priority queue management
  - Concurrent task execution

## Quick Start

```python
from omnixan.in_memory_computing_cloud.fog_computing_module.module import (
    FogComputingModule,
    FogConfig,
    NodeType,
    TaskPriority,
    OffloadStrategy
)

# Initialize
config = FogConfig(
    offload_strategy=OffloadStrategy.BALANCED,
    latency_threshold_ms=100
)

module = FogComputingModule(config)
await module.initialize()

# Register nodes
gateway = await module.register_node(
    name="Edge Gateway",
    node_type=NodeType.GATEWAY,
    location=(40.7128, -74.0060),
    cpu_cores=4,
    memory_mb=8192,
    bandwidth_mbps=1000,
    latency_ms=5
)

fog = await module.register_node(
    name="Fog Server",
    node_type=NodeType.FOG,
    location=(40.7580, -73.9855),
    cpu_cores=16,
    memory_mb=32768,
    bandwidth_mbps=10000,
    latency_ms=10
)

# Submit tasks
task = await module.submit_task(
    name="DataProcessing",
    priority=TaskPriority.HIGH,
    compute_units=5,
    memory_mb=512,
    data_size_kb=100,
    deadline_ms=500
)

# Wait for processing
await asyncio.sleep(1)

# Get metrics
metrics = module.get_metrics()
print(f"Tasks completed: {metrics['completed_tasks']}")
print(f"Avg latency: {metrics['avg_latency_ms']:.2f}ms")

await module.shutdown()
```

## Node Types

| Type | Description | Latency | Resources |
|------|-------------|---------|-----------|
| Edge | End-user devices | <5ms | Low |
| Gateway | Edge aggregation | 5-10ms | Medium |
| Fog | Regional servers | 10-50ms | High |
| Cloud | Datacenter | 50-200ms | Very High |

## Offloading Strategies

- **Latency First**: Minimize response time
- **Energy First**: Minimize power consumption
- **Cost First**: Minimize monetary cost
- **Deadline Aware**: Meet task deadlines
- **Balanced**: Optimize across all factors

## Task Priorities

| Priority | Use Case | Queue Position |
|----------|----------|----------------|
| Critical | Emergency/Safety | First |
| High | Real-time apps | Second |
| Normal | Standard requests | Third |
| Low | Deferred work | Fourth |
| Background | Batch processing | Last |

## Metrics

```python
{
    "total_nodes": 10,
    "online_nodes": 9,
    "total_tasks": 1000,
    "completed_tasks": 985,
    "failed_tasks": 15,
    "success_rate": 0.985,
    "avg_latency_ms": 25.5,
    "total_compute_time_ms": 25000,
    "avg_resource_utilization": 0.65,
    "offload_to_cloud_ratio": 0.15,
    "queue_status": {"queued": 5, "running": 3}
}
```

## Integration

Part of OMNIXAN In-Memory Computing Cloud for distributed edge-to-cloud computing.
