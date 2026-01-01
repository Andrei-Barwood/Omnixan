# Low Latency Routing Module

**Status: ✅ IMPLEMENTED**

Production-ready low-latency routing implementation optimized for real-time applications with adaptive path selection, QoS guarantees, and congestion avoidance.

## Features

- **Routing Algorithms**
  - Dijkstra's shortest path
  - Bellman-Ford (negative weights)
  - ECMP (Equal-Cost Multi-Path)
  - Segment routing
  - SDN-based routing

- **QoS Classes**
  - Realtime: Ultra-low latency
  - Interactive: Low latency
  - Bulk: Best effort
  - Background: Lowest priority

- **Network Management**
  - Dynamic topology updates
  - Congestion detection
  - Path caching
  - Automatic rerouting

## Quick Start

```python
from omnixan.in_memory_computing_cloud.low_latency_routing_module.module import (
    LowLatencyRoutingModule,
    RoutingConfig,
    RoutingAlgorithm,
    QoSClass
)

# Initialize
config = RoutingConfig(
    algorithm=RoutingAlgorithm.DIJKSTRA,
    max_latency_us=500,
    enable_ecmp=True
)

module = LowLatencyRoutingModule(config)
await module.initialize()

# Build topology
nodes = {}
for name in ["edge", "switch", "router", "server"]:
    node = await module.add_node(name=name)
    nodes[name] = node.node_id

# Add links (microsecond latencies)
await module.add_link(nodes["edge"], nodes["switch"], latency_us=50)
await module.add_link(nodes["switch"], nodes["router"], latency_us=30)
await module.add_link(nodes["router"], nodes["server"], latency_us=20)

# Find path
path = await module.find_path(
    nodes["edge"],
    nodes["server"],
    qos=QoSClass.REALTIME
)

print(f"Latency: {path.total_latency_us:.2f}µs")
print(f"Hops: {path.total_hops}")

# Find ECMP paths
paths = await module.find_ecmp_paths(
    nodes["edge"],
    nodes["server"]
)
print(f"Alternative paths: {len(paths)}")

await module.shutdown()
```

## Routing Algorithms

| Algorithm | Best For | Complexity |
|-----------|----------|------------|
| Dijkstra | General purpose | O(E + V log V) |
| Bellman-Ford | Negative weights | O(V × E) |
| ECMP | Load balancing | O(k × Dijkstra) |
| Segment | Traffic engineering | O(V + E) |

## QoS Classes

| Class | Max Latency | Use Case |
|-------|-------------|----------|
| Realtime | 100µs | Gaming, VoIP |
| Interactive | 1ms | Web, API |
| Bulk | 10ms | Downloads |
| Background | 100ms | Updates |

## Link Properties

```python
NetworkLink(
    source="node_a",
    target="node_b",
    latency_us=100,        # Propagation delay
    bandwidth_gbps=10,     # Link capacity
    utilization=0.3,       # Current load
    jitter_us=5,           # Latency variance
    packet_loss=0.001      # Loss rate
)
```

## Path Selection Factors

1. **Total Latency**: Sum of link delays + processing
2. **Congestion**: Penalize loaded links
3. **Jitter**: Important for realtime
4. **Hop Count**: Fewer is better
5. **Bandwidth**: Meet minimum requirements

## Metrics

```python
{
    "total_nodes": 100,
    "total_links": 250,
    "total_routes": 5000,
    "cached_paths": 150,
    "avg_latency_us": 125.5,
    "min_latency_us": 50.0,
    "max_latency_us": 500.0,
    "path_failures": 5,
    "reroutes": 25
}
```

## Congestion Handling

```
Link Utilization    →    Weight Multiplier
     < 50%          →         1.0x
     50-70%         →         1.5x
     70-90%         →         2.0x
     > 90%          →         5.0x
```

## Integration

Part of OMNIXAN In-Memory Computing Cloud for ultra-low latency networking.
