# Local Traffic Shunting Module

**Status: ✅ IMPLEMENTED**

Production-ready local traffic shunting implementation that optimizes data routing by directing traffic locally when possible, reducing backhaul and improving latency.

## Features

- **Traffic Types**
  - Local: Same-site traffic
  - Regional: Nearby sites
  - Backhaul: Core network
  - Internet: External traffic
  - CDN: Content delivery
  - P2P: Peer-to-peer

- **Shunting Policies**
  - Always Local preference
  - Latency-based routing
  - Bandwidth-based selection
  - Cost-based optimization
  - Hybrid approach

- **Local Endpoints**
  - Service registration
  - Capacity tracking
  - Availability monitoring

## Quick Start

```python
from omnixan.in_memory_computing_cloud.local_traffic_shunting_module.module import (
    LocalTrafficShuntingModule,
    ShuntingConfig,
    ShuntingPolicy,
    TrafficType
)

# Initialize
config = ShuntingConfig(
    default_policy=ShuntingPolicy.HYBRID,
    local_preference_weight=0.7
)

module = LocalTrafficShuntingModule(config)
await module.initialize()

# Add routes
await module.add_route(
    source="edge_1",
    destination="cdn.local",
    route_type=TrafficType.LOCAL,
    latency_ms=2.0,
    bandwidth_mbps=10000,
    cost_per_gb=0.0
)

await module.add_route(
    source="edge_1",
    destination="cdn.local",
    route_type=TrafficType.BACKHAUL,
    latency_ms=50.0,
    bandwidth_mbps=1000,
    cost_per_gb=0.02
)

# Add local endpoint
await module.add_local_endpoint(
    name="Local CDN",
    address="192.168.1.100",
    services={"cdn.local", "video.local"}
)

# Route traffic
flow = TrafficFlow(
    flow_id="flow_1",
    source="edge_1",
    destination="cdn.local",
    size_bytes=10 * 1024 * 1024  # 10 MB
)

decision = await module.route_traffic(flow)
print(f"Routed via: {decision.route_type.value}")
print(f"Is local: {decision.is_local}")
print(f"Latency: {decision.estimated_latency_ms}ms")

# Get metrics
metrics = module.get_metrics()
print(f"Local shunt ratio: {metrics['local_shunt_ratio']*100:.1f}%")

await module.shutdown()
```

## Shunting Policies

| Policy | Description | Priority |
|--------|-------------|----------|
| Always Local | Force local routing | Local > All |
| Latency Based | Lowest latency wins | Speed |
| Bandwidth Based | Highest bandwidth | Throughput |
| Cost Based | Lowest cost | Economics |
| Hybrid | Balanced approach | All factors |

## Traffic Types

| Type | Latency | Cost | Use Case |
|------|---------|------|----------|
| Local | <5ms | Free | Same site |
| Regional | 5-20ms | Low | Nearby DC |
| Backhaul | 20-100ms | Medium | Core network |
| Internet | 50-200ms | High | External |
| CDN | Variable | Medium | Content |

## Benefits

- **Backhaul Reduction**: Keep traffic local
- **Latency Improvement**: Shorter paths
- **Cost Savings**: Avoid transit costs
- **Bandwidth Optimization**: Use local capacity

## Metrics

```python
{
    "total_flows": 10000,
    "local_flows": 7500,
    "regional_flows": 1500,
    "backhaul_flows": 1000,
    "local_shunt_ratio": 0.75,
    "total_bytes_shunted": 107374182400,  # 100 GB
    "backhaul_bytes_saved": 80530636800,  # 75 GB
    "cost_savings": 1.50,
    "total_routes": 25,
    "local_endpoints": 5
}
```

## Integration

Part of OMNIXAN In-Memory Computing Cloud for optimized traffic routing.
