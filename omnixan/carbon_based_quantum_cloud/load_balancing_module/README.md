# OMNIXAN Load Balancing Module

Production-ready load balancing module for the carbon_based_quantum_cloud block of OMNIXAN.

## Features

- **Multiple Load Balancing Algorithms**
  - Round Robin
  - Least Connections
  - Weighted Round Robin
  - Quantum-Aware (optimized for quantum workload distribution)
  - IP Hash (for session affinity)
  - Least Latency

- **Health Monitoring**
  - Configurable health check intervals
  - Automatic backend health detection
  - Degraded state handling
  - Consecutive failure/success thresholds

- **Resilience & Fault Tolerance**
  - Circuit breaker pattern for backend protection
  - Automatic failover to healthy backends
  - Request retry logic with exponential backoff
  - Graceful degradation

- **Performance & Scalability**
  - Fully asynchronous with async/await (Python 3.10+)
  - Rate limiting with token bucket algorithm
  - DDoS protection
  - Request queuing and timeout handling
  - Connection pooling awareness

- **Session Management**
  - Session affinity (sticky sessions)
  - Automatic session cleanup

- **Observability**
  - Comprehensive metrics collection
  - Request rate, latency, and error rate tracking
  - Load distribution monitoring
  - Structured logging

- **Integration Ready**
  - Designed for integration with auto_scaling_module
  - Compatible with redundant_deployment_module
  - Extensible architecture

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import asyncio
from load_balancing_module import (
    LoadBalancingModule,
    LoadBalancingModuleConfig,
    LoadBalancingAlgorithm,
    LoadBalancingAlgorithmType,
    BackendConfig,
    Request,
    WorkloadType,
)

async def main():
    # Create configuration
    config = LoadBalancingModuleConfig(
        algorithm=LoadBalancingAlgorithm(
            algorithm_type=LoadBalancingAlgorithmType.QUANTUM_AWARE
        ),
        session_affinity=True,
        metrics_enabled=True
    )

    # Initialize module
    lb_module = LoadBalancingModule(config)
    await lb_module.initialize()

    try:
        # Add backends
        backend1 = BackendConfig(
            host="backend1.omnixan.local",
            port=8080,
            weight=1.0,
            quantum_capable=True,
            priority=10
        )
        backend1_id = await lb_module.add_backend(backend1)

        # Route a request
        request = Request(
            client_ip="192.168.1.100",
            workload_type=WorkloadType.QUANTUM_SIMULATION,
            session_id="user_session_123"
        )

        result = await lb_module.route_request(request)
        print(f"Routed to: {result.backend_id}")
        print(f"Latency: {result.latency_ms:.2f}ms")

        # Get load distribution
        distribution = await lb_module.get_load_distribution()
        print(f"Total requests: {distribution.total_requests}")

    finally:
        await lb_module.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration

### Load Balancing Module Configuration

```python
from load_balancing_module import (
    LoadBalancingModuleConfig,
    LoadBalancingAlgorithm,
    LoadBalancingAlgorithmType,
    HealthCheckConfig,
    CircuitBreakerConfig,
    RateLimitConfig,
)

config = LoadBalancingModuleConfig(
    algorithm=LoadBalancingAlgorithm(
        algorithm_type=LoadBalancingAlgorithmType.QUANTUM_AWARE,
        config={}  # Algorithm-specific configuration
    ),
    health_check=HealthCheckConfig(
        interval=5.0,  # Check every 5 seconds
        timeout=2.0,   # 2 second timeout
        unhealthy_threshold=3,  # 3 failures = unhealthy
        healthy_threshold=2     # 2 successes = healthy
    ),
    circuit_breaker=CircuitBreakerConfig(
        failure_threshold=5,      # Open after 5 failures
        success_threshold=2,      # Close after 2 successes
        timeout=60.0,            # Try again after 60 seconds
        half_open_max_calls=3    # Max calls in half-open state
    ),
    rate_limit=RateLimitConfig(
        enabled=True,
        requests_per_second=1000,
        burst_size=1500,
        ddos_threshold=5000
    ),
    request_timeout=30.0,
    max_retries=3,
    session_affinity=True,
    metrics_enabled=True
)
```

### Backend Configuration

```python
backend = BackendConfig(
    host="backend.omnixan.local",
    port=8080,
    weight=1.5,              # Routing weight (higher = more traffic)
    max_connections=1000,     # Maximum concurrent connections
    quantum_capable=True,     # Supports quantum workloads
    priority=10,             # Backend priority (higher = preferred)
    metadata={               # Custom metadata
        "datacenter": "us-east-1",
        "rack": "rack-42"
    }
)
```

## API Reference

### LoadBalancingModule

#### Methods

##### `async def initialize() -> None`
Initialize the module and start background tasks.

##### `async def shutdown() -> None`
Gracefully shutdown the module and cleanup resources.

##### `async def add_backend(backend: BackendConfig) -> BackendID`
Add a new backend to the load balancer.

**Parameters:**
- `backend`: Backend configuration

**Returns:** Backend ID (string)

##### `async def remove_backend(backend_id: BackendID) -> None`
Remove a backend from the load balancer.

**Parameters:**
- `backend_id`: Backend identifier

##### `async def route_request(request: Request) -> RouteResult`
Route a request to an appropriate backend.

**Parameters:**
- `request`: Request object

**Returns:** Routing result with backend ID and latency

**Raises:**
- `RoutingError`: If routing fails
- `BackendUnavailableError`: If no backends available

##### `async def health_check(backend_id: BackendID) -> HealthStatus`
Perform health check on a specific backend.

**Parameters:**
- `backend_id`: Backend identifier

**Returns:** Health status object

##### `async def get_load_distribution() -> LoadDistribution`
Get current load distribution across all backends.

**Returns:** Load distribution with metrics

##### `async def configure_algorithm(algorithm: LoadBalancingAlgorithm) -> None`
Change the load balancing algorithm.

**Parameters:**
- `algorithm`: Algorithm configuration

##### `async def execute(params: dict[str, Any]) -> dict[str, Any]`
Execute operations via dictionary interface (for integration).

**Parameters:**
- `params`: Operation parameters with "operation" key

**Returns:** Operation result dictionary

## Load Balancing Algorithms

### Round Robin
Distributes requests evenly across all backends in circular order.

```python
algorithm = LoadBalancingAlgorithm(
    algorithm_type=LoadBalancingAlgorithmType.ROUND_ROBIN
)
```

### Least Connections
Routes requests to the backend with the fewest active connections.

```python
algorithm = LoadBalancingAlgorithm(
    algorithm_type=LoadBalancingAlgorithmType.LEAST_CONNECTIONS
)
```

### Weighted Round Robin
Distributes requests based on backend weights.

```python
algorithm = LoadBalancingAlgorithm(
    algorithm_type=LoadBalancingAlgorithmType.WEIGHTED
)
```

### Quantum-Aware
Optimizes routing based on workload type and backend capabilities.
Considers quantum capability, priority, weight, load, and latency.

```python
algorithm = LoadBalancingAlgorithm(
    algorithm_type=LoadBalancingAlgorithmType.QUANTUM_AWARE
)
```

### IP Hash
Routes based on client IP hash for consistent session routing.

```python
algorithm = LoadBalancingAlgorithm(
    algorithm_type=LoadBalancingAlgorithmType.IP_HASH
)
```

### Least Latency
Routes to the backend with the lowest average latency.

```python
algorithm = LoadBalancingAlgorithm(
    algorithm_type=LoadBalancingAlgorithmType.LEAST_LATENCY
)
```

## Monitoring & Metrics

The module collects comprehensive metrics:

- **Per Backend:**
  - Active connections
  - Total requests
  - Successful requests
  - Failed requests
  - Average latency
  - Error rate
  - Health status
  - Circuit breaker state

- **Global:**
  - Total requests across all backends
  - Load distribution
  - Algorithm in use

Access metrics via:

```python
distribution = await lb_module.get_load_distribution()

for backend_id, info in distribution.backends.items():
    print(f"Backend: {info['host']}:{info['port']}")
    print(f"  Requests: {info['total_requests']}")
    print(f"  Success Rate: {info['successful_requests']}/{info['total_requests']}")
    print(f"  Avg Latency: {info['avg_latency_ms']:.2f}ms")
    print(f"  Error Rate: {info['error_rate']:.2%}")
    print(f"  Health: {info['health_status']}")
```

## Error Handling

The module defines custom exceptions:

- `LoadBalancingError`: Base exception
- `RoutingError`: Request routing failed
- `BackendUnavailableError`: No healthy backends available
- `AlgorithmError`: Algorithm configuration error
- `CircuitBreakerOpenError`: Circuit breaker is open

Example error handling:

```python
try:
    result = await lb_module.route_request(request)
except BackendUnavailableError:
    # No backends available, trigger scaling or alert
    await trigger_auto_scaling()
except RoutingError as e:
    # Routing failed, log and retry
    logger.error(f"Routing failed: {e}")
    await retry_with_backoff(request)
```

## Integration

### Auto Scaling Module

```python
from auto_scaling_module import AutoScalingModule

auto_scaler = AutoScalingModule()
await lb_module.integrate_auto_scaling(auto_scaler)
```

### Redundant Deployment Module

```python
from redundant_deployment_module import RedundantDeploymentModule

deployment = RedundantDeploymentModule()
await lb_module.integrate_redundant_deployment(deployment)
```

## Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
pytest test_load_balancing_module.py -v

# Run specific test class
pytest test_load_balancing_module.py::TestBackendManagement -v

# Run with coverage
pytest test_load_balancing_module.py --cov=load_balancing_module --cov-report=html
```

## Performance Considerations

- **Async Operations**: All I/O operations are asynchronous for maximum throughput
- **Lock Contention**: Minimal lock usage, granular locking where necessary
- **Memory Management**: Automatic cleanup of stale sessions and metrics
- **Background Tasks**: Health checks and metrics collection run in separate tasks

## Production Deployment

### Logging Configuration

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('load_balancer.log'),
        logging.StreamHandler()
    ]
)
```

### Monitoring Integration

Integrate with your monitoring system:

```python
async def export_metrics():
    """Export metrics to monitoring system"""
    while True:
        distribution = await lb_module.get_load_distribution()

        # Export to Prometheus, Grafana, etc.
        metrics_exporter.export(distribution)

        await asyncio.sleep(60)
```

### Health Check Endpoint

```python
from aiohttp import web

async def health_handler(request):
    """HTTP health check endpoint"""
    distribution = await lb_module.get_load_distribution()

    healthy_backends = sum(
        1 for b in distribution.backends.values()
        if b['health_status'] == 'healthy'
    )

    if healthy_backends > 0:
        return web.Response(text='OK', status=200)
    else:
        return web.Response(text='No healthy backends', status=503)
```

## License

Part of the OMNIXAN project.

## Contributing

For contributions to the OMNIXAN project, please refer to the main repository guidelines.
