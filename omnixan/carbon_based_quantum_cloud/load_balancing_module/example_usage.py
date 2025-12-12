#!/usr/bin/env python3
"""
Example usage of OMNIXAN Load Balancing Module
This script demonstrates common usage patterns.
"""

import asyncio
import logging
from load_balancing_module import (
    LoadBalancingModule,
    BackendConfig,
    Request,
    WorkloadType,
    create_quantum_aware_module,
    create_high_availability_module,
)


async def basic_example():
    """Basic usage example"""
    print("\n" + "="*80)
    print("BASIC EXAMPLE")
    print("="*80)

    # Create module with default settings
    lb = LoadBalancingModule()
    await lb.initialize()

    try:
        # Add backends
        backend1 = await lb.add_backend(BackendConfig(
            host="backend1.local",
            port=8080,
            weight=1.0
        ))

        backend2 = await lb.add_backend(BackendConfig(
            host="backend2.local",
            port=8080,
            weight=1.5
        ))

        print(f"Added backends: {backend1[:8]}..., {backend2[:8]}...")

        # Wait for health checks
        await asyncio.sleep(0.5)

        # Route some requests
        for i in range(5):
            request = Request(
                client_ip=f"192.168.1.{i}",
                workload_type=WorkloadType.CLASSICAL_COMPUTE
            )
            result = await lb.route_request(request)
            print(f"Request {i} -> Backend {result.backend_id[:8]}... "
                  f"(latency: {result.latency_ms:.2f}ms)")

        # Get load distribution
        distribution = await lb.get_load_distribution()
        print(f"\nTotal requests: {distribution.total_requests}")
        print(f"Algorithm: {distribution.algorithm.value}")

    finally:
        await lb.shutdown()


async def quantum_aware_example():
    """Quantum-aware load balancing example"""
    print("\n" + "="*80)
    print("QUANTUM-AWARE EXAMPLE")
    print("="*80)

    # Create quantum-aware module
    lb = create_quantum_aware_module(session_affinity=True)
    await lb.initialize()

    try:
        # Add quantum-capable backends
        quantum_backend = await lb.add_backend(BackendConfig(
            host="quantum1.local",
            port=8080,
            quantum_capable=True,
            priority=10,
            weight=2.0
        ))

        # Add regular backend
        regular_backend = await lb.add_backend(BackendConfig(
            host="regular1.local",
            port=8080,
            quantum_capable=False,
            priority=5,
            weight=1.0
        ))

        print(f"Quantum backend: {quantum_backend[:8]}...")
        print(f"Regular backend: {regular_backend[:8]}...")

        await asyncio.sleep(0.5)

        # Route quantum workloads
        print("\nRouting quantum workloads:")
        for i in range(3):
            request = Request(
                client_ip="192.168.1.100",
                workload_type=WorkloadType.QUANTUM_SIMULATION,
                session_id="quantum_session_1"
            )
            result = await lb.route_request(request)
            print(f"  Quantum request {i} -> {result.backend_id[:8]}...")

        # Route classical workloads
        print("\nRouting classical workloads:")
        for i in range(3):
            request = Request(
                client_ip="192.168.1.101",
                workload_type=WorkloadType.CLASSICAL_COMPUTE
            )
            result = await lb.route_request(request)
            print(f"  Classical request {i} -> {result.backend_id[:8]}...")

    finally:
        await lb.shutdown()


async def high_availability_example():
    """High availability configuration example"""
    print("\n" + "="*80)
    print("HIGH AVAILABILITY EXAMPLE")
    print("="*80)

    # Create HA module
    lb = create_high_availability_module(
        max_retries=5,
        health_check_interval=2.0
    )
    await lb.initialize()

    try:
        # Add multiple backends
        backends = []
        for i in range(4):
            backend_id = await lb.add_backend(BackendConfig(
                host=f"ha-backend{i}.local",
                port=8080,
                max_connections=1000
            ))
            backends.append(backend_id)
            print(f"Added HA backend {i}: {backend_id[:8]}...")

        await asyncio.sleep(0.5)

        # Simulate traffic
        print("\nSimulating traffic with automatic failover...")
        for i in range(10):
            request = Request(client_ip=f"192.168.1.{i % 5}")
            result = await lb.route_request(request)
            print(f"Request {i:2d} -> {result.backend_id[:8]}... "
                  f"({result.latency_ms:.2f}ms)")

        # Show distribution
        distribution = await lb.get_load_distribution()
        print("\nLoad distribution:")
        for backend_id, info in distribution.backends.items():
            print(f"  {backend_id[:8]}...: {info['total_requests']} requests, "
                  f"health={info['health_status']}")

    finally:
        await lb.shutdown()


async def metrics_example():
    """Metrics and monitoring example"""
    print("\n" + "="*80)
    print("METRICS & MONITORING EXAMPLE")
    print("="*80)

    lb = LoadBalancingModule()
    await lb.initialize()

    try:
        # Add backends
        for i in range(3):
            await lb.add_backend(BackendConfig(
                host=f"backend{i}.local",
                port=8080
            ))

        await asyncio.sleep(0.5)

        # Generate some traffic
        for _ in range(20):
            request = Request(client_ip="192.168.1.100")
            await lb.route_request(request)

        # Get detailed metrics
        distribution = await lb.get_load_distribution()

        print("\nDetailed Metrics:")
        print(f"Total Requests: {distribution.total_requests}")
        print(f"Algorithm: {distribution.algorithm.value}")
        print(f"Number of Backends: {len(distribution.backends)}")

        print("\nPer-Backend Metrics:")
        for backend_id, info in distribution.backends.items():
            print(f"\n  Backend: {info['host']}:{info['port']}")
            print(f"    ID: {backend_id[:16]}...")
            print(f"    Health: {info['health_status']}")
            print(f"    Active Connections: {info['active_connections']}")
            print(f"    Total Requests: {info['total_requests']}")
            print(f"    Success Rate: {info['successful_requests']}/{info['total_requests']}")
            print(f"    Failed Requests: {info['failed_requests']}")
            print(f"    Avg Latency: {info['avg_latency_ms']:.2f}ms")
            print(f"    Error Rate: {info['error_rate']:.2%}")
            print(f"    Circuit Breaker: {info['circuit_breaker_state']}")

    finally:
        await lb.shutdown()


async def main():
    """Run all examples"""
    # Setup logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise in examples
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*80)
    print("OMNIXAN LOAD BALANCING MODULE - EXAMPLES")
    print("="*80)

    # Run examples
    await basic_example()
    await quantum_aware_example()
    await high_availability_example()
    await metrics_example()

    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
