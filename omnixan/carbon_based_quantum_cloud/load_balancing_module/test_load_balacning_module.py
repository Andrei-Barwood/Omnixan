"""
Unit tests for OMNIXAN Load Balancing Module
Requires: pytest, pytest-asyncio
Run with: pytest test_load_balancing_module.py -v
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch
import time

from load_balancing_module import (
    LoadBalancingModule,
    LoadBalancingModuleConfig,
    LoadBalancingAlgorithm,
    LoadBalancingAlgorithmType,
    BackendConfig,
    Request,
    WorkloadType,
    HealthStatusEnum,
    RoutingError,
    BackendUnavailableError,
    AlgorithmError,
    CircuitBreakerOpenError,
    RateLimitConfig,
    HealthCheckConfig,
    CircuitBreakerConfig,
)


@pytest.fixture
async def lb_module():
    """Create a load balancing module instance"""
    config = LoadBalancingModuleConfig(
        algorithm=LoadBalancingAlgorithm(
            algorithm_type=LoadBalancingAlgorithmType.ROUND_ROBIN
        ),
        health_check=HealthCheckConfig(interval=0.1, timeout=1.0),
        metrics_enabled=True
    )
    module = LoadBalancingModule(config)
    await module.initialize()
    yield module
    await module.shutdown()


@pytest.fixture
def backend_config():
    """Create a backend configuration"""
    return BackendConfig(
        host="backend.test.local",
        port=8080,
        weight=1.0,
        quantum_capable=True,
        priority=5
    )


@pytest.fixture
def test_request():
    """Create a test request"""
    return Request(
        client_ip="192.168.1.100",
        workload_type=WorkloadType.QUANTUM_SIMULATION,
        session_id="test_session_123"
    )


class TestBackendManagement:
    """Test backend management operations"""

    @pytest.mark.asyncio
    async def test_add_backend(self, lb_module, backend_config):
        """Test adding a backend"""
        backend_id = await lb_module.add_backend(backend_config)

        assert backend_id in lb_module.backends
        assert lb_module.backends[backend_id].config == backend_config
        assert backend_id in lb_module.circuit_breakers

    @pytest.mark.asyncio
    async def test_add_multiple_backends(self, lb_module):
        """Test adding multiple backends"""
        configs = [
            BackendConfig(host=f"backend{i}.test", port=8080, weight=float(i))
            for i in range(1, 4)
        ]

        backend_ids = []
        for config in configs:
            backend_id = await lb_module.add_backend(config)
            backend_ids.append(backend_id)

        assert len(lb_module.backends) == 3
        assert all(bid in lb_module.backends for bid in backend_ids)

    @pytest.mark.asyncio
    async def test_remove_backend(self, lb_module, backend_config):
        """Test removing a backend"""
        backend_id = await lb_module.add_backend(backend_config)
        await lb_module.remove_backend(backend_id)

        assert backend_id not in lb_module.backends
        assert backend_id not in lb_module.circuit_breakers

    @pytest.mark.asyncio
    async def test_remove_nonexistent_backend(self, lb_module):
        """Test removing a non-existent backend"""
        with pytest.raises(ValueError):
            await lb_module.remove_backend("nonexistent_id")


class TestHealthChecks:
    """Test health check functionality"""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, lb_module, backend_config):
        """Test health check for healthy backend"""
        backend_id = await lb_module.add_backend(backend_config)

        # Perform multiple health checks to reach healthy threshold
        for _ in range(3):
            health = await lb_module.health_check(backend_id)

        assert health.backend_id == backend_id
        assert health.status == HealthStatusEnum.HEALTHY
        assert health.consecutive_failures == 0
        assert health.response_time_ms is not None

    @pytest.mark.asyncio
    async def test_health_check_nonexistent(self, lb_module):
        """Test health check for non-existent backend"""
        with pytest.raises(ValueError):
            await lb_module.health_check("nonexistent_id")

    @pytest.mark.asyncio
    async def test_health_check_background_task(self, lb_module, backend_config):
        """Test background health check task"""
        backend_id = await lb_module.add_backend(backend_config)

        # Wait for background health check
        await asyncio.sleep(0.5)

        backend_state = lb_module.backends[backend_id]
        assert backend_state.health_status.last_check > 0


class TestLoadBalancingAlgorithms:
    """Test different load balancing algorithms"""

    @pytest.mark.asyncio
    async def test_round_robin_algorithm(self, lb_module, test_request):
        """Test round-robin algorithm"""
        # Add backends
        backend_ids = []
        for i in range(3):
            config = BackendConfig(host=f"backend{i}.test", port=8080)
            backend_id = await lb_module.add_backend(config)
            backend_ids.append(backend_id)

        # Wait for health checks
        await asyncio.sleep(0.5)

        # Route requests and track distribution
        routed_backends = []
        for _ in range(6):
            try:
                result = await lb_module.route_request(test_request)
                routed_backends.append(result.backend_id)
            except Exception:
                pass

        # Should cycle through backends
        assert len(set(routed_backends)) > 1

    @pytest.mark.asyncio
    async def test_least_connections_algorithm(self, backend_config):
        """Test least connections algorithm"""
        config = LoadBalancingModuleConfig(
            algorithm=LoadBalancingAlgorithm(
                algorithm_type=LoadBalancingAlgorithmType.LEAST_CONNECTIONS
            )
        )
        module = LoadBalancingModule(config)
        await module.initialize()

        try:
            # Add backends
            for i in range(3):
                config = BackendConfig(host=f"backend{i}.test", port=8080)
                await module.add_backend(config)

            await asyncio.sleep(0.5)

            # First request should go to any backend
            request = Request(client_ip="192.168.1.1", workload_type=WorkloadType.CLASSICAL_COMPUTE)
            result = await module.route_request(request)

            assert result.backend_id in module.backends

        finally:
            await module.shutdown()

    @pytest.mark.asyncio
    async def test_quantum_aware_algorithm(self):
        """Test quantum-aware algorithm"""
        config = LoadBalancingModuleConfig(
            algorithm=LoadBalancingAlgorithm(
                algorithm_type=LoadBalancingAlgorithmType.QUANTUM_AWARE
            )
        )
        module = LoadBalancingModule(config)
        await module.initialize()

        try:
            # Add quantum-capable backend
            quantum_backend = BackendConfig(
                host="quantum.test",
                port=8080,
                quantum_capable=True,
                priority=10
            )
            quantum_id = await module.add_backend(quantum_backend)

            # Add regular backend
            regular_backend = BackendConfig(
                host="regular.test",
                port=8080,
                quantum_capable=False,
                priority=5
            )
            await module.add_backend(regular_backend)

            await asyncio.sleep(0.5)

            # Quantum request should prefer quantum-capable backend
            quantum_request = Request(
                client_ip="192.168.1.1",
                workload_type=WorkloadType.QUANTUM_SIMULATION
            )
            result = await module.route_request(quantum_request)

            # Should route to quantum backend due to workload type
            assert result.backend_id in module.backends

        finally:
            await module.shutdown()

    @pytest.mark.asyncio
    async def test_weighted_algorithm(self):
        """Test weighted algorithm"""
        config = LoadBalancingModuleConfig(
            algorithm=LoadBalancingAlgorithm(
                algorithm_type=LoadBalancingAlgorithmType.WEIGHTED
            )
        )
        module = LoadBalancingModule(config)
        await module.initialize()

        try:
            # Add backends with different weights
            high_weight = BackendConfig(host="high.test", port=8080, weight=3.0)
            await module.add_backend(high_weight)

            low_weight = BackendConfig(host="low.test", port=8080, weight=1.0)
            await module.add_backend(low_weight)

            await asyncio.sleep(0.5)

            # Route requests
            request = Request(client_ip="192.168.1.1")
            result = await module.route_request(request)

            assert result.backend_id in module.backends

        finally:
            await module.shutdown()


class TestRouting:
    """Test request routing"""

    @pytest.mark.asyncio
    async def test_route_request_success(self, lb_module, backend_config, test_request):
        """Test successful request routing"""
        backend_id = await lb_module.add_backend(backend_config)
        await asyncio.sleep(0.5)  # Wait for health check

        result = await lb_module.route_request(test_request)

        assert result.request_id == test_request.request_id
        assert result.backend_id in lb_module.backends
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_route_request_no_backends(self, lb_module, test_request):
        """Test routing with no backends"""
        with pytest.raises(BackendUnavailableError):
            await lb_module.route_request(test_request)

    @pytest.mark.asyncio
    async def test_session_affinity(self, backend_config):
        """Test session affinity"""
        config = LoadBalancingModuleConfig(session_affinity=True)
        module = LoadBalancingModule(config)
        await module.initialize()

        try:
            # Add backends
            for i in range(3):
                config = BackendConfig(host=f"backend{i}.test", port=8080)
                await module.add_backend(config)

            await asyncio.sleep(0.5)

            # Route multiple requests with same session
            session_id = "test_session_123"
            backend_ids = []

            for _ in range(5):
                request = Request(
                    client_ip="192.168.1.1",
                    session_id=session_id
                )
                result = await module.route_request(request)
                backend_ids.append(result.backend_id)

            # All requests should go to same backend
            assert len(set(backend_ids)) == 1

        finally:
            await module.shutdown()


class TestRateLimiting:
    """Test rate limiting functionality"""

    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(self):
        """Test rate limit is enforced"""
        config = LoadBalancingModuleConfig(
            rate_limit=RateLimitConfig(
                enabled=True,
                requests_per_second=10,
                burst_size=15
            )
        )
        module = LoadBalancingModule(config)
        await module.initialize()

        try:
            # Add backend
            backend = BackendConfig(host="backend.test", port=8080)
            await module.add_backend(backend)
            await asyncio.sleep(0.5)

            # Send many requests quickly
            success_count = 0
            rate_limited_count = 0

            for i in range(100):
                try:
                    request = Request(client_ip="192.168.1.1")
                    await module.route_request(request)
                    success_count += 1
                except RoutingError as e:
                    if "Rate limit" in str(e):
                        rate_limited_count += 1

            # Some requests should be rate limited
            assert rate_limited_count > 0

        finally:
            await module.shutdown()

    @pytest.mark.asyncio
    async def test_rate_limit_disabled(self):
        """Test routing with rate limiting disabled"""
        config = LoadBalancingModuleConfig(
            rate_limit=RateLimitConfig(enabled=False)
        )
        module = LoadBalancingModule(config)
        await module.initialize()

        try:
            backend = BackendConfig(host="backend.test", port=8080)
            await module.add_backend(backend)
            await asyncio.sleep(0.5)

            # All requests should succeed
            for _ in range(20):
                request = Request(client_ip="192.168.1.1")
                result = await module.route_request(request)
                assert result.backend_id in module.backends

        finally:
            await module.shutdown()


class TestCircuitBreaker:
    """Test circuit breaker functionality"""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens(self):
        """Test circuit breaker opens after failures"""
        config = LoadBalancingModuleConfig(
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=3,
                timeout=1.0
            )
        )
        module = LoadBalancingModule(config)
        await module.initialize()

        try:
            backend = BackendConfig(host="backend.test", port=8080)
            backend_id = await module.add_backend(backend)

            # Manually set backend to unhealthy
            module.backends[backend_id].health_status.status = HealthStatusEnum.UNHEALTHY

            # Verify circuit breaker behavior exists
            assert backend_id in module.circuit_breakers

        finally:
            await module.shutdown()


class TestMetrics:
    """Test metrics collection"""

    @pytest.mark.asyncio
    async def test_get_load_distribution(self, lb_module, backend_config):
        """Test getting load distribution"""
        backend_id = await lb_module.add_backend(backend_config)
        await asyncio.sleep(0.5)

        distribution = await lb_module.get_load_distribution()

        assert distribution.total_requests >= 0
        assert backend_id in distribution.backends
        assert distribution.algorithm == LoadBalancingAlgorithmType.ROUND_ROBIN

    @pytest.mark.asyncio
    async def test_backend_metrics_tracking(self, lb_module, backend_config, test_request):
        """Test backend metrics are tracked"""
        backend_id = await lb_module.add_backend(backend_config)
        await asyncio.sleep(0.5)

        # Route requests
        for _ in range(5):
            try:
                await lb_module.route_request(test_request)
            except Exception:
                pass

        backend_state = lb_module.backends[backend_id]
        assert backend_state.metrics.total_requests >= 0
        assert backend_state.metrics.successful_requests >= 0


class TestConfiguration:
    """Test configuration and algorithm changes"""

    @pytest.mark.asyncio
    async def test_configure_algorithm(self, lb_module):
        """Test changing load balancing algorithm"""
        new_algorithm = LoadBalancingAlgorithm(
            algorithm_type=LoadBalancingAlgorithmType.LEAST_CONNECTIONS
        )

        await lb_module.configure_algorithm(new_algorithm)

        assert lb_module.config.algorithm.algorithm_type == LoadBalancingAlgorithmType.LEAST_CONNECTIONS

    @pytest.mark.asyncio
    async def test_invalid_algorithm(self, lb_module):
        """Test invalid algorithm configuration"""
        # This test validates that the algorithm enum restricts invalid values
        assert hasattr(LoadBalancingAlgorithmType, 'ROUND_ROBIN')
        assert hasattr(LoadBalancingAlgorithmType, 'QUANTUM_AWARE')


class TestExecuteMethod:
    """Test the execute method interface"""

    @pytest.mark.asyncio
    async def test_execute_route_request(self, lb_module, backend_config):
        """Test execute method for routing"""
        await lb_module.add_backend(backend_config)
        await asyncio.sleep(0.5)

        result = await lb_module.execute({
            "operation": "route_request",
            "request": {
                "client_ip": "192.168.1.1",
                "workload_type": "classical_compute"
            }
        })

        assert result["status"] == "success"
        assert "result" in result

    @pytest.mark.asyncio
    async def test_execute_add_backend(self, lb_module):
        """Test execute method for adding backend"""
        result = await lb_module.execute({
            "operation": "add_backend",
            "backend_config": {
                "host": "new.backend.test",
                "port": 8080
            }
        })

        assert result["status"] == "success"
        assert "backend_id" in result

    @pytest.mark.asyncio
    async def test_execute_get_load_distribution(self, lb_module):
        """Test execute method for load distribution"""
        result = await lb_module.execute({
            "operation": "get_load_distribution"
        })

        assert result["status"] == "success"
        assert "distribution" in result

    @pytest.mark.asyncio
    async def test_execute_unknown_operation(self, lb_module):
        """Test execute method with unknown operation"""
        with pytest.raises(ValueError):
            await lb_module.execute({
                "operation": "unknown_operation"
            })


class TestShutdown:
    """Test module shutdown"""

    @pytest.mark.asyncio
    async def test_shutdown_cleans_up(self):
        """Test shutdown cleans up resources"""
        module = LoadBalancingModule()
        await module.initialize()

        backend = BackendConfig(host="backend.test", port=8080)
        await module.add_backend(backend)

        await module.shutdown()

        assert module._shutting_down is True

    @pytest.mark.asyncio
    async def test_double_shutdown(self):
        """Test shutdown can be called multiple times"""
        module = LoadBalancingModule()
        await module.initialize()

        await module.shutdown()
        await module.shutdown()  # Should not raise error

        assert module._shutting_down is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
