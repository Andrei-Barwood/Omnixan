"""
OMNIXAN Load Balancing Module
==============================

Production-ready load balancing module for carbon_based_quantum_cloud block.

Usage:
    from load_balancing_module import LoadBalancingModule, BackendConfig, Request

    async def main():
        lb = LoadBalancingModule()
        await lb.initialize()

        backend_id = await lb.add_backend(BackendConfig(
            host="backend.omnixan.local",
            port=8080
        ))

        result = await lb.route_request(Request(
            client_ip="192.168.1.100"
        ))

        await lb.shutdown()

Author: Kirtan Teg Singh for the OMNIXAN Project
Version: 1.0.0
Python: 3.10+
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "OMNIXAN Project"
__all__ = [
    # Main module
    "LoadBalancingModule",

    # Configuration models
    "LoadBalancingModuleConfig",
    "LoadBalancingAlgorithm",
    "LoadBalancingAlgorithmType",
    "BackendConfig",
    "HealthCheckConfig",
    "CircuitBreakerConfig",
    "RateLimitConfig",

    # Request/Response models
    "Request",
    "RouteResult",
    "HealthStatus",
    "LoadDistribution",

    # Enums
    "WorkloadType",
    "HealthStatusEnum",

    # Type aliases
    "BackendID",

    # Exceptions
    "LoadBalancingError",
    "RoutingError",
    "BackendUnavailableError",
    "AlgorithmError",
    "CircuitBreakerOpenError",

    # Internal classes (advanced usage)
    "CircuitBreaker",
    "RateLimiter",
    "BackendState",
    "BackendMetrics",
]

# Core module
from .load_balancing_module import LoadBalancingModule

# Configuration models
from .load_balancing_module import (
    LoadBalancingModuleConfig,
    LoadBalancingAlgorithm,
    LoadBalancingAlgorithmType,
    BackendConfig,
    HealthCheckConfig,
    CircuitBreakerConfig,
    RateLimitConfig,
)

# Request/Response models
from .load_balancing_module import (
    Request,
    RouteResult,
    HealthStatus,
    LoadDistribution,
)

# Enums
from .load_balancing_module import (
    WorkloadType,
    HealthStatusEnum,
)

# Type aliases
from .load_balancing_module import BackendID

# Exceptions
from .load_balancing_module import (
    LoadBalancingError,
    RoutingError,
    BackendUnavailableError,
    AlgorithmError,
    CircuitBreakerOpenError,
)

# Internal classes (for advanced usage)
from .load_balancing_module import (
    CircuitBreaker,
    RateLimiter,
    BackendState,
    BackendMetrics,
)


def get_version() -> str:
    """Get the current version of the load balancing module."""
    return __version__


def create_default_module() -> LoadBalancingModule:
    """
    Create a LoadBalancingModule with default configuration.

    Returns:
        LoadBalancingModule: Module instance with default settings

    Example:
        >>> lb = create_default_module()
        >>> await lb.initialize()
    """
    return LoadBalancingModule()


def create_quantum_aware_module(
    session_affinity: bool = True,
    rate_limit_rps: int = 1000
) -> LoadBalancingModule:
    """
    Create a LoadBalancingModule optimized for quantum workloads.

    Args:
        session_affinity: Enable sticky sessions
        rate_limit_rps: Requests per second limit

    Returns:
        LoadBalancingModule: Module configured for quantum workloads

    Example:
        >>> lb = create_quantum_aware_module(session_affinity=True)
        >>> await lb.initialize()
    """
    config = LoadBalancingModuleConfig(
        algorithm=LoadBalancingAlgorithm(
            algorithm_type=LoadBalancingAlgorithmType.QUANTUM_AWARE
        ),
        session_affinity=session_affinity,
        rate_limit=RateLimitConfig(
            enabled=True,
            requests_per_second=rate_limit_rps
        ),
        metrics_enabled=True
    )
    return LoadBalancingModule(config)


def create_high_availability_module(
    max_retries: int = 5,
    health_check_interval: float = 3.0
) -> LoadBalancingModule:
    """
    Create a LoadBalancingModule optimized for high availability.

    Args:
        max_retries: Maximum retry attempts for failed requests
        health_check_interval: Health check interval in seconds

    Returns:
        LoadBalancingModule: Module configured for high availability

    Example:
        >>> lb = create_high_availability_module(max_retries=5)
        >>> await lb.initialize()
    """
    config = LoadBalancingModuleConfig(
        algorithm=LoadBalancingAlgorithm(
            algorithm_type=LoadBalancingAlgorithmType.LEAST_CONNECTIONS
        ),
        health_check=HealthCheckConfig(
            interval=health_check_interval,
            timeout=2.0,
            unhealthy_threshold=2,
            healthy_threshold=2
        ),
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout=30.0
        ),
        max_retries=max_retries,
        metrics_enabled=True
    )
    return LoadBalancingModule(config)


# Package metadata
__doc_url__ = "https://github.com/Andrei-Barwood/Omnixan"
__license__ = "See OMNIXAN Project"

# Convenient imports for common patterns
ALGORITHM_TYPES = LoadBalancingAlgorithmType
WORKLOAD_TYPES = WorkloadType
HEALTH_STATUS = HealthStatusEnum
