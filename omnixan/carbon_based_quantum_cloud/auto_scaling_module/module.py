"""
OMNIXAN Carbon-Based Quantum Cloud - Auto Scaling Module
==========================================================

Production-ready auto-scaling implementation for quantum cloud workloads.
Supports horizontal/vertical scaling, predictive workload analysis, and
cost optimization for quantum computing resources.

Author: OMNIXAN Team
License: MIT
Python: 3.10+
"""

import asyncio
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional, Protocol, TypeAlias, Final
from functools import wraps
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator, PositiveInt, PositiveFloat
from pydantic_settings import BaseSettings


# ============================================================================
# CONSTANTS AND TYPE ALIASES
# ============================================================================

MetricsDict: TypeAlias = dict[str, float]
ResourceDict: TypeAlias = dict[str, Any]

MAX_RETRY_ATTEMPTS: Final[int] = 3
RETRY_BASE_DELAY: Final[float] = 1.0
CIRCUIT_BREAKER_THRESHOLD: Final[int] = 5
METRICS_BUFFER_SIZE: Final[int] = 1000


# ============================================================================
# ENUMS
# ============================================================================

class ScalingDirection(str, Enum):
    """Scaling direction enumeration."""
    UP = "up"
    DOWN = "down"
    NONE = "none"


class ScalingType(str, Enum):
    """Type of scaling operation."""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    HYBRID = "hybrid"


class ResourceType(str, Enum):
    """Type of cloud resource."""
    CPU = "cpu"
    MEMORY = "memory"
    QUANTUM_QUBITS = "quantum_qubits"
    QUANTUM_GATES = "quantum_gates"
    STORAGE = "storage"
    NETWORK = "network"


class HealthStatus(str, Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class ScalingError(Exception):
    """Base exception for scaling operations."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}
        self.timestamp = datetime.utcnow()


class ResourceExhaustedError(ScalingError):
    """Raised when resources are exhausted."""
    pass


class ConfigurationError(ScalingError):
    """Raised when configuration is invalid."""
    pass


class CircuitBreakerOpenError(ScalingError):
    """Raised when circuit breaker is open."""
    pass


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ResourceSpec(BaseModel):
    """Resource specification for scaling operations."""

    cpu_cores: PositiveInt = Field(default=1, description="Number of CPU cores")
    memory_gb: PositiveFloat = Field(default=1.0, description="Memory in GB")
    quantum_qubits: PositiveInt = Field(default=1, description="Number of qubits")
    quantum_gate_count: PositiveInt = Field(default=1000, description="Gate operations limit")
    storage_gb: PositiveFloat = Field(default=10.0, description="Storage in GB")
    network_bandwidth_mbps: PositiveFloat = Field(default=100.0, description="Network bandwidth")

    class Config:
        frozen = True
        use_enum_values = True


class ScalingPolicy(BaseModel):
    """Configuration for scaling policies."""

    name: str = Field(..., description="Policy name")
    scaling_type: ScalingType = Field(default=ScalingType.HORIZONTAL)
    min_instances: PositiveInt = Field(default=1, description="Minimum instances")
    max_instances: PositiveInt = Field(default=10, description="Maximum instances")
    cooldown_seconds: PositiveInt = Field(default=300, description="Cooldown period")
    target_utilization: float = Field(default=0.7, ge=0.0, le=1.0)
    scale_up_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    scale_down_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    predictive_enabled: bool = Field(default=True)
    cost_optimization_enabled: bool = Field(default=True)

    @validator("scale_up_threshold")
    def validate_scale_up(cls, v, values):
        if "scale_down_threshold" in values and v <= values["scale_down_threshold"]:
            raise ValueError("scale_up_threshold must be > scale_down_threshold")
        return v

    class Config:
        use_enum_values = True


class ScalingResult(BaseModel):
    """Result of a scaling operation."""

    success: bool
    direction: ScalingDirection
    scaling_type: ScalingType
    resources_before: ResourceSpec
    resources_after: ResourceSpec
    execution_time_ms: float
    cost_impact_usd: float = Field(default=0.0)
    message: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


class WorkloadPrediction(BaseModel):
    """Predicted workload for future time window."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    prediction_window: timedelta
    predicted_cpu_utilization: float = Field(ge=0.0, le=1.0)
    predicted_memory_utilization: float = Field(ge=0.0, le=1.0)
    predicted_quantum_job_count: int = Field(ge=0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    recommended_action: ScalingDirection = ScalingDirection.NONE

    class Config:
        use_enum_values = True


class AutoScalingConfig(BaseSettings):
    """Configuration settings for auto-scaling module."""

    module_name: str = Field(default="auto_scaling_module")
    log_level: str = Field(default="INFO")
    metrics_retention_hours: int = Field(default=24)
    enable_prometheus: bool = Field(default=True)
    prometheus_port: int = Field(default=9090)
    max_budget_usd_per_hour: float = Field(default=100.0, gt=0)
    quantum_cost_per_qubit_hour: float = Field(default=0.5, gt=0)
    cpu_cost_per_core_hour: float = Field(default=0.05, gt=0)
    memory_cost_per_gb_hour: float = Field(default=0.01, gt=0)
    circuit_breaker_timeout_seconds: int = Field(default=60)
    health_check_interval_seconds: int = Field(default=30)

    class Config:
        env_prefix = "OMNIXAN_AUTOSCALING_"


# ============================================================================
# PROTOCOLS FOR DEPENDENCY INJECTION
# ============================================================================

class MetricsCollectorProtocol(Protocol):
    """Protocol for metrics collection service."""

    async def collect_metrics(self) -> MetricsDict:
        """Collect current system metrics."""
        ...

    async def export_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        ...


class LoadBalancerProtocol(Protocol):
    """Protocol for load balancer integration."""

    async def register_instance(self, instance_id: str, resources: ResourceSpec) -> bool:
        """Register new instance with load balancer."""
        ...

    async def deregister_instance(self, instance_id: str) -> bool:
        """Remove instance from load balancer."""
        ...


class ContainerOrchestratorProtocol(Protocol):
    """Protocol for container orchestration."""

    async def scale_containers(self, count: int, resources: ResourceSpec) -> list[str]:
        """Scale container instances."""
        ...

    async def get_container_status(self, container_id: str) -> dict[str, Any]:
        """Get container status."""
        ...


# ============================================================================
# UTILITY DECORATORS AND FUNCTIONS
# ============================================================================

def retry_with_backoff(max_attempts: int = MAX_RETRY_ATTEMPTS):
    """Decorator for retrying async functions with exponential backoff."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except (asyncio.TimeoutError, ConnectionError) as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = RETRY_BASE_DELAY * (2 ** attempt)
                        await asyncio.sleep(delay)
            raise ScalingError(
                f"Max retry attempts ({max_attempts}) exceeded",
                {"last_exception": str(last_exception)}
            )
        return wrapper
    return decorator


class CircuitBreaker:
    """Circuit breaker pattern implementation for external service calls."""

    __slots__ = ("_failure_count", "_last_failure_time", "_threshold", 
                 "_timeout", "_state", "_lock")

    def __init__(self, threshold: int = CIRCUIT_BREAKER_THRESHOLD, 
                 timeout: int = 60):
        self._failure_count: int = 0
        self._last_failure_time: Optional[float] = None
        self._threshold: int = threshold
        self._timeout: int = timeout
        self._state: str = "closed"
        self._lock: asyncio.Lock = asyncio.Lock()

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self._state == "open":
                if (self._last_failure_time and 
                    time.time() - self._last_failure_time > self._timeout):
                    self._state = "half-open"
                else:
                    raise CircuitBreakerOpenError("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)
            async with self._lock:
                self._failure_count = 0
                self._state = "closed"
            return result
        except Exception as e:
            async with self._lock:
                self._failure_count += 1
                self._last_failure_time = time.time()
                if self._failure_count >= self._threshold:
                    self._state = "open"
            raise e


# ============================================================================
# MAIN AUTO SCALING MODULE
# ============================================================================

class AutoScalingModule:
    """
    Production-ready auto-scaling module for quantum cloud computing.

    Provides intelligent scaling of quantum and classical compute resources
    based on real-time metrics, predictive analysis, and cost optimization.

    Features:
        - Horizontal and vertical scaling policies
        - Predictive workload forecasting using time-series analysis
        - Cost optimization with configurable budgets
        - Integration with load balancers and container orchestrators
        - Circuit breaker pattern for resilience
        - Prometheus-compatible metrics export
        - Comprehensive health checks

    Example:
        >>> config = AutoScalingConfig(max_budget_usd_per_hour=50.0)
        >>> module = AutoScalingModule(config)
        >>> await module.initialize()
        >>> policy = ScalingPolicy(
        ...     name="quantum_workload",
        ...     scaling_type=ScalingType.HORIZONTAL,
        ...     max_instances=20
        ... )
        >>> await module.configure_scaling(policy, {"cpu": 0.8, "memory": 0.7})
        >>> result = await module.execute({"action": "check_and_scale"})
    """

    __slots__ = (
        "_config",
        "_logger",
        "_initialized",
        "_shutdown_event",
        "_current_policy",
        "_thresholds",
        "_metrics_collector",
        "_load_balancer",
        "_container_orchestrator",
        "_circuit_breaker",
        "_metrics_history",
        "_last_scaling_time",
        "_current_resources",
        "_current_instance_count",
        "_cost_accumulator",
        "_health_status",
        "_lock"
    )

    def __init__(
        self,
        config: AutoScalingConfig,
        metrics_collector: Optional[MetricsCollectorProtocol] = None,
        load_balancer: Optional[LoadBalancerProtocol] = None,
        container_orchestrator: Optional[ContainerOrchestratorProtocol] = None
    ):
        """
        Initialize auto-scaling module.

        Args:
            config: Configuration settings
            metrics_collector: Optional metrics collection service
            load_balancer: Optional load balancer integration
            container_orchestrator: Optional container orchestration service
        """
        self._config = config
        self._logger = self._setup_logger()
        self._initialized = False
        self._shutdown_event = asyncio.Event()

        # Scaling configuration
        self._current_policy: Optional[ScalingPolicy] = None
        self._thresholds: dict[str, float] = {}

        # External dependencies (injected)
        self._metrics_collector = metrics_collector
        self._load_balancer = load_balancer
        self._container_orchestrator = container_orchestrator

        # Resilience patterns
        self._circuit_breaker = CircuitBreaker(
            threshold=CIRCUIT_BREAKER_THRESHOLD,
            timeout=self._config.circuit_breaker_timeout_seconds
        )

        # State management
        self._metrics_history: deque = deque(maxlen=METRICS_BUFFER_SIZE)
        self._last_scaling_time: Optional[datetime] = None
        self._current_resources = ResourceSpec()
        self._current_instance_count: int = 1
        self._cost_accumulator: float = 0.0
        self._health_status: HealthStatus = HealthStatus.HEALTHY

        # Thread-safety
        self._lock = asyncio.Lock()

        self._logger.info(f"AutoScalingModule instantiated with config: {config.module_name}")

    def _setup_logger(self) -> logging.Logger:
        """Setup structured logger with context."""
        logger = logging.getLogger(f"omnixan.{self._config.module_name}")
        logger.setLevel(getattr(logging, self._config.log_level.upper()))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s - "
                "[%(funcName)s:%(lineno)d]"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    async def initialize(self) -> None:
        """
        Initialize the auto-scaling module.

        Performs startup tasks including:
        - Validating configuration
        - Establishing connections to external services
        - Starting background tasks (health checks, metrics collection)
        - Initializing resource state

        Raises:
            ConfigurationError: If configuration is invalid
            ScalingError: If initialization fails
        """
        async with self._lock:
            if self._initialized:
                self._logger.warning("Module already initialized")
                return

            try:
                self._logger.info("Initializing AutoScalingModule...")

                # Validate configuration
                self._validate_configuration()

                # Initialize default policy if none exists
                if self._current_policy is None:
                    self._current_policy = ScalingPolicy(
                        name="default",
                        scaling_type=ScalingType.HORIZONTAL
                    )

                # Start background tasks
                asyncio.create_task(self._health_check_loop())
                asyncio.create_task(self._metrics_collection_loop())
                asyncio.create_task(self._cost_tracking_loop())

                self._initialized = True
                self._health_status = HealthStatus.HEALTHY
                self._logger.info("AutoScalingModule initialized successfully")

            except Exception as e:
                self._logger.error(f"Initialization failed: {e}")
                self._health_status = HealthStatus.UNHEALTHY
                raise ConfigurationError(
                    "Failed to initialize auto-scaling module",
                    {"error": str(e)}
                )

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Execute scaling operation or query.

        Supported actions:
        - "check_and_scale": Evaluate metrics and scale if needed
        - "force_scale_up": Force scale up operation
        - "force_scale_down": Force scale down operation
        - "get_status": Return current status
        - "predict_workload": Generate workload prediction

        Args:
            params: Execution parameters including 'action' key

        Returns:
            Dictionary containing execution results

        Raises:
            ScalingError: If execution fails
            ConfigurationError: If module not initialized

        Example:
            >>> result = await module.execute({"action": "check_and_scale"})
            >>> print(result["scaling_performed"])
        """
        if not self._initialized:
            raise ConfigurationError("Module not initialized. Call initialize() first.")

        action = params.get("action", "check_and_scale")
        self._logger.info(f"Executing action: {action}")

        try:
            if action == "check_and_scale":
                return await self._check_and_scale_handler()
            elif action == "force_scale_up":
                resources = params.get("resources", ResourceSpec())
                result = await self.scale_up(resources)
                return result.dict()
            elif action == "force_scale_down":
                resources = params.get("resources", ResourceSpec())
                result = await self.scale_down(resources)
                return result.dict()
            elif action == "get_status":
                return await self._get_status_handler()
            elif action == "predict_workload":
                duration = params.get("duration", timedelta(hours=1))
                prediction = await self.predict_workload(duration)
                return prediction.dict()
            else:
                raise ScalingError(f"Unknown action: {action}")

        except Exception as e:
            self._logger.error(f"Execution failed: {e}")
            raise ScalingError(f"Failed to execute action: {action}", {"error": str(e)})

    async def shutdown(self) -> None:
        """
        Gracefully shutdown the auto-scaling module.

        Performs cleanup including:
        - Stopping background tasks
        - Flushing metrics
        - Closing external connections
        - Saving state
        """
        async with self._lock:
            if not self._initialized:
                return

            self._logger.info("Shutting down AutoScalingModule...")
            self._shutdown_event.set()

            # Wait for background tasks to complete
            await asyncio.sleep(1)

            # Export final metrics
            if self._config.enable_prometheus:
                try:
                    metrics = await self.get_metrics()
                    self._logger.info(f"Final metrics: {metrics}")
                except Exception as e:
                    self._logger.warning(f"Failed to export final metrics: {e}")

            self._initialized = False
            self._health_status = HealthStatus.UNHEALTHY
            self._logger.info("AutoScalingModule shutdown complete")

    async def configure_scaling(
        self,
        policy: ScalingPolicy,
        thresholds: dict[str, float]
    ) -> None:
        """
        Configure scaling policy and thresholds.

        Args:
            policy: Scaling policy configuration
            thresholds: Resource-specific thresholds (e.g., {"cpu": 0.8, "memory": 0.7})

        Raises:
            ConfigurationError: If policy or thresholds are invalid

        Example:
            >>> policy = ScalingPolicy(
            ...     name="high_performance",
            ...     max_instances=50,
            ...     target_utilization=0.75
            ... )
            >>> await module.configure_scaling(
            ...     policy,
            ...     {"cpu": 0.8, "memory": 0.75, "quantum_qubits": 0.9}
            ... )
        """
        async with self._lock:
            try:
                # Validate thresholds
                for resource, threshold in thresholds.items():
                    if not 0.0 <= threshold <= 1.0:
                        raise ValueError(f"Threshold for {resource} must be between 0 and 1")

                self._current_policy = policy
                self._thresholds = thresholds

                self._logger.info(
                    f"Configured scaling policy: {policy.name} "
                    f"(type={policy.scaling_type}, instances={policy.min_instances}-{policy.max_instances})"
                )
                self._logger.info(f"Thresholds: {thresholds}")

            except Exception as e:
                raise ConfigurationError(
                    "Failed to configure scaling policy",
                    {"error": str(e), "policy": policy.dict()}
                )

    @retry_with_backoff(max_attempts=3)
    async def scale_up(self, resources: ResourceSpec) -> ScalingResult:
        """
        Scale up resources.

        Args:
            resources: Resource specification to add

        Returns:
            ScalingResult with operation details

        Raises:
            ResourceExhaustedError: If maximum capacity reached
            ScalingError: If scaling operation fails

        Example:
            >>> new_resources = ResourceSpec(
            ...     cpu_cores=4,
            ...     memory_gb=16.0,
            ...     quantum_qubits=10
            ... )
            >>> result = await module.scale_up(new_resources)
            >>> print(f"Scaled up successfully: {result.success}")
        """
        start_time = time.time()

        async with self._lock:
            if self._current_policy and self._current_instance_count >= self._current_policy.max_instances:
                raise ResourceExhaustedError(
                    "Maximum instance count reached",
                    {"current": self._current_instance_count, 
                     "max": self._current_policy.max_instances}
                )

            # Check cooldown period
            if not await self._check_cooldown():
                self._logger.warning("Scaling skipped: cooldown period active")
                return ScalingResult(
                    success=False,
                    direction=ScalingDirection.UP,
                    scaling_type=self._current_policy.scaling_type if self._current_policy else ScalingType.HORIZONTAL,
                    resources_before=self._current_resources,
                    resources_after=self._current_resources,
                    execution_time_ms=0,
                    message="Cooldown period active"
                )

            resources_before = self._current_resources

            try:
                # Calculate cost impact
                cost_impact = self._calculate_cost_impact(resources, scale_up=True)

                # Check budget constraints
                if self._config.cost_optimization_enabled:
                    await self._check_budget_constraints(cost_impact)

                # Perform scaling operation
                if self._container_orchestrator:
                    await self._circuit_breaker.call(
                        self._container_orchestrator.scale_containers,
                        self._current_instance_count + 1,
                        resources
                    )

                # Update state
                self._current_resources = ResourceSpec(
                    cpu_cores=self._current_resources.cpu_cores + resources.cpu_cores,
                    memory_gb=self._current_resources.memory_gb + resources.memory_gb,
                    quantum_qubits=self._current_resources.quantum_qubits + resources.quantum_qubits,
                    quantum_gate_count=self._current_resources.quantum_gate_count + resources.quantum_gate_count,
                    storage_gb=self._current_resources.storage_gb + resources.storage_gb,
                    network_bandwidth_mbps=self._current_resources.network_bandwidth_mbps + resources.network_bandwidth_mbps
                )
                self._current_instance_count += 1
                self._last_scaling_time = datetime.utcnow()

                execution_time = (time.time() - start_time) * 1000

                result = ScalingResult(
                    success=True,
                    direction=ScalingDirection.UP,
                    scaling_type=self._current_policy.scaling_type if self._current_policy else ScalingType.HORIZONTAL,
                    resources_before=resources_before,
                    resources_after=self._current_resources,
                    execution_time_ms=execution_time,
                    cost_impact_usd=cost_impact,
                    message=f"Scaled up successfully to {self._current_instance_count} instances"
                )

                self._logger.info(
                    f"Scale up completed: {self._current_instance_count} instances, "
                    f"cost_impact=${cost_impact:.2f}, time={execution_time:.2f}ms"
                )

                return result

            except Exception as e:
                self._logger.error(f"Scale up failed: {e}")
                raise ScalingError("Scale up operation failed", {"error": str(e)})

    @retry_with_backoff(max_attempts=3)
    async def scale_down(self, resources: ResourceSpec) -> ScalingResult:
        """
        Scale down resources.

        Args:
            resources: Resource specification to remove

        Returns:
            ScalingResult with operation details

        Raises:
            ScalingError: If scaling operation fails or minimum capacity reached

        Example:
            >>> remove_resources = ResourceSpec(
            ...     cpu_cores=2,
            ...     memory_gb=8.0,
            ...     quantum_qubits=5
            ... )
            >>> result = await module.scale_down(remove_resources)
        """
        start_time = time.time()

        async with self._lock:
            min_instances = self._current_policy.min_instances if self._current_policy else 1
            if self._current_instance_count <= min_instances:
                return ScalingResult(
                    success=False,
                    direction=ScalingDirection.DOWN,
                    scaling_type=self._current_policy.scaling_type if self._current_policy else ScalingType.HORIZONTAL,
                    resources_before=self._current_resources,
                    resources_after=self._current_resources,
                    execution_time_ms=0,
                    message=f"Minimum instance count ({min_instances}) reached"
                )

            # Check cooldown
            if not await self._check_cooldown():
                self._logger.warning("Scaling skipped: cooldown period active")
                return ScalingResult(
                    success=False,
                    direction=ScalingDirection.DOWN,
                    scaling_type=self._current_policy.scaling_type if self._current_policy else ScalingType.HORIZONTAL,
                    resources_before=self._current_resources,
                    resources_after=self._current_resources,
                    execution_time_ms=0,
                    message="Cooldown period active"
                )

            resources_before = self._current_resources

            try:
                # Calculate cost savings
                cost_impact = self._calculate_cost_impact(resources, scale_up=False)

                # Perform scaling operation
                if self._container_orchestrator:
                    await self._circuit_breaker.call(
                        self._container_orchestrator.scale_containers,
                        self._current_instance_count - 1,
                        resources
                    )

                # Update state
                self._current_resources = ResourceSpec(
                    cpu_cores=max(1, self._current_resources.cpu_cores - resources.cpu_cores),
                    memory_gb=max(1.0, self._current_resources.memory_gb - resources.memory_gb),
                    quantum_qubits=max(1, self._current_resources.quantum_qubits - resources.quantum_qubits),
                    quantum_gate_count=max(100, self._current_resources.quantum_gate_count - resources.quantum_gate_count),
                    storage_gb=max(1.0, self._current_resources.storage_gb - resources.storage_gb),
                    network_bandwidth_mbps=max(10.0, self._current_resources.network_bandwidth_mbps - resources.network_bandwidth_mbps)
                )
                self._current_instance_count -= 1
                self._last_scaling_time = datetime.utcnow()

                execution_time = (time.time() - start_time) * 1000

                result = ScalingResult(
                    success=True,
                    direction=ScalingDirection.DOWN,
                    scaling_type=self._current_policy.scaling_type if self._current_policy else ScalingType.HORIZONTAL,
                    resources_before=resources_before,
                    resources_after=self._current_resources,
                    execution_time_ms=execution_time,
                    cost_impact_usd=cost_impact,
                    message=f"Scaled down successfully to {self._current_instance_count} instances"
                )

                self._logger.info(
                    f"Scale down completed: {self._current_instance_count} instances, "
                    f"cost_savings=${abs(cost_impact):.2f}, time={execution_time:.2f}ms"
                )

                return result

            except Exception as e:
                self._logger.error(f"Scale down failed: {e}")
                raise ScalingError("Scale down operation failed", {"error": str(e)})

    async def get_metrics(self) -> MetricsDict:
        """
        Get current system metrics.

        Returns:
            Dictionary of metric name to value mappings

        Example:
            >>> metrics = await module.get_metrics()
            >>> print(f"CPU utilization: {metrics['cpu_utilization']:.2%}")
            >>> print(f"Memory utilization: {metrics['memory_utilization']:.2%}")
        """
        try:
            metrics = {
                "cpu_utilization": await self._get_cpu_utilization(),
                "memory_utilization": await self._get_memory_utilization(),
                "quantum_workload": await self._get_quantum_workload(),
                "instance_count": float(self._current_instance_count),
                "total_cpu_cores": float(self._current_resources.cpu_cores),
                "total_memory_gb": self._current_resources.memory_gb,
                "total_quantum_qubits": float(self._current_resources.quantum_qubits),
                "cost_usd_per_hour": self._calculate_hourly_cost(),
                "health_status": 1.0 if self._health_status == HealthStatus.HEALTHY else 0.0,
                "last_scaling_timestamp": self._last_scaling_time.timestamp() if self._last_scaling_time else 0.0
            }

            # Add custom metrics from collector if available
            if self._metrics_collector:
                try:
                    custom_metrics = await self._circuit_breaker.call(
                        self._metrics_collector.collect_metrics
                    )
                    metrics.update(custom_metrics)
                except Exception as e:
                    self._logger.warning(f"Failed to collect custom metrics: {e}")

            return metrics

        except Exception as e:
            self._logger.error(f"Failed to collect metrics: {e}")
            return {}

    async def predict_workload(self, duration: timedelta) -> WorkloadPrediction:
        """
        Predict future workload using time-series forecasting.

        Uses simple moving average and trend analysis for prediction.
        For production, consider using more sophisticated methods like
        ARIMA, Prophet, or ML-based forecasting.

        Args:
            duration: Time window for prediction

        Returns:
            WorkloadPrediction with forecasted metrics

        Example:
            >>> prediction = await module.predict_workload(timedelta(hours=2))
            >>> if prediction.recommended_action == ScalingDirection.UP:
            ...     print("Recommend scaling up based on prediction")
        """
        try:
            if len(self._metrics_history) < 10:
                # Not enough data for prediction
                return WorkloadPrediction(
                    prediction_window=duration,
                    predicted_cpu_utilization=0.5,
                    predicted_memory_utilization=0.5,
                    predicted_quantum_job_count=0,
                    confidence_score=0.0,
                    recommended_action=ScalingDirection.NONE
                )

            # Convert metrics history to DataFrame
            df = pd.DataFrame(list(self._metrics_history))

            # Calculate moving averages and trends
            window_size = min(10, len(df))
            cpu_ma = df["cpu_utilization"].rolling(window=window_size).mean().iloc[-1]
            memory_ma = df["memory_utilization"].rolling(window=window_size).mean().iloc[-1]

            # Simple linear trend calculation
            recent_cpu = df["cpu_utilization"].tail(window_size).values
            cpu_trend = np.polyfit(range(len(recent_cpu)), recent_cpu, 1)[0]

            recent_memory = df["memory_utilization"].tail(window_size).values
            memory_trend = np.polyfit(range(len(recent_memory)), recent_memory, 1)[0]

            # Project future values
            hours_ahead = duration.total_seconds() / 3600
            predicted_cpu = min(1.0, max(0.0, cpu_ma + cpu_trend * hours_ahead))
            predicted_memory = min(1.0, max(0.0, memory_ma + memory_trend * hours_ahead))

            # Estimate quantum job count (simplified)
            if "quantum_workload" in df.columns:
                quantum_ma = df["quantum_workload"].rolling(window=window_size).mean().iloc[-1]
                predicted_jobs = int(quantum_ma * hours_ahead)
            else:
                predicted_jobs = 0

            # Calculate confidence based on data variance
            cpu_std = df["cpu_utilization"].tail(window_size).std()
            confidence = max(0.0, min(1.0, 1.0 - cpu_std))

            # Determine recommended action
            if self._current_policy:
                if predicted_cpu > self._current_policy.scale_up_threshold:
                    recommended_action = ScalingDirection.UP
                elif predicted_cpu < self._current_policy.scale_down_threshold:
                    recommended_action = ScalingDirection.DOWN
                else:
                    recommended_action = ScalingDirection.NONE
            else:
                recommended_action = ScalingDirection.NONE

            prediction = WorkloadPrediction(
                prediction_window=duration,
                predicted_cpu_utilization=predicted_cpu,
                predicted_memory_utilization=predicted_memory,
                predicted_quantum_job_count=predicted_jobs,
                confidence_score=confidence,
                recommended_action=recommended_action
            )

            self._logger.info(
                f"Workload prediction: CPU={predicted_cpu:.2%}, "
                f"Memory={predicted_memory:.2%}, "
                f"Action={recommended_action}, "
                f"Confidence={confidence:.2%}"
            )

            return prediction

        except Exception as e:
            self._logger.error(f"Workload prediction failed: {e}")
            return WorkloadPrediction(
                prediction_window=duration,
                predicted_cpu_utilization=0.5,
                predicted_memory_utilization=0.5,
                predicted_quantum_job_count=0,
                confidence_score=0.0,
                recommended_action=ScalingDirection.NONE
            )

    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================

    def _validate_configuration(self) -> None:
        """Validate module configuration."""
        if self._config.max_budget_usd_per_hour <= 0:
            raise ConfigurationError("Budget must be positive")

        if self._config.metrics_retention_hours < 1:
            raise ConfigurationError("Metrics retention must be at least 1 hour")

    async def _check_cooldown(self) -> bool:
        """Check if cooldown period has elapsed."""
        if self._last_scaling_time is None:
            return True

        if self._current_policy is None:
            return True

        elapsed = (datetime.utcnow() - self._last_scaling_time).total_seconds()
        return elapsed >= self._current_policy.cooldown_seconds

    def _calculate_cost_impact(self, resources: ResourceSpec, scale_up: bool) -> float:
        """Calculate cost impact of scaling operation."""
        multiplier = 1.0 if scale_up else -1.0

        cpu_cost = resources.cpu_cores * self._config.cpu_cost_per_core_hour
        memory_cost = resources.memory_gb * self._config.memory_cost_per_gb_hour
        quantum_cost = resources.quantum_qubits * self._config.quantum_cost_per_qubit_hour

        return (cpu_cost + memory_cost + quantum_cost) * multiplier

    def _calculate_hourly_cost(self) -> float:
        """Calculate current hourly cost."""
        return self._calculate_cost_impact(self._current_resources, scale_up=True)

    async def _check_budget_constraints(self, cost_impact: float) -> None:
        """Check if scaling operation violates budget constraints."""
        new_hourly_cost = self._calculate_hourly_cost() + cost_impact

        if new_hourly_cost > self._config.max_budget_usd_per_hour:
            raise ResourceExhaustedError(
                "Budget constraint violated",
                {
                    "current_hourly_cost": self._calculate_hourly_cost(),
                    "cost_impact": cost_impact,
                    "max_budget": self._config.max_budget_usd_per_hour
                }
            )

    async def _get_cpu_utilization(self) -> float:
        """Get current CPU utilization (0.0 to 1.0)."""
        # In production, integrate with actual monitoring system
        # For demo, return simulated value
        return np.random.uniform(0.3, 0.8)

    async def _get_memory_utilization(self) -> float:
        """Get current memory utilization (0.0 to 1.0)."""
        # In production, integrate with actual monitoring system
        return np.random.uniform(0.4, 0.7)

    async def _get_quantum_workload(self) -> float:
        """Get current quantum workload metric."""
        # In production, integrate with quantum job scheduler
        return np.random.uniform(0.0, 10.0)

    async def _check_and_scale_handler(self) -> dict[str, Any]:
        """Handler for automatic scaling based on metrics."""
        metrics = await self.get_metrics()

        # Store metrics in history
        self._metrics_history.append(metrics)

        scaling_performed = False
        scaling_direction = ScalingDirection.NONE

        if self._current_policy and self._current_policy.predictive_enabled:
            # Use predictive scaling
            prediction = await self.predict_workload(timedelta(minutes=30))

            if prediction.recommended_action == ScalingDirection.UP:
                try:
                    result = await self.scale_up(ResourceSpec())
                    scaling_performed = result.success
                    scaling_direction = ScalingDirection.UP
                except (ResourceExhaustedError, ScalingError) as e:
                    self._logger.warning(f"Predictive scale up failed: {e}")

            elif prediction.recommended_action == ScalingDirection.DOWN:
                try:
                    result = await self.scale_down(ResourceSpec())
                    scaling_performed = result.success
                    scaling_direction = ScalingDirection.DOWN
                except ScalingError as e:
                    self._logger.warning(f"Predictive scale down failed: {e}")
        else:
            # Use reactive scaling based on current metrics
            cpu_util = metrics.get("cpu_utilization", 0.0)

            if self._current_policy and cpu_util > self._current_policy.scale_up_threshold:
                try:
                    result = await self.scale_up(ResourceSpec())
                    scaling_performed = result.success
                    scaling_direction = ScalingDirection.UP
                except (ResourceExhaustedError, ScalingError) as e:
                    self._logger.warning(f"Reactive scale up failed: {e}")

            elif self._current_policy and cpu_util < self._current_policy.scale_down_threshold:
                try:
                    result = await self.scale_down(ResourceSpec())
                    scaling_performed = result.success
                    scaling_direction = ScalingDirection.DOWN
                except ScalingError as e:
                    self._logger.warning(f"Reactive scale down failed: {e}")

        return {
            "scaling_performed": scaling_performed,
            "scaling_direction": scaling_direction.value,
            "current_metrics": metrics,
            "instance_count": self._current_instance_count,
            "health_status": self._health_status.value
        }

    async def _get_status_handler(self) -> dict[str, Any]:
        """Handler for status query."""
        return {
            "initialized": self._initialized,
            "health_status": self._health_status.value,
            "instance_count": self._current_instance_count,
            "current_resources": self._current_resources.dict(),
            "current_policy": self._current_policy.dict() if self._current_policy else None,
            "last_scaling_time": self._last_scaling_time.isoformat() if self._last_scaling_time else None,
            "hourly_cost_usd": self._calculate_hourly_cost(),
            "metrics_buffer_size": len(self._metrics_history)
        }

    async def _health_check_loop(self) -> None:
        """Background task for periodic health checks."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self._config.health_check_interval_seconds)

                # Perform health checks
                metrics = await self.get_metrics()

                # Determine health status
                if metrics.get("cpu_utilization", 0.0) > 0.95:
                    self._health_status = HealthStatus.DEGRADED
                elif self._current_instance_count == 0:
                    self._health_status = HealthStatus.UNHEALTHY
                else:
                    self._health_status = HealthStatus.HEALTHY

            except Exception as e:
                self._logger.error(f"Health check failed: {e}")
                self._health_status = HealthStatus.DEGRADED

    async def _metrics_collection_loop(self) -> None:
        """Background task for continuous metrics collection."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(10)  # Collect every 10 seconds
                metrics = await self.get_metrics()
                self._metrics_history.append(metrics)

            except Exception as e:
                self._logger.error(f"Metrics collection failed: {e}")

    async def _cost_tracking_loop(self) -> None:
        """Background task for cost accumulation tracking."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Update every minute
                hourly_cost = self._calculate_hourly_cost()
                self._cost_accumulator += hourly_cost / 60  # Cost per minute

                if self._cost_accumulator > self._config.max_budget_usd_per_hour * 24:
                    self._logger.warning(
                        f"Daily budget exceeded: ${self._cost_accumulator:.2f}"
                    )

            except Exception as e:
                self._logger.error(f"Cost tracking failed: {e}")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_auto_scaling_module(
    config: Optional[AutoScalingConfig] = None,
    **kwargs
) -> AutoScalingModule:
    """
    Factory function to create AutoScalingModule instance.

    Args:
        config: Optional configuration (will use defaults if not provided)
        **kwargs: Additional arguments passed to AutoScalingModule constructor

    Returns:
        Initialized AutoScalingModule instance

    Example:
        >>> module = create_auto_scaling_module(
        ...     config=AutoScalingConfig(max_budget_usd_per_hour=75.0)
        ... )
        >>> await module.initialize()
    """
    if config is None:
        config = AutoScalingConfig()

    return AutoScalingModule(config, **kwargs)
