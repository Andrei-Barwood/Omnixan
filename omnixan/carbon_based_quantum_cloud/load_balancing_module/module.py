"""
OMNIXAN Load Balancing Module
Production-ready implementation for carbon_based_quantum_cloud block
Author: Kirtan Teg Singh for OMNIXAN Project
Python Version: 3.10+
"""

from __future__ import annotations

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Set, Literal
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from uuid import uuid4, UUID

from pydantic import BaseModel, Field, ConfigDict, field_validator
from pydantic.types import PositiveInt, PositiveFloat, NonNegativeFloat


# ==================== Type Definitions ====================

class BackendID(str):
    """Type alias for backend identifier"""
    pass


class LoadBalancingAlgorithmType(str, Enum):
    """Supported load balancing algorithms"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    QUANTUM_AWARE = "quantum_aware"
    IP_HASH = "ip_hash"
    LEAST_LATENCY = "least_latency"


class HealthStatusEnum(str, Enum):
    """Backend health states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class WorkloadType(str, Enum):
    """Quantum workload types"""
    QUANTUM_SIMULATION = "quantum_simulation"
    CLASSICAL_COMPUTE = "classical_compute"
    HYBRID = "hybrid"
    DATA_PROCESSING = "data_processing"


# ==================== Custom Exceptions ====================

class LoadBalancingError(Exception):
    """Base exception for load balancing errors"""
    pass


class RoutingError(LoadBalancingError):
    """Exception raised when routing fails"""
    pass


class BackendUnavailableError(LoadBalancingError):
    """Exception raised when no backends are available"""
    pass


class AlgorithmError(LoadBalancingError):
    """Exception raised for algorithm configuration errors"""
    pass


class CircuitBreakerOpenError(LoadBalancingError):
    """Exception raised when circuit breaker is open"""
    pass


# ==================== Configuration Models ====================

class BackendConfig(BaseModel):
    """Configuration for a backend server"""
    model_config = ConfigDict(frozen=True)

    host: str = Field(..., description="Backend host address")
    port: PositiveInt = Field(..., description="Backend port")
    weight: PositiveFloat = Field(default=1.0, description="Routing weight")
    max_connections: PositiveInt = Field(default=1000, description="Max concurrent connections")
    quantum_capable: bool = Field(default=False, description="Supports quantum workloads")
    priority: int = Field(default=0, description="Backend priority (higher = preferred)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator('host')
    @classmethod
    def validate_host(cls, v: str) -> str:
        if not v or v.isspace():
            raise ValueError("Host cannot be empty")
        return v


class LoadBalancingAlgorithm(BaseModel):
    """Load balancing algorithm configuration"""
    model_config = ConfigDict(frozen=True)

    algorithm_type: LoadBalancingAlgorithmType
    config: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('config')
    @classmethod
    def validate_config(cls, v: Dict[str, Any], info) -> Dict[str, Any]:
        # Validate algorithm-specific configuration
        return v


class HealthCheckConfig(BaseModel):
    """Health check configuration"""
    model_config = ConfigDict(frozen=True)

    interval: PositiveFloat = Field(default=5.0, description="Check interval in seconds")
    timeout: PositiveFloat = Field(default=2.0, description="Check timeout in seconds")
    unhealthy_threshold: PositiveInt = Field(default=3, description="Failures before unhealthy")
    healthy_threshold: PositiveInt = Field(default=2, description="Successes before healthy")


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration"""
    model_config = ConfigDict(frozen=True)

    failure_threshold: PositiveInt = Field(default=5, description="Failures to open circuit")
    success_threshold: PositiveInt = Field(default=2, description="Successes to close circuit")
    timeout: PositiveFloat = Field(default=60.0, description="Timeout before half-open (seconds)")
    half_open_max_calls: PositiveInt = Field(default=3, description="Max calls in half-open state")


class RateLimitConfig(BaseModel):
    """Rate limiting configuration"""
    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=True)
    requests_per_second: PositiveInt = Field(default=1000)
    burst_size: PositiveInt = Field(default=1500)
    ddos_threshold: PositiveInt = Field(default=5000, description="DDoS detection threshold")


class LoadBalancingModuleConfig(BaseModel):
    """Main module configuration"""
    model_config = ConfigDict(frozen=False)

    algorithm: LoadBalancingAlgorithm = Field(
        default_factory=lambda: LoadBalancingAlgorithm(
            algorithm_type=LoadBalancingAlgorithmType.ROUND_ROBIN
        )
    )
    health_check: HealthCheckConfig = Field(default_factory=HealthCheckConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    request_timeout: PositiveFloat = Field(default=30.0, description="Request timeout (seconds)")
    max_retries: PositiveInt = Field(default=3, description="Maximum retry attempts")
    session_affinity: bool = Field(default=False, description="Enable sticky sessions")
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")


# ==================== Data Models ====================

class Request(BaseModel):
    """Incoming request model"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str = Field(default_factory=lambda: str(uuid4()))
    client_ip: str
    workload_type: WorkloadType = Field(default=WorkloadType.CLASSICAL_COMPUTE)
    payload: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=0)
    session_id: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)


class RouteResult(BaseModel):
    """Routing result"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str
    backend_id: BackendID
    backend_config: BackendConfig
    latency_ms: float
    algorithm_used: LoadBalancingAlgorithmType
    timestamp: float = Field(default_factory=time.time)


class HealthStatus(BaseModel):
    """Backend health status"""
    backend_id: BackendID
    status: HealthStatusEnum
    last_check: float
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None


class LoadDistribution(BaseModel):
    """Current load distribution across backends"""
    timestamp: float = Field(default_factory=time.time)
    backends: Dict[BackendID, Dict[str, Any]]
    total_requests: int
    algorithm: LoadBalancingAlgorithmType


@dataclass
class BackendMetrics:
    """Runtime metrics for a backend"""
    active_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    last_request_time: Optional[float] = None
    error_rate: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests


@dataclass
class BackendState:
    """Complete backend state"""
    config: BackendConfig
    backend_id: BackendID
    health_status: HealthStatus
    metrics: BackendMetrics
    circuit_breaker_state: Literal["closed", "open", "half_open"] = "closed"
    circuit_breaker_open_until: Optional[float] = None
    circuit_breaker_failure_count: int = 0
    circuit_breaker_success_count: int = 0
    session_affinity_map: Dict[str, str] = field(default_factory=dict)


# ==================== Circuit Breaker ====================

class CircuitBreaker:
    """Circuit breaker for backend protection"""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state: Literal["closed", "open", "half_open"] = "closed"
        self.failure_count = 0
        self.success_count = 0
        self.open_until: Optional[float] = None
        self.half_open_calls = 0
        self._lock = asyncio.Lock()

    async def call(self, func, *args, **kwargs):
        """Execute function through circuit breaker"""
        async with self._lock:
            current_time = time.time()

            # Check if circuit should transition from open to half-open
            if self.state == "open":
                if self.open_until and current_time >= self.open_until:
                    self.state = "half_open"
                    self.half_open_calls = 0
                    self.success_count = 0
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker open until {self.open_until}"
                    )

            # Limit calls in half-open state
            if self.state == "half_open":
                if self.half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitBreakerOpenError("Half-open call limit reached")
                self.half_open_calls += 1

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise

    async def _on_success(self):
        """Handle successful call"""
        async with self._lock:
            self.failure_count = 0

            if self.state == "half_open":
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = "closed"
                    self.success_count = 0

    async def _on_failure(self):
        """Handle failed call"""
        async with self._lock:
            self.success_count = 0
            self.failure_count += 1

            if self.failure_count >= self.config.failure_threshold:
                self.state = "open"
                self.open_until = time.time() + self.config.timeout


# ==================== Rate Limiter ====================

class RateLimiter:
    """Token bucket rate limiter with DDoS protection"""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.tokens = float(config.burst_size)
        self.last_update = time.time()
        self._lock = asyncio.Lock()
        self.ip_request_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

    async def acquire(self, client_ip: str, count: int = 1) -> bool:
        """Acquire tokens for request"""
        if not self.config.enabled:
            return True

        # DDoS detection
        if self._check_ddos(client_ip):
            return False

        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update

            # Refill tokens
            self.tokens = min(
                self.config.burst_size,
                self.tokens + elapsed * self.config.requests_per_second
            )
            self.last_update = now

            if self.tokens >= count:
                self.tokens -= count
                return True
            return False

    def _check_ddos(self, client_ip: str) -> bool:
        """Check for DDoS patterns"""
        now = time.time()
        request_times = self.ip_request_counts[client_ip]

        # Remove old requests (older than 1 second)
        while request_times and now - request_times[0] > 1.0:
            request_times.popleft()

        # Add current request
        request_times.append(now)

        # Check if exceeds threshold
        return len(request_times) > self.config.ddos_threshold


# ==================== Load Balancing Algorithms ====================

class LoadBalancingAlgorithmBase:
    """Base class for load balancing algorithms"""

    def __init__(self, backends: Dict[BackendID, BackendState]):
        self.backends = backends

    async def select_backend(
        self,
        request: Request,
        available_backends: List[BackendID]
    ) -> BackendID:
        """Select backend for request"""
        raise NotImplementedError


class RoundRobinAlgorithm(LoadBalancingAlgorithmBase):
    """Round-robin algorithm"""

    def __init__(self, backends: Dict[BackendID, BackendState]):
        super().__init__(backends)
        self.current_index = 0
        self._lock = asyncio.Lock()

    async def select_backend(
        self,
        request: Request,
        available_backends: List[BackendID]
    ) -> BackendID:
        if not available_backends:
            raise BackendUnavailableError("No available backends")

        async with self._lock:
            backend_id = available_backends[self.current_index % len(available_backends)]
            self.current_index += 1
            return backend_id


class LeastConnectionsAlgorithm(LoadBalancingAlgorithmBase):
    """Least connections algorithm"""

    async def select_backend(
        self,
        request: Request,
        available_backends: List[BackendID]
    ) -> BackendID:
        if not available_backends:
            raise BackendUnavailableError("No available backends")

        min_connections = float('inf')
        selected_backend = None

        for backend_id in available_backends:
            backend = self.backends[backend_id]
            connections = backend.metrics.active_connections

            if connections < min_connections:
                min_connections = connections
                selected_backend = backend_id

        if selected_backend is None:
            return available_backends[0]

        return selected_backend


class WeightedAlgorithm(LoadBalancingAlgorithmBase):
    """Weighted round-robin algorithm"""

    def __init__(self, backends: Dict[BackendID, BackendState]):
        super().__init__(backends)
        self.current_weights: Dict[BackendID, float] = {}
        self._lock = asyncio.Lock()

    async def select_backend(
        self,
        request: Request,
        available_backends: List[BackendID]
    ) -> BackendID:
        if not available_backends:
            raise BackendUnavailableError("No available backends")

        async with self._lock:
            # Initialize weights if needed
            for backend_id in available_backends:
                if backend_id not in self.current_weights:
                    self.current_weights[backend_id] = 0.0

            # Find backend with highest current weight
            max_weight = -1.0
            selected_backend = None
            total_weight = 0.0

            for backend_id in available_backends:
                backend = self.backends[backend_id]
                weight = backend.config.weight
                total_weight += weight

                self.current_weights[backend_id] += weight

                if self.current_weights[backend_id] > max_weight:
                    max_weight = self.current_weights[backend_id]
                    selected_backend = backend_id

            if selected_backend:
                self.current_weights[selected_backend] -= total_weight
                return selected_backend

            return available_backends[0]


class QuantumAwareAlgorithm(LoadBalancingAlgorithmBase):
    """Quantum-aware load balancing"""

    async def select_backend(
        self,
        request: Request,
        available_backends: List[BackendID]
    ) -> BackendID:
        if not available_backends:
            raise BackendUnavailableError("No available backends")

        # Filter quantum-capable backends for quantum workloads
        if request.workload_type == WorkloadType.QUANTUM_SIMULATION:
            quantum_backends = [
                bid for bid in available_backends
                if self.backends[bid].config.quantum_capable
            ]
            if quantum_backends:
                available_backends = quantum_backends

        # Select based on priority, weight, and current load
        best_score = -1.0
        selected_backend = None

        for backend_id in available_backends:
            backend = self.backends[backend_id]

            # Calculate score based on multiple factors
            load_factor = 1.0 - (
                backend.metrics.active_connections / backend.config.max_connections
            )
            weight_factor = backend.config.weight
            priority_factor = backend.config.priority
            latency_factor = 1.0 / (1.0 + backend.metrics.avg_latency_ms)

            score = (
                load_factor * 0.3 +
                weight_factor * 0.2 +
                priority_factor * 0.3 +
                latency_factor * 0.2
            )

            if score > best_score:
                best_score = score
                selected_backend = backend_id

        return selected_backend if selected_backend else available_backends[0]


class IPHashAlgorithm(LoadBalancingAlgorithmBase):
    """IP hash algorithm for session affinity"""

    async def select_backend(
        self,
        request: Request,
        available_backends: List[BackendID]
    ) -> BackendID:
        if not available_backends:
            raise BackendUnavailableError("No available backends")

        # Hash client IP to select backend
        hash_value = hash(request.client_ip)
        index = hash_value % len(available_backends)
        return available_backends[index]


class LeastLatencyAlgorithm(LoadBalancingAlgorithmBase):
    """Least latency algorithm"""

    async def select_backend(
        self,
        request: Request,
        available_backends: List[BackendID]
    ) -> BackendID:
        if not available_backends:
            raise BackendUnavailableError("No available backends")

        min_latency = float('inf')
        selected_backend = None

        for backend_id in available_backends:
            backend = self.backends[backend_id]
            latency = backend.metrics.avg_latency_ms

            if latency < min_latency or (latency == 0 and min_latency == float('inf')):
                min_latency = latency
                selected_backend = backend_id

        return selected_backend if selected_backend else available_backends[0]


# ==================== Main Load Balancing Module ====================

class LoadBalancingModule:
    """Production-ready load balancing module for OMNIXAN"""

    def __init__(self, config: Optional[LoadBalancingModuleConfig] = None):
        self.config = config or LoadBalancingModuleConfig()
        self.backends: Dict[BackendID, BackendState] = {}
        self.circuit_breakers: Dict[BackendID, CircuitBreaker] = {}
        self.rate_limiter = RateLimiter(self.config.rate_limit)
        self.algorithm: Optional[LoadBalancingAlgorithmBase] = None
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.session_affinity_map: Dict[str, BackendID] = {}

        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None

        # State
        self._initialized = False
        self._shutting_down = False
        self._lock = asyncio.Lock()

        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    async def initialize(self) -> None:
        """Initialize the load balancing module"""
        if self._initialized:
            self.logger.warning("Module already initialized")
            return

        self.logger.info("Initializing LoadBalancingModule")

        try:
            # Set up algorithm
            await self.configure_algorithm(self.config.algorithm)

            # Start background tasks
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            if self.config.metrics_enabled:
                self._metrics_task = asyncio.create_task(self._metrics_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

            self._initialized = True
            self.logger.info("LoadBalancingModule initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize module: {e}")
            raise

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute load balancing operation"""
        if not self._initialized:
            raise LoadBalancingError("Module not initialized")

        operation = params.get("operation")

        if operation == "route_request":
            request = Request(**params.get("request", {}))
            result = await self.route_request(request)
            return {"status": "success", "result": result.model_dump()}

        elif operation == "add_backend":
            backend_config = BackendConfig(**params.get("backend_config", {}))
            backend_id = await self.add_backend(backend_config)
            return {"status": "success", "backend_id": backend_id}

        elif operation == "remove_backend":
            backend_id = params.get("backend_id")
            await self.remove_backend(backend_id)
            return {"status": "success"}

        elif operation == "get_load_distribution":
            distribution = await self.get_load_distribution()
            return {"status": "success", "distribution": distribution.model_dump()}

        elif operation == "health_check":
            backend_id = params.get("backend_id")
            health = await self.health_check(backend_id)
            return {"status": "success", "health": health.model_dump()}

        else:
            raise ValueError(f"Unknown operation: {operation}")

    async def shutdown(self) -> None:
        """Shutdown the load balancing module"""
        if not self._initialized or self._shutting_down:
            return

        self.logger.info("Shutting down LoadBalancingModule")
        self._shutting_down = True

        try:
            # Cancel background tasks
            tasks = [
                self._health_check_task,
                self._metrics_task,
                self._cleanup_task
            ]
            for task in tasks:
                if task:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            # Wait for pending requests
            while not self.request_queue.empty():
                await asyncio.sleep(0.1)

            self.logger.info("LoadBalancingModule shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            raise

    async def add_backend(self, backend: BackendConfig) -> BackendID:
        """Add a new backend"""
        backend_id = BackendID(str(uuid4()))

        async with self._lock:
            backend_state = BackendState(
                config=backend,
                backend_id=backend_id,
                health_status=HealthStatus(
                    backend_id=backend_id,
                    status=HealthStatusEnum.UNKNOWN,
                    last_check=time.time()
                ),
                metrics=BackendMetrics()
            )

            self.backends[backend_id] = backend_state
            self.circuit_breakers[backend_id] = CircuitBreaker(
                self.config.circuit_breaker
            )

            self.logger.info(
                f"Added backend {backend_id}: {backend.host}:{backend.port}"
            )

        # Perform initial health check
        await self.health_check(backend_id)

        return backend_id

    async def remove_backend(self, backend_id: BackendID) -> None:
        """Remove a backend"""
        async with self._lock:
            if backend_id not in self.backends:
                raise ValueError(f"Backend {backend_id} not found")

            # Remove from session affinity map
            self.session_affinity_map = {
                k: v for k, v in self.session_affinity_map.items()
                if v != backend_id
            }

            del self.backends[backend_id]
            del self.circuit_breakers[backend_id]

            self.logger.info(f"Removed backend {backend_id}")

    async def get_load_distribution(self) -> LoadDistribution:
        """Get current load distribution"""
        backends_info = {}
        total_requests = 0

        for backend_id, backend_state in self.backends.items():
            metrics = backend_state.metrics
            total_requests += metrics.total_requests

            backends_info[backend_id] = {
                "host": backend_state.config.host,
                "port": backend_state.config.port,
                "active_connections": metrics.active_connections,
                "total_requests": metrics.total_requests,
                "successful_requests": metrics.successful_requests,
                "failed_requests": metrics.failed_requests,
                "avg_latency_ms": metrics.avg_latency_ms,
                "error_rate": metrics.error_rate,
                "health_status": backend_state.health_status.status.value,
                "circuit_breaker_state": backend_state.circuit_breaker_state
            }

        return LoadDistribution(
            backends=backends_info,
            total_requests=total_requests,
            algorithm=self.config.algorithm.algorithm_type
        )

    async def route_request(self, request: Request) -> RouteResult:
        """Route a request to a backend"""
        start_time = time.time()

        # Rate limiting
        if not await self.rate_limiter.acquire(request.client_ip):
            raise RoutingError("Rate limit exceeded")

        # Session affinity check
        if self.config.session_affinity and request.session_id:
            if request.session_id in self.session_affinity_map:
                backend_id = self.session_affinity_map[request.session_id]
                if backend_id in self.backends:
                    backend_state = self.backends[backend_id]
                    if backend_state.health_status.status == HealthStatusEnum.HEALTHY:
                        return await self._execute_route(
                            request, backend_id, start_time
                        )

        # Get available backends
        available_backends = await self._get_available_backends()
        if not available_backends:
            raise BackendUnavailableError("No healthy backends available")

        # Retry logic
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                # Select backend using algorithm
                backend_id = await self.algorithm.select_backend(
                    request, available_backends
                )

                # Execute route with circuit breaker
                result = await self._execute_route_with_circuit_breaker(
                    request, backend_id, start_time
                )

                # Update session affinity
                if self.config.session_affinity and request.session_id:
                    self.session_affinity_map[request.session_id] = backend_id

                return result

            except (CircuitBreakerOpenError, RoutingError) as e:
                last_error = e
                self.logger.warning(
                    f"Routing attempt {attempt + 1} failed: {e}"
                )

                # Remove failed backend from available list
                if backend_id in available_backends:
                    available_backends.remove(backend_id)

                if not available_backends:
                    break

                # Exponential backoff
                await asyncio.sleep(0.1 * (2 ** attempt))

        raise RoutingError(
            f"Failed to route request after {self.config.max_retries} attempts: {last_error}"
        )

    async def _execute_route_with_circuit_breaker(
        self,
        request: Request,
        backend_id: BackendID,
        start_time: float
    ) -> RouteResult:
        """Execute routing with circuit breaker"""
        circuit_breaker = self.circuit_breakers[backend_id]

        return await circuit_breaker.call(
            self._execute_route, request, backend_id, start_time
        )

    async def _execute_route(
        self,
        request: Request,
        backend_id: BackendID,
        start_time: float
    ) -> RouteResult:
        """Execute actual routing to backend"""
        backend_state = self.backends[backend_id]

        # Check connection limit
        if (backend_state.metrics.active_connections >=
                backend_state.config.max_connections):
            raise RoutingError("Backend connection limit reached")

        # Update metrics
        backend_state.metrics.active_connections += 1
        backend_state.metrics.total_requests += 1

        try:
            # Simulate backend call (replace with actual implementation)
            await asyncio.sleep(0.01)  # Simulated latency

            # Success metrics
            latency_ms = (time.time() - start_time) * 1000
            backend_state.metrics.successful_requests += 1
            backend_state.metrics.total_latency_ms += latency_ms
            backend_state.metrics.last_request_time = time.time()

            return RouteResult(
                request_id=request.request_id,
                backend_id=backend_id,
                backend_config=backend_state.config,
                latency_ms=latency_ms,
                algorithm_used=self.config.algorithm.algorithm_type
            )

        except Exception as e:
            backend_state.metrics.failed_requests += 1
            raise RoutingError(f"Backend error: {e}")

        finally:
            backend_state.metrics.active_connections -= 1

            # Update error rate
            total = backend_state.metrics.total_requests
            if total > 0:
                backend_state.metrics.error_rate = (
                    backend_state.metrics.failed_requests / total
                )

    async def health_check(self, backend_id: BackendID) -> HealthStatus:
        """Perform health check on a backend"""
        if backend_id not in self.backends:
            raise ValueError(f"Backend {backend_id} not found")

        backend_state = self.backends[backend_id]
        start_time = time.time()

        try:
            # Simulate health check (replace with actual implementation)
            await asyncio.wait_for(
                asyncio.sleep(0.01),
                timeout=self.config.health_check.timeout
            )

            response_time = (time.time() - start_time) * 1000

            # Update health status
            backend_state.health_status.consecutive_successes += 1
            backend_state.health_status.consecutive_failures = 0
            backend_state.health_status.response_time_ms = response_time
            backend_state.health_status.error_message = None

            # Determine status
            if (backend_state.health_status.consecutive_successes >=
                    self.config.health_check.healthy_threshold):
                backend_state.health_status.status = HealthStatusEnum.HEALTHY

        except asyncio.TimeoutError:
            backend_state.health_status.consecutive_failures += 1
            backend_state.health_status.consecutive_successes = 0
            backend_state.health_status.error_message = "Health check timeout"

            # Determine status
            if (backend_state.health_status.consecutive_failures >=
                    self.config.health_check.unhealthy_threshold):
                backend_state.health_status.status = HealthStatusEnum.UNHEALTHY
            else:
                backend_state.health_status.status = HealthStatusEnum.DEGRADED

        except Exception as e:
            backend_state.health_status.consecutive_failures += 1
            backend_state.health_status.consecutive_successes = 0
            backend_state.health_status.error_message = str(e)
            backend_state.health_status.status = HealthStatusEnum.UNHEALTHY

        finally:
            backend_state.health_status.last_check = time.time()

        return backend_state.health_status

    async def configure_algorithm(
        self,
        algorithm: LoadBalancingAlgorithm
    ) -> None:
        """Configure load balancing algorithm"""
        algorithm_map = {
            LoadBalancingAlgorithmType.ROUND_ROBIN: RoundRobinAlgorithm,
            LoadBalancingAlgorithmType.LEAST_CONNECTIONS: LeastConnectionsAlgorithm,
            LoadBalancingAlgorithmType.WEIGHTED: WeightedAlgorithm,
            LoadBalancingAlgorithmType.QUANTUM_AWARE: QuantumAwareAlgorithm,
            LoadBalancingAlgorithmType.IP_HASH: IPHashAlgorithm,
            LoadBalancingAlgorithmType.LEAST_LATENCY: LeastLatencyAlgorithm,
        }

        algorithm_class = algorithm_map.get(algorithm.algorithm_type)
        if not algorithm_class:
            raise AlgorithmError(
                f"Unknown algorithm type: {algorithm.algorithm_type}"
            )

        self.algorithm = algorithm_class(self.backends)
        self.config.algorithm = algorithm

        self.logger.info(f"Configured algorithm: {algorithm.algorithm_type.value}")

    async def _get_available_backends(self) -> List[BackendID]:
        """Get list of available backends"""
        available = []

        for backend_id, backend_state in self.backends.items():
            # Check health status
            if backend_state.health_status.status != HealthStatusEnum.HEALTHY:
                continue

            # Check circuit breaker
            circuit_breaker = self.circuit_breakers[backend_id]
            if circuit_breaker.state == "open":
                continue

            available.append(backend_id)

        return available

    async def _health_check_loop(self) -> None:
        """Background task for periodic health checks"""
        while not self._shutting_down:
            try:
                for backend_id in list(self.backends.keys()):
                    if not self._shutting_down:
                        await self.health_check(backend_id)

                await asyncio.sleep(self.config.health_check.interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(1.0)

    async def _metrics_loop(self) -> None:
        """Background task for metrics collection"""
        while not self._shutting_down:
            try:
                # Log current metrics
                distribution = await self.get_load_distribution()
                self.logger.info(
                    f"Load distribution - Total requests: {distribution.total_requests}, "
                    f"Active backends: {len([b for b in distribution.backends.values() if b['health_status'] == 'healthy'])}"
                )

                await asyncio.sleep(60.0)  # Log every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics loop error: {e}")
                await asyncio.sleep(1.0)

    async def _cleanup_loop(self) -> None:
        """Background task for cleanup operations"""
        while not self._shutting_down:
            try:
                # Clean up old session affinity entries
                current_time = time.time()
                expired_sessions = []

                for session_id, backend_id in self.session_affinity_map.items():
                    if backend_id not in self.backends:
                        expired_sessions.append(session_id)

                for session_id in expired_sessions:
                    del self.session_affinity_map[session_id]

                await asyncio.sleep(300.0)  # Cleanup every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(1.0)

    # Integration methods

    async def integrate_auto_scaling(
        self,
        auto_scaling_module: Any
    ) -> None:
        """Integrate with auto scaling module"""
        # Placeholder for integration logic
        self.logger.info("Integrated with auto_scaling_module")

    async def integrate_redundant_deployment(
        self,
        redundant_deployment_module: Any
    ) -> None:
        """Integrate with redundant deployment module"""
        # Placeholder for integration logic
        self.logger.info("Integrated with redundant_deployment_module")


# ==================== Usage Example ====================

async def main():
    """Example usage of LoadBalancingModule"""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

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

        backend2 = BackendConfig(
            host="backend2.omnixan.local",
            port=8080,
            weight=1.5,
            quantum_capable=True,
            priority=5
        )
        backend2_id = await lb_module.add_backend(backend2)

        backend3 = BackendConfig(
            host="backend3.omnixan.local",
            port=8080,
            weight=0.8,
            quantum_capable=False,
            priority=3
        )
        backend3_id = await lb_module.add_backend(backend3)

        # Wait for initial health checks
        await asyncio.sleep(1.0)

        # Route some requests
        for i in range(10):
            request = Request(
                client_ip=f"192.168.1.{i % 10}",
                workload_type=WorkloadType.QUANTUM_SIMULATION if i % 2 == 0 else WorkloadType.CLASSICAL_COMPUTE,
                session_id=f"session_{i % 3}"
            )

            try:
                result = await lb_module.route_request(request)
                print(f"Request {i} routed to {result.backend_id} "
                      f"(latency: {result.latency_ms:.2f}ms)")
            except Exception as e:
                print(f"Request {i} failed: {e}")

        # Get load distribution
        distribution = await lb_module.get_load_distribution()
        print(f"Load Distribution:")
        print(f"Total Requests: {distribution.total_requests}")
        print(f"Algorithm: {distribution.algorithm.value}")
        for backend_id, info in distribution.backends.items():
            print(f"Backend {backend_id[:8]}:")
            print(f"  Host: {info['host']}:{info['port']}")
            print(f"  Requests: {info['total_requests']}")
            print(f"  Success Rate: {info['successful_requests']}/{info['total_requests']}")
            print(f"  Avg Latency: {info['avg_latency_ms']:.2f}ms")
            print(f"  Health: {info['health_status']}")

        # Keep running for a bit
        await asyncio.sleep(10.0)

    finally:
        await lb_module.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
