"""
OMNIXAN Local Traffic Shunting Module
in_memory_computing_cloud/local_traffic_shunting_module

Production-ready local traffic shunting implementation that optimizes data
routing by directing traffic locally when possible, reducing backhaul
and improving latency.
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4
import heapq

from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficType(str, Enum):
    """Types of network traffic"""
    LOCAL = "local"
    REGIONAL = "regional"
    BACKHAUL = "backhaul"
    INTERNET = "internet"
    CDN = "cdn"
    P2P = "p2p"


class ShuntingPolicy(str, Enum):
    """Traffic shunting policies"""
    ALWAYS_LOCAL = "always_local"
    LATENCY_BASED = "latency_based"
    BANDWIDTH_BASED = "bandwidth_based"
    COST_BASED = "cost_based"
    HYBRID = "hybrid"


class RouteStatus(str, Enum):
    """Route status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEGRADED = "degraded"
    CONGESTED = "congested"


@dataclass
class TrafficRoute:
    """A traffic route"""
    route_id: str
    source: str
    destination: str
    route_type: TrafficType
    status: RouteStatus = RouteStatus.ACTIVE
    latency_ms: float = 0.0
    bandwidth_mbps: float = 1000.0
    cost_per_gb: float = 0.0
    packet_loss: float = 0.0
    hop_count: int = 1
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    bytes_transferred: int = 0


@dataclass
class TrafficFlow:
    """A traffic flow to be routed"""
    flow_id: str
    source: str
    destination: str
    size_bytes: int
    priority: int = 0
    deadline_ms: Optional[float] = None
    traffic_type: Optional[TrafficType] = None


@dataclass 
class ShuntingDecision:
    """Traffic shunting decision"""
    flow_id: str
    route_id: str
    route_type: TrafficType
    estimated_latency_ms: float
    estimated_cost: float
    is_local: bool
    reason: str


@dataclass
class LocalEndpoint:
    """Local endpoint for traffic"""
    endpoint_id: str
    name: str
    address: str
    services: Set[str]
    capacity_mbps: float = 1000.0
    current_load_mbps: float = 0.0
    is_available: bool = True


@dataclass
class ShuntingMetrics:
    """Traffic shunting metrics"""
    total_flows: int = 0
    local_flows: int = 0
    regional_flows: int = 0
    backhaul_flows: int = 0
    total_bytes_shunted: int = 0
    backhaul_bytes_saved: int = 0
    avg_latency_reduction_ms: float = 0.0
    cost_savings: float = 0.0


class ShuntingConfig(BaseModel):
    """Configuration for traffic shunting"""
    default_policy: ShuntingPolicy = Field(
        default=ShuntingPolicy.HYBRID,
        description="Default shunting policy"
    )
    local_preference_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for local routing preference"
    )
    latency_threshold_ms: float = Field(
        default=10.0,
        gt=0.0,
        description="Latency threshold for local shunting"
    )
    bandwidth_threshold_mbps: float = Field(
        default=100.0,
        gt=0.0,
        description="Minimum bandwidth for route"
    )
    enable_caching: bool = Field(
        default=True,
        description="Enable route caching"
    )
    cache_ttl_seconds: float = Field(
        default=300.0,
        gt=0.0,
        description="Route cache TTL"
    )
    max_routes: int = Field(
        default=10000,
        ge=100,
        description="Maximum cached routes"
    )


class ShuntingError(Exception):
    """Base exception for shunting errors"""
    pass


class RouteNotFoundError(ShuntingError):
    """Raised when no route is available"""
    pass


# ============================================================================
# Route Manager
# ============================================================================

class RouteManager:
    """Manages traffic routes"""
    
    def __init__(self):
        self.routes: Dict[str, TrafficRoute] = {}
        self.route_cache: Dict[Tuple[str, str], List[str]] = {}  # (src, dst) -> [route_ids]
        self.local_endpoints: Dict[str, LocalEndpoint] = {}
    
    def add_route(self, route: TrafficRoute) -> None:
        """Add a route"""
        self.routes[route.route_id] = route
        
        # Update cache
        key = (route.source, route.destination)
        if key not in self.route_cache:
            self.route_cache[key] = []
        self.route_cache[key].append(route.route_id)
    
    def remove_route(self, route_id: str) -> bool:
        """Remove a route"""
        if route_id not in self.routes:
            return False
        
        route = self.routes[route_id]
        key = (route.source, route.destination)
        
        if key in self.route_cache:
            self.route_cache[key] = [
                r for r in self.route_cache[key] if r != route_id
            ]
        
        del self.routes[route_id]
        return True
    
    def get_routes(self, source: str, destination: str) -> List[TrafficRoute]:
        """Get routes between source and destination"""
        key = (source, destination)
        route_ids = self.route_cache.get(key, [])
        
        routes = []
        for route_id in route_ids:
            if route_id in self.routes:
                route = self.routes[route_id]
                if route.status == RouteStatus.ACTIVE:
                    routes.append(route)
        
        return routes
    
    def add_local_endpoint(self, endpoint: LocalEndpoint) -> None:
        """Add a local endpoint"""
        self.local_endpoints[endpoint.endpoint_id] = endpoint
    
    def find_local_endpoint(self, service: str) -> Optional[LocalEndpoint]:
        """Find local endpoint for service"""
        for endpoint in self.local_endpoints.values():
            if service in endpoint.services and endpoint.is_available:
                return endpoint
        return None
    
    def update_route_stats(
        self,
        route_id: str,
        bytes_transferred: int,
        latency_ms: float
    ) -> None:
        """Update route statistics"""
        if route_id in self.routes:
            route = self.routes[route_id]
            route.bytes_transferred += bytes_transferred
            route.last_used = time.time()
            
            # Update latency with moving average
            route.latency_ms = route.latency_ms * 0.9 + latency_ms * 0.1


# ============================================================================
# Shunting Decision Engine
# ============================================================================

class ShuntingDecisionEngine:
    """Makes traffic shunting decisions"""
    
    def __init__(self, config: ShuntingConfig, route_manager: RouteManager):
        self.config = config
        self.route_manager = route_manager
    
    def decide(
        self,
        flow: TrafficFlow,
        available_routes: List[TrafficRoute]
    ) -> Optional[ShuntingDecision]:
        """Decide how to route traffic"""
        if not available_routes:
            return None
        
        # Check for local endpoint
        local_endpoint = self.route_manager.find_local_endpoint(flow.destination)
        
        if self.config.default_policy == ShuntingPolicy.ALWAYS_LOCAL:
            return self._always_local(flow, available_routes, local_endpoint)
        elif self.config.default_policy == ShuntingPolicy.LATENCY_BASED:
            return self._latency_based(flow, available_routes, local_endpoint)
        elif self.config.default_policy == ShuntingPolicy.BANDWIDTH_BASED:
            return self._bandwidth_based(flow, available_routes, local_endpoint)
        elif self.config.default_policy == ShuntingPolicy.COST_BASED:
            return self._cost_based(flow, available_routes, local_endpoint)
        else:
            return self._hybrid(flow, available_routes, local_endpoint)
    
    def _always_local(
        self,
        flow: TrafficFlow,
        routes: List[TrafficRoute],
        local_endpoint: Optional[LocalEndpoint]
    ) -> ShuntingDecision:
        """Always prefer local routes"""
        local_routes = [r for r in routes if r.route_type == TrafficType.LOCAL]
        
        if local_routes:
            best = min(local_routes, key=lambda r: r.latency_ms)
        else:
            best = min(routes, key=lambda r: r.latency_ms)
        
        return ShuntingDecision(
            flow_id=flow.flow_id,
            route_id=best.route_id,
            route_type=best.route_type,
            estimated_latency_ms=best.latency_ms,
            estimated_cost=best.cost_per_gb * flow.size_bytes / (1024**3),
            is_local=best.route_type == TrafficType.LOCAL,
            reason="Local preference policy"
        )
    
    def _latency_based(
        self,
        flow: TrafficFlow,
        routes: List[TrafficRoute],
        local_endpoint: Optional[LocalEndpoint]
    ) -> ShuntingDecision:
        """Select route with lowest latency"""
        best = min(routes, key=lambda r: r.latency_ms)
        
        return ShuntingDecision(
            flow_id=flow.flow_id,
            route_id=best.route_id,
            route_type=best.route_type,
            estimated_latency_ms=best.latency_ms,
            estimated_cost=best.cost_per_gb * flow.size_bytes / (1024**3),
            is_local=best.route_type == TrafficType.LOCAL,
            reason="Lowest latency route"
        )
    
    def _bandwidth_based(
        self,
        flow: TrafficFlow,
        routes: List[TrafficRoute],
        local_endpoint: Optional[LocalEndpoint]
    ) -> ShuntingDecision:
        """Select route with highest bandwidth"""
        # Filter by minimum bandwidth
        suitable = [
            r for r in routes
            if r.bandwidth_mbps >= self.config.bandwidth_threshold_mbps
        ]
        
        if not suitable:
            suitable = routes
        
        best = max(suitable, key=lambda r: r.bandwidth_mbps)
        
        return ShuntingDecision(
            flow_id=flow.flow_id,
            route_id=best.route_id,
            route_type=best.route_type,
            estimated_latency_ms=best.latency_ms,
            estimated_cost=best.cost_per_gb * flow.size_bytes / (1024**3),
            is_local=best.route_type == TrafficType.LOCAL,
            reason="Highest bandwidth route"
        )
    
    def _cost_based(
        self,
        flow: TrafficFlow,
        routes: List[TrafficRoute],
        local_endpoint: Optional[LocalEndpoint]
    ) -> ShuntingDecision:
        """Select lowest cost route"""
        best = min(routes, key=lambda r: r.cost_per_gb)
        
        return ShuntingDecision(
            flow_id=flow.flow_id,
            route_id=best.route_id,
            route_type=best.route_type,
            estimated_latency_ms=best.latency_ms,
            estimated_cost=best.cost_per_gb * flow.size_bytes / (1024**3),
            is_local=best.route_type == TrafficType.LOCAL,
            reason="Lowest cost route"
        )
    
    def _hybrid(
        self,
        flow: TrafficFlow,
        routes: List[TrafficRoute],
        local_endpoint: Optional[LocalEndpoint]
    ) -> ShuntingDecision:
        """Hybrid decision combining multiple factors"""
        def score(route: TrafficRoute) -> float:
            # Normalize metrics
            latency_score = route.latency_ms / 100.0
            cost_score = route.cost_per_gb / 0.1
            local_bonus = 0 if route.route_type != TrafficType.LOCAL else -0.5
            
            return (
                (1 - self.config.local_preference_weight) * latency_score +
                0.2 * cost_score +
                self.config.local_preference_weight * local_bonus
            )
        
        best = min(routes, key=score)
        
        return ShuntingDecision(
            flow_id=flow.flow_id,
            route_id=best.route_id,
            route_type=best.route_type,
            estimated_latency_ms=best.latency_ms,
            estimated_cost=best.cost_per_gb * flow.size_bytes / (1024**3),
            is_local=best.route_type == TrafficType.LOCAL,
            reason="Hybrid optimization"
        )


# ============================================================================
# Main Module Implementation
# ============================================================================

class LocalTrafficShuntingModule:
    """
    Production-ready local traffic shunting module for OMNIXAN.
    
    Provides:
    - Intelligent traffic routing
    - Local traffic optimization
    - Backhaul reduction
    - Cost optimization
    - Route caching
    """
    
    def __init__(self, config: Optional[ShuntingConfig] = None):
        """Initialize the Local Traffic Shunting Module"""
        self.config = config or ShuntingConfig()
        self.route_manager = RouteManager()
        self.decision_engine = ShuntingDecisionEngine(self.config, self.route_manager)
        self.metrics = ShuntingMetrics()
        
        self._initialized = False
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(__name__)
        
        # Track latency improvements
        self._latency_savings: List[float] = []
    
    async def initialize(self) -> None:
        """Initialize the shunting module"""
        if self._initialized:
            self._logger.warning("Module already initialized")
            return
        
        try:
            self._logger.info("Initializing LocalTrafficShuntingModule...")
            
            self._initialized = True
            self._logger.info("LocalTrafficShuntingModule initialized successfully")
        
        except Exception as e:
            self._logger.error(f"Initialization failed: {str(e)}")
            raise ShuntingError(f"Failed to initialize module: {str(e)}")
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute shunting operation"""
        if not self._initialized:
            raise ShuntingError("Module not initialized")
        
        operation = params.get("operation")
        
        if operation == "add_route":
            route = await self.add_route(
                source=params["source"],
                destination=params["destination"],
                route_type=TrafficType(params.get("type", "local")),
                latency_ms=params.get("latency_ms", 1.0),
                bandwidth_mbps=params.get("bandwidth_mbps", 1000.0),
                cost_per_gb=params.get("cost_per_gb", 0.0)
            )
            return {"route_id": route.route_id}
        
        elif operation == "remove_route":
            success = await self.remove_route(params["route_id"])
            return {"success": success}
        
        elif operation == "add_local_endpoint":
            endpoint = await self.add_local_endpoint(
                name=params["name"],
                address=params["address"],
                services=set(params.get("services", []))
            )
            return {"endpoint_id": endpoint.endpoint_id}
        
        elif operation == "route_traffic":
            flow = TrafficFlow(
                flow_id=str(uuid4()),
                source=params["source"],
                destination=params["destination"],
                size_bytes=params.get("size_bytes", 1024),
                priority=params.get("priority", 0)
            )
            decision = await self.route_traffic(flow)
            if decision:
                return {
                    "flow_id": decision.flow_id,
                    "route_id": decision.route_id,
                    "route_type": decision.route_type.value,
                    "is_local": decision.is_local,
                    "estimated_latency_ms": decision.estimated_latency_ms,
                    "estimated_cost": decision.estimated_cost
                }
            return {"error": "No route available"}
        
        elif operation == "get_metrics":
            return self.get_metrics()
        
        elif operation == "list_routes":
            return {
                "routes": [
                    {
                        "route_id": r.route_id,
                        "source": r.source,
                        "destination": r.destination,
                        "type": r.route_type.value,
                        "status": r.status.value,
                        "latency_ms": r.latency_ms
                    }
                    for r in self.route_manager.routes.values()
                ]
            }
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def add_route(
        self,
        source: str,
        destination: str,
        route_type: TrafficType = TrafficType.LOCAL,
        latency_ms: float = 1.0,
        bandwidth_mbps: float = 1000.0,
        cost_per_gb: float = 0.0,
        hop_count: int = 1
    ) -> TrafficRoute:
        """Add a traffic route"""
        async with self._lock:
            route = TrafficRoute(
                route_id=str(uuid4()),
                source=source,
                destination=destination,
                route_type=route_type,
                latency_ms=latency_ms,
                bandwidth_mbps=bandwidth_mbps,
                cost_per_gb=cost_per_gb,
                hop_count=hop_count
            )
            
            self.route_manager.add_route(route)
            self._logger.info(f"Added route {route.route_id}: {source} -> {destination}")
            return route
    
    async def remove_route(self, route_id: str) -> bool:
        """Remove a traffic route"""
        async with self._lock:
            success = self.route_manager.remove_route(route_id)
            if success:
                self._logger.info(f"Removed route {route_id}")
            return success
    
    async def add_local_endpoint(
        self,
        name: str,
        address: str,
        services: Set[str],
        capacity_mbps: float = 1000.0
    ) -> LocalEndpoint:
        """Add a local endpoint"""
        async with self._lock:
            endpoint = LocalEndpoint(
                endpoint_id=str(uuid4()),
                name=name,
                address=address,
                services=services,
                capacity_mbps=capacity_mbps
            )
            
            self.route_manager.add_local_endpoint(endpoint)
            self._logger.info(f"Added local endpoint {endpoint.endpoint_id}: {name}")
            return endpoint
    
    async def route_traffic(self, flow: TrafficFlow) -> Optional[ShuntingDecision]:
        """Route a traffic flow"""
        async with self._lock:
            # Get available routes
            routes = self.route_manager.get_routes(flow.source, flow.destination)
            
            if not routes:
                # Try to find any route to destination
                routes = [
                    r for r in self.route_manager.routes.values()
                    if r.destination == flow.destination
                    and r.status == RouteStatus.ACTIVE
                ]
            
            if not routes:
                self._logger.warning(f"No route for {flow.source} -> {flow.destination}")
                return None
            
            # Get decision
            decision = self.decision_engine.decide(flow, routes)
            
            if decision:
                # Update metrics
                self.metrics.total_flows += 1
                self.metrics.total_bytes_shunted += flow.size_bytes
                
                if decision.is_local:
                    self.metrics.local_flows += 1
                    # Estimate backhaul savings
                    self.metrics.backhaul_bytes_saved += flow.size_bytes
                elif decision.route_type == TrafficType.REGIONAL:
                    self.metrics.regional_flows += 1
                else:
                    self.metrics.backhaul_flows += 1
                
                # Update route stats
                self.route_manager.update_route_stats(
                    decision.route_id,
                    flow.size_bytes,
                    decision.estimated_latency_ms
                )
                
                self.metrics.cost_savings += (
                    0.01 * flow.size_bytes / (1024**3) - decision.estimated_cost
                )
            
            return decision
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get shunting metrics"""
        local_ratio = (
            self.metrics.local_flows / max(self.metrics.total_flows, 1)
        )
        
        return {
            "total_flows": self.metrics.total_flows,
            "local_flows": self.metrics.local_flows,
            "regional_flows": self.metrics.regional_flows,
            "backhaul_flows": self.metrics.backhaul_flows,
            "local_shunt_ratio": round(local_ratio, 4),
            "total_bytes_shunted": self.metrics.total_bytes_shunted,
            "backhaul_bytes_saved": self.metrics.backhaul_bytes_saved,
            "cost_savings": round(self.metrics.cost_savings, 4),
            "total_routes": len(self.route_manager.routes),
            "local_endpoints": len(self.route_manager.local_endpoints)
        }
    
    async def shutdown(self) -> None:
        """Shutdown the shunting module"""
        self._logger.info("Shutting down LocalTrafficShuntingModule...")
        
        self.route_manager.routes.clear()
        self.route_manager.local_endpoints.clear()
        self._initialized = False
        
        self._logger.info("LocalTrafficShuntingModule shutdown complete")


# Example usage
async def main():
    """Example usage of LocalTrafficShuntingModule"""
    
    config = ShuntingConfig(
        default_policy=ShuntingPolicy.HYBRID,
        local_preference_weight=0.7
    )
    
    module = LocalTrafficShuntingModule(config)
    await module.initialize()
    
    try:
        # Add local routes
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
            route_type=TrafficType.REGIONAL,
            latency_ms=15.0,
            bandwidth_mbps=5000,
            cost_per_gb=0.005
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
        
        print("Routing traffic flows...")
        
        # Route multiple flows
        for i in range(100):
            flow = TrafficFlow(
                flow_id=str(uuid4()),
                source="edge_1",
                destination="cdn.local",
                size_bytes=1024 * 1024 * 10  # 10 MB
            )
            
            decision = await module.route_traffic(flow)
            if i < 5 and decision:
                print(f"  Flow {i+1}: {decision.route_type.value} "
                      f"(local={decision.is_local}, "
                      f"latency={decision.estimated_latency_ms}ms)")
        
        # Get metrics
        metrics = module.get_metrics()
        print(f"\nMetrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        
        print(f"\nLocal shunt ratio: {metrics['local_shunt_ratio']*100:.1f}%")
        print(f"Backhaul bytes saved: {metrics['backhaul_bytes_saved'] / (1024**3):.2f} GB")
    
    finally:
        await module.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

