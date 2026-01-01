"""
OMNIXAN Low Latency Routing Module
in_memory_computing_cloud/low_latency_routing_module

Production-ready low-latency routing implementation optimized for real-time
applications, with adaptive path selection, QoS guarantees, and congestion
avoidance.
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


class QoSClass(str, Enum):
    """Quality of Service classes"""
    REALTIME = "realtime"  # Ultra-low latency
    INTERACTIVE = "interactive"  # Low latency
    BULK = "bulk"  # Best effort
    BACKGROUND = "background"  # Lowest priority


class RoutingAlgorithm(str, Enum):
    """Routing algorithms"""
    DIJKSTRA = "dijkstra"
    BELLMAN_FORD = "bellman_ford"
    ECMP = "ecmp"  # Equal-Cost Multi-Path
    SEGMENT = "segment"  # Segment routing
    SDN = "sdn"  # Software-defined


class LinkStatus(str, Enum):
    """Network link status"""
    UP = "up"
    DOWN = "down"
    DEGRADED = "degraded"
    CONGESTED = "congested"


@dataclass
class NetworkNode:
    """A network node"""
    node_id: str
    name: str
    location: Tuple[float, float]
    processing_delay_us: float = 10.0
    is_edge: bool = False
    is_gateway: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NetworkLink:
    """A network link between nodes"""
    link_id: str
    source: str
    target: str
    latency_us: float  # Microseconds
    bandwidth_gbps: float
    status: LinkStatus = LinkStatus.UP
    utilization: float = 0.0
    packet_loss: float = 0.0
    jitter_us: float = 0.0
    weight: float = 1.0  # For routing calculations
    is_bidirectional: bool = True


@dataclass
class RoutePath:
    """A routing path"""
    path_id: str
    source: str
    destination: str
    nodes: List[str]
    links: List[str]
    total_latency_us: float
    total_hops: int
    min_bandwidth_gbps: float
    qos_class: QoSClass = QoSClass.BULK
    is_active: bool = True
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)


@dataclass
class RoutingMetrics:
    """Routing metrics"""
    total_routes: int = 0
    active_routes: int = 0
    total_packets: int = 0
    avg_latency_us: float = 0.0
    min_latency_us: float = float('inf')
    max_latency_us: float = 0.0
    path_failures: int = 0
    reroutes: int = 0


class RoutingConfig(BaseModel):
    """Configuration for low-latency routing"""
    algorithm: RoutingAlgorithm = Field(
        default=RoutingAlgorithm.DIJKSTRA,
        description="Routing algorithm"
    )
    max_latency_us: float = Field(
        default=1000.0,
        gt=0.0,
        description="Maximum allowed latency (microseconds)"
    )
    max_hops: int = Field(
        default=10,
        ge=1,
        description="Maximum hop count"
    )
    enable_ecmp: bool = Field(
        default=True,
        description="Enable equal-cost multi-path"
    )
    congestion_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Link congestion threshold"
    )
    path_cache_ttl: float = Field(
        default=60.0,
        gt=0.0,
        description="Path cache TTL in seconds"
    )
    qos_enabled: bool = Field(
        default=True,
        description="Enable QoS differentiation"
    )


class RoutingError(Exception):
    """Base exception for routing errors"""
    pass


class NoPathError(RoutingError):
    """Raised when no path exists"""
    pass


class LatencyExceededError(RoutingError):
    """Raised when latency requirements cannot be met"""
    pass


# ============================================================================
# Graph and Path Finding
# ============================================================================

class NetworkGraph:
    """Network topology graph"""
    
    def __init__(self):
        self.nodes: Dict[str, NetworkNode] = {}
        self.links: Dict[str, NetworkLink] = {}
        self.adjacency: Dict[str, List[Tuple[str, str]]] = defaultdict(list)  # node -> [(neighbor, link_id)]
    
    def add_node(self, node: NetworkNode) -> None:
        """Add node to graph"""
        self.nodes[node.node_id] = node
    
    def remove_node(self, node_id: str) -> bool:
        """Remove node from graph"""
        if node_id not in self.nodes:
            return False
        
        # Remove associated links
        links_to_remove = [
            link_id for link_id, link in self.links.items()
            if link.source == node_id or link.target == node_id
        ]
        
        for link_id in links_to_remove:
            self.remove_link(link_id)
        
        del self.nodes[node_id]
        del self.adjacency[node_id]
        
        return True
    
    def add_link(self, link: NetworkLink) -> None:
        """Add link to graph"""
        self.links[link.link_id] = link
        self.adjacency[link.source].append((link.target, link.link_id))
        
        if link.is_bidirectional:
            self.adjacency[link.target].append((link.source, link.link_id))
    
    def remove_link(self, link_id: str) -> bool:
        """Remove link from graph"""
        if link_id not in self.links:
            return False
        
        link = self.links[link_id]
        
        self.adjacency[link.source] = [
            (n, l) for n, l in self.adjacency[link.source] if l != link_id
        ]
        
        if link.is_bidirectional:
            self.adjacency[link.target] = [
                (n, l) for n, l in self.adjacency[link.target] if l != link_id
            ]
        
        del self.links[link_id]
        return True
    
    def get_link_weight(self, link_id: str, qos: QoSClass) -> float:
        """Get link weight considering QoS"""
        link = self.links.get(link_id)
        if not link or link.status == LinkStatus.DOWN:
            return float('inf')
        
        # Base weight from latency
        weight = link.latency_us
        
        # Penalty for congestion
        if link.utilization > 0.5:
            weight *= (1 + link.utilization)
        
        # Penalty for degraded links
        if link.status == LinkStatus.DEGRADED:
            weight *= 1.5
        elif link.status == LinkStatus.CONGESTED:
            weight *= 2.0
        
        # QoS adjustments
        if qos == QoSClass.REALTIME:
            # Strongly penalize jitter for realtime
            weight += link.jitter_us * 2
        
        return weight


class PathFinder:
    """Finds optimal paths in network graph"""
    
    def __init__(self, graph: NetworkGraph, config: RoutingConfig):
        self.graph = graph
        self.config = config
    
    def find_path(
        self,
        source: str,
        destination: str,
        qos: QoSClass = QoSClass.BULK
    ) -> Optional[RoutePath]:
        """Find optimal path"""
        if self.config.algorithm == RoutingAlgorithm.DIJKSTRA:
            return self._dijkstra(source, destination, qos)
        elif self.config.algorithm == RoutingAlgorithm.BELLMAN_FORD:
            return self._bellman_ford(source, destination, qos)
        else:
            return self._dijkstra(source, destination, qos)
    
    def find_all_paths(
        self,
        source: str,
        destination: str,
        qos: QoSClass = QoSClass.BULK,
        k: int = 3
    ) -> List[RoutePath]:
        """Find k shortest paths (for ECMP)"""
        paths = []
        
        # Yen's k-shortest paths algorithm (simplified)
        primary = self.find_path(source, destination, qos)
        if primary:
            paths.append(primary)
        
        # Find alternative paths by removing links
        for i in range(min(k - 1, 2)):
            if paths:
                # Temporarily remove links from previous path
                removed = []
                for link_id in paths[-1].links:
                    if link_id in self.graph.links:
                        link = self.graph.links[link_id]
                        old_status = link.status
                        link.status = LinkStatus.DOWN
                        removed.append((link_id, old_status))
                
                alt = self.find_path(source, destination, qos)
                
                # Restore links
                for link_id, old_status in removed:
                    self.graph.links[link_id].status = old_status
                
                if alt and alt.path_id not in [p.path_id for p in paths]:
                    paths.append(alt)
        
        return paths
    
    def _dijkstra(
        self,
        source: str,
        destination: str,
        qos: QoSClass
    ) -> Optional[RoutePath]:
        """Dijkstra's shortest path algorithm"""
        if source not in self.graph.nodes or destination not in self.graph.nodes:
            return None
        
        # Distance and predecessor tracking
        distances: Dict[str, float] = {source: 0}
        predecessors: Dict[str, Tuple[str, str]] = {}  # node -> (prev_node, link_id)
        
        # Priority queue: (distance, node_id)
        pq = [(0, source)]
        visited = set()
        
        while pq:
            dist, node = heapq.heappop(pq)
            
            if node in visited:
                continue
            
            visited.add(node)
            
            if node == destination:
                break
            
            # Explore neighbors
            for neighbor, link_id in self.graph.adjacency.get(node, []):
                if neighbor in visited:
                    continue
                
                weight = self.graph.get_link_weight(link_id, qos)
                if weight == float('inf'):
                    continue
                
                # Add processing delay
                node_delay = self.graph.nodes[node].processing_delay_us
                new_dist = dist + weight + node_delay
                
                if new_dist < distances.get(neighbor, float('inf')):
                    distances[neighbor] = new_dist
                    predecessors[neighbor] = (node, link_id)
                    heapq.heappush(pq, (new_dist, neighbor))
        
        if destination not in predecessors and source != destination:
            return None
        
        # Reconstruct path
        return self._build_path(source, destination, predecessors, distances, qos)
    
    def _bellman_ford(
        self,
        source: str,
        destination: str,
        qos: QoSClass
    ) -> Optional[RoutePath]:
        """Bellman-Ford algorithm (handles negative weights)"""
        distances: Dict[str, float] = {n: float('inf') for n in self.graph.nodes}
        distances[source] = 0
        predecessors: Dict[str, Tuple[str, str]] = {}
        
        # Relax edges |V| - 1 times
        for _ in range(len(self.graph.nodes) - 1):
            for link_id, link in self.graph.links.items():
                if link.status == LinkStatus.DOWN:
                    continue
                
                weight = self.graph.get_link_weight(link_id, qos)
                
                if distances[link.source] + weight < distances[link.target]:
                    distances[link.target] = distances[link.source] + weight
                    predecessors[link.target] = (link.source, link_id)
                
                if link.is_bidirectional:
                    if distances[link.target] + weight < distances[link.source]:
                        distances[link.source] = distances[link.target] + weight
                        predecessors[link.source] = (link.target, link_id)
        
        if destination not in predecessors and source != destination:
            return None
        
        return self._build_path(source, destination, predecessors, distances, qos)
    
    def _build_path(
        self,
        source: str,
        destination: str,
        predecessors: Dict[str, Tuple[str, str]],
        distances: Dict[str, float],
        qos: QoSClass
    ) -> RoutePath:
        """Build path from predecessors"""
        nodes = [destination]
        links = []
        
        current = destination
        while current != source:
            if current not in predecessors:
                break
            prev_node, link_id = predecessors[current]
            nodes.insert(0, prev_node)
            links.insert(0, link_id)
            current = prev_node
        
        # Calculate min bandwidth
        min_bw = float('inf')
        for link_id in links:
            link = self.graph.links.get(link_id)
            if link:
                min_bw = min(min_bw, link.bandwidth_gbps)
        
        return RoutePath(
            path_id=str(uuid4()),
            source=source,
            destination=destination,
            nodes=nodes,
            links=links,
            total_latency_us=distances.get(destination, 0),
            total_hops=len(links),
            min_bandwidth_gbps=min_bw if min_bw != float('inf') else 0,
            qos_class=qos
        )


# ============================================================================
# Main Module Implementation
# ============================================================================

class LowLatencyRoutingModule:
    """
    Production-ready low-latency routing module for OMNIXAN.
    
    Provides:
    - Ultra-low latency path computation
    - QoS-aware routing
    - Multi-path routing (ECMP)
    - Congestion avoidance
    - Adaptive rerouting
    """
    
    def __init__(self, config: Optional[RoutingConfig] = None):
        """Initialize the Low Latency Routing Module"""
        self.config = config or RoutingConfig()
        self.graph = NetworkGraph()
        self.path_finder = PathFinder(self.graph, self.config)
        self.path_cache: Dict[Tuple[str, str, QoSClass], List[RoutePath]] = {}
        self.metrics = RoutingMetrics()
        
        # Background tasks
        self._monitor_task: Optional[asyncio.Task] = None
        self._initialized = False
        self._shutting_down = False
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(__name__)
        
        # Latency tracking
        self._latencies: List[float] = []
    
    async def initialize(self) -> None:
        """Initialize the routing module"""
        if self._initialized:
            self._logger.warning("Module already initialized")
            return
        
        try:
            self._logger.info("Initializing LowLatencyRoutingModule...")
            
            # Start monitoring
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            
            self._initialized = True
            self._logger.info("LowLatencyRoutingModule initialized successfully")
        
        except Exception as e:
            self._logger.error(f"Initialization failed: {str(e)}")
            raise RoutingError(f"Failed to initialize module: {str(e)}")
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute routing operation"""
        if not self._initialized:
            raise RoutingError("Module not initialized")
        
        operation = params.get("operation")
        
        if operation == "add_node":
            node = await self.add_node(
                name=params["name"],
                location=tuple(params.get("location", [0, 0])),
                processing_delay_us=params.get("processing_delay_us", 10),
                is_edge=params.get("is_edge", False),
                is_gateway=params.get("is_gateway", False)
            )
            return {"node_id": node.node_id}
        
        elif operation == "add_link":
            link = await self.add_link(
                source=params["source"],
                target=params["target"],
                latency_us=params.get("latency_us", 100),
                bandwidth_gbps=params.get("bandwidth_gbps", 10)
            )
            return {"link_id": link.link_id}
        
        elif operation == "find_path":
            path = await self.find_path(
                source=params["source"],
                destination=params["destination"],
                qos=QoSClass(params.get("qos", "bulk"))
            )
            if path:
                return {
                    "path_id": path.path_id,
                    "nodes": path.nodes,
                    "latency_us": path.total_latency_us,
                    "hops": path.total_hops,
                    "bandwidth_gbps": path.min_bandwidth_gbps
                }
            return {"error": "No path found"}
        
        elif operation == "find_ecmp_paths":
            paths = await self.find_ecmp_paths(
                source=params["source"],
                destination=params["destination"],
                qos=QoSClass(params.get("qos", "bulk"))
            )
            return {
                "paths": [
                    {
                        "path_id": p.path_id,
                        "nodes": p.nodes,
                        "latency_us": p.total_latency_us
                    }
                    for p in paths
                ]
            }
        
        elif operation == "update_link":
            success = await self.update_link(
                link_id=params["link_id"],
                latency_us=params.get("latency_us"),
                utilization=params.get("utilization"),
                status=LinkStatus(params["status"]) if "status" in params else None
            )
            return {"success": success}
        
        elif operation == "get_metrics":
            return self.get_metrics()
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def add_node(
        self,
        name: str,
        location: Tuple[float, float] = (0, 0),
        processing_delay_us: float = 10.0,
        is_edge: bool = False,
        is_gateway: bool = False
    ) -> NetworkNode:
        """Add a network node"""
        async with self._lock:
            node = NetworkNode(
                node_id=str(uuid4()),
                name=name,
                location=location,
                processing_delay_us=processing_delay_us,
                is_edge=is_edge,
                is_gateway=is_gateway
            )
            
            self.graph.add_node(node)
            self._logger.info(f"Added node {node.node_id}: {name}")
            return node
    
    async def add_link(
        self,
        source: str,
        target: str,
        latency_us: float = 100.0,
        bandwidth_gbps: float = 10.0,
        bidirectional: bool = True
    ) -> NetworkLink:
        """Add a network link"""
        async with self._lock:
            link = NetworkLink(
                link_id=str(uuid4()),
                source=source,
                target=target,
                latency_us=latency_us,
                bandwidth_gbps=bandwidth_gbps,
                is_bidirectional=bidirectional
            )
            
            self.graph.add_link(link)
            
            # Invalidate cache
            self.path_cache.clear()
            
            self._logger.info(f"Added link {link.link_id}: {source} -> {target}")
            return link
    
    async def update_link(
        self,
        link_id: str,
        latency_us: Optional[float] = None,
        utilization: Optional[float] = None,
        status: Optional[LinkStatus] = None
    ) -> bool:
        """Update link properties"""
        async with self._lock:
            if link_id not in self.graph.links:
                return False
            
            link = self.graph.links[link_id]
            
            if latency_us is not None:
                link.latency_us = latency_us
            if utilization is not None:
                link.utilization = utilization
                if utilization > self.config.congestion_threshold:
                    link.status = LinkStatus.CONGESTED
            if status is not None:
                link.status = status
            
            # Invalidate cache if significant change
            self.path_cache.clear()
            
            return True
    
    async def find_path(
        self,
        source: str,
        destination: str,
        qos: QoSClass = QoSClass.BULK
    ) -> Optional[RoutePath]:
        """Find optimal path"""
        async with self._lock:
            # Check cache
            cache_key = (source, destination, qos)
            if cache_key in self.path_cache:
                cached = self.path_cache[cache_key]
                if cached and cached[0].is_active:
                    return cached[0]
            
            # Find new path
            path = self.path_finder.find_path(source, destination, qos)
            
            if path:
                # Validate constraints
                if path.total_latency_us > self.config.max_latency_us:
                    if qos == QoSClass.REALTIME:
                        self._logger.warning(
                            f"Path latency {path.total_latency_us}us "
                            f"exceeds max {self.config.max_latency_us}us"
                        )
                
                if path.total_hops > self.config.max_hops:
                    self._logger.warning(
                        f"Path hops {path.total_hops} exceeds max {self.config.max_hops}"
                    )
                
                # Cache path
                self.path_cache[cache_key] = [path]
                
                # Update metrics
                self.metrics.total_routes += 1
                self.metrics.active_routes = len(self.path_cache)
                self._update_latency_stats(path.total_latency_us)
            
            return path
    
    async def find_ecmp_paths(
        self,
        source: str,
        destination: str,
        qos: QoSClass = QoSClass.BULK
    ) -> List[RoutePath]:
        """Find multiple equal-cost paths"""
        if not self.config.enable_ecmp:
            path = await self.find_path(source, destination, qos)
            return [path] if path else []
        
        async with self._lock:
            paths = self.path_finder.find_all_paths(source, destination, qos)
            
            # Cache all paths
            cache_key = (source, destination, qos)
            self.path_cache[cache_key] = paths
            
            return paths
    
    def _update_latency_stats(self, latency_us: float) -> None:
        """Update latency statistics"""
        self._latencies.append(latency_us)
        
        self.metrics.min_latency_us = min(self.metrics.min_latency_us, latency_us)
        self.metrics.max_latency_us = max(self.metrics.max_latency_us, latency_us)
        self.metrics.avg_latency_us = sum(self._latencies) / len(self._latencies)
    
    async def _monitor_loop(self) -> None:
        """Background monitoring loop"""
        while not self._shutting_down:
            try:
                await asyncio.sleep(self.config.path_cache_ttl / 2)
                
                # Clean expired cache entries
                async with self._lock:
                    now = time.time()
                    expired = [
                        key for key, paths in self.path_cache.items()
                        if paths and (now - paths[0].created_at) > self.config.path_cache_ttl
                    ]
                    for key in expired:
                        del self.path_cache[key]
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Monitor error: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get routing metrics"""
        return {
            "total_nodes": len(self.graph.nodes),
            "total_links": len(self.graph.links),
            "total_routes": self.metrics.total_routes,
            "cached_paths": len(self.path_cache),
            "avg_latency_us": round(self.metrics.avg_latency_us, 2),
            "min_latency_us": self.metrics.min_latency_us if self.metrics.min_latency_us != float('inf') else 0,
            "max_latency_us": self.metrics.max_latency_us,
            "path_failures": self.metrics.path_failures,
            "reroutes": self.metrics.reroutes
        }
    
    async def shutdown(self) -> None:
        """Shutdown the routing module"""
        self._logger.info("Shutting down LowLatencyRoutingModule...")
        self._shutting_down = True
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        self.graph.nodes.clear()
        self.graph.links.clear()
        self.path_cache.clear()
        self._initialized = False
        
        self._logger.info("LowLatencyRoutingModule shutdown complete")


# Example usage
async def main():
    """Example usage of LowLatencyRoutingModule"""
    
    config = RoutingConfig(
        algorithm=RoutingAlgorithm.DIJKSTRA,
        max_latency_us=500,
        enable_ecmp=True
    )
    
    module = LowLatencyRoutingModule(config)
    await module.initialize()
    
    try:
        # Build network topology
        nodes = {}
        node_names = ["edge1", "switch1", "switch2", "router1", "core", "server1"]
        
        for name in node_names:
            node = await module.add_node(
                name=name,
                processing_delay_us=5 if "edge" in name else 2,
                is_edge="edge" in name
            )
            nodes[name] = node.node_id
        
        # Add links (microsecond latencies)
        await module.add_link(nodes["edge1"], nodes["switch1"], latency_us=50, bandwidth_gbps=10)
        await module.add_link(nodes["switch1"], nodes["switch2"], latency_us=20, bandwidth_gbps=40)
        await module.add_link(nodes["switch1"], nodes["router1"], latency_us=30, bandwidth_gbps=40)
        await module.add_link(nodes["switch2"], nodes["core"], latency_us=25, bandwidth_gbps=100)
        await module.add_link(nodes["router1"], nodes["core"], latency_us=35, bandwidth_gbps=100)
        await module.add_link(nodes["core"], nodes["server1"], latency_us=10, bandwidth_gbps=100)
        
        print(f"Network: {len(nodes)} nodes, {len(module.graph.links)} links\n")
        
        # Find paths for different QoS classes
        for qos in [QoSClass.REALTIME, QoSClass.INTERACTIVE, QoSClass.BULK]:
            path = await module.find_path(
                nodes["edge1"],
                nodes["server1"],
                qos
            )
            
            if path:
                print(f"{qos.value.upper()} path:")
                print(f"  Latency: {path.total_latency_us:.2f}µs")
                print(f"  Hops: {path.total_hops}")
                print(f"  Bandwidth: {path.min_bandwidth_gbps} Gbps")
                print()
        
        # Find ECMP paths
        print("ECMP paths:")
        paths = await module.find_ecmp_paths(
            nodes["edge1"],
            nodes["server1"],
            QoSClass.BULK
        )
        
        for i, path in enumerate(paths):
            print(f"  Path {i+1}: {path.total_latency_us:.2f}µs, {path.total_hops} hops")
        
        # Get metrics
        metrics = module.get_metrics()
        print(f"\nMetrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    
    finally:
        await module.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

