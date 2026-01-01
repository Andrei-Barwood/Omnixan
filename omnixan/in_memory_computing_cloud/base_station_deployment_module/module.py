"""
OMNIXAN Base Station Deployment Module
in_memory_computing_cloud/base_station_deployment_module

Production-ready base station deployment and management for edge computing
infrastructure. Handles placement optimization, capacity planning, coverage
analysis, and dynamic scaling of edge base stations.
"""

import asyncio
import logging
import time
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4
import heapq
import random

import numpy as np

from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseStationType(str, Enum):
    """Types of base stations"""
    MACRO = "macro"  # Large coverage, high capacity
    MICRO = "micro"  # Medium coverage
    PICO = "pico"  # Small coverage, low power
    FEMTO = "femto"  # Very small, indoor
    SMALL_CELL = "small_cell"  # 5G small cell


class StationStatus(str, Enum):
    """Base station operational status"""
    OFFLINE = "offline"
    STARTING = "starting"
    ONLINE = "online"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    OVERLOADED = "overloaded"


class DeploymentStrategy(str, Enum):
    """Deployment strategies"""
    COVERAGE_FIRST = "coverage_first"
    CAPACITY_FIRST = "capacity_first"
    COST_OPTIMIZED = "cost_optimized"
    LATENCY_OPTIMIZED = "latency_optimized"
    HYBRID = "hybrid"


@dataclass
class Location:
    """Geographic location"""
    latitude: float
    longitude: float
    altitude: float = 0.0
    
    def distance_to(self, other: "Location") -> float:
        """Calculate distance in km using Haversine formula"""
        R = 6371  # Earth radius in km
        
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c


@dataclass
class BaseStation:
    """Represents a base station"""
    station_id: str
    name: str
    location: Location
    station_type: BaseStationType
    status: StationStatus = StationStatus.OFFLINE
    coverage_radius_km: float = 1.0
    capacity: int = 1000  # Max concurrent connections
    current_load: int = 0
    power_watts: float = 100.0
    frequency_mhz: float = 2100.0
    created_at: float = field(default_factory=time.time)
    last_health_check: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentPlan:
    """Deployment plan for base stations"""
    plan_id: str
    stations: List[BaseStation]
    total_coverage_area: float
    estimated_capacity: int
    estimated_cost: float
    deployment_time_hours: float
    strategy: DeploymentStrategy


@dataclass
class CoverageArea:
    """Coverage analysis result"""
    center: Location
    radius_km: float
    coverage_quality: float  # 0-1
    overlapping_stations: List[str]


@dataclass
class DeploymentMetrics:
    """Deployment metrics"""
    total_stations: int = 0
    online_stations: int = 0
    total_coverage_area_km2: float = 0.0
    total_capacity: int = 0
    current_total_load: int = 0
    avg_station_utilization: float = 0.0
    power_consumption_watts: float = 0.0


class DeploymentConfig(BaseModel):
    """Configuration for base station deployment"""
    max_stations: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of stations"
    )
    default_coverage_radius: float = Field(
        default=1.0,
        gt=0.0,
        description="Default coverage radius in km"
    )
    overlap_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Acceptable coverage overlap ratio"
    )
    auto_scaling: bool = Field(
        default=True,
        description="Enable auto-scaling based on load"
    )
    health_check_interval: float = Field(
        default=30.0,
        gt=0.0,
        description="Health check interval in seconds"
    )
    strategy: DeploymentStrategy = Field(
        default=DeploymentStrategy.HYBRID,
        description="Deployment strategy"
    )


class DeploymentError(Exception):
    """Base exception for deployment errors"""
    pass


class CapacityError(DeploymentError):
    """Raised when capacity limits are exceeded"""
    pass


class CoverageError(DeploymentError):
    """Raised when coverage requirements aren't met"""
    pass


# ============================================================================
# Placement Optimization
# ============================================================================

class PlacementOptimizer:
    """Optimizes base station placement"""
    
    def __init__(self, strategy: DeploymentStrategy):
        self.strategy = strategy
    
    def optimize_placement(
        self,
        demand_points: List[Tuple[Location, int]],  # (location, demand)
        existing_stations: List[BaseStation],
        budget: float,
        min_coverage: float = 0.9
    ) -> List[Location]:
        """Find optimal locations for new stations"""
        
        if self.strategy == DeploymentStrategy.COVERAGE_FIRST:
            return self._coverage_first_placement(demand_points, existing_stations, budget)
        elif self.strategy == DeploymentStrategy.CAPACITY_FIRST:
            return self._capacity_first_placement(demand_points, existing_stations, budget)
        elif self.strategy == DeploymentStrategy.COST_OPTIMIZED:
            return self._cost_optimized_placement(demand_points, existing_stations, budget)
        elif self.strategy == DeploymentStrategy.LATENCY_OPTIMIZED:
            return self._latency_optimized_placement(demand_points, existing_stations, budget)
        else:
            return self._hybrid_placement(demand_points, existing_stations, budget, min_coverage)
    
    def _coverage_first_placement(
        self,
        demand_points: List[Tuple[Location, int]],
        existing_stations: List[BaseStation],
        budget: float
    ) -> List[Location]:
        """Maximize coverage area"""
        # Greedy algorithm: place stations to cover most uncovered area
        new_locations = []
        uncovered = set(range(len(demand_points)))
        station_cost = budget / 10  # Estimate
        
        while uncovered and len(new_locations) * station_cost < budget:
            best_location = None
            best_coverage = 0
            
            for idx in uncovered:
                point, _ = demand_points[idx]
                
                # Count how many points this location would cover
                coverage = sum(
                    1 for i in uncovered
                    if demand_points[i][0].distance_to(point) < 1.0  # 1km radius
                )
                
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_location = point
            
            if best_location:
                new_locations.append(best_location)
                # Remove covered points
                uncovered = {
                    i for i in uncovered
                    if demand_points[i][0].distance_to(best_location) >= 1.0
                }
        
        return new_locations
    
    def _capacity_first_placement(
        self,
        demand_points: List[Tuple[Location, int]],
        existing_stations: List[BaseStation],
        budget: float
    ) -> List[Location]:
        """Place stations at highest demand points"""
        # Sort by demand
        sorted_points = sorted(demand_points, key=lambda x: x[1], reverse=True)
        
        new_locations = []
        station_cost = budget / 10
        
        for point, demand in sorted_points:
            if len(new_locations) * station_cost >= budget:
                break
            
            # Check if already covered by existing or new stations
            is_covered = any(
                s.location.distance_to(point) < s.coverage_radius_km
                for s in existing_stations
            )
            
            if not is_covered:
                new_locations.append(point)
        
        return new_locations
    
    def _cost_optimized_placement(
        self,
        demand_points: List[Tuple[Location, int]],
        existing_stations: List[BaseStation],
        budget: float
    ) -> List[Location]:
        """Minimize cost per user covered"""
        # K-means style clustering
        n_clusters = max(1, int(budget / 5000))  # Estimate stations per budget
        
        if not demand_points:
            return []
        
        # Simple clustering
        locations = [p[0] for p in demand_points]
        weights = [p[1] for p in demand_points]
        
        # Initialize centroids
        centroids = random.sample(locations, min(n_clusters, len(locations)))
        
        # Iterate
        for _ in range(10):
            clusters = defaultdict(list)
            
            for loc, weight in zip(locations, weights):
                # Assign to nearest centroid
                nearest = min(range(len(centroids)), 
                            key=lambda i: loc.distance_to(centroids[i]))
                clusters[nearest].append((loc, weight))
            
            # Update centroids (weighted average)
            new_centroids = []
            for i, cluster in clusters.items():
                if cluster:
                    total_weight = sum(w for _, w in cluster)
                    avg_lat = sum(l.latitude * w for l, w in cluster) / total_weight
                    avg_lon = sum(l.longitude * w for l, w in cluster) / total_weight
                    new_centroids.append(Location(avg_lat, avg_lon))
            
            centroids = new_centroids if new_centroids else centroids
        
        return centroids
    
    def _latency_optimized_placement(
        self,
        demand_points: List[Tuple[Location, int]],
        existing_stations: List[BaseStation],
        budget: float
    ) -> List[Location]:
        """Minimize average latency (distance)"""
        # Similar to coverage but weight by demand
        new_locations = []
        
        for point, demand in sorted(demand_points, key=lambda x: x[1], reverse=True):
            # Check distance to nearest station
            min_distance = float('inf')
            
            for station in existing_stations:
                dist = point.distance_to(station.location)
                min_distance = min(min_distance, dist)
            
            for loc in new_locations:
                dist = point.distance_to(loc)
                min_distance = min(min_distance, dist)
            
            # Add new station if too far
            if min_distance > 0.5:  # 500m threshold
                new_locations.append(point)
                
            if len(new_locations) >= int(budget / 5000):
                break
        
        return new_locations
    
    def _hybrid_placement(
        self,
        demand_points: List[Tuple[Location, int]],
        existing_stations: List[BaseStation],
        budget: float,
        min_coverage: float
    ) -> List[Location]:
        """Balanced approach"""
        # Combine coverage and capacity strategies
        coverage_locations = self._coverage_first_placement(
            demand_points, existing_stations, budget * 0.6
        )
        capacity_locations = self._capacity_first_placement(
            demand_points, existing_stations, budget * 0.4
        )
        
        # Merge and deduplicate
        all_locations = coverage_locations + capacity_locations
        unique_locations = []
        
        for loc in all_locations:
            is_duplicate = any(
                loc.distance_to(existing) < 0.1
                for existing in unique_locations
            )
            if not is_duplicate:
                unique_locations.append(loc)
        
        return unique_locations


# ============================================================================
# Coverage Analyzer
# ============================================================================

class CoverageAnalyzer:
    """Analyzes network coverage"""
    
    def analyze_coverage(
        self,
        stations: List[BaseStation],
        area_bounds: Tuple[Location, Location],
        resolution: float = 0.1
    ) -> Dict[str, Any]:
        """Analyze coverage of an area"""
        min_loc, max_loc = area_bounds
        
        # Grid-based analysis
        lat_steps = int((max_loc.latitude - min_loc.latitude) / resolution) + 1
        lon_steps = int((max_loc.longitude - min_loc.longitude) / resolution) + 1
        
        covered_cells = 0
        total_cells = lat_steps * lon_steps
        
        coverage_map = []
        
        for i in range(lat_steps):
            for j in range(lon_steps):
                lat = min_loc.latitude + i * resolution
                lon = min_loc.longitude + j * resolution
                point = Location(lat, lon)
                
                # Check coverage
                covering_stations = [
                    s for s in stations
                    if s.status == StationStatus.ONLINE
                    and point.distance_to(s.location) <= s.coverage_radius_km
                ]
                
                if covering_stations:
                    covered_cells += 1
                    coverage_map.append({
                        "location": (lat, lon),
                        "covered": True,
                        "stations": [s.station_id for s in covering_stations],
                        "signal_strength": self._calculate_signal_strength(
                            point, covering_stations[0]
                        )
                    })
        
        coverage_ratio = covered_cells / total_cells if total_cells > 0 else 0
        
        return {
            "coverage_ratio": coverage_ratio,
            "covered_cells": covered_cells,
            "total_cells": total_cells,
            "coverage_map": coverage_map[:100],  # Limit output
            "gaps": self._identify_gaps(stations, area_bounds, resolution)
        }
    
    def _calculate_signal_strength(
        self,
        point: Location,
        station: BaseStation
    ) -> float:
        """Calculate signal strength (simplified model)"""
        distance = point.distance_to(station.location)
        max_distance = station.coverage_radius_km
        
        if distance >= max_distance:
            return 0.0
        
        # Simple inverse square falloff
        return 1.0 - (distance / max_distance) ** 2
    
    def _identify_gaps(
        self,
        stations: List[BaseStation],
        area_bounds: Tuple[Location, Location],
        resolution: float
    ) -> List[Location]:
        """Identify coverage gaps"""
        min_loc, max_loc = area_bounds
        gaps = []
        
        lat_steps = int((max_loc.latitude - min_loc.latitude) / resolution) + 1
        lon_steps = int((max_loc.longitude - min_loc.longitude) / resolution) + 1
        
        for i in range(lat_steps):
            for j in range(lon_steps):
                lat = min_loc.latitude + i * resolution
                lon = min_loc.longitude + j * resolution
                point = Location(lat, lon)
                
                is_covered = any(
                    point.distance_to(s.location) <= s.coverage_radius_km
                    for s in stations if s.status == StationStatus.ONLINE
                )
                
                if not is_covered:
                    gaps.append(point)
        
        return gaps[:50]  # Limit output


# ============================================================================
# Main Module Implementation
# ============================================================================

class BaseStationDeploymentModule:
    """
    Production-ready base station deployment module for OMNIXAN.
    
    Provides:
    - Station lifecycle management
    - Placement optimization
    - Coverage analysis
    - Capacity planning
    - Auto-scaling
    """
    
    def __init__(self, config: Optional[DeploymentConfig] = None):
        """Initialize the Base Station Deployment Module"""
        self.config = config or DeploymentConfig()
        self.stations: Dict[str, BaseStation] = {}
        self.optimizer = PlacementOptimizer(self.config.strategy)
        self.analyzer = CoverageAnalyzer()
        self.metrics = DeploymentMetrics()
        
        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._auto_scale_task: Optional[asyncio.Task] = None
        self._initialized = False
        self._shutting_down = False
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the deployment module"""
        if self._initialized:
            self._logger.warning("Module already initialized")
            return
        
        try:
            self._logger.info("Initializing BaseStationDeploymentModule...")
            
            # Start background tasks
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            if self.config.auto_scaling:
                self._auto_scale_task = asyncio.create_task(self._auto_scale_loop())
            
            self._initialized = True
            self._logger.info("BaseStationDeploymentModule initialized successfully")
        
        except Exception as e:
            self._logger.error(f"Initialization failed: {str(e)}")
            raise DeploymentError(f"Failed to initialize module: {str(e)}")
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute deployment operation"""
        if not self._initialized:
            raise DeploymentError("Module not initialized")
        
        operation = params.get("operation")
        
        if operation == "deploy_station":
            station = await self.deploy_station(
                name=params["name"],
                location=Location(**params["location"]),
                station_type=BaseStationType(params.get("type", "micro")),
                coverage_radius=params.get("coverage_radius", self.config.default_coverage_radius),
                capacity=params.get("capacity", 1000)
            )
            return {"station_id": station.station_id, "status": station.status.value}
        
        elif operation == "remove_station":
            success = await self.remove_station(params["station_id"])
            return {"success": success}
        
        elif operation == "start_station":
            success = await self.start_station(params["station_id"])
            return {"success": success}
        
        elif operation == "stop_station":
            success = await self.stop_station(params["station_id"])
            return {"success": success}
        
        elif operation == "optimize_placement":
            demand_points = [
                (Location(**p["location"]), p["demand"])
                for p in params.get("demand_points", [])
            ]
            budget = params.get("budget", 100000)
            locations = self.optimizer.optimize_placement(
                demand_points,
                list(self.stations.values()),
                budget
            )
            return {
                "suggested_locations": [
                    {"latitude": l.latitude, "longitude": l.longitude}
                    for l in locations
                ]
            }
        
        elif operation == "analyze_coverage":
            bounds = (
                Location(**params["min_bounds"]),
                Location(**params["max_bounds"])
            )
            analysis = self.analyzer.analyze_coverage(
                list(self.stations.values()),
                bounds,
                params.get("resolution", 0.1)
            )
            return analysis
        
        elif operation == "get_metrics":
            return self.get_metrics()
        
        elif operation == "list_stations":
            return {
                "stations": [
                    {
                        "station_id": s.station_id,
                        "name": s.name,
                        "type": s.station_type.value,
                        "status": s.status.value,
                        "location": {
                            "latitude": s.location.latitude,
                            "longitude": s.location.longitude
                        },
                        "load": s.current_load,
                        "capacity": s.capacity
                    }
                    for s in self.stations.values()
                ]
            }
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def deploy_station(
        self,
        name: str,
        location: Location,
        station_type: BaseStationType = BaseStationType.MICRO,
        coverage_radius: Optional[float] = None,
        capacity: int = 1000
    ) -> BaseStation:
        """Deploy a new base station"""
        async with self._lock:
            if len(self.stations) >= self.config.max_stations:
                raise CapacityError("Maximum station limit reached")
            
            station = BaseStation(
                station_id=str(uuid4()),
                name=name,
                location=location,
                station_type=station_type,
                coverage_radius_km=coverage_radius or self.config.default_coverage_radius,
                capacity=capacity,
                power_watts=self._get_power_for_type(station_type)
            )
            
            self.stations[station.station_id] = station
            self._update_metrics()
            
            self._logger.info(f"Deployed station {station.station_id}: {name}")
            return station
    
    async def remove_station(self, station_id: str) -> bool:
        """Remove a base station"""
        async with self._lock:
            if station_id not in self.stations:
                return False
            
            del self.stations[station_id]
            self._update_metrics()
            
            self._logger.info(f"Removed station {station_id}")
            return True
    
    async def start_station(self, station_id: str) -> bool:
        """Start a base station"""
        async with self._lock:
            if station_id not in self.stations:
                return False
            
            station = self.stations[station_id]
            station.status = StationStatus.STARTING
            
            # Simulate startup
            await asyncio.sleep(0.1)
            
            station.status = StationStatus.ONLINE
            self._update_metrics()
            
            self._logger.info(f"Started station {station_id}")
            return True
    
    async def stop_station(self, station_id: str) -> bool:
        """Stop a base station"""
        async with self._lock:
            if station_id not in self.stations:
                return False
            
            station = self.stations[station_id]
            station.status = StationStatus.OFFLINE
            station.current_load = 0
            self._update_metrics()
            
            self._logger.info(f"Stopped station {station_id}")
            return True
    
    async def update_load(self, station_id: str, load: int) -> bool:
        """Update station load"""
        async with self._lock:
            if station_id not in self.stations:
                return False
            
            station = self.stations[station_id]
            station.current_load = min(load, station.capacity)
            
            # Update status based on load
            if station.current_load >= station.capacity * 0.9:
                station.status = StationStatus.OVERLOADED
            elif station.status == StationStatus.OVERLOADED:
                station.status = StationStatus.ONLINE
            
            self._update_metrics()
            return True
    
    def _get_power_for_type(self, station_type: BaseStationType) -> float:
        """Get power consumption by station type"""
        power_map = {
            BaseStationType.MACRO: 1000.0,
            BaseStationType.MICRO: 200.0,
            BaseStationType.PICO: 50.0,
            BaseStationType.FEMTO: 10.0,
            BaseStationType.SMALL_CELL: 100.0,
        }
        return power_map.get(station_type, 100.0)
    
    def _update_metrics(self) -> None:
        """Update deployment metrics"""
        stations = list(self.stations.values())
        
        self.metrics.total_stations = len(stations)
        self.metrics.online_stations = sum(
            1 for s in stations if s.status == StationStatus.ONLINE
        )
        
        self.metrics.total_coverage_area_km2 = sum(
            math.pi * s.coverage_radius_km ** 2
            for s in stations if s.status == StationStatus.ONLINE
        )
        
        self.metrics.total_capacity = sum(s.capacity for s in stations)
        self.metrics.current_total_load = sum(s.current_load for s in stations)
        
        if self.metrics.total_capacity > 0:
            self.metrics.avg_station_utilization = (
                self.metrics.current_total_load / self.metrics.total_capacity
            )
        
        self.metrics.power_consumption_watts = sum(
            s.power_watts for s in stations if s.status == StationStatus.ONLINE
        )
    
    async def _health_check_loop(self) -> None:
        """Background health check loop"""
        while not self._shutting_down:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                async with self._lock:
                    for station in self.stations.values():
                        if station.status == StationStatus.ONLINE:
                            station.last_health_check = time.time()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Health check error: {e}")
    
    async def _auto_scale_loop(self) -> None:
        """Background auto-scaling loop"""
        while not self._shutting_down:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                async with self._lock:
                    # Check for overloaded stations
                    overloaded = [
                        s for s in self.stations.values()
                        if s.status == StationStatus.OVERLOADED
                    ]
                    
                    if overloaded:
                        self._logger.warning(
                            f"{len(overloaded)} stations overloaded, "
                            "consider deploying more stations"
                        )
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Auto-scale error: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get deployment metrics"""
        return {
            "total_stations": self.metrics.total_stations,
            "online_stations": self.metrics.online_stations,
            "total_coverage_area_km2": round(self.metrics.total_coverage_area_km2, 2),
            "total_capacity": self.metrics.total_capacity,
            "current_total_load": self.metrics.current_total_load,
            "avg_station_utilization": round(self.metrics.avg_station_utilization, 4),
            "power_consumption_watts": self.metrics.power_consumption_watts
        }
    
    async def shutdown(self) -> None:
        """Shutdown the deployment module"""
        self._logger.info("Shutting down BaseStationDeploymentModule...")
        self._shutting_down = True
        
        # Cancel background tasks
        for task in [self._health_check_task, self._auto_scale_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.stations.clear()
        self._initialized = False
        self._logger.info("BaseStationDeploymentModule shutdown complete")


# Example usage
async def main():
    """Example usage of BaseStationDeploymentModule"""
    
    config = DeploymentConfig(
        max_stations=100,
        default_coverage_radius=1.0,
        strategy=DeploymentStrategy.HYBRID
    )
    
    module = BaseStationDeploymentModule(config)
    await module.initialize()
    
    try:
        # Deploy stations
        station1 = await module.deploy_station(
            name="Downtown Tower",
            location=Location(40.7128, -74.0060),
            station_type=BaseStationType.MACRO,
            coverage_radius=2.0,
            capacity=5000
        )
        
        station2 = await module.deploy_station(
            name="Midtown Cell",
            location=Location(40.7549, -73.9840),
            station_type=BaseStationType.MICRO,
            coverage_radius=0.5,
            capacity=1000
        )
        
        # Start stations
        await module.start_station(station1.station_id)
        await module.start_station(station2.station_id)
        
        # Update load
        await module.update_load(station1.station_id, 3500)
        await module.update_load(station2.station_id, 800)
        
        # Analyze coverage
        analysis = module.analyzer.analyze_coverage(
            list(module.stations.values()),
            (Location(40.70, -74.02), Location(40.76, -73.97)),
            resolution=0.01
        )
        
        print(f"Coverage ratio: {analysis['coverage_ratio']:.1%}")
        
        # Get metrics
        metrics = module.get_metrics()
        print(f"\nMetrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        
        # Optimize placement for new demand
        demand_points = [
            (Location(40.72, -74.00), 500),
            (Location(40.73, -73.99), 800),
            (Location(40.74, -73.98), 300),
        ]
        
        suggested = module.optimizer.optimize_placement(
            demand_points,
            list(module.stations.values()),
            budget=50000
        )
        
        print(f"\nSuggested new locations: {len(suggested)}")
        for loc in suggested:
            print(f"  ({loc.latitude:.4f}, {loc.longitude:.4f})")
    
    finally:
        await module.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

