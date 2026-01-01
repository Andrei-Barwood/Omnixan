"""
OMNIXAN Compute-Storage Integrated Module
supercomputing_interconnect_cloud/compute_storage_integrated_module

Production-ready compute-storage integration module that provides unified
access to compute and storage resources, enabling near-data computing,
intelligent data placement, and optimized data pipelines.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4
import hashlib

import numpy as np

from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StorageTier(str, Enum):
    """Storage tiers"""
    L1_CACHE = "l1_cache"  # GPU L1/L2 cache
    HBM = "hbm"  # High Bandwidth Memory
    DRAM = "dram"  # System DRAM
    NVME = "nvme"  # NVMe SSD
    OBJECT = "object"  # Object storage


class ComputeType(str, Enum):
    """Compute resource types"""
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    FPGA = "fpga"
    DPU = "dpu"  # Data Processing Unit


class DataPlacement(str, Enum):
    """Data placement strategies"""
    COMPUTE_LOCAL = "compute_local"  # Near compute
    STORAGE_LOCAL = "storage_local"  # Near storage
    BALANCED = "balanced"
    HOT_COLD = "hot_cold"  # Tiered based on access
    REPLICATED = "replicated"


class OperationType(str, Enum):
    """Operation types"""
    READ = "read"
    WRITE = "write"
    COMPUTE = "compute"
    TRANSFORM = "transform"
    AGGREGATE = "aggregate"


@dataclass
class DataObject:
    """A data object in the system"""
    object_id: str
    name: str
    size_bytes: int
    tier: StorageTier
    location: str  # Node/device ID
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    checksum: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    data: Optional[np.ndarray] = None


@dataclass
class ComputeNode:
    """A compute node"""
    node_id: str
    name: str
    compute_type: ComputeType
    cores: int
    memory_gb: float
    bandwidth_gbps: float
    available_memory_gb: float
    utilization: float = 0.0
    is_online: bool = True


@dataclass
class StorageNode:
    """A storage node"""
    node_id: str
    name: str
    tier: StorageTier
    capacity_gb: float
    used_gb: float = 0.0
    bandwidth_gbps: float = 1.0
    iops: int = 10000
    latency_us: float = 100.0
    is_online: bool = True


@dataclass
class DataPipeline:
    """A data processing pipeline"""
    pipeline_id: str
    stages: List[Tuple[OperationType, Dict[str, Any]]]
    source_objects: List[str]
    target_tier: StorageTier
    status: str = "pending"
    created_at: float = field(default_factory=time.time)


@dataclass
class CSIMetrics:
    """Compute-storage integration metrics"""
    total_objects: int = 0
    total_bytes_stored: int = 0
    total_reads: int = 0
    total_writes: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_read_latency_ms: float = 0.0
    avg_write_latency_ms: float = 0.0
    data_movement_bytes: int = 0
    compute_utilization: float = 0.0


class CSIConfig(BaseModel):
    """Configuration for compute-storage integration"""
    cache_size_mb: int = Field(
        default=1024,
        ge=64,
        description="Cache size in MB"
    )
    placement_strategy: DataPlacement = Field(
        default=DataPlacement.BALANCED,
        description="Default data placement strategy"
    )
    enable_prefetch: bool = Field(
        default=True,
        description="Enable data prefetching"
    )
    prefetch_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Access frequency threshold for prefetch"
    )
    enable_tiering: bool = Field(
        default=True,
        description="Enable automatic data tiering"
    )
    hot_data_threshold: int = Field(
        default=10,
        ge=1,
        description="Access count for hot data classification"
    )


class CSIError(Exception):
    """Base exception for CSI errors"""
    pass


# ============================================================================
# Cache Manager
# ============================================================================

class CacheManager:
    """Manages data caching"""
    
    def __init__(self, max_size_bytes: int):
        self.max_size = max_size_bytes
        self.current_size = 0
        self.cache: Dict[str, Tuple[np.ndarray, float]] = {}  # id -> (data, timestamp)
        self.access_order: List[str] = []
    
    def get(self, object_id: str) -> Optional[np.ndarray]:
        """Get object from cache"""
        if object_id in self.cache:
            data, _ = self.cache[object_id]
            self.cache[object_id] = (data, time.time())
            
            # Update LRU order
            if object_id in self.access_order:
                self.access_order.remove(object_id)
            self.access_order.append(object_id)
            
            return data
        return None
    
    def put(self, object_id: str, data: np.ndarray) -> bool:
        """Put object in cache"""
        size = data.nbytes
        
        # Evict if necessary
        while self.current_size + size > self.max_size and self.access_order:
            self._evict_lru()
        
        if self.current_size + size <= self.max_size:
            self.cache[object_id] = (data.copy(), time.time())
            self.current_size += size
            self.access_order.append(object_id)
            return True
        
        return False
    
    def _evict_lru(self) -> None:
        """Evict least recently used"""
        if self.access_order:
            lru_id = self.access_order.pop(0)
            if lru_id in self.cache:
                data, _ = self.cache.pop(lru_id)
                self.current_size -= data.nbytes
    
    def invalidate(self, object_id: str) -> None:
        """Invalidate cache entry"""
        if object_id in self.cache:
            data, _ = self.cache.pop(object_id)
            self.current_size -= data.nbytes
            if object_id in self.access_order:
                self.access_order.remove(object_id)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size_bytes": self.current_size,
            "max_size_bytes": self.max_size,
            "utilization": self.current_size / self.max_size,
            "entries": len(self.cache)
        }


# ============================================================================
# Data Placement Engine
# ============================================================================

class DataPlacementEngine:
    """Decides optimal data placement"""
    
    def __init__(self, config: CSIConfig):
        self.config = config
    
    def compute_placement(
        self,
        obj: DataObject,
        compute_nodes: List[ComputeNode],
        storage_nodes: List[StorageNode]
    ) -> Tuple[str, StorageTier]:
        """Compute optimal placement for object"""
        if self.config.placement_strategy == DataPlacement.COMPUTE_LOCAL:
            return self._compute_local(obj, compute_nodes, storage_nodes)
        elif self.config.placement_strategy == DataPlacement.STORAGE_LOCAL:
            return self._storage_local(obj, storage_nodes)
        elif self.config.placement_strategy == DataPlacement.HOT_COLD:
            return self._hot_cold(obj, storage_nodes)
        else:
            return self._balanced(obj, compute_nodes, storage_nodes)
    
    def _compute_local(
        self,
        obj: DataObject,
        compute_nodes: List[ComputeNode],
        storage_nodes: List[StorageNode]
    ) -> Tuple[str, StorageTier]:
        """Place data near compute"""
        # Find compute node with best bandwidth
        best_node = max(
            [n for n in compute_nodes if n.is_online],
            key=lambda n: n.bandwidth_gbps,
            default=None
        )
        
        if best_node:
            return best_node.node_id, StorageTier.HBM
        
        return storage_nodes[0].node_id if storage_nodes else "", StorageTier.DRAM
    
    def _storage_local(
        self,
        obj: DataObject,
        storage_nodes: List[StorageNode]
    ) -> Tuple[str, StorageTier]:
        """Place data on storage tier"""
        # Find storage node with capacity
        for tier in [StorageTier.NVME, StorageTier.DRAM, StorageTier.OBJECT]:
            for node in storage_nodes:
                if node.tier == tier and node.is_online:
                    free_gb = node.capacity_gb - node.used_gb
                    if free_gb * 1e9 >= obj.size_bytes:
                        return node.node_id, tier
        
        return storage_nodes[0].node_id if storage_nodes else "", StorageTier.OBJECT
    
    def _hot_cold(
        self,
        obj: DataObject,
        storage_nodes: List[StorageNode]
    ) -> Tuple[str, StorageTier]:
        """Place based on access patterns"""
        if obj.access_count >= self.config.hot_data_threshold:
            tier = StorageTier.HBM
        elif obj.access_count >= self.config.hot_data_threshold // 2:
            tier = StorageTier.DRAM
        else:
            tier = StorageTier.NVME
        
        for node in storage_nodes:
            if node.tier == tier and node.is_online:
                return node.node_id, tier
        
        return storage_nodes[0].node_id if storage_nodes else "", StorageTier.DRAM
    
    def _balanced(
        self,
        obj: DataObject,
        compute_nodes: List[ComputeNode],
        storage_nodes: List[StorageNode]
    ) -> Tuple[str, StorageTier]:
        """Balanced placement"""
        # Consider both compute and storage proximity
        if obj.access_count > 5:
            return self._compute_local(obj, compute_nodes, storage_nodes)
        else:
            return self._storage_local(obj, storage_nodes)


# ============================================================================
# Main Module Implementation
# ============================================================================

class ComputeStorageIntegratedModule:
    """
    Production-ready Compute-Storage Integration module for OMNIXAN.
    
    Provides:
    - Unified compute-storage access
    - Intelligent data placement
    - Multi-tier caching
    - Data pipelines
    - Near-data computing
    """
    
    def __init__(self, config: Optional[CSIConfig] = None):
        """Initialize the CSI Module"""
        self.config = config or CSIConfig()
        
        self.cache = CacheManager(self.config.cache_size_mb * 1024 * 1024)
        self.placement_engine = DataPlacementEngine(self.config)
        
        self.objects: Dict[str, DataObject] = {}
        self.compute_nodes: Dict[str, ComputeNode] = {}
        self.storage_nodes: Dict[str, StorageNode] = {}
        self.pipelines: Dict[str, DataPipeline] = {}
        
        self.metrics = CSIMetrics()
        self._initialized = False
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the CSI module"""
        if self._initialized:
            self._logger.warning("Module already initialized")
            return
        
        try:
            self._logger.info("Initializing ComputeStorageIntegratedModule...")
            
            # Initialize default storage tiers
            await self._init_default_nodes()
            
            self._initialized = True
            self._logger.info("ComputeStorageIntegratedModule initialized successfully")
        
        except Exception as e:
            self._logger.error(f"Initialization failed: {str(e)}")
            raise CSIError(f"Failed to initialize module: {str(e)}")
    
    async def _init_default_nodes(self) -> None:
        """Initialize default compute and storage nodes"""
        # Default compute node
        await self.register_compute_node(
            name="default_compute",
            compute_type=ComputeType.CPU,
            cores=8,
            memory_gb=32.0,
            bandwidth_gbps=100.0
        )
        
        # Default storage tiers
        for tier, capacity, latency in [
            (StorageTier.HBM, 16.0, 10.0),
            (StorageTier.DRAM, 128.0, 50.0),
            (StorageTier.NVME, 1024.0, 100.0),
            (StorageTier.OBJECT, 10240.0, 1000.0)
        ]:
            await self.register_storage_node(
                name=f"default_{tier.value}",
                tier=tier,
                capacity_gb=capacity,
                latency_us=latency
            )
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute CSI operation"""
        if not self._initialized:
            raise CSIError("Module not initialized")
        
        operation = params.get("operation")
        
        if operation == "store":
            name = params["name"]
            data = np.array(params["data"])
            tier = StorageTier(params.get("tier", "dram"))
            obj = await self.store(name, data, tier)
            return {"object_id": obj.object_id, "size_bytes": obj.size_bytes}
        
        elif operation == "load":
            object_id = params["object_id"]
            data = await self.load(object_id)
            if data is not None:
                return {"data": data.tolist(), "shape": list(data.shape)}
            return {"error": "Object not found"}
        
        elif operation == "delete":
            object_id = params["object_id"]
            success = await self.delete(object_id)
            return {"success": success}
        
        elif operation == "migrate":
            object_id = params["object_id"]
            target_tier = StorageTier(params["target_tier"])
            success = await self.migrate(object_id, target_tier)
            return {"success": success}
        
        elif operation == "create_pipeline":
            stages = [
                (OperationType(s["op"]), s.get("params", {}))
                for s in params["stages"]
            ]
            source_ids = params["source_objects"]
            target_tier = StorageTier(params.get("target_tier", "dram"))
            pipeline = await self.create_pipeline(stages, source_ids, target_tier)
            return {"pipeline_id": pipeline.pipeline_id}
        
        elif operation == "run_pipeline":
            pipeline_id = params["pipeline_id"]
            result = await self.run_pipeline(pipeline_id)
            if result is not None:
                return {"result": result.tolist(), "shape": list(result.shape)}
            return {"error": "Pipeline failed"}
        
        elif operation == "get_metrics":
            return self.get_metrics()
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def register_compute_node(
        self,
        name: str,
        compute_type: ComputeType,
        cores: int,
        memory_gb: float,
        bandwidth_gbps: float
    ) -> ComputeNode:
        """Register a compute node"""
        async with self._lock:
            node = ComputeNode(
                node_id=str(uuid4()),
                name=name,
                compute_type=compute_type,
                cores=cores,
                memory_gb=memory_gb,
                bandwidth_gbps=bandwidth_gbps,
                available_memory_gb=memory_gb
            )
            self.compute_nodes[node.node_id] = node
            return node
    
    async def register_storage_node(
        self,
        name: str,
        tier: StorageTier,
        capacity_gb: float,
        latency_us: float = 100.0,
        bandwidth_gbps: float = 10.0
    ) -> StorageNode:
        """Register a storage node"""
        async with self._lock:
            node = StorageNode(
                node_id=str(uuid4()),
                name=name,
                tier=tier,
                capacity_gb=capacity_gb,
                latency_us=latency_us,
                bandwidth_gbps=bandwidth_gbps
            )
            self.storage_nodes[node.node_id] = node
            return node
    
    async def store(
        self,
        name: str,
        data: np.ndarray,
        tier: StorageTier = StorageTier.DRAM
    ) -> DataObject:
        """Store data object"""
        async with self._lock:
            start_time = time.time()
            
            object_id = str(uuid4())
            checksum = hashlib.md5(data.tobytes()).hexdigest()
            
            # Determine placement
            node_id, actual_tier = self.placement_engine.compute_placement(
                DataObject(
                    object_id=object_id,
                    name=name,
                    size_bytes=data.nbytes,
                    tier=tier,
                    location=""
                ),
                list(self.compute_nodes.values()),
                list(self.storage_nodes.values())
            )
            
            obj = DataObject(
                object_id=object_id,
                name=name,
                size_bytes=data.nbytes,
                tier=actual_tier,
                location=node_id,
                checksum=checksum,
                data=data.copy()
            )
            
            self.objects[object_id] = obj
            
            # Update storage node
            for node in self.storage_nodes.values():
                if node.node_id == node_id:
                    node.used_gb += data.nbytes / 1e9
                    break
            
            # Cache if hot tier
            if actual_tier in [StorageTier.HBM, StorageTier.DRAM]:
                self.cache.put(object_id, data)
            
            elapsed = (time.time() - start_time) * 1000
            
            self.metrics.total_objects += 1
            self.metrics.total_bytes_stored += data.nbytes
            self.metrics.total_writes += 1
            self._update_avg_write_latency(elapsed)
            
            return obj
    
    async def load(self, object_id: str) -> Optional[np.ndarray]:
        """Load data object"""
        async with self._lock:
            start_time = time.time()
            
            # Check cache first
            cached = self.cache.get(object_id)
            if cached is not None:
                self.metrics.cache_hits += 1
                self.metrics.total_reads += 1
                return cached
            
            self.metrics.cache_misses += 1
            
            if object_id not in self.objects:
                return None
            
            obj = self.objects[object_id]
            obj.access_count += 1
            obj.last_accessed = time.time()
            
            # Cache the data
            if obj.data is not None:
                self.cache.put(object_id, obj.data)
            
            elapsed = (time.time() - start_time) * 1000
            self.metrics.total_reads += 1
            self._update_avg_read_latency(elapsed)
            
            # Consider promotion if frequently accessed
            if self.config.enable_tiering:
                await self._consider_promotion(obj)
            
            return obj.data
    
    async def delete(self, object_id: str) -> bool:
        """Delete data object"""
        async with self._lock:
            if object_id not in self.objects:
                return False
            
            obj = self.objects[object_id]
            
            # Update storage node
            for node in self.storage_nodes.values():
                if node.node_id == obj.location:
                    node.used_gb -= obj.size_bytes / 1e9
                    break
            
            # Invalidate cache
            self.cache.invalidate(object_id)
            
            self.metrics.total_bytes_stored -= obj.size_bytes
            self.metrics.total_objects -= 1
            
            del self.objects[object_id]
            
            return True
    
    async def migrate(
        self,
        object_id: str,
        target_tier: StorageTier
    ) -> bool:
        """Migrate object to different tier"""
        async with self._lock:
            if object_id not in self.objects:
                return False
            
            obj = self.objects[object_id]
            
            # Find target storage node
            target_node = None
            for node in self.storage_nodes.values():
                if node.tier == target_tier and node.is_online:
                    free_gb = node.capacity_gb - node.used_gb
                    if free_gb * 1e9 >= obj.size_bytes:
                        target_node = node
                        break
            
            if not target_node:
                return False
            
            # Update old node
            for node in self.storage_nodes.values():
                if node.node_id == obj.location:
                    node.used_gb -= obj.size_bytes / 1e9
                    break
            
            # Update object
            obj.tier = target_tier
            obj.location = target_node.node_id
            target_node.used_gb += obj.size_bytes / 1e9
            
            self.metrics.data_movement_bytes += obj.size_bytes
            
            return True
    
    async def _consider_promotion(self, obj: DataObject) -> None:
        """Consider promoting hot data to faster tier"""
        if obj.access_count >= self.config.hot_data_threshold:
            if obj.tier not in [StorageTier.HBM, StorageTier.L1_CACHE]:
                await self.migrate(obj.object_id, StorageTier.HBM)
    
    async def create_pipeline(
        self,
        stages: List[Tuple[OperationType, Dict[str, Any]]],
        source_objects: List[str],
        target_tier: StorageTier
    ) -> DataPipeline:
        """Create a data pipeline"""
        pipeline = DataPipeline(
            pipeline_id=str(uuid4()),
            stages=stages,
            source_objects=source_objects,
            target_tier=target_tier
        )
        self.pipelines[pipeline.pipeline_id] = pipeline
        return pipeline
    
    async def run_pipeline(self, pipeline_id: str) -> Optional[np.ndarray]:
        """Execute a data pipeline"""
        if pipeline_id not in self.pipelines:
            return None
        
        pipeline = self.pipelines[pipeline_id]
        pipeline.status = "running"
        
        try:
            # Load source data
            data_list = []
            for obj_id in pipeline.source_objects:
                data = await self.load(obj_id)
                if data is not None:
                    data_list.append(data)
            
            if not data_list:
                pipeline.status = "failed"
                return None
            
            result = data_list[0] if len(data_list) == 1 else np.concatenate(data_list)
            
            # Execute stages
            for op_type, params in pipeline.stages:
                result = self._execute_stage(result, op_type, params)
            
            pipeline.status = "completed"
            return result
        
        except Exception as e:
            self._logger.error(f"Pipeline failed: {e}")
            pipeline.status = "failed"
            return None
    
    def _execute_stage(
        self,
        data: np.ndarray,
        op_type: OperationType,
        params: Dict[str, Any]
    ) -> np.ndarray:
        """Execute single pipeline stage"""
        if op_type == OperationType.TRANSFORM:
            func = params.get("func", "identity")
            if func == "normalize":
                return (data - np.mean(data)) / np.std(data)
            elif func == "sqrt":
                return np.sqrt(np.abs(data))
            elif func == "log":
                return np.log1p(np.abs(data))
        
        elif op_type == OperationType.AGGREGATE:
            axis = params.get("axis", None)
            agg = params.get("agg", "sum")
            if agg == "sum":
                return np.sum(data, axis=axis, keepdims=True)
            elif agg == "mean":
                return np.mean(data, axis=axis, keepdims=True)
            elif agg == "max":
                return np.max(data, axis=axis, keepdims=True)
        
        return data
    
    def _update_avg_read_latency(self, elapsed: float) -> None:
        """Update average read latency"""
        n = self.metrics.total_reads
        self.metrics.avg_read_latency_ms = (
            (self.metrics.avg_read_latency_ms * (n - 1) + elapsed) / n
        )
    
    def _update_avg_write_latency(self, elapsed: float) -> None:
        """Update average write latency"""
        n = self.metrics.total_writes
        self.metrics.avg_write_latency_ms = (
            (self.metrics.avg_write_latency_ms * (n - 1) + elapsed) / n
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get CSI metrics"""
        hit_rate = 0
        if self.metrics.cache_hits + self.metrics.cache_misses > 0:
            hit_rate = self.metrics.cache_hits / (
                self.metrics.cache_hits + self.metrics.cache_misses
            )
        
        return {
            "total_objects": self.metrics.total_objects,
            "total_bytes_stored": self.metrics.total_bytes_stored,
            "total_reads": self.metrics.total_reads,
            "total_writes": self.metrics.total_writes,
            "cache_hit_rate": round(hit_rate, 4),
            "avg_read_latency_ms": round(self.metrics.avg_read_latency_ms, 4),
            "avg_write_latency_ms": round(self.metrics.avg_write_latency_ms, 4),
            "data_movement_bytes": self.metrics.data_movement_bytes,
            "cache_stats": self.cache.stats(),
            "compute_nodes": len(self.compute_nodes),
            "storage_nodes": len(self.storage_nodes)
        }
    
    async def shutdown(self) -> None:
        """Shutdown the CSI module"""
        self._logger.info("Shutting down ComputeStorageIntegratedModule...")
        
        self.objects.clear()
        self.compute_nodes.clear()
        self.storage_nodes.clear()
        self.pipelines.clear()
        self._initialized = False
        
        self._logger.info("ComputeStorageIntegratedModule shutdown complete")


# Example usage
async def main():
    """Example usage of ComputeStorageIntegratedModule"""
    
    config = CSIConfig(
        cache_size_mb=256,
        placement_strategy=DataPlacement.BALANCED,
        enable_tiering=True
    )
    
    module = ComputeStorageIntegratedModule(config)
    await module.initialize()
    
    try:
        # Store data
        data1 = np.random.randn(1000, 256).astype(np.float32)
        obj1 = await module.store("dataset1", data1, StorageTier.DRAM)
        print(f"Stored object: {obj1.object_id}, size: {obj1.size_bytes} bytes")
        
        data2 = np.random.randn(500, 256).astype(np.float32)
        obj2 = await module.store("dataset2", data2, StorageTier.NVME)
        print(f"Stored object: {obj2.object_id}, size: {obj2.size_bytes} bytes")
        
        # Load data (cache miss)
        loaded = await module.load(obj1.object_id)
        print(f"Loaded data shape: {loaded.shape}")
        
        # Load again (cache hit)
        loaded = await module.load(obj1.object_id)
        print(f"Loaded from cache: {loaded.shape}")
        
        # Create and run pipeline
        pipeline = await module.create_pipeline(
            stages=[
                (OperationType.TRANSFORM, {"func": "normalize"}),
                (OperationType.AGGREGATE, {"agg": "mean", "axis": 0})
            ],
            source_objects=[obj1.object_id],
            target_tier=StorageTier.DRAM
        )
        
        result = await module.run_pipeline(pipeline.pipeline_id)
        print(f"Pipeline result shape: {result.shape}")
        
        # Migrate data
        await module.migrate(obj2.object_id, StorageTier.DRAM)
        print("Migrated data to DRAM")
        
        # Get metrics
        metrics = module.get_metrics()
        print(f"\nMetrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    
    finally:
        await module.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

