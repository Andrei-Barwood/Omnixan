"""
OMNIXAN Tensor Slicing Module
supercomputing_interconnect_cloud/tensor_slicing_module

Production-ready tensor slicing and partitioning module for distributed
tensor operations, model parallelism, and efficient memory management
across multiple devices.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np

from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SlicingStrategy(str, Enum):
    """Tensor slicing strategies"""
    ROW = "row"  # Slice along rows
    COLUMN = "column"  # Slice along columns
    BLOCK = "block"  # 2D block slicing
    CHANNEL = "channel"  # Slice along channel dimension
    BATCH = "batch"  # Slice along batch dimension
    CUSTOM = "custom"  # Custom slicing pattern


class ParallelismType(str, Enum):
    """Types of parallelism"""
    DATA = "data"  # Data parallelism
    MODEL = "model"  # Model/tensor parallelism
    PIPELINE = "pipeline"  # Pipeline parallelism
    EXPERT = "expert"  # Mixture of experts


class ReduceOperation(str, Enum):
    """Reduction operations for gathering slices"""
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    CONCAT = "concat"


@dataclass
class SliceSpec:
    """Specification for a tensor slice"""
    slice_id: str
    original_shape: Tuple[int, ...]
    slice_shape: Tuple[int, ...]
    start_indices: Tuple[int, ...]
    end_indices: Tuple[int, ...]
    device_id: int = 0
    
    @property
    def slices(self) -> Tuple[slice, ...]:
        return tuple(
            slice(start, end)
            for start, end in zip(self.start_indices, self.end_indices)
        )


@dataclass
class TensorPartition:
    """A partitioned tensor"""
    partition_id: str
    slices: List[SliceSpec]
    strategy: SlicingStrategy
    num_devices: int
    original_shape: Tuple[int, ...]


@dataclass
class SlicingMetrics:
    """Tensor slicing metrics"""
    total_slices_created: int = 0
    total_bytes_sliced: int = 0
    total_gather_operations: int = 0
    avg_slice_time_ms: float = 0.0
    avg_gather_time_ms: float = 0.0
    communication_overhead_ms: float = 0.0


class SlicingConfig(BaseModel):
    """Configuration for tensor slicing"""
    num_devices: int = Field(
        default=1,
        ge=1,
        description="Number of devices for distribution"
    )
    default_strategy: SlicingStrategy = Field(
        default=SlicingStrategy.ROW,
        description="Default slicing strategy"
    )
    min_slice_size: int = Field(
        default=1024,
        ge=1,
        description="Minimum elements per slice"
    )
    enable_overlap: bool = Field(
        default=False,
        description="Enable overlapping slices for halo exchange"
    )
    overlap_size: int = Field(
        default=0,
        ge=0,
        description="Overlap size for halo exchange"
    )
    balance_workload: bool = Field(
        default=True,
        description="Balance slice sizes across devices"
    )


class SlicingError(Exception):
    """Base exception for slicing errors"""
    pass


# ============================================================================
# Slicing Strategies
# ============================================================================

class SlicerBase(ABC):
    """Abstract base class for slicing strategies"""
    
    @abstractmethod
    def slice(
        self,
        tensor: np.ndarray,
        num_slices: int,
        overlap: int = 0
    ) -> List[Tuple[np.ndarray, SliceSpec]]:
        """Slice tensor into parts"""
        pass
    
    @abstractmethod
    def gather(
        self,
        slices: List[Tuple[np.ndarray, SliceSpec]],
        reduce_op: ReduceOperation = ReduceOperation.CONCAT
    ) -> np.ndarray:
        """Gather slices back into full tensor"""
        pass


class RowSlicer(SlicerBase):
    """Slice tensor along row dimension"""
    
    def slice(
        self,
        tensor: np.ndarray,
        num_slices: int,
        overlap: int = 0
    ) -> List[Tuple[np.ndarray, SliceSpec]]:
        if tensor.ndim < 1:
            raise SlicingError("Tensor must have at least 1 dimension")
        
        rows = tensor.shape[0]
        base_size = rows // num_slices
        remainder = rows % num_slices
        
        slices = []
        start = 0
        
        for i in range(num_slices):
            # Distribute remainder evenly
            size = base_size + (1 if i < remainder else 0)
            
            # Add overlap
            actual_start = max(0, start - overlap)
            actual_end = min(rows, start + size + overlap)
            
            slice_data = tensor[actual_start:actual_end]
            
            spec = SliceSpec(
                slice_id=str(uuid4()),
                original_shape=tensor.shape,
                slice_shape=slice_data.shape,
                start_indices=(actual_start,) + (0,) * (tensor.ndim - 1),
                end_indices=(actual_end,) + tensor.shape[1:],
                device_id=i
            )
            
            slices.append((slice_data.copy(), spec))
            start += size
        
        return slices
    
    def gather(
        self,
        slices: List[Tuple[np.ndarray, SliceSpec]],
        reduce_op: ReduceOperation = ReduceOperation.CONCAT
    ) -> np.ndarray:
        if reduce_op == ReduceOperation.CONCAT:
            # Sort by start index
            sorted_slices = sorted(slices, key=lambda x: x[1].start_indices[0])
            return np.concatenate([s[0] for s in sorted_slices], axis=0)
        elif reduce_op == ReduceOperation.SUM:
            return sum(s[0] for s in slices)
        elif reduce_op == ReduceOperation.MEAN:
            return sum(s[0] for s in slices) / len(slices)
        elif reduce_op == ReduceOperation.MAX:
            return np.maximum.reduce([s[0] for s in slices])
        elif reduce_op == ReduceOperation.MIN:
            return np.minimum.reduce([s[0] for s in slices])


class ColumnSlicer(SlicerBase):
    """Slice tensor along column dimension"""
    
    def slice(
        self,
        tensor: np.ndarray,
        num_slices: int,
        overlap: int = 0
    ) -> List[Tuple[np.ndarray, SliceSpec]]:
        if tensor.ndim < 2:
            raise SlicingError("Tensor must have at least 2 dimensions")
        
        cols = tensor.shape[1]
        base_size = cols // num_slices
        remainder = cols % num_slices
        
        slices = []
        start = 0
        
        for i in range(num_slices):
            size = base_size + (1 if i < remainder else 0)
            
            actual_start = max(0, start - overlap)
            actual_end = min(cols, start + size + overlap)
            
            slice_data = tensor[:, actual_start:actual_end]
            
            spec = SliceSpec(
                slice_id=str(uuid4()),
                original_shape=tensor.shape,
                slice_shape=slice_data.shape,
                start_indices=(0, actual_start) + (0,) * (tensor.ndim - 2),
                end_indices=(tensor.shape[0], actual_end) + tensor.shape[2:],
                device_id=i
            )
            
            slices.append((slice_data.copy(), spec))
            start += size
        
        return slices
    
    def gather(
        self,
        slices: List[Tuple[np.ndarray, SliceSpec]],
        reduce_op: ReduceOperation = ReduceOperation.CONCAT
    ) -> np.ndarray:
        if reduce_op == ReduceOperation.CONCAT:
            sorted_slices = sorted(slices, key=lambda x: x[1].start_indices[1])
            return np.concatenate([s[0] for s in sorted_slices], axis=1)
        elif reduce_op == ReduceOperation.SUM:
            return sum(s[0] for s in slices)
        elif reduce_op == ReduceOperation.MEAN:
            return sum(s[0] for s in slices) / len(slices)


class BlockSlicer(SlicerBase):
    """2D block slicing for matrices"""
    
    def slice(
        self,
        tensor: np.ndarray,
        num_slices: int,
        overlap: int = 0
    ) -> List[Tuple[np.ndarray, SliceSpec]]:
        if tensor.ndim < 2:
            raise SlicingError("Tensor must have at least 2 dimensions")
        
        # Find optimal grid
        import math
        grid_size = int(math.sqrt(num_slices))
        if grid_size * grid_size != num_slices:
            grid_size = max(1, grid_size)
            num_slices = grid_size * grid_size
        
        rows, cols = tensor.shape[:2]
        row_size = rows // grid_size
        col_size = cols // grid_size
        
        slices = []
        device_id = 0
        
        for i in range(grid_size):
            for j in range(grid_size):
                row_start = i * row_size
                row_end = (i + 1) * row_size if i < grid_size - 1 else rows
                col_start = j * col_size
                col_end = (j + 1) * col_size if j < grid_size - 1 else cols
                
                slice_data = tensor[row_start:row_end, col_start:col_end]
                
                spec = SliceSpec(
                    slice_id=str(uuid4()),
                    original_shape=tensor.shape,
                    slice_shape=slice_data.shape,
                    start_indices=(row_start, col_start) + (0,) * (tensor.ndim - 2),
                    end_indices=(row_end, col_end) + tensor.shape[2:],
                    device_id=device_id
                )
                
                slices.append((slice_data.copy(), spec))
                device_id += 1
        
        return slices
    
    def gather(
        self,
        slices: List[Tuple[np.ndarray, SliceSpec]],
        reduce_op: ReduceOperation = ReduceOperation.CONCAT
    ) -> np.ndarray:
        if not slices:
            return np.array([])
        
        original_shape = slices[0][1].original_shape
        result = np.zeros(original_shape)
        
        for slice_data, spec in slices:
            result[spec.slices] = slice_data
        
        return result


class BatchSlicer(SlicerBase):
    """Slice along batch dimension"""
    
    def slice(
        self,
        tensor: np.ndarray,
        num_slices: int,
        overlap: int = 0
    ) -> List[Tuple[np.ndarray, SliceSpec]]:
        if tensor.ndim < 1:
            raise SlicingError("Tensor must have at least 1 dimension")
        
        batch_size = tensor.shape[0]
        base_size = batch_size // num_slices
        remainder = batch_size % num_slices
        
        slices = []
        start = 0
        
        for i in range(num_slices):
            size = base_size + (1 if i < remainder else 0)
            end = start + size
            
            slice_data = tensor[start:end]
            
            spec = SliceSpec(
                slice_id=str(uuid4()),
                original_shape=tensor.shape,
                slice_shape=slice_data.shape,
                start_indices=(start,) + (0,) * (tensor.ndim - 1),
                end_indices=(end,) + tensor.shape[1:],
                device_id=i
            )
            
            slices.append((slice_data.copy(), spec))
            start = end
        
        return slices
    
    def gather(
        self,
        slices: List[Tuple[np.ndarray, SliceSpec]],
        reduce_op: ReduceOperation = ReduceOperation.CONCAT
    ) -> np.ndarray:
        sorted_slices = sorted(slices, key=lambda x: x[1].start_indices[0])
        return np.concatenate([s[0] for s in sorted_slices], axis=0)


# ============================================================================
# Main Module Implementation
# ============================================================================

class TensorSlicingModule:
    """
    Production-ready Tensor Slicing module for OMNIXAN.
    
    Provides:
    - Multiple slicing strategies (row, column, block, batch)
    - Model and data parallelism support
    - Efficient gather operations
    - Overlap for halo exchange
    - Workload balancing
    """
    
    def __init__(self, config: Optional[SlicingConfig] = None):
        """Initialize the Tensor Slicing Module"""
        self.config = config or SlicingConfig()
        
        # Slicers registry
        self.slicers: Dict[SlicingStrategy, SlicerBase] = {
            SlicingStrategy.ROW: RowSlicer(),
            SlicingStrategy.COLUMN: ColumnSlicer(),
            SlicingStrategy.BLOCK: BlockSlicer(),
            SlicingStrategy.BATCH: BatchSlicer(),
        }
        
        self.metrics = SlicingMetrics()
        self.partitions: Dict[str, TensorPartition] = {}
        self._initialized = False
        self._logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the tensor slicing module"""
        if self._initialized:
            self._logger.warning("Module already initialized")
            return
        
        try:
            self._logger.info("Initializing TensorSlicingModule...")
            self._initialized = True
            self._logger.info("TensorSlicingModule initialized successfully")
        
        except Exception as e:
            self._logger.error(f"Initialization failed: {str(e)}")
            raise SlicingError(f"Failed to initialize module: {str(e)}")
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute slicing operation"""
        if not self._initialized:
            raise SlicingError("Module not initialized")
        
        operation = params.get("operation")
        
        if operation == "slice":
            tensor = np.array(params["tensor"])
            strategy = SlicingStrategy(params.get("strategy", "row"))
            num_slices = params.get("num_slices", self.config.num_devices)
            
            partition = await self.slice_tensor(tensor, strategy, num_slices)
            
            return {
                "partition_id": partition.partition_id,
                "num_slices": len(partition.slices),
                "slice_shapes": [list(s.slice_shape) for s in partition.slices]
            }
        
        elif operation == "gather":
            partition_id = params["partition_id"]
            reduce_op = ReduceOperation(params.get("reduce_op", "concat"))
            
            result = await self.gather_partition(partition_id, reduce_op)
            return {"result": result.tolist(), "shape": list(result.shape)}
        
        elif operation == "get_slice":
            partition_id = params["partition_id"]
            slice_index = params["slice_index"]
            
            slice_data = self.get_slice(partition_id, slice_index)
            if slice_data is not None:
                return {"slice": slice_data.tolist(), "shape": list(slice_data.shape)}
            return {"error": "Slice not found"}
        
        elif operation == "get_metrics":
            return self.get_metrics()
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def slice_tensor(
        self,
        tensor: np.ndarray,
        strategy: SlicingStrategy = None,
        num_slices: int = None,
        overlap: int = None
    ) -> TensorPartition:
        """Slice tensor using specified strategy"""
        strategy = strategy or self.config.default_strategy
        num_slices = num_slices or self.config.num_devices
        overlap = overlap if overlap is not None else (
            self.config.overlap_size if self.config.enable_overlap else 0
        )
        
        if strategy not in self.slicers:
            raise SlicingError(f"Unknown slicing strategy: {strategy}")
        
        start_time = time.time()
        
        slicer = self.slicers[strategy]
        slices = slicer.slice(tensor, num_slices, overlap)
        
        elapsed = (time.time() - start_time) * 1000
        
        # Create partition
        partition = TensorPartition(
            partition_id=str(uuid4()),
            slices=[spec for _, spec in slices],
            strategy=strategy,
            num_devices=num_slices,
            original_shape=tensor.shape
        )
        
        # Store slices
        self.partitions[partition.partition_id] = partition
        self._slice_data: Dict[str, np.ndarray] = getattr(self, '_slice_data', {})
        for slice_data, spec in slices:
            self._slice_data[spec.slice_id] = slice_data
        
        # Update metrics
        self.metrics.total_slices_created += len(slices)
        self.metrics.total_bytes_sliced += tensor.nbytes
        self._update_avg_slice_time(elapsed)
        
        self._logger.info(
            f"Sliced tensor {tensor.shape} into {len(slices)} parts "
            f"using {strategy.value} strategy"
        )
        
        return partition
    
    async def gather_partition(
        self,
        partition_id: str,
        reduce_op: ReduceOperation = ReduceOperation.CONCAT
    ) -> np.ndarray:
        """Gather partition slices back into full tensor"""
        if partition_id not in self.partitions:
            raise SlicingError(f"Partition {partition_id} not found")
        
        partition = self.partitions[partition_id]
        start_time = time.time()
        
        # Collect slice data
        slices = []
        for spec in partition.slices:
            slice_data = self._slice_data.get(spec.slice_id)
            if slice_data is not None:
                slices.append((slice_data, spec))
        
        if not slices:
            raise SlicingError("No slice data found")
        
        # Gather using appropriate slicer
        slicer = self.slicers[partition.strategy]
        result = slicer.gather(slices, reduce_op)
        
        elapsed = (time.time() - start_time) * 1000
        
        self.metrics.total_gather_operations += 1
        self._update_avg_gather_time(elapsed)
        
        return result
    
    def get_slice(self, partition_id: str, slice_index: int) -> Optional[np.ndarray]:
        """Get specific slice from partition"""
        if partition_id not in self.partitions:
            return None
        
        partition = self.partitions[partition_id]
        
        if slice_index >= len(partition.slices):
            return None
        
        spec = partition.slices[slice_index]
        return self._slice_data.get(spec.slice_id)
    
    def _update_avg_slice_time(self, elapsed: float) -> None:
        """Update average slicing time"""
        n = self.metrics.total_slices_created
        if n > 0:
            self.metrics.avg_slice_time_ms = (
                (self.metrics.avg_slice_time_ms * (n - 1) + elapsed) / n
            )
    
    def _update_avg_gather_time(self, elapsed: float) -> None:
        """Update average gather time"""
        n = self.metrics.total_gather_operations
        if n > 0:
            self.metrics.avg_gather_time_ms = (
                (self.metrics.avg_gather_time_ms * (n - 1) + elapsed) / n
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get slicing metrics"""
        return {
            "total_slices_created": self.metrics.total_slices_created,
            "total_bytes_sliced": self.metrics.total_bytes_sliced,
            "total_gather_operations": self.metrics.total_gather_operations,
            "avg_slice_time_ms": round(self.metrics.avg_slice_time_ms, 4),
            "avg_gather_time_ms": round(self.metrics.avg_gather_time_ms, 4),
            "active_partitions": len(self.partitions),
            "num_devices": self.config.num_devices
        }
    
    async def shutdown(self) -> None:
        """Shutdown the tensor slicing module"""
        self._logger.info("Shutting down TensorSlicingModule...")
        
        self.partitions.clear()
        if hasattr(self, '_slice_data'):
            self._slice_data.clear()
        self._initialized = False
        
        self._logger.info("TensorSlicingModule shutdown complete")


# Example usage
async def main():
    """Example usage of TensorSlicingModule"""
    
    config = SlicingConfig(
        num_devices=4,
        default_strategy=SlicingStrategy.ROW,
        balance_workload=True
    )
    
    module = TensorSlicingModule(config)
    await module.initialize()
    
    try:
        # Create test tensor
        tensor = np.random.randn(1000, 512).astype(np.float32)
        print(f"Original tensor shape: {tensor.shape}")
        
        # Row slicing
        partition = await module.slice_tensor(
            tensor,
            strategy=SlicingStrategy.ROW,
            num_slices=4
        )
        
        print(f"\nRow slicing into {len(partition.slices)} parts:")
        for i, spec in enumerate(partition.slices):
            print(f"  Slice {i}: shape={spec.slice_shape}, device={spec.device_id}")
        
        # Gather back
        reconstructed = await module.gather_partition(partition.partition_id)
        print(f"Reconstructed shape: {reconstructed.shape}")
        print(f"Reconstruction error: {np.max(np.abs(tensor - reconstructed)):.6f}")
        
        # Column slicing
        partition = await module.slice_tensor(
            tensor,
            strategy=SlicingStrategy.COLUMN,
            num_slices=4
        )
        
        print(f"\nColumn slicing into {len(partition.slices)} parts:")
        for i, spec in enumerate(partition.slices):
            print(f"  Slice {i}: shape={spec.slice_shape}")
        
        # Block slicing
        tensor_2d = np.random.randn(512, 512).astype(np.float32)
        partition = await module.slice_tensor(
            tensor_2d,
            strategy=SlicingStrategy.BLOCK,
            num_slices=4
        )
        
        print(f"\nBlock slicing into {len(partition.slices)} parts:")
        for i, spec in enumerate(partition.slices):
            print(f"  Block {i}: shape={spec.slice_shape}")
        
        # Get metrics
        metrics = module.get_metrics()
        print(f"\nMetrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    
    finally:
        await module.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

