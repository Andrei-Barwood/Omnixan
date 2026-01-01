"""
OMNIXAN Memory Pooling Module
edge_computing_network/memory_pooling_module

Production-ready shared memory pool implementation with efficient allocation,
deallocation, defragmentation, and multi-tenant memory management for
edge computing environments.
"""

import asyncio
import logging
import time
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4
import weakref

from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AllocationStrategy(str, Enum):
    """Memory allocation strategies"""
    FIRST_FIT = "first_fit"
    BEST_FIT = "best_fit"
    WORST_FIT = "worst_fit"
    BUDDY = "buddy"
    SLAB = "slab"


class MemoryTier(str, Enum):
    """Memory tier types"""
    HOT = "hot"  # Fastest, most expensive
    WARM = "warm"  # Medium speed
    COLD = "cold"  # Slowest, cheapest
    PMEM = "pmem"  # Persistent memory


class BlockState(str, Enum):
    """Memory block states"""
    FREE = "free"
    ALLOCATED = "allocated"
    RESERVED = "reserved"
    PINNED = "pinned"


@dataclass
class MemoryBlock:
    """Represents a memory block"""
    block_id: str
    offset: int
    size: int
    state: BlockState
    owner: Optional[str] = None
    tier: MemoryTier = MemoryTier.HOT
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    pinned: bool = False
    data: Optional[bytearray] = None


@dataclass
class AllocationResult:
    """Result of memory allocation"""
    success: bool
    block_id: Optional[str] = None
    offset: Optional[int] = None
    size: Optional[int] = None
    error: Optional[str] = None


@dataclass
class PoolMetrics:
    """Memory pool metrics"""
    total_size: int = 0
    used_size: int = 0
    free_size: int = 0
    num_allocations: int = 0
    num_deallocations: int = 0
    num_blocks: int = 0
    fragmentation_ratio: float = 0.0
    peak_usage: int = 0
    allocation_failures: int = 0


class MemoryPoolConfig(BaseModel):
    """Configuration for memory pool"""
    pool_size: int = Field(
        default=1024 * 1024 * 1024,  # 1 GB
        ge=1024 * 1024,
        description="Total pool size in bytes"
    )
    min_block_size: int = Field(
        default=64,
        ge=1,
        description="Minimum block size"
    )
    max_block_size: int = Field(
        default=1024 * 1024 * 100,  # 100 MB
        description="Maximum single allocation"
    )
    allocation_strategy: AllocationStrategy = Field(
        default=AllocationStrategy.BEST_FIT,
        description="Allocation strategy"
    )
    enable_defragmentation: bool = Field(
        default=True,
        description="Enable automatic defragmentation"
    )
    defrag_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Fragmentation threshold for defrag"
    )
    enable_tiering: bool = Field(
        default=False,
        description="Enable memory tiering"
    )
    alignment: int = Field(
        default=8,
        ge=1,
        description="Memory alignment in bytes"
    )


class MemoryPoolError(Exception):
    """Base exception for memory pool errors"""
    pass


class AllocationError(MemoryPoolError):
    """Raised when allocation fails"""
    pass


class DeallocationError(MemoryPoolError):
    """Raised when deallocation fails"""
    pass


class OutOfMemoryError(MemoryPoolError):
    """Raised when pool is exhausted"""
    pass


# ============================================================================
# Allocation Strategy Implementations
# ============================================================================

class AllocationStrategyBase(ABC):
    """Abstract base class for allocation strategies"""
    
    @abstractmethod
    def find_block(
        self,
        free_blocks: List[MemoryBlock],
        size: int
    ) -> Optional[MemoryBlock]:
        """Find suitable block for allocation"""
        pass


class FirstFitStrategy(AllocationStrategyBase):
    """First-fit allocation - use first block that fits"""
    
    def find_block(
        self,
        free_blocks: List[MemoryBlock],
        size: int
    ) -> Optional[MemoryBlock]:
        for block in free_blocks:
            if block.size >= size:
                return block
        return None


class BestFitStrategy(AllocationStrategyBase):
    """Best-fit allocation - use smallest block that fits"""
    
    def find_block(
        self,
        free_blocks: List[MemoryBlock],
        size: int
    ) -> Optional[MemoryBlock]:
        suitable = [b for b in free_blocks if b.size >= size]
        if not suitable:
            return None
        return min(suitable, key=lambda b: b.size)


class WorstFitStrategy(AllocationStrategyBase):
    """Worst-fit allocation - use largest block"""
    
    def find_block(
        self,
        free_blocks: List[MemoryBlock],
        size: int
    ) -> Optional[MemoryBlock]:
        suitable = [b for b in free_blocks if b.size >= size]
        if not suitable:
            return None
        return max(suitable, key=lambda b: b.size)


class BuddyAllocator(AllocationStrategyBase):
    """Buddy system allocation"""
    
    def __init__(self, min_block_size: int = 64):
        self.min_block_size = min_block_size
    
    def find_block(
        self,
        free_blocks: List[MemoryBlock],
        size: int
    ) -> Optional[MemoryBlock]:
        # Round up to nearest power of 2
        target_size = self._next_power_of_2(max(size, self.min_block_size))
        
        # Find smallest block that's power of 2 and fits
        suitable = [b for b in free_blocks if b.size >= target_size]
        if not suitable:
            return None
        
        return min(suitable, key=lambda b: b.size)
    
    def _next_power_of_2(self, n: int) -> int:
        if n <= 0:
            return 1
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        return n + 1


# ============================================================================
# Slab Allocator for Fixed-Size Objects
# ============================================================================

class SlabCache:
    """Slab cache for fixed-size objects"""
    
    def __init__(self, object_size: int, slab_size: int = 4096):
        self.object_size = object_size
        self.slab_size = slab_size
        self.objects_per_slab = slab_size // object_size
        
        self.slabs: List[bytearray] = []
        self.free_objects: List[Tuple[int, int]] = []  # (slab_idx, obj_idx)
        self.allocated: Set[Tuple[int, int]] = set()
        self._lock = threading.Lock()
    
    def allocate(self) -> Optional[Tuple[int, int]]:
        """Allocate object from slab"""
        with self._lock:
            if self.free_objects:
                obj_ref = self.free_objects.pop()
                self.allocated.add(obj_ref)
                return obj_ref
            
            # Need new slab
            slab_idx = len(self.slabs)
            self.slabs.append(bytearray(self.slab_size))
            
            # Add all objects except first to free list
            for i in range(1, self.objects_per_slab):
                self.free_objects.append((slab_idx, i))
            
            obj_ref = (slab_idx, 0)
            self.allocated.add(obj_ref)
            return obj_ref
    
    def deallocate(self, obj_ref: Tuple[int, int]) -> None:
        """Return object to slab"""
        with self._lock:
            if obj_ref in self.allocated:
                self.allocated.remove(obj_ref)
                self.free_objects.append(obj_ref)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get slab statistics"""
        return {
            "object_size": self.object_size,
            "num_slabs": len(self.slabs),
            "allocated_objects": len(self.allocated),
            "free_objects": len(self.free_objects),
            "utilization": len(self.allocated) / max(
                len(self.slabs) * self.objects_per_slab, 1
            )
        }


# ============================================================================
# Main Memory Pool Implementation
# ============================================================================

class MemoryPool:
    """
    High-performance memory pool with multiple allocation strategies.
    """
    
    def __init__(self, config: MemoryPoolConfig):
        self.config = config
        self.total_size = config.pool_size
        
        # Memory storage
        self._memory = bytearray(config.pool_size)
        
        # Block tracking
        self.blocks: Dict[str, MemoryBlock] = {}
        self.free_blocks: List[MemoryBlock] = []
        self.allocated_blocks: Dict[str, MemoryBlock] = {}
        
        # Initialize with single free block
        initial_block = MemoryBlock(
            block_id=str(uuid4()),
            offset=0,
            size=config.pool_size,
            state=BlockState.FREE
        )
        self.blocks[initial_block.block_id] = initial_block
        self.free_blocks.append(initial_block)
        
        # Allocation strategy
        self.strategy = self._create_strategy()
        
        # Slab caches for common sizes
        self.slab_caches: Dict[int, SlabCache] = {}
        
        # Metrics
        self.metrics = PoolMetrics(
            total_size=config.pool_size,
            free_size=config.pool_size
        )
        
        # Thread safety
        self._lock = asyncio.Lock()
    
    def _create_strategy(self) -> AllocationStrategyBase:
        """Create allocation strategy"""
        strategies = {
            AllocationStrategy.FIRST_FIT: FirstFitStrategy,
            AllocationStrategy.BEST_FIT: BestFitStrategy,
            AllocationStrategy.WORST_FIT: WorstFitStrategy,
            AllocationStrategy.BUDDY: lambda: BuddyAllocator(self.config.min_block_size),
        }
        
        factory = strategies.get(
            self.config.allocation_strategy,
            BestFitStrategy
        )
        return factory() if callable(factory) else factory
    
    async def allocate(
        self,
        size: int,
        owner: Optional[str] = None,
        pinned: bool = False
    ) -> AllocationResult:
        """Allocate memory from pool"""
        async with self._lock:
            # Align size
            aligned_size = self._align(size)
            
            if aligned_size > self.config.max_block_size:
                return AllocationResult(
                    success=False,
                    error=f"Size {size} exceeds max block size"
                )
            
            if aligned_size > self.metrics.free_size:
                # Try defragmentation
                if self.config.enable_defragmentation:
                    await self._defragment()
                
                if aligned_size > self.metrics.free_size:
                    self.metrics.allocation_failures += 1
                    return AllocationResult(
                        success=False,
                        error="Out of memory"
                    )
            
            # Find suitable block
            block = self.strategy.find_block(self.free_blocks, aligned_size)
            
            if block is None:
                # Try defragmentation
                if self.config.enable_defragmentation:
                    await self._defragment()
                    block = self.strategy.find_block(self.free_blocks, aligned_size)
                
                if block is None:
                    self.metrics.allocation_failures += 1
                    return AllocationResult(
                        success=False,
                        error="No suitable block found"
                    )
            
            # Split block if necessary
            if block.size > aligned_size:
                new_block = await self._split_block(block, aligned_size)
                allocated_block = block
            else:
                allocated_block = block
            
            # Mark as allocated
            allocated_block.state = BlockState.PINNED if pinned else BlockState.ALLOCATED
            allocated_block.owner = owner
            allocated_block.pinned = pinned
            allocated_block.last_access = time.time()
            
            self.free_blocks.remove(allocated_block)
            self.allocated_blocks[allocated_block.block_id] = allocated_block
            
            # Update metrics
            self.metrics.used_size += allocated_block.size
            self.metrics.free_size -= allocated_block.size
            self.metrics.num_allocations += 1
            self.metrics.peak_usage = max(
                self.metrics.peak_usage,
                self.metrics.used_size
            )
            
            return AllocationResult(
                success=True,
                block_id=allocated_block.block_id,
                offset=allocated_block.offset,
                size=allocated_block.size
            )
    
    async def deallocate(self, block_id: str) -> bool:
        """Deallocate memory block"""
        async with self._lock:
            if block_id not in self.allocated_blocks:
                return False
            
            block = self.allocated_blocks[block_id]
            
            if block.pinned:
                # Cannot deallocate pinned memory
                return False
            
            # Mark as free
            block.state = BlockState.FREE
            block.owner = None
            
            del self.allocated_blocks[block_id]
            self.free_blocks.append(block)
            
            # Update metrics
            self.metrics.used_size -= block.size
            self.metrics.free_size += block.size
            self.metrics.num_deallocations += 1
            
            # Try to coalesce with neighbors
            await self._coalesce(block)
            
            # Update fragmentation
            self._update_fragmentation()
            
            return True
    
    async def _split_block(
        self,
        block: MemoryBlock,
        size: int
    ) -> MemoryBlock:
        """Split a block into two"""
        remaining_size = block.size - size
        
        # Create new block for remaining space
        new_block = MemoryBlock(
            block_id=str(uuid4()),
            offset=block.offset + size,
            size=remaining_size,
            state=BlockState.FREE
        )
        
        # Update original block
        block.size = size
        
        # Add new block
        self.blocks[new_block.block_id] = new_block
        self.free_blocks.append(new_block)
        
        return new_block
    
    async def _coalesce(self, block: MemoryBlock) -> None:
        """Coalesce adjacent free blocks"""
        # Sort free blocks by offset
        self.free_blocks.sort(key=lambda b: b.offset)
        
        i = 0
        while i < len(self.free_blocks) - 1:
            current = self.free_blocks[i]
            next_block = self.free_blocks[i + 1]
            
            # Check if adjacent
            if current.offset + current.size == next_block.offset:
                # Merge blocks
                current.size += next_block.size
                
                # Remove merged block
                self.free_blocks.remove(next_block)
                del self.blocks[next_block.block_id]
            else:
                i += 1
    
    async def _defragment(self) -> None:
        """Defragment memory pool"""
        if not self.free_blocks:
            return
        
        # Sort all blocks by offset
        all_blocks = sorted(self.blocks.values(), key=lambda b: b.offset)
        
        # Compact allocated blocks
        current_offset = 0
        
        for block in all_blocks:
            if block.state != BlockState.FREE and not block.pinned:
                if block.offset != current_offset:
                    # Move block data
                    old_offset = block.offset
                    self._memory[current_offset:current_offset + block.size] = \
                        self._memory[old_offset:old_offset + block.size]
                    block.offset = current_offset
                
                current_offset += block.size
            elif block.pinned:
                # Can't move pinned blocks
                current_offset = block.offset + block.size
        
        # Rebuild free list
        self.free_blocks.clear()
        
        if current_offset < self.total_size:
            # Create single free block at end
            free_block = MemoryBlock(
                block_id=str(uuid4()),
                offset=current_offset,
                size=self.total_size - current_offset,
                state=BlockState.FREE
            )
            self.blocks[free_block.block_id] = free_block
            self.free_blocks.append(free_block)
        
        self._update_fragmentation()
    
    def _update_fragmentation(self) -> None:
        """Update fragmentation ratio"""
        if not self.free_blocks:
            self.metrics.fragmentation_ratio = 0.0
            return
        
        # Fragmentation = 1 - (largest_free_block / total_free)
        largest_free = max(b.size for b in self.free_blocks)
        total_free = sum(b.size for b in self.free_blocks)
        
        if total_free > 0:
            self.metrics.fragmentation_ratio = 1.0 - (largest_free / total_free)
        else:
            self.metrics.fragmentation_ratio = 0.0
    
    def _align(self, size: int) -> int:
        """Align size to configured alignment"""
        alignment = self.config.alignment
        return ((size + alignment - 1) // alignment) * alignment
    
    def write(self, block_id: str, data: bytes, offset: int = 0) -> bool:
        """Write data to allocated block"""
        if block_id not in self.allocated_blocks:
            return False
        
        block = self.allocated_blocks[block_id]
        
        if offset + len(data) > block.size:
            return False
        
        start = block.offset + offset
        self._memory[start:start + len(data)] = data
        block.last_access = time.time()
        block.access_count += 1
        
        return True
    
    def read(self, block_id: str, size: int, offset: int = 0) -> Optional[bytes]:
        """Read data from allocated block"""
        if block_id not in self.allocated_blocks:
            return None
        
        block = self.allocated_blocks[block_id]
        
        if offset + size > block.size:
            return None
        
        start = block.offset + offset
        block.last_access = time.time()
        block.access_count += 1
        
        return bytes(self._memory[start:start + size])
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pool metrics"""
        return {
            "total_size": self.metrics.total_size,
            "used_size": self.metrics.used_size,
            "free_size": self.metrics.free_size,
            "utilization": self.metrics.used_size / self.metrics.total_size,
            "num_allocations": self.metrics.num_allocations,
            "num_deallocations": self.metrics.num_deallocations,
            "num_free_blocks": len(self.free_blocks),
            "num_allocated_blocks": len(self.allocated_blocks),
            "fragmentation_ratio": self.metrics.fragmentation_ratio,
            "peak_usage": self.metrics.peak_usage,
            "allocation_failures": self.metrics.allocation_failures
        }


# ============================================================================
# Main Module Implementation
# ============================================================================

class MemoryPoolingModule:
    """
    Production-ready memory pooling module for OMNIXAN.
    
    Provides efficient shared memory management with:
    - Multiple allocation strategies
    - Automatic defragmentation
    - Slab allocation for fixed-size objects
    - Multi-tenant support
    - Memory tiering
    """
    
    def __init__(self, config: Optional[MemoryPoolConfig] = None):
        """Initialize the Memory Pooling Module"""
        self.config = config or MemoryPoolConfig()
        self.pools: Dict[str, MemoryPool] = {}
        self.slab_caches: Dict[int, SlabCache] = {}
        self.tenant_allocations: Dict[str, Set[str]] = defaultdict(set)
        self._initialized = False
        self._logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the memory pooling module"""
        if self._initialized:
            self._logger.warning("Module already initialized")
            return
        
        try:
            self._logger.info("Initializing MemoryPoolingModule...")
            
            # Create default pool
            self.pools["default"] = MemoryPool(self.config)
            
            # Create slab caches for common sizes
            for size in [64, 128, 256, 512, 1024, 4096]:
                self.slab_caches[size] = SlabCache(size)
            
            self._initialized = True
            self._logger.info("MemoryPoolingModule initialized successfully")
        
        except Exception as e:
            self._logger.error(f"Initialization failed: {str(e)}")
            raise MemoryPoolError(f"Failed to initialize module: {str(e)}")
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute memory pool operation"""
        if not self._initialized:
            raise MemoryPoolError("Module not initialized")
        
        operation = params.get("operation")
        pool_name = params.get("pool", "default")
        
        if operation == "create_pool":
            pool_config = MemoryPoolConfig(**params.get("config", {}))
            self.create_pool(pool_name, pool_config)
            return {"success": True, "pool": pool_name}
        
        elif operation == "allocate":
            size = params["size"]
            owner = params.get("owner")
            pinned = params.get("pinned", False)
            result = await self.allocate(pool_name, size, owner, pinned)
            return {
                "success": result.success,
                "block_id": result.block_id,
                "offset": result.offset,
                "size": result.size,
                "error": result.error
            }
        
        elif operation == "deallocate":
            block_id = params["block_id"]
            success = await self.deallocate(pool_name, block_id)
            return {"success": success}
        
        elif operation == "write":
            block_id = params["block_id"]
            data = params["data"]
            offset = params.get("offset", 0)
            success = self.write(pool_name, block_id, data, offset)
            return {"success": success}
        
        elif operation == "read":
            block_id = params["block_id"]
            size = params["size"]
            offset = params.get("offset", 0)
            data = self.read(pool_name, block_id, size, offset)
            return {"data": data}
        
        elif operation == "get_metrics":
            metrics = self.get_metrics(pool_name)
            return metrics
        
        elif operation == "defragment":
            await self.defragment(pool_name)
            return {"success": True}
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def create_pool(self, name: str, config: Optional[MemoryPoolConfig] = None) -> None:
        """Create a new memory pool"""
        if name in self.pools:
            raise MemoryPoolError(f"Pool {name} already exists")
        
        self.pools[name] = MemoryPool(config or self.config)
        self._logger.info(f"Created memory pool: {name}")
    
    async def allocate(
        self,
        pool_name: str,
        size: int,
        owner: Optional[str] = None,
        pinned: bool = False
    ) -> AllocationResult:
        """Allocate memory from pool"""
        if pool_name not in self.pools:
            return AllocationResult(success=False, error=f"Pool {pool_name} not found")
        
        pool = self.pools[pool_name]
        result = await pool.allocate(size, owner, pinned)
        
        if result.success and owner:
            self.tenant_allocations[owner].add(result.block_id)
        
        return result
    
    async def deallocate(self, pool_name: str, block_id: str) -> bool:
        """Deallocate memory block"""
        if pool_name not in self.pools:
            return False
        
        pool = self.pools[pool_name]
        
        # Remove from tenant tracking
        for tenant, blocks in self.tenant_allocations.items():
            blocks.discard(block_id)
        
        return await pool.deallocate(block_id)
    
    def write(
        self,
        pool_name: str,
        block_id: str,
        data: bytes,
        offset: int = 0
    ) -> bool:
        """Write data to block"""
        if pool_name not in self.pools:
            return False
        return self.pools[pool_name].write(block_id, data, offset)
    
    def read(
        self,
        pool_name: str,
        block_id: str,
        size: int,
        offset: int = 0
    ) -> Optional[bytes]:
        """Read data from block"""
        if pool_name not in self.pools:
            return None
        return self.pools[pool_name].read(block_id, size, offset)
    
    async def defragment(self, pool_name: str) -> None:
        """Defragment memory pool"""
        if pool_name in self.pools:
            await self.pools[pool_name]._defragment()
    
    def get_metrics(self, pool_name: Optional[str] = None) -> Dict[str, Any]:
        """Get pool metrics"""
        if pool_name:
            if pool_name not in self.pools:
                return {"error": f"Pool {pool_name} not found"}
            return self.pools[pool_name].get_metrics()
        
        # Aggregate all pools
        total_metrics = {
            "num_pools": len(self.pools),
            "pools": {}
        }
        
        for name, pool in self.pools.items():
            total_metrics["pools"][name] = pool.get_metrics()
        
        return total_metrics
    
    async def shutdown(self) -> None:
        """Shutdown the memory pooling module"""
        self._logger.info("Shutting down MemoryPoolingModule...")
        
        self.pools.clear()
        self.slab_caches.clear()
        self.tenant_allocations.clear()
        self._initialized = False
        
        self._logger.info("MemoryPoolingModule shutdown complete")


# Example usage
async def main():
    """Example usage of MemoryPoolingModule"""
    
    config = MemoryPoolConfig(
        pool_size=1024 * 1024,  # 1 MB
        allocation_strategy=AllocationStrategy.BEST_FIT,
        enable_defragmentation=True
    )
    
    module = MemoryPoolingModule(config)
    await module.initialize()
    
    try:
        # Allocate memory
        result1 = await module.allocate("default", 1024, owner="tenant1")
        print(f"Allocation 1: {result1}")
        
        result2 = await module.allocate("default", 2048, owner="tenant1")
        print(f"Allocation 2: {result2}")
        
        result3 = await module.allocate("default", 512, owner="tenant2")
        print(f"Allocation 3: {result3}")
        
        # Write data
        if result1.success:
            data = b"Hello, Memory Pool!"
            module.write("default", result1.block_id, data)
            
            # Read back
            read_data = module.read("default", result1.block_id, len(data))
            print(f"Read data: {read_data}")
        
        # Deallocate
        if result2.success:
            await module.deallocate("default", result2.block_id)
        
        # Get metrics
        metrics = module.get_metrics("default")
        print(f"\nPool Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        # Defragment
        await module.defragment("default")
        
        metrics_after = module.get_metrics("default")
        print(f"\nAfter defragmentation:")
        print(f"  Fragmentation: {metrics_after['fragmentation_ratio']:.2%}")
    
    finally:
        await module.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

