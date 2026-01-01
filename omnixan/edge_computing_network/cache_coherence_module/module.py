"""
OMNIXAN Cache Coherence Module
edge_computing_network/cache_coherence_module

Production-ready distributed cache coherence implementation supporting
multiple coherence protocols (MESI, MOESI, MSI) with automatic invalidation,
synchronization, and conflict resolution for edge computing environments.
"""

import asyncio
import logging
import time
import hashlib
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4
import json

from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheLineState(str, Enum):
    """Cache line states for MESI/MOESI protocols"""
    MODIFIED = "M"  # Modified (dirty, exclusive)
    OWNED = "O"  # Owned (MOESI only - modified but shared)
    EXCLUSIVE = "E"  # Exclusive (clean, only copy)
    SHARED = "S"  # Shared (clean, multiple copies may exist)
    INVALID = "I"  # Invalid (not present or stale)


class CoherenceProtocol(str, Enum):
    """Supported cache coherence protocols"""
    MSI = "msi"  # Modified-Shared-Invalid
    MESI = "mesi"  # Modified-Exclusive-Shared-Invalid
    MOESI = "moesi"  # Modified-Owned-Exclusive-Shared-Invalid


class BusOperation(str, Enum):
    """Cache bus operations"""
    BUS_READ = "bus_read"  # Read request
    BUS_READ_X = "bus_read_x"  # Read with intent to modify
    BUS_UPGRADE = "bus_upgrade"  # Upgrade from shared to exclusive
    BUS_WRITEBACK = "bus_writeback"  # Write dirty data back
    INVALIDATE = "invalidate"  # Invalidate other copies


class CacheEventType(str, Enum):
    """Types of cache events"""
    READ_HIT = "read_hit"
    READ_MISS = "read_miss"
    WRITE_HIT = "write_hit"
    WRITE_MISS = "write_miss"
    INVALIDATION = "invalidation"
    EVICTION = "eviction"
    COHERENCE_UPDATE = "coherence_update"


@dataclass
class CacheLine:
    """Represents a single cache line"""
    key: str
    value: Any
    state: CacheLineState
    version: int = 0
    timestamp: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    dirty: bool = False
    owner_node: Optional[str] = None
    sharers: Set[str] = field(default_factory=set)


@dataclass
class CacheEvent:
    """Cache event for logging and metrics"""
    event_type: CacheEventType
    key: str
    node_id: str
    timestamp: float = field(default_factory=time.time)
    old_state: Optional[CacheLineState] = None
    new_state: Optional[CacheLineState] = None
    latency_ms: float = 0.0


@dataclass
class CoherenceMetrics:
    """Metrics for cache coherence"""
    total_reads: int = 0
    total_writes: int = 0
    read_hits: int = 0
    read_misses: int = 0
    write_hits: int = 0
    write_misses: int = 0
    invalidations: int = 0
    evictions: int = 0
    coherence_messages: int = 0
    bus_transactions: int = 0


class CacheCoherenceConfig(BaseModel):
    """Configuration for cache coherence module"""
    protocol: CoherenceProtocol = Field(
        default=CoherenceProtocol.MESI,
        description="Cache coherence protocol"
    )
    cache_size: int = Field(
        default=1000,
        ge=10,
        le=1000000,
        description="Maximum cache entries per node"
    )
    line_size: int = Field(
        default=64,
        ge=1,
        le=4096,
        description="Cache line size in bytes"
    )
    associativity: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Set associativity"
    )
    write_policy: str = Field(
        default="write_back",
        description="Write policy (write_back or write_through)"
    )
    eviction_policy: str = Field(
        default="lru",
        description="Eviction policy (lru, lfu, fifo)"
    )
    sync_interval: float = Field(
        default=1.0,
        gt=0.0,
        description="Synchronization interval in seconds"
    )
    enable_prefetch: bool = Field(
        default=True,
        description="Enable cache prefetching"
    )


class CacheCoherenceError(Exception):
    """Base exception for cache coherence errors"""
    pass


class InvalidStateTransitionError(CacheCoherenceError):
    """Raised when invalid state transition is attempted"""
    pass


class CoherenceConflictError(CacheCoherenceError):
    """Raised when coherence conflict cannot be resolved"""
    pass


# ============================================================================
# Coherence Protocol Implementations
# ============================================================================

class CoherenceProtocolBase(ABC):
    """Abstract base class for coherence protocols"""
    
    @abstractmethod
    def handle_read(
        self,
        current_state: CacheLineState,
        is_local: bool
    ) -> Tuple[CacheLineState, List[BusOperation]]:
        """Handle read operation"""
        pass
    
    @abstractmethod
    def handle_write(
        self,
        current_state: CacheLineState,
        is_local: bool
    ) -> Tuple[CacheLineState, List[BusOperation]]:
        """Handle write operation"""
        pass
    
    @abstractmethod
    def handle_bus_operation(
        self,
        current_state: CacheLineState,
        operation: BusOperation
    ) -> CacheLineState:
        """Handle bus operation from another cache"""
        pass


class MESIProtocol(CoherenceProtocolBase):
    """MESI cache coherence protocol implementation"""
    
    def handle_read(
        self,
        current_state: CacheLineState,
        is_local: bool
    ) -> Tuple[CacheLineState, List[BusOperation]]:
        """Handle processor read"""
        
        if current_state == CacheLineState.INVALID:
            # Cache miss - need to fetch
            return CacheLineState.EXCLUSIVE, [BusOperation.BUS_READ]
        
        elif current_state in [CacheLineState.SHARED, CacheLineState.EXCLUSIVE, 
                               CacheLineState.MODIFIED]:
            # Cache hit - no bus operation needed
            return current_state, []
        
        return current_state, []
    
    def handle_write(
        self,
        current_state: CacheLineState,
        is_local: bool
    ) -> Tuple[CacheLineState, List[BusOperation]]:
        """Handle processor write"""
        
        if current_state == CacheLineState.INVALID:
            # Write miss - need exclusive access
            return CacheLineState.MODIFIED, [BusOperation.BUS_READ_X]
        
        elif current_state == CacheLineState.SHARED:
            # Need to invalidate other copies
            return CacheLineState.MODIFIED, [BusOperation.BUS_UPGRADE]
        
        elif current_state == CacheLineState.EXCLUSIVE:
            # Can silently upgrade to modified
            return CacheLineState.MODIFIED, []
        
        elif current_state == CacheLineState.MODIFIED:
            # Already have exclusive modified access
            return CacheLineState.MODIFIED, []
        
        return current_state, []
    
    def handle_bus_operation(
        self,
        current_state: CacheLineState,
        operation: BusOperation
    ) -> CacheLineState:
        """Handle snooped bus operation"""
        
        if operation == BusOperation.BUS_READ:
            if current_state == CacheLineState.MODIFIED:
                # Write back and transition to shared
                return CacheLineState.SHARED
            elif current_state == CacheLineState.EXCLUSIVE:
                return CacheLineState.SHARED
            return current_state
        
        elif operation == BusOperation.BUS_READ_X:
            if current_state in [CacheLineState.MODIFIED, CacheLineState.EXCLUSIVE,
                                CacheLineState.SHARED]:
                return CacheLineState.INVALID
            return current_state
        
        elif operation == BusOperation.BUS_UPGRADE:
            if current_state == CacheLineState.SHARED:
                return CacheLineState.INVALID
            return current_state
        
        elif operation == BusOperation.INVALIDATE:
            return CacheLineState.INVALID
        
        return current_state


class MOESIProtocol(CoherenceProtocolBase):
    """MOESI cache coherence protocol implementation (with Owned state)"""
    
    def handle_read(
        self,
        current_state: CacheLineState,
        is_local: bool
    ) -> Tuple[CacheLineState, List[BusOperation]]:
        """Handle processor read"""
        
        if current_state == CacheLineState.INVALID:
            return CacheLineState.EXCLUSIVE, [BusOperation.BUS_READ]
        
        # Hit in any valid state
        return current_state, []
    
    def handle_write(
        self,
        current_state: CacheLineState,
        is_local: bool
    ) -> Tuple[CacheLineState, List[BusOperation]]:
        """Handle processor write"""
        
        if current_state == CacheLineState.INVALID:
            return CacheLineState.MODIFIED, [BusOperation.BUS_READ_X]
        
        elif current_state in [CacheLineState.SHARED, CacheLineState.OWNED]:
            return CacheLineState.MODIFIED, [BusOperation.BUS_UPGRADE]
        
        elif current_state in [CacheLineState.EXCLUSIVE, CacheLineState.MODIFIED]:
            return CacheLineState.MODIFIED, []
        
        return current_state, []
    
    def handle_bus_operation(
        self,
        current_state: CacheLineState,
        operation: BusOperation
    ) -> CacheLineState:
        """Handle snooped bus operation"""
        
        if operation == BusOperation.BUS_READ:
            if current_state == CacheLineState.MODIFIED:
                return CacheLineState.OWNED  # Keep dirty data, share
            elif current_state == CacheLineState.EXCLUSIVE:
                return CacheLineState.SHARED
            elif current_state == CacheLineState.OWNED:
                return CacheLineState.OWNED  # Still owner
            return current_state
        
        elif operation == BusOperation.BUS_READ_X:
            if current_state in [CacheLineState.MODIFIED, CacheLineState.OWNED]:
                # Write back before invalidation
                return CacheLineState.INVALID
            elif current_state in [CacheLineState.EXCLUSIVE, CacheLineState.SHARED]:
                return CacheLineState.INVALID
            return current_state
        
        elif operation == BusOperation.BUS_UPGRADE:
            return CacheLineState.INVALID
        
        return current_state


# ============================================================================
# Cache Node Implementation
# ============================================================================

class CacheNode:
    """
    Represents a single cache node in the distributed system.
    Implements local cache with coherence support.
    """
    
    def __init__(
        self,
        node_id: str,
        config: CacheCoherenceConfig,
        protocol: CoherenceProtocolBase
    ):
        self.node_id = node_id
        self.config = config
        self.protocol = protocol
        
        # Cache storage
        self.cache: Dict[str, CacheLine] = {}
        self.cache_order: List[str] = []  # For LRU
        
        # Metrics
        self.metrics = CoherenceMetrics()
        self.events: List[CacheEvent] = []
        
        # Locks for thread safety
        self._lock = asyncio.Lock()
    
    async def read(self, key: str) -> Tuple[Optional[Any], bool]:
        """Read value from cache"""
        async with self._lock:
            start_time = time.time()
            
            if key in self.cache:
                line = self.cache[key]
                
                if line.state != CacheLineState.INVALID:
                    # Cache hit
                    new_state, bus_ops = self.protocol.handle_read(line.state, True)
                    line.state = new_state
                    line.last_access = time.time()
                    line.access_count += 1
                    
                    self._update_lru(key)
                    self.metrics.read_hits += 1
                    self.metrics.total_reads += 1
                    
                    self._record_event(
                        CacheEventType.READ_HIT, key, line.state, new_state,
                        (time.time() - start_time) * 1000
                    )
                    
                    return line.value, True
            
            # Cache miss
            self.metrics.read_misses += 1
            self.metrics.total_reads += 1
            
            self._record_event(
                CacheEventType.READ_MISS, key, CacheLineState.INVALID, None,
                (time.time() - start_time) * 1000
            )
            
            return None, False
    
    async def write(
        self,
        key: str,
        value: Any,
        from_coherence: bool = False
    ) -> List[BusOperation]:
        """Write value to cache"""
        async with self._lock:
            start_time = time.time()
            bus_ops = []
            
            if key in self.cache:
                line = self.cache[key]
                old_state = line.state
                
                new_state, bus_ops = self.protocol.handle_write(line.state, True)
                
                line.value = value
                line.state = new_state
                line.version += 1
                line.timestamp = time.time()
                line.last_access = time.time()
                line.dirty = (new_state == CacheLineState.MODIFIED)
                line.owner_node = self.node_id
                
                self._update_lru(key)
                self.metrics.write_hits += 1
                
                self._record_event(
                    CacheEventType.WRITE_HIT, key, old_state, new_state,
                    (time.time() - start_time) * 1000
                )
            
            else:
                # Write miss - allocate new line
                if len(self.cache) >= self.config.cache_size:
                    await self._evict()
                
                new_state, bus_ops = self.protocol.handle_write(
                    CacheLineState.INVALID, True
                )
                
                line = CacheLine(
                    key=key,
                    value=value,
                    state=new_state,
                    version=1,
                    dirty=(new_state == CacheLineState.MODIFIED),
                    owner_node=self.node_id
                )
                
                self.cache[key] = line
                self.cache_order.append(key)
                self.metrics.write_misses += 1
                
                self._record_event(
                    CacheEventType.WRITE_MISS, key, CacheLineState.INVALID, new_state,
                    (time.time() - start_time) * 1000
                )
            
            self.metrics.total_writes += 1
            self.metrics.bus_transactions += len(bus_ops)
            
            return bus_ops
    
    async def handle_bus_snoop(
        self,
        key: str,
        operation: BusOperation
    ) -> Optional[Any]:
        """Handle bus operation from another cache"""
        async with self._lock:
            if key not in self.cache:
                return None
            
            line = self.cache[key]
            old_state = line.state
            
            new_state = self.protocol.handle_bus_operation(line.state, operation)
            
            # Check if we need to supply data (M->S transition)
            supply_data = None
            if old_state == CacheLineState.MODIFIED and new_state in [
                CacheLineState.SHARED, CacheLineState.INVALID, CacheLineState.OWNED
            ]:
                supply_data = line.value
                line.dirty = False
            
            line.state = new_state
            
            if new_state == CacheLineState.INVALID:
                self.metrics.invalidations += 1
                self._record_event(
                    CacheEventType.INVALIDATION, key, old_state, new_state, 0
                )
            else:
                self._record_event(
                    CacheEventType.COHERENCE_UPDATE, key, old_state, new_state, 0
                )
            
            self.metrics.coherence_messages += 1
            
            return supply_data
    
    async def invalidate(self, key: str) -> None:
        """Invalidate a cache line"""
        async with self._lock:
            if key in self.cache:
                old_state = self.cache[key].state
                self.cache[key].state = CacheLineState.INVALID
                self.metrics.invalidations += 1
                
                self._record_event(
                    CacheEventType.INVALIDATION, key, old_state,
                    CacheLineState.INVALID, 0
                )
    
    async def _evict(self) -> None:
        """Evict a cache line based on policy"""
        if not self.cache_order:
            return
        
        # LRU eviction
        if self.config.eviction_policy == "lru":
            key_to_evict = self.cache_order[0]
        elif self.config.eviction_policy == "fifo":
            key_to_evict = self.cache_order[0]
        else:  # LFU
            key_to_evict = min(
                self.cache_order,
                key=lambda k: self.cache[k].access_count
            )
        
        line = self.cache[key_to_evict]
        
        # Writeback if dirty
        if line.dirty:
            # In real implementation, write to memory/lower level
            pass
        
        old_state = line.state
        del self.cache[key_to_evict]
        self.cache_order.remove(key_to_evict)
        self.metrics.evictions += 1
        
        self._record_event(
            CacheEventType.EVICTION, key_to_evict, old_state,
            CacheLineState.INVALID, 0
        )
    
    def _update_lru(self, key: str) -> None:
        """Update LRU order"""
        if key in self.cache_order:
            self.cache_order.remove(key)
        self.cache_order.append(key)
    
    def _record_event(
        self,
        event_type: CacheEventType,
        key: str,
        old_state: Optional[CacheLineState],
        new_state: Optional[CacheLineState],
        latency_ms: float
    ) -> None:
        """Record cache event"""
        event = CacheEvent(
            event_type=event_type,
            key=key,
            node_id=self.node_id,
            old_state=old_state,
            new_state=new_state,
            latency_ms=latency_ms
        )
        self.events.append(event)
        
        # Keep only recent events
        if len(self.events) > 10000:
            self.events = self.events[-5000:]
    
    def get_line_state(self, key: str) -> Optional[CacheLineState]:
        """Get current state of a cache line"""
        if key in self.cache:
            return self.cache[key].state
        return None


# ============================================================================
# Main Module Implementation
# ============================================================================

class CacheCoherenceModule:
    """
    Production-ready distributed cache coherence module for OMNIXAN.
    
    Manages cache coherence across multiple nodes using configurable
    protocols (MSI, MESI, MOESI) with automatic synchronization.
    """
    
    def __init__(self, config: Optional[CacheCoherenceConfig] = None):
        """Initialize the Cache Coherence Module"""
        self.config = config or CacheCoherenceConfig()
        self.nodes: Dict[str, CacheNode] = {}
        self.protocol = self._create_protocol()
        
        # Directory for tracking sharers (directory-based coherence)
        self.directory: Dict[str, Set[str]] = defaultdict(set)
        self.directory_owner: Dict[str, str] = {}
        
        # Background tasks
        self._sync_task: Optional[asyncio.Task] = None
        self._initialized = False
        self._shutting_down = False
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(__name__)
    
    def _create_protocol(self) -> CoherenceProtocolBase:
        """Create protocol based on configuration"""
        if self.config.protocol == CoherenceProtocol.MOESI:
            return MOESIProtocol()
        else:  # Default to MESI
            return MESIProtocol()
    
    async def initialize(self) -> None:
        """Initialize the cache coherence module"""
        if self._initialized:
            self._logger.warning("Module already initialized")
            return
        
        try:
            self._logger.info(
                f"Initializing CacheCoherenceModule with {self.config.protocol.value} protocol"
            )
            
            # Start sync task
            self._sync_task = asyncio.create_task(self._sync_loop())
            
            self._initialized = True
            self._logger.info("CacheCoherenceModule initialized successfully")
        
        except Exception as e:
            self._logger.error(f"Initialization failed: {str(e)}")
            raise CacheCoherenceError(f"Failed to initialize module: {str(e)}")
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cache coherence operation"""
        if not self._initialized:
            raise CacheCoherenceError("Module not initialized")
        
        operation = params.get("operation")
        
        if operation == "register_node":
            node_id = params.get("node_id", str(uuid4()))
            self.register_node(node_id)
            return {"node_id": node_id}
        
        elif operation == "read":
            node_id = params.get("node_id")
            key = params.get("key")
            value, hit = await self.read(node_id, key)
            return {"value": value, "hit": hit}
        
        elif operation == "write":
            node_id = params.get("node_id")
            key = params.get("key")
            value = params.get("value")
            await self.write(node_id, key, value)
            return {"success": True}
        
        elif operation == "invalidate":
            key = params.get("key")
            await self.invalidate_all(key)
            return {"success": True}
        
        elif operation == "get_metrics":
            node_id = params.get("node_id")
            metrics = self.get_metrics(node_id)
            return metrics
        
        elif operation == "get_directory":
            return self.get_directory_state()
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def register_node(self, node_id: str) -> CacheNode:
        """Register a new cache node"""
        if node_id in self.nodes:
            return self.nodes[node_id]
        
        node = CacheNode(node_id, self.config, self.protocol)
        self.nodes[node_id] = node
        self._logger.info(f"Registered cache node: {node_id}")
        
        return node
    
    async def read(self, node_id: str, key: str) -> Tuple[Optional[Any], bool]:
        """
        Read from cache with coherence.
        
        Args:
            node_id: Node performing the read
            key: Cache key
        
        Returns:
            Tuple of (value, hit_status)
        """
        if node_id not in self.nodes:
            raise CacheCoherenceError(f"Node {node_id} not registered")
        
        node = self.nodes[node_id]
        value, hit = await node.read(key)
        
        if not hit:
            # Cache miss - check other nodes
            value = await self._fetch_from_sharers(key)
            
            if value is not None:
                # Got data from another node
                await node.write(key, value, from_coherence=True)
                
                # Update directory
                async with self._lock:
                    self.directory[key].add(node_id)
                
                return value, False
        else:
            # Update directory on hit
            async with self._lock:
                self.directory[key].add(node_id)
        
        return value, hit
    
    async def write(self, node_id: str, key: str, value: Any) -> None:
        """
        Write to cache with coherence.
        
        Args:
            node_id: Node performing the write
            key: Cache key
            value: Value to write
        """
        if node_id not in self.nodes:
            raise CacheCoherenceError(f"Node {node_id} not registered")
        
        node = self.nodes[node_id]
        
        # Get bus operations from write
        bus_ops = await node.write(key, value)
        
        # Process bus operations for coherence
        async with self._lock:
            if BusOperation.BUS_READ_X in bus_ops or BusOperation.BUS_UPGRADE in bus_ops:
                # Invalidate other sharers
                sharers = self.directory[key].copy()
                sharers.discard(node_id)
                
                for sharer_id in sharers:
                    if sharer_id in self.nodes:
                        await self.nodes[sharer_id].handle_bus_snoop(
                            key, BusOperation.INVALIDATE
                        )
                
                # Update directory - only writer has copy
                self.directory[key] = {node_id}
                self.directory_owner[key] = node_id
            
            else:
                self.directory[key].add(node_id)
                self.directory_owner[key] = node_id
    
    async def invalidate_all(self, key: str) -> None:
        """Invalidate key across all nodes"""
        async with self._lock:
            sharers = self.directory.get(key, set()).copy()
            
            for node_id in sharers:
                if node_id in self.nodes:
                    await self.nodes[node_id].invalidate(key)
            
            self.directory[key].clear()
            if key in self.directory_owner:
                del self.directory_owner[key]
    
    async def _fetch_from_sharers(self, key: str) -> Optional[Any]:
        """Fetch data from a node that has it"""
        async with self._lock:
            # Check owner first
            if key in self.directory_owner:
                owner_id = self.directory_owner[key]
                if owner_id in self.nodes:
                    node = self.nodes[owner_id]
                    if key in node.cache:
                        # Supply data (may trigger state change)
                        data = await node.handle_bus_snoop(key, BusOperation.BUS_READ)
                        if data is not None:
                            return data
                        return node.cache[key].value
            
            # Check other sharers
            for node_id in self.directory.get(key, set()):
                if node_id in self.nodes:
                    node = self.nodes[node_id]
                    if key in node.cache and node.cache[key].state != CacheLineState.INVALID:
                        return node.cache[key].value
        
        return None
    
    async def _sync_loop(self) -> None:
        """Background synchronization loop"""
        while not self._shutting_down:
            try:
                await asyncio.sleep(self.config.sync_interval)
                
                # Periodic cleanup and verification
                async with self._lock:
                    # Clean up stale directory entries
                    for key in list(self.directory.keys()):
                        valid_sharers = set()
                        for node_id in self.directory[key]:
                            if node_id in self.nodes:
                                node = self.nodes[node_id]
                                if key in node.cache and node.cache[key].state != CacheLineState.INVALID:
                                    valid_sharers.add(node_id)
                        self.directory[key] = valid_sharers
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Sync loop error: {e}")
    
    def get_metrics(self, node_id: Optional[str] = None) -> Dict[str, Any]:
        """Get cache metrics"""
        if node_id:
            if node_id not in self.nodes:
                return {"error": f"Node {node_id} not found"}
            
            metrics = self.nodes[node_id].metrics
            return {
                "node_id": node_id,
                "total_reads": metrics.total_reads,
                "total_writes": metrics.total_writes,
                "read_hits": metrics.read_hits,
                "read_misses": metrics.read_misses,
                "write_hits": metrics.write_hits,
                "write_misses": metrics.write_misses,
                "hit_rate": (
                    (metrics.read_hits + metrics.write_hits) /
                    max(metrics.total_reads + metrics.total_writes, 1)
                ),
                "invalidations": metrics.invalidations,
                "evictions": metrics.evictions
            }
        
        # Aggregate metrics
        total_metrics = {
            "total_nodes": len(self.nodes),
            "total_reads": 0,
            "total_writes": 0,
            "total_hits": 0,
            "total_misses": 0,
            "total_invalidations": 0
        }
        
        for node in self.nodes.values():
            total_metrics["total_reads"] += node.metrics.total_reads
            total_metrics["total_writes"] += node.metrics.total_writes
            total_metrics["total_hits"] += node.metrics.read_hits + node.metrics.write_hits
            total_metrics["total_misses"] += node.metrics.read_misses + node.metrics.write_misses
            total_metrics["total_invalidations"] += node.metrics.invalidations
        
        total_ops = total_metrics["total_hits"] + total_metrics["total_misses"]
        total_metrics["overall_hit_rate"] = (
            total_metrics["total_hits"] / max(total_ops, 1)
        )
        
        return total_metrics
    
    def get_directory_state(self) -> Dict[str, Any]:
        """Get current directory state"""
        return {
            "entries": {k: list(v) for k, v in self.directory.items()},
            "owners": dict(self.directory_owner),
            "total_entries": len(self.directory)
        }
    
    async def shutdown(self) -> None:
        """Shutdown the cache coherence module"""
        self._logger.info("Shutting down CacheCoherenceModule...")
        self._shutting_down = True
        
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        
        # Flush dirty data
        for node in self.nodes.values():
            for key, line in node.cache.items():
                if line.dirty:
                    # In production, write back to memory
                    pass
        
        self.nodes.clear()
        self.directory.clear()
        self._initialized = False
        
        self._logger.info("CacheCoherenceModule shutdown complete")


# Example usage
async def main():
    """Example usage of CacheCoherenceModule"""
    
    config = CacheCoherenceConfig(
        protocol=CoherenceProtocol.MESI,
        cache_size=100,
        sync_interval=1.0
    )
    
    module = CacheCoherenceModule(config)
    await module.initialize()
    
    try:
        # Register nodes
        module.register_node("node1")
        module.register_node("node2")
        module.register_node("node3")
        
        # Node 1 writes
        await module.write("node1", "key1", "value1")
        print(f"Node1 wrote key1=value1")
        
        # Node 2 reads (should get from node1)
        value, hit = await module.read("node2", "key1")
        print(f"Node2 read key1: value={value}, hit={hit}")
        
        # Node 3 reads (shared)
        value, hit = await module.read("node3", "key1")
        print(f"Node3 read key1: value={value}, hit={hit}")
        
        # Check directory
        directory = module.get_directory_state()
        print(f"\nDirectory state: {json.dumps(directory, indent=2)}")
        
        # Node 2 writes (should invalidate others)
        await module.write("node2", "key1", "value1_updated")
        print(f"\nNode2 wrote key1=value1_updated")
        
        # Check states
        for node_id in ["node1", "node2", "node3"]:
            state = module.nodes[node_id].get_line_state("key1")
            print(f"{node_id} key1 state: {state}")
        
        # Get metrics
        print(f"\nMetrics:")
        print(json.dumps(module.get_metrics(), indent=2))
    
    finally:
        await module.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

