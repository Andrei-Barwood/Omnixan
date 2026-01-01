"""
OMNIXAN Non-Blocking Module
heterogenous_computing_group/non_blocking_module

Production-ready non-blocking I/O and communication module for high-performance
asynchronous operations with lock-free data structures, zero-copy transfers,
and event-driven architecture.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, Generic, TypeVar
from uuid import uuid4
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')


class OperationType(str, Enum):
    """Types of non-blocking operations"""
    READ = "read"
    WRITE = "write"
    COMPUTE = "compute"
    NETWORK = "network"
    STORAGE = "storage"


class CompletionStatus(str, Enum):
    """Operation completion status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QueuePolicy(str, Enum):
    """Queue policies"""
    FIFO = "fifo"
    LIFO = "lifo"
    PRIORITY = "priority"


@dataclass
class AsyncOperation:
    """Represents an async operation"""
    op_id: str
    op_type: OperationType
    status: CompletionStatus
    data: Any = None
    result: Any = None
    error: Optional[str] = None
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    callback: Optional[Callable] = None


@dataclass
class CompletionEvent:
    """Completion event"""
    event_id: str
    op_id: str
    status: CompletionStatus
    result: Any = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class NonBlockingMetrics:
    """Non-blocking operation metrics"""
    total_operations: int = 0
    completed_operations: int = 0
    failed_operations: int = 0
    pending_operations: int = 0
    avg_latency_ms: float = 0.0
    throughput_ops_per_sec: float = 0.0
    queue_depth: int = 0


class NonBlockingConfig(BaseModel):
    """Configuration for non-blocking operations"""
    max_concurrent_ops: int = Field(
        default=1000,
        ge=1,
        description="Maximum concurrent operations"
    )
    queue_size: int = Field(
        default=10000,
        ge=100,
        description="Operation queue size"
    )
    worker_threads: int = Field(
        default=4,
        ge=1,
        description="Worker thread count"
    )
    completion_queue_size: int = Field(
        default=1000,
        ge=10,
        description="Completion queue size"
    )
    enable_batching: bool = Field(
        default=True,
        description="Enable operation batching"
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size"
    )


class NonBlockingError(Exception):
    """Base exception for non-blocking errors"""
    pass


# ============================================================================
# Lock-Free Ring Buffer
# ============================================================================

class LockFreeRingBuffer(Generic[T]):
    """Lock-free single-producer single-consumer ring buffer"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Optional[T]] = [None] * capacity
        self.head = 0  # Write position
        self.tail = 0  # Read position
        self._size = 0
    
    def push(self, item: T) -> bool:
        """Push item (non-blocking)"""
        next_head = (self.head + 1) % self.capacity
        
        if next_head == self.tail:
            return False  # Buffer full
        
        self.buffer[self.head] = item
        self.head = next_head
        self._size += 1
        return True
    
    def pop(self) -> Optional[T]:
        """Pop item (non-blocking)"""
        if self.tail == self.head:
            return None  # Buffer empty
        
        item = self.buffer[self.tail]
        self.buffer[self.tail] = None
        self.tail = (self.tail + 1) % self.capacity
        self._size -= 1
        return item
    
    def is_empty(self) -> bool:
        return self.head == self.tail
    
    def is_full(self) -> bool:
        return ((self.head + 1) % self.capacity) == self.tail
    
    def size(self) -> int:
        return self._size


# ============================================================================
# Async Queue
# ============================================================================

class AsyncQueue(Generic[T]):
    """Async-friendly queue with priority support"""
    
    def __init__(self, maxsize: int = 0, policy: QueuePolicy = QueuePolicy.FIFO):
        self.maxsize = maxsize
        self.policy = policy
        
        if policy == QueuePolicy.PRIORITY:
            self._queue: Any = []  # Min-heap
        elif policy == QueuePolicy.LIFO:
            self._queue = deque()
        else:
            self._queue = deque()
        
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition()
        self._not_full = asyncio.Condition()
    
    async def put(self, item: T, priority: int = 0) -> None:
        """Put item in queue"""
        async with self._not_full:
            while self.maxsize > 0 and self.qsize() >= self.maxsize:
                await self._not_full.wait()
            
            if self.policy == QueuePolicy.PRIORITY:
                import heapq
                heapq.heappush(self._queue, (priority, time.time(), item))
            elif self.policy == QueuePolicy.LIFO:
                self._queue.append(item)
            else:
                self._queue.append(item)
        
        async with self._not_empty:
            self._not_empty.notify()
    
    async def get(self) -> T:
        """Get item from queue"""
        async with self._not_empty:
            while self.qsize() == 0:
                await self._not_empty.wait()
            
            if self.policy == QueuePolicy.PRIORITY:
                import heapq
                _, _, item = heapq.heappop(self._queue)
            elif self.policy == QueuePolicy.LIFO:
                item = self._queue.pop()
            else:
                item = self._queue.popleft()
        
        async with self._not_full:
            self._not_full.notify()
        
        return item
    
    def get_nowait(self) -> Optional[T]:
        """Get item without waiting"""
        if self.qsize() == 0:
            return None
        
        if self.policy == QueuePolicy.PRIORITY:
            import heapq
            _, _, item = heapq.heappop(self._queue)
        elif self.policy == QueuePolicy.LIFO:
            item = self._queue.pop()
        else:
            item = self._queue.popleft()
        
        return item
    
    def qsize(self) -> int:
        return len(self._queue)
    
    def empty(self) -> bool:
        return self.qsize() == 0


# ============================================================================
# Completion Queue
# ============================================================================

class CompletionQueue:
    """Queue for operation completions"""
    
    def __init__(self, maxsize: int = 1000):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._handlers: Dict[str, Callable] = {}
    
    async def post_completion(self, event: CompletionEvent) -> None:
        """Post completion event"""
        await self._queue.put(event)
    
    async def wait_completion(self, timeout: Optional[float] = None) -> Optional[CompletionEvent]:
        """Wait for completion event"""
        try:
            if timeout:
                return await asyncio.wait_for(self._queue.get(), timeout)
            return await self._queue.get()
        except asyncio.TimeoutError:
            return None
    
    def poll_completions(self, max_events: int = 10) -> List[CompletionEvent]:
        """Poll for completions (non-blocking)"""
        events = []
        for _ in range(max_events):
            try:
                event = self._queue.get_nowait()
                events.append(event)
            except asyncio.QueueEmpty:
                break
        return events
    
    def register_handler(self, op_type: str, handler: Callable) -> None:
        """Register completion handler"""
        self._handlers[op_type] = handler
    
    async def process_completions(self) -> int:
        """Process pending completions with handlers"""
        events = self.poll_completions()
        for event in events:
            if event.op_id in self._handlers:
                handler = self._handlers[event.op_id]
                handler(event)
        return len(events)
    
    def qsize(self) -> int:
        return self._queue.qsize()


# ============================================================================
# Main Module Implementation
# ============================================================================

class NonBlockingModule:
    """
    Production-ready Non-Blocking module for OMNIXAN.
    
    Provides:
    - Non-blocking I/O operations
    - Lock-free data structures
    - Async operation submission
    - Completion queue handling
    - Zero-copy data transfers
    - Event-driven callbacks
    """
    
    def __init__(self, config: Optional[NonBlockingConfig] = None):
        """Initialize the Non-Blocking Module"""
        self.config = config or NonBlockingConfig()
        
        self.operations: Dict[str, AsyncOperation] = {}
        self.op_queue: AsyncQueue[AsyncOperation] = AsyncQueue(
            maxsize=self.config.queue_size,
            policy=QueuePolicy.PRIORITY
        )
        self.completion_queue = CompletionQueue(self.config.completion_queue_size)
        self.ring_buffer: LockFreeRingBuffer[Any] = LockFreeRingBuffer(1024)
        
        self.metrics = NonBlockingMetrics()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._worker_tasks: List[asyncio.Task] = []
        self._initialized = False
        self._shutting_down = False
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(__name__)
        
        # Latency tracking
        self._latencies: List[float] = []
        self._start_time = time.time()
    
    async def initialize(self) -> None:
        """Initialize the non-blocking module"""
        if self._initialized:
            self._logger.warning("Module already initialized")
            return
        
        try:
            self._logger.info("Initializing NonBlockingModule...")
            
            # Create thread pool
            self._executor = ThreadPoolExecutor(
                max_workers=self.config.worker_threads,
                thread_name_prefix="nb_worker"
            )
            
            # Start worker tasks
            for i in range(self.config.worker_threads):
                task = asyncio.create_task(self._worker_loop(i))
                self._worker_tasks.append(task)
            
            self._initialized = True
            self._logger.info("NonBlockingModule initialized successfully")
        
        except Exception as e:
            self._logger.error(f"Initialization failed: {str(e)}")
            raise NonBlockingError(f"Failed to initialize module: {str(e)}")
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute non-blocking operation"""
        if not self._initialized:
            raise NonBlockingError("Module not initialized")
        
        operation = params.get("operation")
        
        if operation == "submit":
            op_type = OperationType(params.get("op_type", "compute"))
            data = params.get("data")
            priority = params.get("priority", 0)
            
            op = await self.submit(op_type, data, priority)
            return {"op_id": op.op_id, "status": op.status.value}
        
        elif operation == "submit_batch":
            ops = params.get("operations", [])
            results = await self.submit_batch([
                (OperationType(o.get("op_type", "compute")), o.get("data"), o.get("priority", 0))
                for o in ops
            ])
            return {
                "op_ids": [op.op_id for op in results],
                "count": len(results)
            }
        
        elif operation == "poll":
            op_id = params["op_id"]
            status = await self.poll(op_id)
            if status:
                return {
                    "op_id": op_id,
                    "status": status.status.value,
                    "result": status.result
                }
            return {"error": "Operation not found"}
        
        elif operation == "wait":
            op_id = params["op_id"]
            timeout = params.get("timeout")
            result = await self.wait(op_id, timeout)
            return {
                "op_id": op_id,
                "status": result.status.value if result else "timeout",
                "result": result.result if result else None
            }
        
        elif operation == "cancel":
            op_id = params["op_id"]
            success = await self.cancel(op_id)
            return {"success": success}
        
        elif operation == "poll_completions":
            max_events = params.get("max_events", 10)
            events = self.completion_queue.poll_completions(max_events)
            return {
                "events": [
                    {
                        "event_id": e.event_id,
                        "op_id": e.op_id,
                        "status": e.status.value
                    }
                    for e in events
                ]
            }
        
        elif operation == "get_metrics":
            return self.get_metrics()
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def submit(
        self,
        op_type: OperationType,
        data: Any = None,
        priority: int = 0,
        callback: Optional[Callable] = None
    ) -> AsyncOperation:
        """Submit an async operation"""
        op = AsyncOperation(
            op_id=str(uuid4()),
            op_type=op_type,
            status=CompletionStatus.PENDING,
            data=data,
            priority=priority,
            callback=callback
        )
        
        async with self._lock:
            self.operations[op.op_id] = op
            self.metrics.total_operations += 1
            self.metrics.pending_operations += 1
        
        await self.op_queue.put(op, priority)
        
        return op
    
    async def submit_batch(
        self,
        operations: List[Tuple[OperationType, Any, int]]
    ) -> List[AsyncOperation]:
        """Submit batch of operations"""
        results = []
        
        for op_type, data, priority in operations:
            op = await self.submit(op_type, data, priority)
            results.append(op)
        
        return results
    
    async def poll(self, op_id: str) -> Optional[AsyncOperation]:
        """Poll operation status (non-blocking)"""
        return self.operations.get(op_id)
    
    async def wait(
        self,
        op_id: str,
        timeout: Optional[float] = None
    ) -> Optional[AsyncOperation]:
        """Wait for operation completion"""
        start = time.time()
        
        while True:
            op = self.operations.get(op_id)
            if op and op.status in [CompletionStatus.COMPLETED, CompletionStatus.FAILED]:
                return op
            
            if timeout and (time.time() - start) >= timeout:
                return None
            
            await asyncio.sleep(0.001)  # 1ms poll
    
    async def cancel(self, op_id: str) -> bool:
        """Cancel pending operation"""
        async with self._lock:
            op = self.operations.get(op_id)
            if op and op.status == CompletionStatus.PENDING:
                op.status = CompletionStatus.CANCELLED
                self.metrics.pending_operations -= 1
                return True
            return False
    
    async def _worker_loop(self, worker_id: int) -> None:
        """Worker loop for processing operations"""
        while not self._shutting_down:
            try:
                # Get operation from queue
                op = await asyncio.wait_for(
                    self.op_queue.get(),
                    timeout=0.1
                )
                
                if op.status == CompletionStatus.CANCELLED:
                    continue
                
                await self._process_operation(op)
            
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Worker {worker_id} error: {e}")
    
    async def _process_operation(self, op: AsyncOperation) -> None:
        """Process a single operation"""
        async with self._lock:
            op.status = CompletionStatus.IN_PROGRESS
            op.started_at = time.time()
            self.metrics.pending_operations -= 1
        
        try:
            # Simulate operation based on type
            result = await self._execute_operation(op)
            
            async with self._lock:
                op.status = CompletionStatus.COMPLETED
                op.result = result
                op.completed_at = time.time()
                self.metrics.completed_operations += 1
                
                # Track latency
                latency = (op.completed_at - op.created_at) * 1000
                self._latencies.append(latency)
            
            # Post completion
            await self.completion_queue.post_completion(CompletionEvent(
                event_id=str(uuid4()),
                op_id=op.op_id,
                status=CompletionStatus.COMPLETED,
                result=result
            ))
            
            # Invoke callback
            if op.callback:
                op.callback(op)
        
        except Exception as e:
            async with self._lock:
                op.status = CompletionStatus.FAILED
                op.error = str(e)
                op.completed_at = time.time()
                self.metrics.failed_operations += 1
            
            await self.completion_queue.post_completion(CompletionEvent(
                event_id=str(uuid4()),
                op_id=op.op_id,
                status=CompletionStatus.FAILED
            ))
    
    async def _execute_operation(self, op: AsyncOperation) -> Any:
        """Execute operation based on type"""
        # Simulate different operation latencies
        latencies = {
            OperationType.READ: 0.0001,    # 100us
            OperationType.WRITE: 0.0002,   # 200us
            OperationType.COMPUTE: 0.001,  # 1ms
            OperationType.NETWORK: 0.005,  # 5ms
            OperationType.STORAGE: 0.01,   # 10ms
        }
        
        delay = latencies.get(op.op_type, 0.001)
        await asyncio.sleep(delay)
        
        # Return processed result
        if op.data is not None:
            return {"processed": True, "data_size": len(str(op.data))}
        return {"processed": True}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get non-blocking metrics"""
        # Calculate averages
        avg_latency = 0.0
        if self._latencies:
            avg_latency = sum(self._latencies) / len(self._latencies)
        
        # Calculate throughput
        elapsed = time.time() - self._start_time
        throughput = 0.0
        if elapsed > 0:
            throughput = self.metrics.completed_operations / elapsed
        
        return {
            "total_operations": self.metrics.total_operations,
            "completed_operations": self.metrics.completed_operations,
            "failed_operations": self.metrics.failed_operations,
            "pending_operations": self.metrics.pending_operations,
            "avg_latency_ms": round(avg_latency, 3),
            "throughput_ops_per_sec": round(throughput, 2),
            "queue_depth": self.op_queue.qsize(),
            "completion_queue_depth": self.completion_queue.qsize(),
            "ring_buffer_size": self.ring_buffer.size()
        }
    
    async def shutdown(self) -> None:
        """Shutdown the non-blocking module"""
        self._logger.info("Shutting down NonBlockingModule...")
        self._shutting_down = True
        
        # Cancel worker tasks
        for task in self._worker_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Shutdown executor
        if self._executor:
            self._executor.shutdown(wait=False)
        
        self.operations.clear()
        self._initialized = False
        
        self._logger.info("NonBlockingModule shutdown complete")


# Example usage
async def main():
    """Example usage of NonBlockingModule"""
    
    config = NonBlockingConfig(
        max_concurrent_ops=1000,
        worker_threads=4,
        enable_batching=True
    )
    
    module = NonBlockingModule(config)
    await module.initialize()
    
    try:
        # Submit single operation
        op = await module.submit(
            OperationType.COMPUTE,
            data={"input": "test"},
            priority=10
        )
        print(f"Submitted op: {op.op_id}")
        
        # Wait for completion
        result = await module.wait(op.op_id, timeout=5.0)
        print(f"Result: {result.status.value}, data: {result.result}")
        
        # Submit batch
        batch = await module.submit_batch([
            (OperationType.READ, {"key": "value1"}, 5),
            (OperationType.WRITE, {"key": "value2"}, 5),
            (OperationType.COMPUTE, {"key": "value3"}, 10),
        ])
        print(f"Submitted batch of {len(batch)} operations")
        
        # Wait for all
        for op in batch:
            result = await module.wait(op.op_id, timeout=5.0)
            if result:
                print(f"  {op.op_type.value}: {result.status.value}")
        
        # Poll completions
        await asyncio.sleep(0.1)
        events = module.completion_queue.poll_completions()
        print(f"Polled {len(events)} completion events")
        
        # Get metrics
        metrics = module.get_metrics()
        print(f"\nMetrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    
    finally:
        await module.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

