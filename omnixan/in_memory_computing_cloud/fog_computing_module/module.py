"""
OMNIXAN Fog Computing Module
in_memory_computing_cloud/fog_computing_module

Production-ready fog computing implementation providing distributed computation
between edge devices and cloud, with intelligent task offloading, resource
management, and latency optimization.
"""

import asyncio
import logging
import time
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


class NodeType(str, Enum):
    """Types of fog nodes"""
    EDGE = "edge"  # Edge device
    FOG = "fog"  # Fog node
    CLOUD = "cloud"  # Cloud datacenter
    GATEWAY = "gateway"  # Edge gateway


class TaskPriority(str, Enum):
    """Task priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OffloadStrategy(str, Enum):
    """Task offloading strategies"""
    LATENCY_FIRST = "latency_first"
    ENERGY_FIRST = "energy_first"
    COST_FIRST = "cost_first"
    BALANCED = "balanced"
    DEADLINE_AWARE = "deadline_aware"


@dataclass
class FogNode:
    """Represents a fog computing node"""
    node_id: str
    name: str
    node_type: NodeType
    location: Tuple[float, float]  # (lat, lon)
    cpu_cores: int
    memory_mb: int
    bandwidth_mbps: float
    latency_ms: float  # Latency to cloud
    available_cpu: float = 1.0  # 0-1
    available_memory: float = 1.0  # 0-1
    energy_cost: float = 1.0  # Cost per computation unit
    is_online: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FogTask:
    """A task to be executed in fog network"""
    task_id: str
    name: str
    priority: TaskPriority
    compute_units: int
    memory_mb: int
    data_size_kb: float
    deadline_ms: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING
    assigned_node: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    callback: Optional[Callable] = None


@dataclass
class OffloadDecision:
    """Offloading decision result"""
    task_id: str
    target_node: str
    estimated_latency_ms: float
    estimated_energy: float
    estimated_cost: float
    reason: str


@dataclass
class FogMetrics:
    """Fog computing metrics"""
    total_nodes: int = 0
    online_nodes: int = 0
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_latency_ms: float = 0.0
    total_compute_time_ms: float = 0.0
    avg_resource_utilization: float = 0.0
    offload_to_cloud_ratio: float = 0.0


class FogConfig(BaseModel):
    """Configuration for fog computing"""
    max_nodes: int = Field(
        default=100,
        ge=1,
        description="Maximum fog nodes"
    )
    max_queue_size: int = Field(
        default=1000,
        ge=10,
        description="Maximum task queue size"
    )
    offload_strategy: OffloadStrategy = Field(
        default=OffloadStrategy.BALANCED,
        description="Offloading strategy"
    )
    latency_threshold_ms: float = Field(
        default=100.0,
        gt=0.0,
        description="Latency threshold for local execution"
    )
    cloud_fallback: bool = Field(
        default=True,
        description="Allow cloud fallback"
    )
    task_timeout_ms: float = Field(
        default=30000.0,
        gt=0.0,
        description="Task timeout"
    )
    resource_check_interval: float = Field(
        default=5.0,
        gt=0.0,
        description="Resource check interval"
    )


class FogError(Exception):
    """Base exception for fog computing errors"""
    pass


class NodeNotFoundError(FogError):
    """Raised when node is not found"""
    pass


class TaskOffloadError(FogError):
    """Raised when task offloading fails"""
    pass


# ============================================================================
# Offload Decision Engine
# ============================================================================

class OffloadDecisionEngine:
    """Makes intelligent offloading decisions"""
    
    def __init__(self, strategy: OffloadStrategy):
        self.strategy = strategy
    
    def decide(
        self,
        task: FogTask,
        nodes: List[FogNode]
    ) -> Optional[OffloadDecision]:
        """Decide where to offload task"""
        if not nodes:
            return None
        
        # Filter online nodes with sufficient resources
        suitable_nodes = [
            n for n in nodes
            if n.is_online
            and n.available_memory * n.memory_mb >= task.memory_mb
            and n.available_cpu >= 0.1
        ]
        
        if not suitable_nodes:
            return None
        
        if self.strategy == OffloadStrategy.LATENCY_FIRST:
            return self._latency_first(task, suitable_nodes)
        elif self.strategy == OffloadStrategy.ENERGY_FIRST:
            return self._energy_first(task, suitable_nodes)
        elif self.strategy == OffloadStrategy.COST_FIRST:
            return self._cost_first(task, suitable_nodes)
        elif self.strategy == OffloadStrategy.DEADLINE_AWARE:
            return self._deadline_aware(task, suitable_nodes)
        else:
            return self._balanced(task, suitable_nodes)
    
    def _latency_first(
        self,
        task: FogTask,
        nodes: List[FogNode]
    ) -> OffloadDecision:
        """Select node with lowest latency"""
        best = min(nodes, key=lambda n: self._estimate_latency(task, n))
        
        return OffloadDecision(
            task_id=task.task_id,
            target_node=best.node_id,
            estimated_latency_ms=self._estimate_latency(task, best),
            estimated_energy=self._estimate_energy(task, best),
            estimated_cost=self._estimate_cost(task, best),
            reason="Lowest latency node"
        )
    
    def _energy_first(
        self,
        task: FogTask,
        nodes: List[FogNode]
    ) -> OffloadDecision:
        """Select node with lowest energy consumption"""
        best = min(nodes, key=lambda n: self._estimate_energy(task, n))
        
        return OffloadDecision(
            task_id=task.task_id,
            target_node=best.node_id,
            estimated_latency_ms=self._estimate_latency(task, best),
            estimated_energy=self._estimate_energy(task, best),
            estimated_cost=self._estimate_cost(task, best),
            reason="Lowest energy consumption"
        )
    
    def _cost_first(
        self,
        task: FogTask,
        nodes: List[FogNode]
    ) -> OffloadDecision:
        """Select node with lowest cost"""
        best = min(nodes, key=lambda n: self._estimate_cost(task, n))
        
        return OffloadDecision(
            task_id=task.task_id,
            target_node=best.node_id,
            estimated_latency_ms=self._estimate_latency(task, best),
            estimated_energy=self._estimate_energy(task, best),
            estimated_cost=self._estimate_cost(task, best),
            reason="Lowest cost"
        )
    
    def _deadline_aware(
        self,
        task: FogTask,
        nodes: List[FogNode]
    ) -> OffloadDecision:
        """Select node that can meet deadline"""
        deadline = task.deadline_ms or float('inf')
        
        # Filter nodes that can meet deadline
        valid = [
            n for n in nodes
            if self._estimate_latency(task, n) <= deadline
        ]
        
        if not valid:
            # Fall back to fastest
            valid = nodes
        
        # Among valid, pick lowest cost
        best = min(valid, key=lambda n: self._estimate_cost(task, n))
        
        return OffloadDecision(
            task_id=task.task_id,
            target_node=best.node_id,
            estimated_latency_ms=self._estimate_latency(task, best),
            estimated_energy=self._estimate_energy(task, best),
            estimated_cost=self._estimate_cost(task, best),
            reason="Meets deadline with lowest cost"
        )
    
    def _balanced(
        self,
        task: FogTask,
        nodes: List[FogNode]
    ) -> OffloadDecision:
        """Balanced decision considering all factors"""
        def score(node: FogNode) -> float:
            latency = self._estimate_latency(task, node) / 100.0  # Normalize
            energy = self._estimate_energy(task, node)
            cost = self._estimate_cost(task, node)
            
            # Priority weights
            priority_weight = {
                TaskPriority.CRITICAL: 0.6,
                TaskPriority.HIGH: 0.4,
                TaskPriority.NORMAL: 0.3,
                TaskPriority.LOW: 0.2,
                TaskPriority.BACKGROUND: 0.1,
            }
            
            latency_weight = priority_weight.get(task.priority, 0.3)
            
            return (
                latency_weight * latency +
                0.3 * energy +
                (1 - latency_weight - 0.3) * cost
            )
        
        best = min(nodes, key=score)
        
        return OffloadDecision(
            task_id=task.task_id,
            target_node=best.node_id,
            estimated_latency_ms=self._estimate_latency(task, best),
            estimated_energy=self._estimate_energy(task, best),
            estimated_cost=self._estimate_cost(task, best),
            reason="Balanced optimization"
        )
    
    def _estimate_latency(self, task: FogTask, node: FogNode) -> float:
        """Estimate task latency on node"""
        # Transmission latency
        transmission_time = (task.data_size_kb * 8) / node.bandwidth_mbps
        
        # Compute latency
        compute_time = task.compute_units / (node.cpu_cores * node.available_cpu)
        
        return node.latency_ms + transmission_time + compute_time
    
    def _estimate_energy(self, task: FogTask, node: FogNode) -> float:
        """Estimate energy consumption"""
        return task.compute_units * node.energy_cost
    
    def _estimate_cost(self, task: FogTask, node: FogNode) -> float:
        """Estimate monetary cost"""
        base_cost = node.energy_cost * task.compute_units
        
        # Cloud nodes are more expensive
        if node.node_type == NodeType.CLOUD:
            base_cost *= 2.0
        
        return base_cost


# ============================================================================
# Task Scheduler
# ============================================================================

class FogScheduler:
    """Schedules tasks across fog nodes"""
    
    def __init__(self):
        self.task_queue: List[Tuple[int, FogTask]] = []  # Priority queue
        self.running_tasks: Dict[str, Tuple[str, FogTask]] = {}  # task_id -> (node_id, task)
    
    def enqueue(self, task: FogTask) -> None:
        """Add task to queue"""
        # Priority: lower number = higher priority
        priority_map = {
            TaskPriority.CRITICAL: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.NORMAL: 2,
            TaskPriority.LOW: 3,
            TaskPriority.BACKGROUND: 4,
        }
        
        priority = priority_map.get(task.priority, 2)
        heapq.heappush(self.task_queue, (priority, task))
        task.status = TaskStatus.QUEUED
    
    def dequeue(self) -> Optional[FogTask]:
        """Get next task from queue"""
        if self.task_queue:
            _, task = heapq.heappop(self.task_queue)
            return task
        return None
    
    def assign(self, task: FogTask, node_id: str) -> None:
        """Assign task to node"""
        task.assigned_node = node_id
        task.started_at = time.time()
        task.status = TaskStatus.RUNNING
        self.running_tasks[task.task_id] = (node_id, task)
    
    def complete(self, task_id: str, result: Any = None) -> Optional[FogTask]:
        """Mark task as complete"""
        if task_id in self.running_tasks:
            node_id, task = self.running_tasks.pop(task_id)
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            task.result = result
            return task
        return None
    
    def fail(self, task_id: str, error: str) -> Optional[FogTask]:
        """Mark task as failed"""
        if task_id in self.running_tasks:
            node_id, task = self.running_tasks.pop(task_id)
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
            task.result = {"error": error}
            return task
        return None
    
    def get_queue_status(self) -> Dict[str, int]:
        """Get queue status"""
        return {
            "queued": len(self.task_queue),
            "running": len(self.running_tasks)
        }


# ============================================================================
# Main Module Implementation
# ============================================================================

class FogComputingModule:
    """
    Production-ready fog computing module for OMNIXAN.
    
    Provides:
    - Distributed fog node management
    - Intelligent task offloading
    - Priority-based scheduling
    - Latency and energy optimization
    - Cloud fallback
    """
    
    def __init__(self, config: Optional[FogConfig] = None):
        """Initialize the Fog Computing Module"""
        self.config = config or FogConfig()
        self.nodes: Dict[str, FogNode] = {}
        self.decision_engine = OffloadDecisionEngine(self.config.offload_strategy)
        self.scheduler = FogScheduler()
        self.metrics = FogMetrics()
        
        # Background tasks
        self._scheduler_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._initialized = False
        self._shutting_down = False
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(__name__)
        
        # Latency tracking
        self._latencies: List[float] = []
    
    async def initialize(self) -> None:
        """Initialize the fog computing module"""
        if self._initialized:
            self._logger.warning("Module already initialized")
            return
        
        try:
            self._logger.info("Initializing FogComputingModule...")
            
            # Start background tasks
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            
            self._initialized = True
            self._logger.info("FogComputingModule initialized successfully")
        
        except Exception as e:
            self._logger.error(f"Initialization failed: {str(e)}")
            raise FogError(f"Failed to initialize module: {str(e)}")
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute fog computing operation"""
        if not self._initialized:
            raise FogError("Module not initialized")
        
        operation = params.get("operation")
        
        if operation == "register_node":
            node = await self.register_node(
                name=params["name"],
                node_type=NodeType(params.get("type", "fog")),
                location=tuple(params.get("location", [0, 0])),
                cpu_cores=params.get("cpu_cores", 4),
                memory_mb=params.get("memory_mb", 4096),
                bandwidth_mbps=params.get("bandwidth_mbps", 100),
                latency_ms=params.get("latency_ms", 10)
            )
            return {"node_id": node.node_id}
        
        elif operation == "unregister_node":
            success = await self.unregister_node(params["node_id"])
            return {"success": success}
        
        elif operation == "submit_task":
            task = await self.submit_task(
                name=params["name"],
                priority=TaskPriority(params.get("priority", "normal")),
                compute_units=params.get("compute_units", 1),
                memory_mb=params.get("memory_mb", 256),
                data_size_kb=params.get("data_size_kb", 100),
                deadline_ms=params.get("deadline_ms")
            )
            return {"task_id": task.task_id, "status": task.status.value}
        
        elif operation == "get_task_status":
            status = self.get_task_status(params["task_id"])
            return status or {"error": "Task not found"}
        
        elif operation == "cancel_task":
            success = await self.cancel_task(params["task_id"])
            return {"success": success}
        
        elif operation == "get_metrics":
            return self.get_metrics()
        
        elif operation == "list_nodes":
            return {
                "nodes": [
                    {
                        "node_id": n.node_id,
                        "name": n.name,
                        "type": n.node_type.value,
                        "is_online": n.is_online,
                        "cpu_available": n.available_cpu,
                        "memory_available": n.available_memory
                    }
                    for n in self.nodes.values()
                ]
            }
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def register_node(
        self,
        name: str,
        node_type: NodeType,
        location: Tuple[float, float],
        cpu_cores: int,
        memory_mb: int,
        bandwidth_mbps: float,
        latency_ms: float
    ) -> FogNode:
        """Register a fog node"""
        async with self._lock:
            if len(self.nodes) >= self.config.max_nodes:
                raise FogError("Maximum node limit reached")
            
            node = FogNode(
                node_id=str(uuid4()),
                name=name,
                node_type=node_type,
                location=location,
                cpu_cores=cpu_cores,
                memory_mb=memory_mb,
                bandwidth_mbps=bandwidth_mbps,
                latency_ms=latency_ms
            )
            
            self.nodes[node.node_id] = node
            self._update_metrics()
            
            self._logger.info(f"Registered fog node {node.node_id}: {name}")
            return node
    
    async def unregister_node(self, node_id: str) -> bool:
        """Unregister a fog node"""
        async with self._lock:
            if node_id not in self.nodes:
                return False
            
            del self.nodes[node_id]
            self._update_metrics()
            
            self._logger.info(f"Unregistered fog node {node_id}")
            return True
    
    async def submit_task(
        self,
        name: str,
        priority: TaskPriority = TaskPriority.NORMAL,
        compute_units: int = 1,
        memory_mb: int = 256,
        data_size_kb: float = 100,
        deadline_ms: Optional[float] = None,
        callback: Optional[Callable] = None
    ) -> FogTask:
        """Submit a task for execution"""
        task = FogTask(
            task_id=str(uuid4()),
            name=name,
            priority=priority,
            compute_units=compute_units,
            memory_mb=memory_mb,
            data_size_kb=data_size_kb,
            deadline_ms=deadline_ms,
            callback=callback
        )
        
        self.scheduler.enqueue(task)
        self.metrics.total_tasks += 1
        
        self._logger.debug(f"Submitted task {task.task_id}: {name}")
        return task
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        # Check if in queue
        for i, (_, task) in enumerate(self.scheduler.task_queue):
            if task.task_id == task_id:
                self.scheduler.task_queue.pop(i)
                heapq.heapify(self.scheduler.task_queue)
                task.status = TaskStatus.CANCELLED
                return True
        
        return False
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        # Check running
        if task_id in self.scheduler.running_tasks:
            node_id, task = self.scheduler.running_tasks[task_id]
            return {
                "task_id": task_id,
                "status": task.status.value,
                "assigned_node": node_id,
                "started_at": task.started_at
            }
        
        # Check queue
        for _, task in self.scheduler.task_queue:
            if task.task_id == task_id:
                return {
                    "task_id": task_id,
                    "status": task.status.value,
                    "queue_position": self.scheduler.task_queue.index((_, task))
                }
        
        return None
    
    async def _scheduler_loop(self) -> None:
        """Background scheduler loop"""
        while not self._shutting_down:
            try:
                await asyncio.sleep(0.01)  # Check every 10ms
                
                async with self._lock:
                    task = self.scheduler.dequeue()
                    if task:
                        await self._process_task(task)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Scheduler error: {e}")
    
    async def _process_task(self, task: FogTask) -> None:
        """Process a task"""
        # Get offload decision
        decision = self.decision_engine.decide(task, list(self.nodes.values()))
        
        if not decision:
            if self.config.cloud_fallback:
                # Would offload to cloud
                self._logger.warning(f"No suitable fog node, would fall back to cloud")
            
            self.scheduler.fail(task.task_id, "No suitable node available")
            self.metrics.failed_tasks += 1
            return
        
        # Assign to node
        node = self.nodes.get(decision.target_node)
        if not node:
            self.scheduler.fail(task.task_id, "Node not available")
            self.metrics.failed_tasks += 1
            return
        
        self.scheduler.assign(task, node.node_id)
        
        # Update node resources
        node.available_cpu = max(0, node.available_cpu - 0.1)
        node.available_memory = max(
            0,
            node.available_memory - task.memory_mb / node.memory_mb
        )
        
        # Simulate task execution
        execution_time = decision.estimated_latency_ms / 1000.0
        await asyncio.sleep(min(execution_time, 0.1))  # Cap for demo
        
        # Complete task
        completed_task = self.scheduler.complete(task.task_id, {"success": True})
        
        if completed_task:
            latency = (completed_task.completed_at - completed_task.started_at) * 1000
            self._latencies.append(latency)
            self.metrics.completed_tasks += 1
            self.metrics.total_compute_time_ms += latency
            
            # Track cloud offloads
            if node.node_type == NodeType.CLOUD:
                self.metrics.offload_to_cloud_ratio = (
                    self.metrics.offload_to_cloud_ratio * 0.9 + 0.1
                )
            else:
                self.metrics.offload_to_cloud_ratio *= 0.9
        
        # Restore node resources
        node.available_cpu = min(1.0, node.available_cpu + 0.1)
        node.available_memory = min(
            1.0,
            node.available_memory + task.memory_mb / node.memory_mb
        )
    
    async def _monitor_loop(self) -> None:
        """Background resource monitoring"""
        while not self._shutting_down:
            try:
                await asyncio.sleep(self.config.resource_check_interval)
                
                async with self._lock:
                    self._update_metrics()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Monitor error: {e}")
    
    def _update_metrics(self) -> None:
        """Update metrics"""
        self.metrics.total_nodes = len(self.nodes)
        self.metrics.online_nodes = sum(1 for n in self.nodes.values() if n.is_online)
        
        if self._latencies:
            self.metrics.avg_latency_ms = sum(self._latencies) / len(self._latencies)
        
        # Average resource utilization
        if self.nodes:
            self.metrics.avg_resource_utilization = 1.0 - (
                sum(n.available_cpu for n in self.nodes.values()) / len(self.nodes)
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get fog computing metrics"""
        return {
            "total_nodes": self.metrics.total_nodes,
            "online_nodes": self.metrics.online_nodes,
            "total_tasks": self.metrics.total_tasks,
            "completed_tasks": self.metrics.completed_tasks,
            "failed_tasks": self.metrics.failed_tasks,
            "success_rate": (
                self.metrics.completed_tasks /
                max(self.metrics.total_tasks, 1)
            ),
            "avg_latency_ms": round(self.metrics.avg_latency_ms, 2),
            "total_compute_time_ms": round(self.metrics.total_compute_time_ms, 2),
            "avg_resource_utilization": round(self.metrics.avg_resource_utilization, 4),
            "offload_to_cloud_ratio": round(self.metrics.offload_to_cloud_ratio, 4),
            "queue_status": self.scheduler.get_queue_status()
        }
    
    async def shutdown(self) -> None:
        """Shutdown the fog computing module"""
        self._logger.info("Shutting down FogComputingModule...")
        self._shutting_down = True
        
        for task in [self._scheduler_task, self._monitor_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.nodes.clear()
        self._initialized = False
        self._logger.info("FogComputingModule shutdown complete")


# Example usage
async def main():
    """Example usage of FogComputingModule"""
    
    config = FogConfig(
        offload_strategy=OffloadStrategy.BALANCED,
        latency_threshold_ms=100
    )
    
    module = FogComputingModule(config)
    await module.initialize()
    
    try:
        # Register fog nodes
        node1 = await module.register_node(
            name="Edge Gateway 1",
            node_type=NodeType.GATEWAY,
            location=(40.7128, -74.0060),
            cpu_cores=4,
            memory_mb=8192,
            bandwidth_mbps=1000,
            latency_ms=5
        )
        
        node2 = await module.register_node(
            name="Fog Server 1",
            node_type=NodeType.FOG,
            location=(40.7580, -73.9855),
            cpu_cores=16,
            memory_mb=32768,
            bandwidth_mbps=10000,
            latency_ms=10
        )
        
        node3 = await module.register_node(
            name="Cloud Backend",
            node_type=NodeType.CLOUD,
            location=(39.0438, -77.4874),
            cpu_cores=64,
            memory_mb=131072,
            bandwidth_mbps=100000,
            latency_ms=50
        )
        
        print(f"Registered {len(module.nodes)} nodes")
        
        # Submit tasks
        tasks = []
        for i in range(10):
            task = await module.submit_task(
                name=f"Task_{i}",
                priority=random.choice(list(TaskPriority)),
                compute_units=random.randint(1, 10),
                memory_mb=random.randint(128, 1024),
                data_size_kb=random.uniform(10, 500)
            )
            tasks.append(task)
        
        print(f"Submitted {len(tasks)} tasks")
        
        # Wait for processing
        await asyncio.sleep(1)
        
        # Get metrics
        metrics = module.get_metrics()
        print(f"\nMetrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    
    finally:
        await module.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

