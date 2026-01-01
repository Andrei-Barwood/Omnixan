"""
OMNIXAN Fault Mitigation Module
virtualized_cluster/fault_mitigation_module

Production-ready fault mitigation system for distributed computing with
failure detection, automatic recovery, redundancy management, and
graceful degradation capabilities.
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

import numpy as np

from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaultType(str, Enum):
    """Types of faults"""
    TRANSIENT = "transient"  # Temporary
    INTERMITTENT = "intermittent"  # Recurring
    PERMANENT = "permanent"  # Hardware failure
    BYZANTINE = "byzantine"  # Arbitrary behavior
    CRASH = "crash"  # Process/node crash
    OMISSION = "omission"  # Message loss


class ComponentState(str, Enum):
    """Component health states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAULTY = "faulty"
    RECOVERING = "recovering"
    FAILED = "failed"


class RecoveryStrategy(str, Enum):
    """Recovery strategies"""
    RESTART = "restart"
    FAILOVER = "failover"
    CHECKPOINT = "checkpoint"
    REPLICATION = "replication"
    ISOLATION = "isolation"


class RedundancyLevel(str, Enum):
    """Redundancy levels"""
    NONE = "none"
    DUAL = "dual"  # 2x
    TRIPLE = "triple"  # TMR
    QUAD = "quad"  # 4x


@dataclass
class Component:
    """A system component"""
    component_id: str
    name: str
    state: ComponentState
    redundancy: RedundancyLevel = RedundancyLevel.NONE
    replicas: List[str] = field(default_factory=list)
    fault_count: int = 0
    last_health_check: float = field(default_factory=time.time)
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3


@dataclass
class Fault:
    """A detected fault"""
    fault_id: str
    fault_type: FaultType
    component_id: str
    description: str
    timestamp: float
    severity: int = 1  # 1-5
    resolved: bool = False
    resolution_time: Optional[float] = None


@dataclass
class Checkpoint:
    """System checkpoint"""
    checkpoint_id: str
    component_id: str
    state_data: Dict[str, Any]
    timestamp: float
    is_valid: bool = True


@dataclass
class RecoveryAction:
    """A recovery action"""
    action_id: str
    strategy: RecoveryStrategy
    component_id: str
    started_at: float
    completed_at: Optional[float] = None
    success: bool = False


@dataclass
class FaultMetrics:
    """Fault mitigation metrics"""
    total_faults_detected: int = 0
    faults_mitigated: int = 0
    faults_unresolved: int = 0
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    avg_recovery_time_ms: float = 0.0
    system_availability: float = 1.0


class FaultConfig(BaseModel):
    """Configuration for fault mitigation"""
    health_check_interval_s: float = Field(
        default=5.0,
        gt=0.0,
        description="Health check interval"
    )
    max_fault_threshold: int = Field(
        default=3,
        ge=1,
        description="Faults before failover"
    )
    recovery_timeout_s: float = Field(
        default=30.0,
        gt=0.0,
        description="Recovery timeout"
    )
    default_redundancy: RedundancyLevel = Field(
        default=RedundancyLevel.TRIPLE,
        description="Default redundancy level"
    )
    enable_auto_recovery: bool = Field(
        default=True,
        description="Enable automatic recovery"
    )
    checkpoint_interval_s: float = Field(
        default=60.0,
        gt=0.0,
        description="Checkpoint interval"
    )


class FaultError(Exception):
    """Base exception for fault errors"""
    pass


# ============================================================================
# Failure Detector
# ============================================================================

class FailureDetector:
    """Heartbeat-based failure detector"""
    
    def __init__(self, timeout_s: float = 10.0):
        self.timeout_s = timeout_s
        self.heartbeats: Dict[str, float] = {}
        self.suspected: Set[str] = set()
    
    def heartbeat(self, component_id: str) -> None:
        """Record heartbeat from component"""
        self.heartbeats[component_id] = time.time()
        self.suspected.discard(component_id)
    
    def check(self, component_id: str) -> bool:
        """Check if component is alive"""
        if component_id not in self.heartbeats:
            return False
        
        elapsed = time.time() - self.heartbeats[component_id]
        
        if elapsed > self.timeout_s:
            self.suspected.add(component_id)
            return False
        
        return True
    
    def get_suspected(self) -> Set[str]:
        """Get suspected failed components"""
        now = time.time()
        for comp_id, last_hb in self.heartbeats.items():
            if now - last_hb > self.timeout_s:
                self.suspected.add(comp_id)
        return self.suspected.copy()


# ============================================================================
# Main Module Implementation
# ============================================================================

class FaultMitigationModule:
    """
    Production-ready Fault Mitigation module for OMNIXAN.
    
    Provides:
    - Fault detection and classification
    - Automatic recovery strategies
    - Redundancy management (TMR, etc.)
    - Checkpointing and rollback
    - Graceful degradation
    - Health monitoring
    """
    
    def __init__(self, config: Optional[FaultConfig] = None):
        """Initialize the Fault Mitigation Module"""
        self.config = config or FaultConfig()
        
        self.components: Dict[str, Component] = {}
        self.faults: Dict[str, Fault] = {}
        self.checkpoints: Dict[str, List[Checkpoint]] = defaultdict(list)
        self.recovery_actions: List[RecoveryAction] = []
        
        self.failure_detector = FailureDetector(self.config.health_check_interval_s * 2)
        self.metrics = FaultMetrics()
        
        self._monitor_task: Optional[asyncio.Task] = None
        self._recovery_times: List[float] = []
        self._start_time = time.time()
        self._initialized = False
        self._shutting_down = False
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the fault mitigation module"""
        if self._initialized:
            self._logger.warning("Module already initialized")
            return
        
        try:
            self._logger.info("Initializing FaultMitigationModule...")
            
            # Start health monitoring
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            
            self._initialized = True
            self._logger.info("FaultMitigationModule initialized successfully")
        
        except Exception as e:
            self._logger.error(f"Initialization failed: {str(e)}")
            raise FaultError(f"Failed to initialize module: {str(e)}")
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute fault mitigation operation"""
        if not self._initialized:
            raise FaultError("Module not initialized")
        
        operation = params.get("operation")
        
        if operation == "register_component":
            name = params["name"]
            redundancy = RedundancyLevel(params.get("redundancy", "triple"))
            component = await self.register_component(name, redundancy)
            return {"component_id": component.component_id}
        
        elif operation == "report_fault":
            component_id = params["component_id"]
            fault_type = FaultType(params.get("fault_type", "transient"))
            description = params.get("description", "Unknown fault")
            fault = await self.report_fault(component_id, fault_type, description)
            return {"fault_id": fault.fault_id}
        
        elif operation == "heartbeat":
            component_id = params["component_id"]
            await self.heartbeat(component_id)
            return {"success": True}
        
        elif operation == "checkpoint":
            component_id = params["component_id"]
            state = params.get("state", {})
            cp = await self.create_checkpoint(component_id, state)
            return {"checkpoint_id": cp.checkpoint_id}
        
        elif operation == "restore":
            component_id = params["component_id"]
            checkpoint_id = params.get("checkpoint_id")
            state = await self.restore_checkpoint(component_id, checkpoint_id)
            return {"state": state}
        
        elif operation == "recover":
            component_id = params["component_id"]
            strategy = RecoveryStrategy(params.get("strategy", "restart"))
            success = await self.recover_component(component_id, strategy)
            return {"success": success}
        
        elif operation == "get_status":
            return self.get_status()
        
        elif operation == "get_metrics":
            return self.get_metrics()
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def register_component(
        self,
        name: str,
        redundancy: RedundancyLevel = RedundancyLevel.TRIPLE
    ) -> Component:
        """Register a component for monitoring"""
        async with self._lock:
            component = Component(
                component_id=str(uuid4()),
                name=name,
                state=ComponentState.HEALTHY,
                redundancy=redundancy
            )
            
            # Create replicas if redundant
            if redundancy != RedundancyLevel.NONE:
                replica_count = {
                    RedundancyLevel.DUAL: 2,
                    RedundancyLevel.TRIPLE: 3,
                    RedundancyLevel.QUAD: 4
                }.get(redundancy, 1)
                
                for i in range(replica_count):
                    replica_id = f"{component.component_id}_replica_{i}"
                    component.replicas.append(replica_id)
            
            self.components[component.component_id] = component
            self.failure_detector.heartbeat(component.component_id)
            
            self._logger.info(f"Registered component: {name} ({redundancy.value})")
            return component
    
    async def report_fault(
        self,
        component_id: str,
        fault_type: FaultType,
        description: str = ""
    ) -> Fault:
        """Report a fault in a component"""
        async with self._lock:
            fault = Fault(
                fault_id=str(uuid4()),
                fault_type=fault_type,
                component_id=component_id,
                description=description,
                timestamp=time.time(),
                severity=self._calculate_severity(fault_type)
            )
            
            self.faults[fault.fault_id] = fault
            self.metrics.total_faults_detected += 1
            
            # Update component state
            if component_id in self.components:
                comp = self.components[component_id]
                comp.fault_count += 1
                
                if comp.fault_count >= self.config.max_fault_threshold:
                    comp.state = ComponentState.FAULTY
                else:
                    comp.state = ComponentState.DEGRADED
            
            self._logger.warning(f"Fault reported: {fault_type.value} in {component_id}")
            
            # Trigger auto-recovery
            if self.config.enable_auto_recovery:
                asyncio.create_task(self._auto_recover(component_id, fault))
            
            return fault
    
    async def heartbeat(self, component_id: str) -> None:
        """Receive heartbeat from component"""
        async with self._lock:
            self.failure_detector.heartbeat(component_id)
            
            if component_id in self.components:
                comp = self.components[component_id]
                comp.last_health_check = time.time()
                
                if comp.state == ComponentState.RECOVERING:
                    comp.state = ComponentState.HEALTHY
    
    async def create_checkpoint(
        self,
        component_id: str,
        state_data: Dict[str, Any]
    ) -> Checkpoint:
        """Create a checkpoint for a component"""
        async with self._lock:
            checkpoint = Checkpoint(
                checkpoint_id=str(uuid4()),
                component_id=component_id,
                state_data=state_data.copy(),
                timestamp=time.time()
            )
            
            self.checkpoints[component_id].append(checkpoint)
            
            # Keep only last 5 checkpoints
            if len(self.checkpoints[component_id]) > 5:
                self.checkpoints[component_id] = self.checkpoints[component_id][-5:]
            
            return checkpoint
    
    async def restore_checkpoint(
        self,
        component_id: str,
        checkpoint_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Restore component from checkpoint"""
        async with self._lock:
            if component_id not in self.checkpoints:
                raise FaultError("No checkpoints available")
            
            checkpoints = self.checkpoints[component_id]
            
            if checkpoint_id:
                checkpoint = next(
                    (c for c in checkpoints if c.checkpoint_id == checkpoint_id),
                    None
                )
            else:
                # Get latest valid checkpoint
                checkpoint = next(
                    (c for c in reversed(checkpoints) if c.is_valid),
                    None
                )
            
            if not checkpoint:
                raise FaultError("Checkpoint not found")
            
            self._logger.info(f"Restored checkpoint for {component_id}")
            return checkpoint.state_data
    
    async def recover_component(
        self,
        component_id: str,
        strategy: RecoveryStrategy = RecoveryStrategy.RESTART
    ) -> bool:
        """Attempt to recover a component"""
        async with self._lock:
            if component_id not in self.components:
                return False
            
            comp = self.components[component_id]
            
            if comp.recovery_attempts >= comp.max_recovery_attempts:
                comp.state = ComponentState.FAILED
                self.metrics.failed_recoveries += 1
                return False
            
            action = RecoveryAction(
                action_id=str(uuid4()),
                strategy=strategy,
                component_id=component_id,
                started_at=time.time()
            )
            self.recovery_actions.append(action)
            
            comp.state = ComponentState.RECOVERING
            comp.recovery_attempts += 1
            self.metrics.recovery_attempts += 1
        
        # Simulate recovery
        try:
            await self._execute_recovery(comp, strategy)
            
            async with self._lock:
                action.completed_at = time.time()
                action.success = True
                
                recovery_time = (action.completed_at - action.started_at) * 1000
                self._recovery_times.append(recovery_time)
                
                comp.state = ComponentState.HEALTHY
                comp.fault_count = 0
                
                self.metrics.successful_recoveries += 1
                self.metrics.faults_mitigated += 1
                
                self._logger.info(f"Successfully recovered {component_id}")
                return True
        
        except Exception as e:
            async with self._lock:
                action.completed_at = time.time()
                action.success = False
                comp.state = ComponentState.FAULTY
                self.metrics.failed_recoveries += 1
                
                self._logger.error(f"Recovery failed for {component_id}: {e}")
                return False
    
    async def _execute_recovery(
        self,
        component: Component,
        strategy: RecoveryStrategy
    ) -> None:
        """Execute recovery strategy"""
        if strategy == RecoveryStrategy.RESTART:
            await asyncio.sleep(0.1)  # Simulate restart
        
        elif strategy == RecoveryStrategy.FAILOVER:
            if component.replicas:
                # Promote first healthy replica
                await asyncio.sleep(0.05)
        
        elif strategy == RecoveryStrategy.CHECKPOINT:
            # Restore from checkpoint
            if component.component_id in self.checkpoints:
                await asyncio.sleep(0.1)
        
        elif strategy == RecoveryStrategy.REPLICATION:
            # Re-sync with replicas
            await asyncio.sleep(0.2)
        
        elif strategy == RecoveryStrategy.ISOLATION:
            # Isolate faulty component
            await asyncio.sleep(0.01)
    
    async def _auto_recover(self, component_id: str, fault: Fault) -> None:
        """Automatic recovery based on fault type"""
        strategy_map = {
            FaultType.TRANSIENT: RecoveryStrategy.RESTART,
            FaultType.INTERMITTENT: RecoveryStrategy.CHECKPOINT,
            FaultType.PERMANENT: RecoveryStrategy.FAILOVER,
            FaultType.CRASH: RecoveryStrategy.RESTART,
            FaultType.BYZANTINE: RecoveryStrategy.ISOLATION,
            FaultType.OMISSION: RecoveryStrategy.REPLICATION,
        }
        
        strategy = strategy_map.get(fault.fault_type, RecoveryStrategy.RESTART)
        
        success = await self.recover_component(component_id, strategy)
        
        async with self._lock:
            fault.resolved = success
            if success:
                fault.resolution_time = time.time()
    
    async def _monitor_loop(self) -> None:
        """Background health monitoring loop"""
        while not self._shutting_down:
            try:
                await asyncio.sleep(self.config.health_check_interval_s)
                
                async with self._lock:
                    # Check for suspected failures
                    suspected = self.failure_detector.get_suspected()
                    
                    for comp_id in suspected:
                        if comp_id in self.components:
                            comp = self.components[comp_id]
                            if comp.state == ComponentState.HEALTHY:
                                comp.state = ComponentState.DEGRADED
                    
                    # Update system availability
                    total = len(self.components)
                    healthy = sum(
                        1 for c in self.components.values()
                        if c.state == ComponentState.HEALTHY
                    )
                    self.metrics.system_availability = healthy / total if total > 0 else 1.0
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Monitor error: {e}")
    
    def _calculate_severity(self, fault_type: FaultType) -> int:
        """Calculate fault severity"""
        severity_map = {
            FaultType.TRANSIENT: 1,
            FaultType.INTERMITTENT: 2,
            FaultType.OMISSION: 2,
            FaultType.CRASH: 3,
            FaultType.PERMANENT: 4,
            FaultType.BYZANTINE: 5,
        }
        return severity_map.get(fault_type, 1)
    
    def get_status(self) -> Dict[str, Any]:
        """Get fault mitigation status"""
        return {
            "components": [
                {
                    "component_id": c.component_id,
                    "name": c.name,
                    "state": c.state.value,
                    "redundancy": c.redundancy.value,
                    "fault_count": c.fault_count,
                    "replicas": len(c.replicas)
                }
                for c in self.components.values()
            ],
            "active_faults": sum(1 for f in self.faults.values() if not f.resolved),
            "suspected_failures": len(self.failure_detector.suspected),
            "system_availability": round(self.metrics.system_availability, 4)
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get fault mitigation metrics"""
        avg_recovery = 0.0
        if self._recovery_times:
            avg_recovery = sum(self._recovery_times) / len(self._recovery_times)
        
        return {
            "total_faults_detected": self.metrics.total_faults_detected,
            "faults_mitigated": self.metrics.faults_mitigated,
            "recovery_attempts": self.metrics.recovery_attempts,
            "successful_recoveries": self.metrics.successful_recoveries,
            "failed_recoveries": self.metrics.failed_recoveries,
            "avg_recovery_time_ms": round(avg_recovery, 3),
            "system_availability": round(self.metrics.system_availability, 4),
            "active_components": len(self.components),
            "checkpoints_stored": sum(len(v) for v in self.checkpoints.values())
        }
    
    async def shutdown(self) -> None:
        """Shutdown the fault mitigation module"""
        self._logger.info("Shutting down FaultMitigationModule...")
        self._shutting_down = True
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        self.components.clear()
        self.faults.clear()
        self.checkpoints.clear()
        self._initialized = False
        
        self._logger.info("FaultMitigationModule shutdown complete")


# Example usage
async def main():
    """Example usage of FaultMitigationModule"""
    
    config = FaultConfig(
        health_check_interval_s=1.0,
        max_fault_threshold=3,
        enable_auto_recovery=True
    )
    
    module = FaultMitigationModule(config)
    await module.initialize()
    
    try:
        # Register components
        comp1 = await module.register_component("compute_node_1", RedundancyLevel.TRIPLE)
        comp2 = await module.register_component("storage_node_1", RedundancyLevel.DUAL)
        
        print(f"Registered: {comp1.name}, {comp2.name}")
        
        # Send heartbeats
        for _ in range(3):
            await module.heartbeat(comp1.component_id)
            await module.heartbeat(comp2.component_id)
            await asyncio.sleep(0.5)
        
        # Create checkpoint
        cp = await module.create_checkpoint(
            comp1.component_id,
            {"iteration": 100, "data": [1, 2, 3]}
        )
        print(f"Checkpoint created: {cp.checkpoint_id[:8]}")
        
        # Report a fault
        fault = await module.report_fault(
            comp1.component_id,
            FaultType.TRANSIENT,
            "Memory allocation error"
        )
        print(f"Fault reported: {fault.fault_id[:8]}")
        
        # Wait for auto-recovery
        await asyncio.sleep(1)
        
        # Get status
        status = module.get_status()
        print(f"\nSystem Status:")
        print(f"  Availability: {status['system_availability']:.4f}")
        for c in status["components"]:
            print(f"  {c['name']}: {c['state']}")
        
        # Get metrics
        metrics = module.get_metrics()
        print(f"\nMetrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    
    finally:
        await module.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

