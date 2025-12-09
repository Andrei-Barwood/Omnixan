
# omnixan/carbon_based_quantum_cloud/auto_scaling_module.py

"""
Auto Scaling Module for Quantum Cloud Computing
Provides intelligent resource scaling based on workload metrics and predictive analytics.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import json
from pathlib import Path


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ScalingPolicy(Enum):
    """Enumeration of available scaling policies."""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    HYBRID = "hybrid"
    PREDICTIVE = "predictive"


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    CPU = "cpu"
    MEMORY = "memory"
    QUANTUM_QUBITS = "quantum_qubits"
    STORAGE = "storage"
    NETWORK = "network"


@dataclass
class ScalingThresholds:
    """Threshold configuration for auto-scaling decisions."""
    cpu_upper: float = 80.0  # Percentage
    cpu_lower: float = 30.0
    memory_upper: float = 75.0
    memory_lower: float = 25.0
    quantum_workload_upper: float = 85.0
    quantum_workload_lower: float = 20.0
    cooldown_period: int = 300  # Seconds

    def validate(self) -> bool:
        """Validate threshold configuration."""
        return (
            self.cpu_lower < self.cpu_upper and
            self.memory_lower < self.memory_upper and
            self.quantum_workload_lower < self.quantum_workload_upper
        )


@dataclass
class ResourceMetrics:
    """Container for current resource utilization metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    quantum_workload: float
    active_tasks: int
    queue_depth: int
    network_throughput: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'quantum_workload': self.quantum_workload,
            'active_tasks': self.active_tasks,
            'queue_depth': self.queue_depth,
            'network_throughput': self.network_throughput
        }


@dataclass
class ScalingAction:
    """Represents a scaling action to be executed."""
    action_type: str  # 'scale_up' or 'scale_down'
    resource_type: ResourceType
    amount: int
    reason: str
    timestamp: datetime
    estimated_cost: float = 0.0


@dataclass
class CostConfig:
    """Cost configuration for resource optimization."""
    cpu_cost_per_unit: float = 0.05  # Per hour
    memory_cost_per_gb: float = 0.01
    quantum_cost_per_qubit: float = 0.50
    storage_cost_per_gb: float = 0.001
    max_budget_per_hour: float = 100.0


class AutoScalingModule:
    """
    Advanced auto-scaling module for quantum cloud computing infrastructure.

    Features:
    - Horizontal and vertical scaling
    - Predictive scaling based on historical data
    - Cost optimization
    - Integration with load balancer and container orchestration
    - Async/await for non-blocking operations
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        monitoring_interval: float = 30.0,
        metrics_history_size: int = 1000
    ):
        """
        Initialize the Auto Scaling Module.

        Args:
            config_path: Path to configuration file
            monitoring_interval: Interval between metric checks (seconds)
            metrics_history_size: Number of historical metrics to retain
        """
        self.config_path = config_path
        self.monitoring_interval = monitoring_interval
        self.metrics_history: deque = deque(maxlen=metrics_history_size)

        # State management
        self.is_initialized: bool = False
        self.is_running: bool = False
        self.last_scaling_action: Optional[datetime] = None

        # Configuration
        self.scaling_policy: ScalingPolicy = ScalingPolicy.HYBRID
        self.thresholds: ScalingThresholds = ScalingThresholds()
        self.cost_config: CostConfig = CostConfig()

        # Resource state
        self.current_resources: Dict[ResourceType, int] = {
            ResourceType.CPU: 4,
            ResourceType.MEMORY: 16,  # GB
            ResourceType.QUANTUM_QUBITS: 10,
            ResourceType.STORAGE: 100,  # GB
        }

        # Integration references (to be set after initialization)
        self.load_balancer = None
        self.container_orchestrator = None

        # Async tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._scaling_lock = asyncio.Lock()

        logger.info("AutoScalingModule instantiated")

    async def initialize(self) -> bool:
        """
        Initialize the auto-scaling module.

        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info("Initializing AutoScalingModule...")

            # Load configuration if path provided
            if self.config_path and self.config_path.exists():
                await self._load_configuration()

            # Validate thresholds
            if not self.thresholds.validate():
                raise ValueError("Invalid threshold configuration")

            # Initialize monitoring
            self.is_initialized = True
            self.is_running = True

            # Start monitoring task
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())

            logger.info("AutoScalingModule initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            return False

    async def execute(self) -> Dict[str, Any]:
        """
        Execute main scaling logic.

        Returns:
            Dict containing execution results
        """
        if not self.is_initialized:
            raise RuntimeError("Module not initialized. Call initialize() first.")

        try:
            # Get current metrics
            metrics = await self.get_metrics()
            self.metrics_history.append(metrics)

            # Analyze metrics and determine if scaling is needed
            scaling_decision = await self._analyze_metrics(metrics)

            if scaling_decision:
                # Execute scaling action
                result = await self._execute_scaling_action(scaling_decision)
                return {
                    'status': 'scaling_executed',
                    'action': scaling_decision,
                    'result': result
                }

            return {
                'status': 'no_scaling_needed',
                'metrics': metrics.to_dict()
            }

        except Exception as e:
            logger.error(f"Execution error: {e}", exc_info=True)
            return {'status': 'error', 'message': str(e)}

    async def shutdown(self) -> bool:
        """
        Gracefully shutdown the auto-scaling module.

        Returns:
            bool: True if shutdown successful
        """
        try:
            logger.info("Shutting down AutoScalingModule...")

            self.is_running = False

            # Cancel monitoring task
            if self._monitoring_task and not self._monitoring_task.done():
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass

            # Save current state
            await self._save_state()

            logger.info("AutoScalingModule shutdown complete")
            return True

        except Exception as e:
            logger.error(f"Shutdown error: {e}", exc_info=True)
            return False

    def configure_scaling(
        self,
        policy: ScalingPolicy,
        thresholds: Optional[ScalingThresholds] = None
    ) -> bool:
        """
        Configure scaling policy and thresholds.

        Args:
            policy: Scaling policy to use
            thresholds: Optional custom thresholds

        Returns:
            bool: True if configuration successful
        """
        try:
            self.scaling_policy = policy

            if thresholds:
                if not thresholds.validate():
                    raise ValueError("Invalid thresholds")
                self.thresholds = thresholds

            logger.info(f"Scaling configured: policy={policy.value}")
            return True

        except Exception as e:
            logger.error(f"Configuration error: {e}")
            return False

    async def scale_up(
        self,
        resources: Dict[ResourceType, int]
    ) -> Dict[str, Any]:
        """
        Scale up resources.

        Args:
            resources: Dictionary of resources to scale up

        Returns:
            Dict containing scaling results
        """
        async with self._scaling_lock:
            try:
                logger.info(f"Scaling up: {resources}")

                # Validate budget
                estimated_cost = self._calculate_cost_increase(resources)
                if not self._within_budget(estimated_cost):
                    return {
                        'status': 'rejected',
                        'reason': 'budget_exceeded',
                        'estimated_cost': estimated_cost
                    }

                # Execute scale up
                results = {}
                for resource_type, amount in resources.items():
                    success = await self._scale_resource_up(resource_type, amount)
                    results[resource_type.value] = success

                    if success:
                        self.current_resources[resource_type] += amount

                # Update integrations
                await self._notify_integrations('scale_up', resources)

                self.last_scaling_action = datetime.now()

                return {
                    'status': 'success',
                    'results': results,
                    'new_resources': self.current_resources.copy()
                }

            except Exception as e:
                logger.error(f"Scale up error: {e}", exc_info=True)
                return {'status': 'error', 'message': str(e)}

    async def scale_down(
        self,
        resources: Dict[ResourceType, int]
    ) -> Dict[str, Any]:
        """
        Scale down resources.

        Args:
            resources: Dictionary of resources to scale down

        Returns:
            Dict containing scaling results
        """
        async with self._scaling_lock:
            try:
                logger.info(f"Scaling down: {resources}")

                # Validate minimum resources
                if not self._validate_minimum_resources(resources):
                    return {
                        'status': 'rejected',
                        'reason': 'minimum_resources_required'
                    }

                # Execute scale down
                results = {}
                for resource_type, amount in resources.items():
                    success = await self._scale_resource_down(resource_type, amount)
                    results[resource_type.value] = success

                    if success:
                        self.current_resources[resource_type] -= amount

                # Update integrations
                await self._notify_integrations('scale_down', resources)

                self.last_scaling_action = datetime.now()

                return {
                    'status': 'success',
                    'results': results,
                    'new_resources': self.current_resources.copy(),
                    'cost_savings': self._calculate_cost_increase(resources)
                }

            except Exception as e:
                logger.error(f"Scale down error: {e}", exc_info=True)
                return {'status': 'error', 'message': str(e)}

    async def get_metrics(self) -> ResourceMetrics:
        """
        Get current resource utilization metrics.

        Returns:
            ResourceMetrics: Current metrics
        """
        try:
            # Simulate metric collection (in production, integrate with actual monitoring)
            import random

            metrics = ResourceMetrics(
                timestamp=datetime.now(),
                cpu_usage=random.uniform(20.0, 90.0),
                memory_usage=random.uniform(30.0, 85.0),
                quantum_workload=random.uniform(10.0, 95.0),
                active_tasks=random.randint(1, 50),
                queue_depth=random.randint(0, 100),
                network_throughput=random.uniform(100.0, 1000.0)
            )

            return metrics

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            raise

    def get_scaling_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get scaling recommendations based on historical data.

        Returns:
            List of scaling recommendations
        """
        recommendations = []

        if len(self.metrics_history) < 10:
            return recommendations

        # Analyze trends
        recent_metrics = list(self.metrics_history)[-10:]

        # CPU trend
        cpu_trend = self._calculate_trend([m.cpu_usage for m in recent_metrics])
        if cpu_trend > 5.0:  # Increasing trend
            recommendations.append({
                'resource': ResourceType.CPU.value,
                'action': 'scale_up',
                'reason': 'increasing_cpu_trend',
                'confidence': min(cpu_trend / 10.0, 1.0)
            })

        # Memory trend
        memory_trend = self._calculate_trend([m.memory_usage for m in recent_metrics])
        if memory_trend > 5.0:
            recommendations.append({
                'resource': ResourceType.MEMORY.value,
                'action': 'scale_up',
                'reason': 'increasing_memory_trend',
                'confidence': min(memory_trend / 10.0, 1.0)
            })

        return recommendations

    # Private methods

    async def _monitoring_loop(self):
        """Continuous monitoring loop."""
        logger.info("Monitoring loop started")

        while self.is_running:
            try:
                await self.execute()
                await asyncio.sleep(self.monitoring_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}", exc_info=True)
                await asyncio.sleep(self.monitoring_interval)

        logger.info("Monitoring loop stopped")

    async def _analyze_metrics(
        self,
        metrics: ResourceMetrics
    ) -> Optional[ScalingAction]:
        """Analyze metrics and determine if scaling is needed."""

        # Check cooldown period
        if not self._is_cooldown_expired():
            return None

        # Check CPU
        if metrics.cpu_usage > self.thresholds.cpu_upper:
            return ScalingAction(
                action_type='scale_up',
                resource_type=ResourceType.CPU,
                amount=self._calculate_scale_amount(metrics.cpu_usage, 'up'),
                reason='cpu_threshold_exceeded',
                timestamp=datetime.now()
            )
        elif metrics.cpu_usage < self.thresholds.cpu_lower:
            return ScalingAction(
                action_type='scale_down',
                resource_type=ResourceType.CPU,
                amount=self._calculate_scale_amount(metrics.cpu_usage, 'down'),
                reason='cpu_underutilized',
                timestamp=datetime.now()
            )

        # Check memory
        if metrics.memory_usage > self.thresholds.memory_upper:
            return ScalingAction(
                action_type='scale_up',
                resource_type=ResourceType.MEMORY,
                amount=4,  # Scale memory in 4GB increments
                reason='memory_threshold_exceeded',
                timestamp=datetime.now()
            )

        # Check quantum workload
        if metrics.quantum_workload > self.thresholds.quantum_workload_upper:
            return ScalingAction(
                action_type='scale_up',
                resource_type=ResourceType.QUANTUM_QUBITS,
                amount=5,
                reason='quantum_workload_high',
                timestamp=datetime.now()
            )

        return None

    async def _execute_scaling_action(
        self,
        action: ScalingAction
    ) -> Dict[str, Any]:
        """Execute a scaling action."""

        resources = {action.resource_type: action.amount}

        if action.action_type == 'scale_up':
            return await self.scale_up(resources)
        else:
            return await self.scale_down(resources)

    async def _scale_resource_up(
        self,
        resource_type: ResourceType,
        amount: int
    ) -> bool:
        """Scale a specific resource up."""
        try:
            # Simulate scaling operation
            await asyncio.sleep(0.5)
            logger.info(f"Scaled up {resource_type.value} by {amount}")
            return True
        except Exception as e:
            logger.error(f"Failed to scale up {resource_type.value}: {e}")
            return False

    async def _scale_resource_down(
        self,
        resource_type: ResourceType,
        amount: int
    ) -> bool:
        """Scale a specific resource down."""
        try:
            # Simulate scaling operation
            await asyncio.sleep(0.5)
            logger.info(f"Scaled down {resource_type.value} by {amount}")
            return True
        except Exception as e:
            logger.error(f"Failed to scale down {resource_type.value}: {e}")
            return False

    async def _notify_integrations(
        self,
        action: str,
        resources: Dict[ResourceType, int]
    ):
        """Notify integrated modules of scaling events."""
        try:
            if self.load_balancer:
                await self.load_balancer.update_resource_pool(resources)

            if self.container_orchestrator:
                await self.container_orchestrator.adjust_replicas(action, resources)

        except Exception as e:
            logger.error(f"Failed to notify integrations: {e}")

    def _calculate_cost_increase(
        self,
        resources: Dict[ResourceType, int]
    ) -> float:
        """Calculate cost increase for scaling action."""
        cost = 0.0

        for resource_type, amount in resources.items():
            if resource_type == ResourceType.CPU:
                cost += amount * self.cost_config.cpu_cost_per_unit
            elif resource_type == ResourceType.MEMORY:
                cost += amount * self.cost_config.memory_cost_per_gb
            elif resource_type == ResourceType.QUANTUM_QUBITS:
                cost += amount * self.cost_config.quantum_cost_per_qubit

        return cost

    def _within_budget(self, additional_cost: float) -> bool:
        """Check if additional cost is within budget."""
        current_cost = sum(
            self.current_resources[ResourceType.CPU] * self.cost_config.cpu_cost_per_unit +
            self.current_resources[ResourceType.MEMORY] * self.cost_config.memory_cost_per_gb +
            self.current_resources[ResourceType.QUANTUM_QUBITS] * self.cost_config.quantum_cost_per_qubit
        )

        return (current_cost + additional_cost) <= self.cost_config.max_budget_per_hour

    def _validate_minimum_resources(
        self,
        resources_to_remove: Dict[ResourceType, int]
    ) -> bool:
        """Validate that minimum resources will be maintained."""
        min_resources = {
            ResourceType.CPU: 2,
            ResourceType.MEMORY: 4,
            ResourceType.QUANTUM_QUBITS: 5
        }

        for resource_type, amount in resources_to_remove.items():
            if resource_type in min_resources:
                new_amount = self.current_resources[resource_type] - amount
                if new_amount < min_resources[resource_type]:
                    return False

        return True

    def _is_cooldown_expired(self) -> bool:
        """Check if cooldown period has expired."""
        if not self.last_scaling_action:
            return True

        elapsed = (datetime.now() - self.last_scaling_action).total_seconds()
        return elapsed >= self.thresholds.cooldown_period

    def _calculate_scale_amount(self, usage: float, direction: str) -> int:
        """Calculate appropriate scaling amount based on usage."""
        if direction == 'up':
            if usage > 90:
                return 4
            elif usage > 80:
                return 2
            else:
                return 1
        else:  # scale down
            if usage < 20:
                return 2
            else:
                return 1

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend from list of values (simple linear regression)."""
        if len(values) < 2:
            return 0.0

        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        slope = numerator / denominator
        return slope

    async def _load_configuration(self):
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)

            # Load thresholds
            if 'thresholds' in config:
                self.thresholds = ScalingThresholds(**config['thresholds'])

            # Load cost config
            if 'cost_config' in config:
                self.cost_config = CostConfig(**config['cost_config'])

            logger.info(f"Configuration loaded from {self.config_path}")

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")

    async def _save_state(self):
        """Save current state to file."""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'current_resources': {k.value: v for k, v in self.current_resources.items()},
                'last_scaling_action': self.last_scaling_action.isoformat() if self.last_scaling_action else None,
                'metrics_count': len(self.metrics_history)
            }

            # In production, save to persistent storage
            logger.info(f"State saved: {state}")

        except Exception as e:
            logger.error(f"Failed to save state: {e}")


# Integration helper classes

class LoadBalancingIntegration:
    """Mock load balancer integration."""

    async def update_resource_pool(self, resources: Dict[ResourceType, int]):
        """Update load balancer resource pool."""
        logger.info(f"Load balancer updated with resources: {resources}")
        await asyncio.sleep(0.1)


class ContainerOrchestratorIntegration:
    """Mock container orchestrator integration."""

    async def adjust_replicas(self, action: str, resources: Dict[ResourceType, int]):
        """Adjust container replicas."""
        logger.info(f"Container orchestrator: {action} - {resources}")
        await asyncio.sleep(0.1)


# Example usage
async def main():
    """Example usage of AutoScalingModule."""

    # Create module instance
    scaling_module = AutoScalingModule(
        monitoring_interval=10.0,
        metrics_history_size=100
    )

    # Set up integrations
    scaling_module.load_balancer = LoadBalancingIntegration()
    scaling_module.container_orchestrator = ContainerOrchestratorIntegration()

    # Configure scaling policy
    custom_thresholds = ScalingThresholds(
        cpu_upper=75.0,
        cpu_lower=25.0,
        memory_upper=70.0,
        memory_lower=20.0
    )
    scaling_module.configure_scaling(ScalingPolicy.HYBRID, custom_thresholds)

    # Initialize
    if await scaling_module.initialize():
        print("Auto scaling module initialized successfully")

        # Run for 60 seconds
        await asyncio.sleep(60)

        # Get recommendations
        recommendations = scaling_module.get_scaling_recommendations()
        print(f"Scaling recommendations: {recommendations}")

        # Shutdown
        await scaling_module.shutdown()
    else:
        print("Failed to initialize auto scaling module")


if __name__ == "__main__":
    asyncio.run(main())
