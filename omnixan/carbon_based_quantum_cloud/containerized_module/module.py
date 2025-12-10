"""
OMNIXAN - Carbon-Based Quantum Cloud - Containerized Module
Production-ready container orchestration for quantum workloads
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Literal
from dataclasses import dataclass, field

import aiodocker
from aiodocker.exceptions import DockerError
from kubernetes import client as k8s_client
from kubernetes import config as k8s_config
from kubernetes.client.rest import ApiException
from pydantic import BaseModel, Field, validator, ConfigDict
from pydantic.v1 import SecretStr


# ==================== Custom Exceptions ====================

class ContainerError(Exception):
    """Base exception for container operations"""
    pass


class ImageNotFoundError(ContainerError):
    """Raised when container image is not found"""
    pass


class ResourceLimitError(ContainerError):
    """Raised when resource limits are exceeded"""
    pass


class OrchestrationError(ContainerError):
    """Raised when orchestration operations fail"""
    pass


# ==================== Enums ====================

class ContainerAction(str, Enum):
    """Container management actions"""
    START = "start"
    STOP = "stop"
    RESTART = "restart"
    PAUSE = "pause"
    UNPAUSE = "unpause"
    REMOVE = "remove"


class ContainerState(str, Enum):
    """Container states"""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    EXITED = "exited"
    DEAD = "dead"
    RESTARTING = "restarting"


class OrchestratorType(str, Enum):
    """Container orchestration platform"""
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    DOCKER_SWARM = "docker_swarm"


class QuantumRuntime(str, Enum):
    """Supported quantum computing runtimes"""
    QISKIT = "qiskit"
    CIRQ = "cirq"
    PENNYLANE = "pennylane"
    BRAKET = "braket"
    QSHARP = "qsharp"


# ==================== Pydantic Models ====================

class ResourceLimits(BaseModel):
    """Container resource limits"""
    model_config = ConfigDict(frozen=False)
    
    cpu_limit: float = Field(default=1.0, ge=0.1, le=32.0)
    memory_limit: str = Field(default="1g")
    gpu_count: int = Field(default=0, ge=0, le=8)
    storage_limit: str = Field(default="10g")
    
    @validator("memory_limit", "storage_limit")
    def validate_size_format(cls, v: str) -> str:
        """Validate size format (e.g., 1g, 512m)"""
        if not v[-1] in ["k", "m", "g", "t"]:
            raise ValueError("Size must end with k, m, g, or t")
        try:
            int(v[:-1])
        except ValueError:
            raise ValueError("Invalid size format")
        return v


class NetworkConfig(BaseModel):
    """Network configuration for containers"""
    model_config = ConfigDict(frozen=False)
    
    network_name: str = Field(default="quantum-net")
    isolation: bool = Field(default=True)
    expose_ports: list[int] = Field(default_factory=list)
    dns_servers: list[str] = Field(default_factory=lambda: ["8.8.8.8", "8.8.4.4"])
    enable_ipv6: bool = Field(default=False)


class SecurityPolicy(BaseModel):
    """Security policies for containers"""
    model_config = ConfigDict(frozen=False)
    
    read_only_root: bool = Field(default=True)
    privileged: bool = Field(default=False)
    no_new_privileges: bool = Field(default=True)
    seccomp_profile: Optional[str] = Field(default="runtime/default")
    apparmor_profile: Optional[str] = Field(default=None)
    capabilities_add: list[str] = Field(default_factory=list)
    capabilities_drop: list[str] = Field(default_factory=lambda: ["ALL"])


class HealthCheck(BaseModel):
    """Health check configuration"""
    model_config = ConfigDict(frozen=False)
    
    enabled: bool = Field(default=True)
    endpoint: str = Field(default="/health")
    interval_seconds: int = Field(default=30, ge=5, le=300)
    timeout_seconds: int = Field(default=5, ge=1, le=60)
    retries: int = Field(default=3, ge=1, le=10)
    start_period_seconds: int = Field(default=60, ge=0)


class ContainerConfig(BaseModel):
    """Complete container configuration"""
    model_config = ConfigDict(frozen=False)
    
    name: str
    image: str
    quantum_runtime: Optional[QuantumRuntime] = None
    command: Optional[list[str]] = None
    environment: dict[str, str] = Field(default_factory=dict)
    resources: ResourceLimits = Field(default_factory=ResourceLimits)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    security: SecurityPolicy = Field(default_factory=SecurityPolicy)
    health_check: HealthCheck = Field(default_factory=HealthCheck)
    volumes: dict[str, dict[str, str]] = Field(default_factory=dict)
    labels: dict[str, str] = Field(default_factory=dict)
    restart_policy: Literal["no", "always", "on-failure", "unless-stopped"] = "on-failure"
    tenant_id: Optional[str] = None


class ContainerStatus(BaseModel):
    """Container status information"""
    model_config = ConfigDict(frozen=False)
    
    container_id: str
    name: str
    state: ContainerState
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    exit_code: Optional[int] = None
    health_status: Optional[str] = None
    resource_usage: dict[str, Any] = Field(default_factory=dict)
    network_stats: dict[str, Any] = Field(default_factory=dict)


class ContainerInstance(BaseModel):
    """Deployed container instance"""
    model_config = ConfigDict(frozen=False)
    
    container_id: str
    name: str
    image: str
    ip_address: Optional[str] = None
    ports: dict[int, int] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    orchestrator: OrchestratorType


class OperationResult(BaseModel):
    """Result of container operation"""
    model_config = ConfigDict(frozen=False)
    
    success: bool
    message: str
    container_id: Optional[str] = None
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ScalingResult(BaseModel):
    """Result of scaling operation"""
    model_config = ConfigDict(frozen=False)
    
    service_id: str
    previous_replicas: int
    current_replicas: int
    target_replicas: int
    success: bool
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


@dataclass
class ContainerMetrics:
    """Container performance metrics"""
    container_id: str
    cpu_percent: float
    memory_usage_mb: float
    memory_limit_mb: float
    network_rx_bytes: int
    network_tx_bytes: int
    block_read_bytes: int
    block_write_bytes: int
    pids: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ==================== Main Module ====================

class ContainerizedModule:
    """
    Production-ready containerized module for OMNIXAN quantum cloud platform.
    
    Supports Docker and Kubernetes orchestration with quantum runtime integration.
    """
    
    # Quantum runtime base images [web:5][web:12]
    QUANTUM_IMAGES: dict[QuantumRuntime, str] = {
        QuantumRuntime.QISKIT: "qiskit/qiskit:latest",
        QuantumRuntime.CIRQ: "quantumai/cirq:latest",
        QuantumRuntime.PENNYLANE: "pennylane/pennylane:latest",
        QuantumRuntime.BRAKET: "amazon/braket-sdk:latest",
        QuantumRuntime.QSHARP: "mcr.microsoft.com/quantum/qsharp-runtime:latest",
    }
    
    def __init__(
        self,
        orchestrator: OrchestratorType = OrchestratorType.DOCKER,
        docker_url: str = "unix://var/run/docker.sock",
        k8s_config_file: Optional[Path] = None,
        registry_url: Optional[str] = None,
        registry_auth: Optional[dict[str, str]] = None,
        enable_monitoring: bool = True,
        log_level: str = "INFO"
    ):
        """
        Initialize containerized module.
        
        Args:
            orchestrator: Container orchestration platform
            docker_url: Docker daemon URL
            k8s_config_file: Kubernetes config file path
            registry_url: Container registry URL
            registry_auth: Registry authentication credentials
            enable_monitoring: Enable metrics collection
            log_level: Logging level
        """
        self.orchestrator = orchestrator
        self.docker_url = docker_url
        self.k8s_config_file = k8s_config_file
        self.registry_url = registry_url
        self.registry_auth = registry_auth
        self.enable_monitoring = enable_monitoring
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level))
        
        # Clients
        self.docker: Optional[aiodocker.Docker] = None
        self.k8s_core_v1: Optional[k8s_client.CoreV1Api] = None
        self.k8s_apps_v1: Optional[k8s_client.AppsV1Api] = None
        
        # Container tracking
        self.containers: dict[str, ContainerInstance] = {}
        self.metrics_buffer: dict[str, list[ContainerMetrics]] = {}
        
        # Integration modules (to be set externally)
        self.auto_scaling_module: Optional[Any] = None
        self.load_balancing_module: Optional[Any] = None
        
        self._initialized = False
        self._shutdown = False
    
    async def initialize(self) -> None:
        """Initialize container orchestration clients"""
        if self._initialized:
            self.logger.warning("Module already initialized")
            return
        
        try:
            self.logger.info(f"Initializing {self.orchestrator.value} orchestrator")
            
            if self.orchestrator == OrchestratorType.DOCKER:
                await self._initialize_docker()
            elif self.orchestrator == OrchestratorType.KUBERNETES:
                await self._initialize_kubernetes()
            elif self.orchestrator == OrchestratorType.DOCKER_SWARM:
                await self._initialize_docker_swarm()
            
            self._initialized = True
            self.logger.info("Containerized module initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}")
            raise ContainerError(f"Initialization failed: {e}") from e
    
    async def _initialize_docker(self) -> None:
        """Initialize Docker client"""
        self.docker = aiodocker.Docker(url=self.docker_url)
        
        # Verify connection
        try:
            await self.docker.version()
            self.logger.info("Docker client connected")
        except Exception as e:
            raise ContainerError(f"Docker connection failed: {e}") from e
    
    async def _initialize_kubernetes(self) -> None:
        """Initialize Kubernetes client"""
        try:
            if self.k8s_config_file:
                k8s_config.load_kube_config(config_file=str(self.k8s_config_file))
            else:
                k8s_config.load_incluster_config()
            
            self.k8s_core_v1 = k8s_client.CoreV1Api()
            self.k8s_apps_v1 = k8s_client.AppsV1Api()
            
            # Verify connection
            await asyncio.to_thread(self.k8s_core_v1.list_namespace)
            self.logger.info("Kubernetes client connected")
            
        except Exception as e:
            raise ContainerError(f"Kubernetes connection failed: {e}") from e
    
    async def _initialize_docker_swarm(self) -> None:
        """Initialize Docker Swarm"""
        await self._initialize_docker()
        
        # Verify swarm mode
        try:
            info = await self.docker.system.info()
            if not info.get("Swarm", {}).get("NodeID"):
                raise ContainerError("Docker Swarm not initialized")
            self.logger.info("Docker Swarm connected")
        except Exception as e:
            raise ContainerError(f"Docker Swarm connection failed: {e}") from e
    
    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Execute container operation.
        
        Args:
            params: Operation parameters
            
        Returns:
            Operation results
        """
        if not self._initialized:
            raise ContainerError("Module not initialized")
        
        operation = params.get("operation")
        
        if operation == "deploy":
            result = await self.deploy_container(
                params["image"],
                ContainerConfig(**params["config"])
            )
            return result.model_dump()
        
        elif operation == "manage":
            result = await self.manage_containers(
                ContainerAction(params["action"]),
                params["container_id"]
            )
            return result.model_dump()
        
        elif operation == "scale":
            result = await self.scale_containers(
                params["service_id"],
                params["replicas"]
            )
            return result.model_dump()
        
        elif operation == "status":
            result = await self.get_status(params["container_id"])
            return result.model_dump()
        
        elif operation == "logs":
            logs = await self.get_logs(
                params["container_id"],
                params.get("tail", 100)
            )
            return {"logs": logs}
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def deploy_container(
        self,
        image: str,
        config: ContainerConfig
    ) -> ContainerInstance:
        """
        Deploy a new container.
        
        Args:
            image: Container image
            config: Container configuration
            
        Returns:
            Deployed container instance
        """
        if not self._initialized:
            raise ContainerError("Module not initialized")
        
        try:
            # Use quantum runtime image if specified
            if config.quantum_runtime:
                image = self.QUANTUM_IMAGES[config.quantum_runtime]
                self.logger.info(f"Using quantum runtime image: {image}")
            
            # Pull image
            await self._pull_image(image)
            
            # Deploy based on orchestrator
            if self.orchestrator == OrchestratorType.DOCKER:
                instance = await self._deploy_docker_container(image, config)
            elif self.orchestrator == OrchestratorType.KUBERNETES:
                instance = await self._deploy_k8s_pod(image, config)
            elif self.orchestrator == OrchestratorType.DOCKER_SWARM:
                instance = await self._deploy_swarm_service(image, config)
            else:
                raise ContainerError(f"Unsupported orchestrator: {self.orchestrator}")
            
            # Track container
            self.containers[instance.container_id] = instance
            
            self.logger.info(f"Container deployed: {instance.container_id}")
            return instance
            
        except DockerError as e:
            if "404" in str(e):
                raise ImageNotFoundError(f"Image not found: {image}") from e
            raise ContainerError(f"Deployment failed: {e}") from e
        except Exception as e:
            self.logger.error(f"Deployment error: {e}")
            raise ContainerError(f"Deployment failed: {e}") from e
    
    async def _pull_image(self, image: str) -> None:
        """Pull container image"""
        if self.orchestrator == OrchestratorType.DOCKER:
            try:
                await self.docker.images.pull(image)
                self.logger.debug(f"Pulled image: {image}")
            except DockerError as e:
                raise ImageNotFoundError(f"Failed to pull image {image}: {e}") from e
    
    async def _deploy_docker_container(
        self,
        image: str,
        config: ContainerConfig
    ) -> ContainerInstance:
        """Deploy Docker container"""
        # Build container configuration
        container_config = {
            "Image": image,
            "name": config.name,
            "Env": [f"{k}={v}" for k, v in config.environment.items()],
            "Labels": {**config.labels, "tenant_id": config.tenant_id or "default"},
            "HostConfig": {
                "CpuQuota": int(config.resources.cpu_limit * 100000),
                "Memory": self._parse_size(config.resources.memory_limit),
                "RestartPolicy": {"Name": config.restart_policy},
                "ReadonlyRootfs": config.security.read_only_root,
                "Privileged": config.security.privileged,
                "CapAdd": config.security.capabilities_add,
                "CapDrop": config.security.capabilities_drop,
                "Binds": [
                    f"{src}:{target['bind']}:{target.get('mode', 'rw')}"
                    for src, target in config.volumes.items()
                ],
            },
        }
        
        if config.command:
            container_config["Cmd"] = config.command
        
        # Add GPU support if needed
        if config.resources.gpu_count > 0:
            container_config["HostConfig"]["DeviceRequests"] = [{
                "Count": config.resources.gpu_count,
                "Capabilities": [["gpu"]],
            }]
        
        # Create and start container
        container = await self.docker.containers.create(container_config)
        await container.start()
        
        # Get container info
        info = await container.show()
        
        return ContainerInstance(
            container_id=info["Id"][:12],
            name=config.name,
            image=image,
            ip_address=info["NetworkSettings"]["IPAddress"],
            ports={},  # Parse from NetworkSettings if needed
            orchestrator=OrchestratorType.DOCKER
        )
    
    async def _deploy_k8s_pod(
        self,
        image: str,
        config: ContainerConfig
    ) -> ContainerInstance:
        """Deploy Kubernetes pod"""
        # Build pod specification
        pod_manifest = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": config.name,
                "labels": {**config.labels, "tenant_id": config.tenant_id or "default"},
            },
            "spec": {
                "containers": [{
                    "name": config.name,
                    "image": image,
                    "env": [{"name": k, "value": v} for k, v in config.environment.items()],
                    "resources": {
                        "limits": {
                            "cpu": str(config.resources.cpu_limit),
                            "memory": config.resources.memory_limit,
                        },
                        "requests": {
                            "cpu": str(config.resources.cpu_limit / 2),
                            "memory": config.resources.memory_limit,
                        },
                    },
                    "securityContext": {
                        "readOnlyRootFilesystem": config.security.read_only_root,
                        "privileged": config.security.privileged,
                        "allowPrivilegeEscalation": not config.security.no_new_privileges,
                    },
                }],
                "restartPolicy": "Always" if config.restart_policy == "always" else "OnFailure",
            },
        }
        
        # Add command if specified
        if config.command:
            pod_manifest["spec"]["containers"][0]["command"] = config.command
        
        # Create pod
        pod = await asyncio.to_thread(
            self.k8s_core_v1.create_namespaced_pod,
            namespace="default",
            body=pod_manifest
        )
        
        return ContainerInstance(
            container_id=pod.metadata.uid,
            name=config.name,
            image=image,
            ip_address=pod.status.pod_ip,
            ports={},
            orchestrator=OrchestratorType.KUBERNETES
        )
    
    async def _deploy_swarm_service(
        self,
        image: str,
        config: ContainerConfig
    ) -> ContainerInstance:
        """Deploy Docker Swarm service"""
        service_spec = {
            "Name": config.name,
            "TaskTemplate": {
                "ContainerSpec": {
                    "Image": image,
                    "Env": [f"{k}={v}" for k, v in config.environment.items()],
                    "Labels": config.labels,
                },
                "Resources": {
                    "Limits": {
                        "NanoCPUs": int(config.resources.cpu_limit * 1e9),
                        "MemoryBytes": self._parse_size(config.resources.memory_limit),
                    },
                },
                "RestartPolicy": {"Condition": config.restart_policy},
            },
            "Mode": {"Replicated": {"Replicas": 1}},
        }
        
        # Create service
        response = await self.docker.services.create(**service_spec)
        service_id = response["ID"]
        
        return ContainerInstance(
            container_id=service_id,
            name=config.name,
            image=image,
            orchestrator=OrchestratorType.DOCKER_SWARM
        )
    
    async def manage_containers(
        self,
        action: ContainerAction,
        container_id: str
    ) -> OperationResult:
        """
        Manage container lifecycle.
        
        Args:
            action: Management action
            container_id: Container identifier
            
        Returns:
            Operation result
        """
        if not self._initialized:
            raise ContainerError("Module not initialized")
        
        try:
            if self.orchestrator == OrchestratorType.DOCKER:
                result = await self._manage_docker_container(action, container_id)
            elif self.orchestrator == OrchestratorType.KUBERNETES:
                result = await self._manage_k8s_pod(action, container_id)
            else:
                result = await self._manage_swarm_service(action, container_id)
            
            self.logger.info(f"Container {container_id} {action.value} completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Management error: {e}")
            return OperationResult(
                success=False,
                message=f"Failed to {action.value} container: {e}",
                container_id=container_id
            )
    
    async def _manage_docker_container(
        self,
        action: ContainerAction,
        container_id: str
    ) -> OperationResult:
        """Manage Docker container"""
        try:
            container = await self.docker.containers.get(container_id)
            
            if action == ContainerAction.START:
                await container.start()
            elif action == ContainerAction.STOP:
                await container.stop()
            elif action == ContainerAction.RESTART:
                await container.restart()
            elif action == ContainerAction.PAUSE:
                await container.pause()
            elif action == ContainerAction.UNPAUSE:
                await container.unpause()
            elif action == ContainerAction.REMOVE:
                await container.delete(force=True)
                self.containers.pop(container_id, None)
            
            return OperationResult(
                success=True,
                message=f"Container {action.value} successful",
                container_id=container_id
            )
            
        except DockerError as e:
            raise ContainerError(f"Docker operation failed: {e}") from e
    
    async def _manage_k8s_pod(
        self,
        action: ContainerAction,
        pod_name: str
    ) -> OperationResult:
        """Manage Kubernetes pod"""
        try:
            if action == ContainerAction.REMOVE:
                await asyncio.to_thread(
                    self.k8s_core_v1.delete_namespaced_pod,
                    name=pod_name,
                    namespace="default"
                )
                self.containers.pop(pod_name, None)
            else:
                # K8s doesn't support pause/unpause directly
                raise ContainerError(f"Action {action.value} not supported for Kubernetes")
            
            return OperationResult(
                success=True,
                message=f"Pod {action.value} successful",
                container_id=pod_name
            )
            
        except ApiException as e:
            raise ContainerError(f"Kubernetes operation failed: {e}") from e
    
    async def _manage_swarm_service(
        self,
        action: ContainerAction,
        service_id: str
    ) -> OperationResult:
        """Manage Docker Swarm service"""
        try:
            service = await self.docker.services.inspect(service_id)
            
            if action == ContainerAction.REMOVE:
                await self.docker.services.delete(service_id)
                self.containers.pop(service_id, None)
            else:
                raise ContainerError(f"Action {action.value} not fully supported for Swarm")
            
            return OperationResult(
                success=True,
                message=f"Service {action.value} successful",
                container_id=service_id
            )
            
        except Exception as e:
            raise ContainerError(f"Swarm operation failed: {e}") from e
    
    async def get_status(self, container_id: str) -> ContainerStatus:
        """
        Get container status.
        
        Args:
            container_id: Container identifier
            
        Returns:
            Container status
        """
        if not self._initialized:
            raise ContainerError("Module not initialized")
        
        try:
            if self.orchestrator == OrchestratorType.DOCKER:
                return await self._get_docker_status(container_id)
            elif self.orchestrator == OrchestratorType.KUBERNETES:
                return await self._get_k8s_status(container_id)
            else:
                return await self._get_swarm_status(container_id)
                
        except Exception as e:
            self.logger.error(f"Status check error: {e}")
            raise ContainerError(f"Failed to get status: {e}") from e
    
    async def _get_docker_status(self, container_id: str) -> ContainerStatus:
        """Get Docker container status"""
        container = await self.docker.containers.get(container_id)
        info = await container.show()
        
        state_map = {
            "created": ContainerState.CREATED,
            "running": ContainerState.RUNNING,
            "paused": ContainerState.PAUSED,
            "restarting": ContainerState.RESTARTING,
            "removing": ContainerState.STOPPED,
            "exited": ContainerState.EXITED,
            "dead": ContainerState.DEAD,
        }
        
        state_info = info["State"]
        state = state_map.get(state_info["Status"], ContainerState.STOPPED)
        
        # Get resource usage if container is running
        resource_usage = {}
        if state == ContainerState.RUNNING and self.enable_monitoring:
            try:
                stats = await container.stats(stream=False)
                resource_usage = self._parse_docker_stats(stats)
            except Exception:
                pass
        
        return ContainerStatus(
            container_id=container_id,
            name=info["Name"].lstrip("/"),
            state=state,
            started_at=datetime.fromisoformat(state_info["StartedAt"].rstrip("Z")) if state_info.get("StartedAt") else None,
            finished_at=datetime.fromisoformat(state_info["FinishedAt"].rstrip("Z")) if state_info.get("FinishedAt") else None,
            exit_code=state_info.get("ExitCode"),
            health_status=state_info.get("Health", {}).get("Status"),
            resource_usage=resource_usage
        )
    
    async def _get_k8s_status(self, pod_name: str) -> ContainerStatus:
        """Get Kubernetes pod status"""
        pod = await asyncio.to_thread(
            self.k8s_core_v1.read_namespaced_pod,
            name=pod_name,
            namespace="default"
        )
        
        phase_map = {
            "Pending": ContainerState.CREATED,
            "Running": ContainerState.RUNNING,
            "Succeeded": ContainerState.EXITED,
            "Failed": ContainerState.EXITED,
            "Unknown": ContainerState.STOPPED,
        }
        
        state = phase_map.get(pod.status.phase, ContainerState.STOPPED)
        
        return ContainerStatus(
            container_id=pod.metadata.uid,
            name=pod_name,
            state=state,
            started_at=pod.status.start_time,
            resource_usage={}
        )
    
    async def _get_swarm_status(self, service_id: str) -> ContainerStatus:
        """Get Docker Swarm service status"""
        service = await self.docker.services.inspect(service_id)
        
        return ContainerStatus(
            container_id=service_id,
            name=service["Spec"]["Name"],
            state=ContainerState.RUNNING,  # Simplified
            resource_usage={}
        )
    
    async def get_logs(
        self,
        container_id: str,
        tail: int = 100
    ) -> list[str]:
        """
        Get container logs.
        
        Args:
            container_id: Container identifier
            tail: Number of log lines to retrieve
            
        Returns:
            List of log lines
        """
        if not self._initialized:
            raise ContainerError("Module not initialized")
        
        try:
            if self.orchestrator == OrchestratorType.DOCKER:
                return await self._get_docker_logs(container_id, tail)
            elif self.orchestrator == OrchestratorType.KUBERNETES:
                return await self._get_k8s_logs(container_id, tail)
            else:
                return await self._get_swarm_logs(container_id, tail)
                
        except Exception as e:
            self.logger.error(f"Log retrieval error: {e}")
            raise ContainerError(f"Failed to get logs: {e}") from e
    
    async def _get_docker_logs(self, container_id: str, tail: int) -> list[str]:
        """Get Docker container logs"""
        container = await self.docker.containers.get(container_id)
        logs = await container.log(stdout=True, stderr=True, tail=tail)
        return [line.decode("utf-8", errors="ignore").strip() for line in logs]
    
    async def _get_k8s_logs(self, pod_name: str, tail: int) -> list[str]:
        """Get Kubernetes pod logs"""
        logs = await asyncio.to_thread(
            self.k8s_core_v1.read_namespaced_pod_log,
            name=pod_name,
            namespace="default",
            tail_lines=tail
        )
        return logs.split("\n")
    
    async def _get_swarm_logs(self, service_id: str, tail: int) -> list[str]:
        """Get Docker Swarm service logs"""
        # Swarm logs require accessing task containers
        logs = []
        try:
            service = await self.docker.services.inspect(service_id)
            # Simplified - would need to get task IDs and their logs
            logs.append(f"Service {service['Spec']['Name']} logs")
        except Exception:
            pass
        return logs
    
    async def scale_containers(
        self,
        service_id: str,
        replicas: int
    ) -> ScalingResult:
        """
        Scale container service.
        
        Args:
            service_id: Service identifier
            replicas: Target number of replicas
            
        Returns:
            Scaling result
        """
        if not self._initialized:
            raise ContainerError("Module not initialized")
        
        if replicas < 0:
            raise ValueError("Replicas must be non-negative")
        
        try:
            if self.orchestrator == OrchestratorType.KUBERNETES:
                return await self._scale_k8s_deployment(service_id, replicas)
            elif self.orchestrator == OrchestratorType.DOCKER_SWARM:
                return await self._scale_swarm_service(service_id, replicas)
            else:
                raise ContainerError("Scaling not supported for Docker standalone")
                
        except Exception as e:
            self.logger.error(f"Scaling error: {e}")
            return ScalingResult(
                service_id=service_id,
                previous_replicas=0,
                current_replicas=0,
                target_replicas=replicas,
                success=False,
                message=f"Scaling failed: {e}"
            )
    
    async def _scale_k8s_deployment(
        self,
        deployment_name: str,
        replicas: int
    ) -> ScalingResult:
        """Scale Kubernetes deployment"""
        try:
            # Get current deployment
            deployment = await asyncio.to_thread(
                self.k8s_apps_v1.read_namespaced_deployment,
                name=deployment_name,
                namespace="default"
            )
            
            previous_replicas = deployment.spec.replicas
            
            # Update replicas
            deployment.spec.replicas = replicas
            await asyncio.to_thread(
                self.k8s_apps_v1.patch_namespaced_deployment,
                name=deployment_name,
                namespace="default",
                body=deployment
            )
            
            # Notify auto-scaling module if available
            if self.auto_scaling_module:
                await self.auto_scaling_module.notify_scaling_event(
                    service_id=deployment_name,
                    replicas=replicas
                )
            
            return ScalingResult(
                service_id=deployment_name,
                previous_replicas=previous_replicas,
                current_replicas=replicas,
                target_replicas=replicas,
                success=True,
                message="Deployment scaled successfully"
            )
            
        except ApiException as e:
            raise OrchestrationError(f"K8s scaling failed: {e}") from e
    
    async def _scale_swarm_service(
        self,
        service_id: str,
        replicas: int
    ) -> ScalingResult:
        """Scale Docker Swarm service"""
        try:
            service = await self.docker.services.inspect(service_id)
            previous_replicas = service["Spec"]["Mode"]["Replicated"]["Replicas"]
            
            # Update service
            service["Spec"]["Mode"]["Replicated"]["Replicas"] = replicas
            await self.docker.services.update(
                service_id,
                version=service["Version"]["Index"],
                spec=service["Spec"]
            )
            
            return ScalingResult(
                service_id=service_id,
                previous_replicas=previous_replicas,
                current_replicas=replicas,
                target_replicas=replicas,
                success=True,
                message="Service scaled successfully"
            )
            
        except Exception as e:
            raise OrchestrationError(f"Swarm scaling failed: {e}") from e
    
    async def collect_metrics(self, container_id: str) -> ContainerMetrics:
        """
        Collect container performance metrics.
        
        Args:
            container_id: Container identifier
            
        Returns:
            Container metrics
        """
        if not self.enable_monitoring:
            raise ContainerError("Monitoring not enabled")
        
        try:
            if self.orchestrator == OrchestratorType.DOCKER:
                container = await self.docker.containers.get(container_id)
                stats = await container.stats(stream=False)
                return self._parse_docker_stats_to_metrics(container_id, stats)
            else:
                # K8s and Swarm metrics would require metrics-server
                raise ContainerError("Metrics collection not implemented for this orchestrator")
                
        except Exception as e:
            self.logger.error(f"Metrics collection error: {e}")
            raise ContainerError(f"Failed to collect metrics: {e}") from e
    
    def _parse_docker_stats(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Parse Docker container stats"""
        try:
            cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - \
                       stats["precpu_stats"]["cpu_usage"]["total_usage"]
            system_delta = stats["cpu_stats"]["system_cpu_usage"] - \
                          stats["precpu_stats"]["system_cpu_usage"]
            cpu_percent = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0.0
            
            memory_usage = stats["memory_stats"]["usage"]
            memory_limit = stats["memory_stats"]["limit"]
            
            return {
                "cpu_percent": round(cpu_percent, 2),
                "memory_usage_mb": round(memory_usage / 1024 / 1024, 2),
                "memory_limit_mb": round(memory_limit / 1024 / 1024, 2),
                "memory_percent": round((memory_usage / memory_limit) * 100, 2)
            }
        except (KeyError, ZeroDivisionError):
            return {}
    
    def _parse_docker_stats_to_metrics(
        self,
        container_id: str,
        stats: dict[str, Any]
    ) -> ContainerMetrics:
        """Parse Docker stats to ContainerMetrics"""
        parsed = self._parse_docker_stats(stats)
        
        networks = stats.get("networks", {})
        network_rx = sum(net.get("rx_bytes", 0) for net in networks.values())
        network_tx = sum(net.get("tx_bytes", 0) for net in networks.values())
        
        block_io = stats.get("blkio_stats", {}).get("io_service_bytes_recursive", [])
        block_read = sum(io.get("value", 0) for io in block_io if io.get("op") == "Read")
        block_write = sum(io.get("value", 0) for io in block_io if io.get("op") == "Write")
        
        return ContainerMetrics(
            container_id=container_id,
            cpu_percent=parsed.get("cpu_percent", 0.0),
            memory_usage_mb=parsed.get("memory_usage_mb", 0.0),
            memory_limit_mb=parsed.get("memory_limit_mb", 0.0),
            network_rx_bytes=network_rx,
            network_tx_bytes=network_tx,
            block_read_bytes=block_read,
            block_write_bytes=block_write,
            pids=stats.get("pids_stats", {}).get("current", 0)
        )
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string to bytes"""
        units = {"k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}
        number = int(size_str[:-1])
        unit = size_str[-1].lower()
        return number * units[unit]
    
    async def shutdown(self) -> None:
        """Shutdown container module and cleanup resources"""
        if self._shutdown:
            return
        
        self.logger.info("Shutting down containerized module")
        self._shutdown = True
        
        try:
            # Close Docker connection
            if self.docker:
                await self.docker.close()
            
            # K8s client doesn't need explicit closure
            
            self._initialized = False
            self.logger.info("Containerized module shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")
            raise ContainerError(f"Shutdown failed: {e}") from e


# ==================== Helper Functions ====================

async def create_quantum_container(
    module: ContainerizedModule,
    runtime: QuantumRuntime,
    circuit_code: str,
    tenant_id: str
) -> ContainerInstance:
    """
    Helper function to create a quantum computing container.
    
    Args:
        module: Containerized module instance
        runtime: Quantum runtime to use
        circuit_code: Quantum circuit code to execute
        tenant_id: Tenant identifier for isolation
        
    Returns:
        Deployed container instance
    """
    config = ContainerConfig(
        name=f"quantum-{runtime.value}-{tenant_id}",
        image="",  # Will be set by quantum_runtime
        quantum_runtime=runtime,
        environment={
            "QUANTUM_RUNTIME": runtime.value,
            "CIRCUIT_CODE": circuit_code,
            "TENANT_ID": tenant_id,
        },
        resources=ResourceLimits(
            cpu_limit=2.0,
            memory_limit="4g",
            gpu_count=0
        ),
        security=SecurityPolicy(
            read_only_root=True,
            privileged=False,
            no_new_privileges=True
        ),
        tenant_id=tenant_id
    )
    
    return await module.deploy_container("", config)


# ==================== Integration Examples ====================

async def main_example():
    """Example usage of ContainerizedModule"""
    
    # Initialize module with Docker
    module = ContainerizedModule(
        orchestrator=OrchestratorType.DOCKER,
        enable_monitoring=True
    )
    
    try:
        await module.initialize()
        
        # Deploy a Qiskit quantum container
        config = ContainerConfig(
            name="qiskit-demo",
            image="qiskit/qiskit:latest",
            quantum_runtime=QuantumRuntime.QISKIT,
            environment={"BACKEND": "qasm_simulator"},
            resources=ResourceLimits(cpu_limit=2.0, memory_limit="2g")
        )
        
        instance = await module.deploy_container("", config)
        print(f"Deployed container: {instance.container_id}")
        
        # Get status
        status = await module.get_status(instance.container_id)
        print(f"Container state: {status.state}")
        
        # Get logs
        logs = await module.get_logs(instance.container_id, tail=50)
        print(f"Logs: {logs[:5]}")
        
        # Cleanup
        await module.manage_containers(
            ContainerAction.REMOVE,
            instance.container_id
        )
        
    finally:
        await module.shutdown()


if __name__ == "__main__":
    asyncio.run(main_example())
