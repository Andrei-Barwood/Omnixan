"""
Canonical product data model for OMNIXAN.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class QuantumBackendMode(str, Enum):
    """Supported execution modes for the canonical quantum flow."""

    SIMULATOR_LOCAL = "simulator_local"
    SIMULATOR_NOISY = "simulator_noisy"
    HYBRID_RUNTIME = "hybrid_runtime"
    EXTERNAL_BACKEND = "external_backend"


class QuantumPipelineStage(str, Enum):
    """Ordered stages for the canonical quantum flow."""

    MISSION = "mission"
    PLANNING = "planning"
    CIRCUIT_DESIGN = "circuit_design"
    OPTIMIZATION = "optimization"
    EXECUTION = "execution"
    MITIGATION = "mitigation"
    REPORTING = "reporting"


class QuantumJobStatus(str, Enum):
    """Canonical lifecycle states for a quantum job."""

    PENDING = "pending"
    PLANNED = "planned"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DEGRADED = "degraded"
    CANCELLED = "cancelled"


class QuantumMetricKind(str, Enum):
    """Canonical metric categories for the product surface."""

    LATENCY = "latency"
    FIDELITY = "fidelity"
    DEPTH = "depth"
    GATE_COUNT = "gate_count"
    SHOTS = "shots"
    COST = "cost"
    AVAILABILITY = "availability"


class QuantumRequest(BaseModel):
    """External request entering the canonical product flow."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(default_factory=lambda: f"request-{uuid4().hex[:12]}")
    title: str
    objective: str
    requester: str | None = None
    source_service: str | None = None
    algorithm_family: str | None = None
    preferred_backend_mode: QuantumBackendMode = QuantumBackendMode.SIMULATOR_LOCAL
    constraints: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
    required_capabilities: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)


class QuantumMission(BaseModel):
    """Accepted mission derived from a request."""

    model_config = ConfigDict(extra="forbid")

    mission_id: str = Field(default_factory=lambda: f"mission-{uuid4().hex[:12]}")
    request_id: str | None = None
    title: str
    objective: str
    algorithm_family: str | None = None
    preferred_backend_mode: QuantumBackendMode = QuantumBackendMode.SIMULATOR_LOCAL
    constraints: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
    required_capabilities: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)


class QuantumBackendProfile(BaseModel):
    """Canonical public description of a backend."""

    model_config = ConfigDict(extra="forbid")

    backend_id: str = Field(default_factory=lambda: f"backend-{uuid4().hex[:12]}")
    mode: QuantumBackendMode
    provider: str = "local"
    backend_name: str = "baseline-simulator"
    availability_status: str = "available"
    supports_noise: bool = False
    supports_hybrid: bool = False
    max_qubits: int | None = None
    capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QuantumExecutionPolicy(BaseModel):
    """Canonical execution policy applied by the planning service."""

    model_config = ConfigDict(extra="forbid")

    policy_id: str = Field(default_factory=lambda: f"policy-{uuid4().hex[:12]}")
    optimization_goal: str = "balanced"
    mitigation_strategy: str = "baseline_mitigation"
    execution_path: str = "simulate"
    max_shots: int = 1024
    strict_environment: bool = False
    allow_fallback: bool = True
    notes: list[str] = Field(default_factory=list)


class QuantumExecutionPlan(BaseModel):
    """Validated execution plan derived from a mission."""

    model_config = ConfigDict(extra="forbid")

    mission_id: str
    backend_mode: QuantumBackendMode
    backend: QuantumBackendProfile
    policy: QuantumExecutionPolicy
    selected_services: list[str]
    target_modules: list[str]
    notes: list[str] = Field(default_factory=list)

    @property
    def optimization_goal(self) -> str:
        return self.policy.optimization_goal

    @property
    def mitigation_strategy(self) -> str:
        return self.policy.mitigation_strategy

    @property
    def execution_path(self) -> str:
        return self.policy.execution_path


class QuantumCircuitArtifact(BaseModel):
    """Circuit artifact shared between design, optimization and execution."""

    model_config = ConfigDict(extra="forbid")

    mission_id: str
    representation: str = "abstract"
    qubit_count: int = 0
    depth: int | None = None
    circuit_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QuantumJob(BaseModel):
    """Canonical runtime job tracked by the official flow."""

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(default_factory=lambda: f"job-{uuid4().hex[:12]}")
    mission_id: str
    stage: QuantumPipelineStage = QuantumPipelineStage.EXECUTION
    status: QuantumJobStatus = QuantumJobStatus.PENDING
    backend_id: str | None = None
    submitted_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QuantumMetricRecord(BaseModel):
    """Canonical metric record for the public product surface."""

    model_config = ConfigDict(extra="forbid")

    metric_id: str = Field(default_factory=lambda: f"metric-{uuid4().hex[:12]}")
    name: str
    kind: QuantumMetricKind
    value: float
    unit: str | None = None
    stage: QuantumPipelineStage | None = None
    mission_id: str | None = None
    job_id: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)


class QuantumResultSummary(BaseModel):
    """Canonical result envelope for mission-level execution output."""

    model_config = ConfigDict(extra="forbid")

    mission_id: str
    job_id: str | None = None
    status: str
    counts: dict[str, int] = Field(default_factory=dict)
    observables: dict[str, float] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class QuantumExecutionRecord(BaseModel):
    """Execution record produced by the canonical pipeline."""

    model_config = ConfigDict(extra="forbid")

    mission_id: str
    backend_mode: QuantumBackendMode
    backend: QuantumBackendProfile
    job: QuantumJob
    execution_path: str
    status: str
    shots: int = 0
    result: QuantumResultSummary | None = None
    metrics: list[QuantumMetricRecord] = Field(default_factory=list)

    @property
    def result_summary(self) -> dict[str, Any]:
        if self.result is None:
            return {}

        summary = dict(self.result.payload)
        if self.result.counts:
            summary.setdefault("counts", dict(self.result.counts))
        if self.result.observables:
            summary.setdefault("observables", dict(self.result.observables))
        if self.result.error is not None:
            summary.setdefault("error", self.result.error)
        return summary


class QuantumMitigationRecord(BaseModel):
    """Mitigation or correction output attached to a mission."""

    model_config = ConfigDict(extra="forbid")

    mission_id: str
    strategy: str
    applied: bool
    fidelity_estimate: float | None = None
    metrics: list[QuantumMetricRecord] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class QuantumPipelineReport(BaseModel):
    """End-to-end report produced by the canonical quantum pipeline."""

    model_config = ConfigDict(extra="forbid")

    mission: QuantumMission
    plan: QuantumExecutionPlan
    circuit: QuantumCircuitArtifact
    execution: QuantumExecutionRecord
    mitigation: QuantumMitigationRecord
    result: QuantumResultSummary | None = None
    metrics: list[QuantumMetricRecord] = Field(default_factory=list)
    stage_health: dict[str, str] = Field(default_factory=dict)
    narrative_status: str = "baseline_ready"

