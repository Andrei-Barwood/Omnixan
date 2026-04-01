"""
Canonical quantum pipeline contracts for OMNIXAN.
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class QuantumBackendMode(str, Enum):
    """Supported execution modes for the canonical quantum pipeline."""

    SIMULATOR_LOCAL = "simulator_local"
    SIMULATOR_NOISY = "simulator_noisy"
    HYBRID_RUNTIME = "hybrid_runtime"
    EXTERNAL_BACKEND = "external_backend"


class QuantumPipelineStage(str, Enum):
    """Ordered canonical stages for OMNIXAN's quantum flow."""

    MISSION = "mission"
    PLANNING = "planning"
    CIRCUIT_DESIGN = "circuit_design"
    OPTIMIZATION = "optimization"
    EXECUTION = "execution"
    MITIGATION = "mitigation"
    REPORTING = "reporting"


class PipelineStageDefinition(BaseModel):
    """Describe one stage in the canonical quantum pipeline."""

    model_config = ConfigDict(extra="forbid")

    stage: QuantumPipelineStage
    service_name: str
    service_alias: str
    purpose: str
    candidate_modules: list[str] = Field(default_factory=list)
    baseline_status: str = Field(
        description="One of: supported, partial, conceptual, degraded"
    )


class QuantumMission(BaseModel):
    """User-facing request that enters the canonical quantum pipeline."""

    model_config = ConfigDict(extra="forbid")

    mission_id: str = Field(default_factory=lambda: f"mission-{uuid4().hex[:12]}")
    title: str
    objective: str
    algorithm_family: str | None = None
    preferred_backend_mode: QuantumBackendMode = QuantumBackendMode.SIMULATOR_LOCAL
    constraints: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
    required_capabilities: list[str] = Field(default_factory=list)


class QuantumExecutionPlan(BaseModel):
    """Validated execution plan derived from a quantum mission."""

    model_config = ConfigDict(extra="forbid")

    mission_id: str
    backend_mode: QuantumBackendMode
    selected_services: list[str]
    target_modules: list[str]
    optimization_goal: str = "balanced"
    mitigation_strategy: str = "baseline_mitigation"
    execution_path: str = "simulate"
    notes: list[str] = Field(default_factory=list)


class QuantumCircuitArtifact(BaseModel):
    """Circuit design output shared between construction and execution stages."""

    model_config = ConfigDict(extra="forbid")

    mission_id: str
    representation: str = "abstract"
    qubit_count: int = 0
    depth: int | None = None
    circuit_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QuantumExecutionRecord(BaseModel):
    """Execution or simulation result for the canonical pipeline."""

    model_config = ConfigDict(extra="forbid")

    mission_id: str
    backend_mode: QuantumBackendMode
    execution_path: str
    status: str
    shots: int = 0
    result_summary: dict[str, Any] = Field(default_factory=dict)


class QuantumMitigationRecord(BaseModel):
    """Mitigation or correction output attached to a pipeline run."""

    model_config = ConfigDict(extra="forbid")

    mission_id: str
    strategy: str
    applied: bool
    fidelity_estimate: float | None = None
    notes: list[str] = Field(default_factory=list)


class QuantumPipelineReport(BaseModel):
    """End-to-end report produced by the canonical quantum pipeline."""

    model_config = ConfigDict(extra="forbid")

    mission: QuantumMission
    plan: QuantumExecutionPlan
    circuit: QuantumCircuitArtifact
    execution: QuantumExecutionRecord
    mitigation: QuantumMitigationRecord
    stage_health: dict[str, str] = Field(default_factory=dict)
    narrative_status: str = "baseline_ready"


def get_canonical_quantum_pipeline() -> list[PipelineStageDefinition]:
    """Return the ordered baseline quantum pipeline for OMNIXAN."""

    return [
        PipelineStageDefinition(
            stage=QuantumPipelineStage.MISSION,
            service_name="Servicio de Mision Cuantica",
            service_alias="mission-service",
            purpose="Traducir una solicitud a una mision cuantica formal.",
            candidate_modules=[],
            baseline_status="conceptual",
        ),
        PipelineStageDefinition(
            stage=QuantumPipelineStage.PLANNING,
            service_name="Servicio de Decreto Operativo",
            service_alias="planning-service",
            purpose="Convertir la mision en un plan ejecutable y verificable.",
            candidate_modules=[],
            baseline_status="conceptual",
        ),
        PipelineStageDefinition(
            stage=QuantumPipelineStage.CIRCUIT_DESIGN,
            service_name="Servicio de Diseno de Circuito",
            service_alias="circuit-design-service",
            purpose="Construir o seleccionar el artefacto cuantico principal.",
            candidate_modules=[
                "omnixan.quantum_cloud_architecture.quantum_algorithm_module",
                "omnixan.quantum_cloud_architecture.quantum_ml_module",
            ],
            baseline_status="partial",
        ),
        PipelineStageDefinition(
            stage=QuantumPipelineStage.OPTIMIZATION,
            service_name="Servicio de Optimizacion Imperial",
            service_alias="optimization-service",
            purpose="Adaptar el circuito al backend o simulador elegido.",
            candidate_modules=[
                "omnixan.quantum_cloud_architecture.quantum_circuit_optimizer_module"
            ],
            baseline_status="partial",
        ),
        PipelineStageDefinition(
            stage=QuantumPipelineStage.EXECUTION,
            service_name="Servicio del Trono de Ejecucion",
            service_alias="execution-service",
            purpose="Simular o ejecutar la mision cuantica.",
            candidate_modules=[
                "omnixan.quantum_cloud_architecture.quantum_simulator_module"
            ],
            baseline_status="partial",
        ),
        PipelineStageDefinition(
            stage=QuantumPipelineStage.MITIGATION,
            service_name="Servicio de Correccion y Mitigacion",
            service_alias="mitigation-service",
            purpose="Proteger fidelidad e integridad de la ejecucion.",
            candidate_modules=[
                "omnixan.quantum_cloud_architecture.quantum_error_correction_module",
                "omnixan.virtualized_cluster.fault_mitigation_module",
            ],
            baseline_status="partial",
        ),
        PipelineStageDefinition(
            stage=QuantumPipelineStage.REPORTING,
            service_name="Servicio de Juicio y Observacion",
            service_alias="observation-service",
            purpose="Reportar estado, soporte, metricas y degradacion.",
            candidate_modules=["omnixan.doctor", "omnixan.validate"],
            baseline_status="supported",
        ),
    ]


def build_baseline_quantum_plan(mission: QuantumMission) -> QuantumExecutionPlan:
    """Create the baseline execution plan for the canonical quantum flow."""

    stages = get_canonical_quantum_pipeline()
    selected_services = [stage.service_alias for stage in stages]
    target_modules: list[str] = []
    for stage in stages:
        if stage.candidate_modules:
            target_modules.append(stage.candidate_modules[0])

    return QuantumExecutionPlan(
        mission_id=mission.mission_id,
        backend_mode=mission.preferred_backend_mode,
        selected_services=selected_services,
        target_modules=target_modules,
        optimization_goal="balanced",
        mitigation_strategy="baseline_mitigation",
        execution_path=(
            "simulate"
            if mission.preferred_backend_mode
            in {
                QuantumBackendMode.SIMULATOR_LOCAL,
                QuantumBackendMode.SIMULATOR_NOISY,
            }
            else "execute"
        ),
        notes=[
            "Baseline canonical quantum pipeline generated from mission contract.",
            "Mission and planning stages remain conceptual but now have stable interfaces.",
        ],
    )
