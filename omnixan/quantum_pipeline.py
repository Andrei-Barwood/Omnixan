"""
Canonical quantum pipeline contracts for OMNIXAN.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnixan.data_model import (
    QuantumBackendMode,
    QuantumBackendProfile,
    QuantumCircuitArtifact,
    QuantumExecutionPlan,
    QuantumExecutionPolicy,
    QuantumExecutionRecord,
    QuantumJob,
    QuantumJobStatus,
    QuantumMetricRecord,
    QuantumMission,
    QuantumMitigationRecord,
    QuantumPipelineReport,
    QuantumPipelineStage,
    QuantumRequest,
    QuantumResultSummary,
)


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


def build_baseline_backend_profile(
    mode: QuantumBackendMode,
) -> QuantumBackendProfile:
    """Build the canonical backend profile for the baseline flow."""

    provider = "local" if mode != QuantumBackendMode.EXTERNAL_BACKEND else "external"
    backend_name_map = {
        QuantumBackendMode.SIMULATOR_LOCAL: "baseline-local-simulator",
        QuantumBackendMode.SIMULATOR_NOISY: "baseline-noisy-simulator",
        QuantumBackendMode.HYBRID_RUNTIME: "baseline-hybrid-runtime",
        QuantumBackendMode.EXTERNAL_BACKEND: "baseline-external-backend",
    }
    supports_noise = mode == QuantumBackendMode.SIMULATOR_NOISY
    supports_hybrid = mode == QuantumBackendMode.HYBRID_RUNTIME
    capabilities: list[str] = []
    if mode in {
        QuantumBackendMode.SIMULATOR_LOCAL,
        QuantumBackendMode.SIMULATOR_NOISY,
    }:
        capabilities.append("simulation")
    if supports_noise:
        capabilities.append("noise-model")
    if supports_hybrid:
        capabilities.append("hybrid-runtime")
    if mode == QuantumBackendMode.EXTERNAL_BACKEND:
        capabilities.append("external-provider")

    return QuantumBackendProfile(
        mode=mode,
        provider=provider,
        backend_name=backend_name_map[mode],
        supports_noise=supports_noise,
        supports_hybrid=supports_hybrid,
        capabilities=capabilities,
    )


def build_baseline_execution_policy(
    mode: QuantumBackendMode,
) -> QuantumExecutionPolicy:
    """Build the canonical execution policy for the baseline flow."""

    execution_path = (
        "simulate"
        if mode in {
            QuantumBackendMode.SIMULATOR_LOCAL,
            QuantumBackendMode.SIMULATOR_NOISY,
        }
        else "execute"
    )
    notes = [
        "Baseline canonical quantum policy generated from backend mode.",
    ]
    if execution_path == "simulate":
        notes.append("The baseline route prefers simulation-first execution.")
    else:
        notes.append("This route expects runtime execution outside the baseline simulator.")

    return QuantumExecutionPolicy(
        optimization_goal="balanced",
        mitigation_strategy="baseline_mitigation",
        execution_path=execution_path,
        max_shots=1024,
        allow_fallback=True,
        notes=notes,
    )


def build_baseline_quantum_plan(mission: QuantumMission) -> QuantumExecutionPlan:
    """Create the baseline execution plan for the canonical quantum flow."""

    stages = get_canonical_quantum_pipeline()
    selected_services = [stage.service_alias for stage in stages]
    target_modules: list[str] = []
    for stage in stages:
        if stage.candidate_modules:
            target_modules.append(stage.candidate_modules[0])

    backend = build_baseline_backend_profile(mission.preferred_backend_mode)
    policy = build_baseline_execution_policy(mission.preferred_backend_mode)

    return QuantumExecutionPlan(
        mission_id=mission.mission_id,
        backend_mode=mission.preferred_backend_mode,
        backend=backend,
        policy=policy,
        selected_services=selected_services,
        target_modules=target_modules,
        notes=[
            "Baseline canonical quantum pipeline generated from mission contract.",
            "Mission and planning stages remain conceptual but now have stable interfaces.",
            "Canonical backend and policy entities are attached to the plan.",
        ],
    )


def build_baseline_quantum_job(
    mission: QuantumMission,
    plan: QuantumExecutionPlan,
    stage: QuantumPipelineStage = QuantumPipelineStage.EXECUTION,
) -> QuantumJob:
    """Create a canonical job record for the baseline flow."""

    return QuantumJob(
        mission_id=mission.mission_id,
        stage=stage,
        status=(
            QuantumJobStatus.PLANNED
            if stage != QuantumPipelineStage.EXECUTION
            else QuantumJobStatus.READY
        ),
        backend_id=plan.backend.backend_id,
        metadata={
            "execution_path": plan.execution_path,
            "backend_name": plan.backend.backend_name,
        },
    )
