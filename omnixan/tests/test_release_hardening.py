from __future__ import annotations

import logging
from pathlib import Path

from omnixan.carbon_based_quantum_cloud.redundant_deployment_module.module import (
    RedundantDeploymentModule,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
HARDENED_MODULES = [
    REPO_ROOT / "omnixan" / "edge_computing_network" / "cache_coherence_module" / "module.py",
    REPO_ROOT / "omnixan" / "in_memory_computing_cloud" / "edge_ai_module" / "module.py",
    REPO_ROOT / "omnixan" / "in_memory_computing_cloud" / "fog_computing_module" / "module.py",
    REPO_ROOT / "omnixan" / "virtualized_cluster" / "fault_mitigation_module" / "module.py",
    REPO_ROOT / "omnixan" / "heterogenous_computing_group" / "non_blocking_module" / "module.py",
    REPO_ROOT / "omnixan" / "heterogenous_computing_group" / "trillion_thread_parallel_module" / "module.py",
    REPO_ROOT / "omnixan" / "supercomputing_interconnect_cloud" / "tensor_core_module" / "module.py",
    REPO_ROOT / "omnixan" / "quantum_cloud_architecture" / "quantum_algorithm_module" / "module.py",
    REPO_ROOT / "omnixan" / "quantum_cloud_architecture" / "quantum_circuit_optimizer_module" / "module.py",
    REPO_ROOT / "omnixan" / "quantum_cloud_architecture" / "quantum_error_correction_module" / "module.py",
    REPO_ROOT / "omnixan" / "quantum_cloud_architecture" / "quantum_ml_module" / "module.py",
    REPO_ROOT / "omnixan" / "quantum_cloud_architecture" / "quantum_simulator_module" / "module.py",
]


def test_release_artifacts_exist() -> None:
    changelog = REPO_ROOT / "CHANGELOG.md"
    support_status = REPO_ROOT / "omnixan" / "docs" / "SUPPORT_STATUS.md"
    internal_release = (
        REPO_ROOT / "omnixan" / "docs" / "INTERNAL_RELEASE_2026-04-01.md"
    )
    daily_tasks = REPO_ROOT / "omnixan" / "docs" / "DAILY_TASKS.md"
    vision = REPO_ROOT / "omnixan" / "docs" / "VISION.md"
    amarr_principles = REPO_ROOT / "omnixan" / "docs" / "AMARR_PRINCIPLES.md"
    service_language = REPO_ROOT / "omnixan" / "docs" / "SERVICE_LANGUAGE.md"
    block_canon_map = REPO_ROOT / "omnixan" / "docs" / "BLOCK_CANON_MAP.md"
    module_classification = (
        REPO_ROOT / "omnixan" / "docs" / "MODULE_CLASSIFICATION.md"
    )
    quantum_pipeline = REPO_ROOT / "omnixan" / "docs" / "QUANTUM_PIPELINE.md"
    quantum_gap_audit = REPO_ROOT / "omnixan" / "docs" / "QUANTUM_GAP_AUDIT.md"
    quantum_block_readme = (
        REPO_ROOT / "omnixan" / "quantum_cloud_architecture" / "README.md"
    )

    assert changelog.exists()
    assert "0.2.0 - 2026-04-01" in changelog.read_text(encoding="utf-8")
    assert support_status.exists()
    assert "| `carbon_based_quantum_cloud` | supported |" in support_status.read_text(
        encoding="utf-8"
    )
    assert internal_release.exists()
    assert "Estado recomendado: `go`" in internal_release.read_text(encoding="utf-8")
    assert daily_tasks.exists()
    assert "## Fase 2 activa: propósito, dominio y experiencia" in daily_tasks.read_text(
        encoding="utf-8"
    )
    assert vision.exists()
    assert "OMNIXAN es una plataforma de orquestación cuántica" in vision.read_text(
        encoding="utf-8"
    )
    assert amarr_principles.exists()
    assert "## Glosario operativo inicial" in amarr_principles.read_text(
        encoding="utf-8"
    )
    assert service_language.exists()
    assert "## Catálogo inicial de servicios Amarr" in service_language.read_text(
        encoding="utf-8"
    )
    assert block_canon_map.exists()
    block_map_text = block_canon_map.read_text(encoding="utf-8")
    assert "`quantum_cloud_architecture`" in block_map_text
    assert "`carbon_based_quantum_cloud`" in block_map_text
    assert module_classification.exists()
    module_text = module_classification.read_text(encoding="utf-8")
    assert "## Criterios de clasificación" in module_text
    assert "`load_balancing_module`" in module_text
    assert "`cold_migration_module`" in module_text
    assert quantum_pipeline.exists()
    assert "## Flujo canónico" in quantum_pipeline.read_text(encoding="utf-8")
    assert quantum_gap_audit.exists()
    quantum_gap_text = quantum_gap_audit.read_text(encoding="utf-8")
    assert "## Tabla de cobertura cuantica" in quantum_gap_text
    assert "## Backlog cuantico priorizado" in quantum_gap_text
    assert quantum_block_readme.exists()
    block_readme_text = quantum_block_readme.read_text(encoding="utf-8")
    assert "QUANTUM_GAP_AUDIT.md" in block_readme_text
    assert "## Ruta feliz actual del bloque" in block_readme_text


def test_hardened_modules_do_not_configure_global_logging_on_import() -> None:
    for module_path in HARDENED_MODULES:
        assert "logging.basicConfig(" not in module_path.read_text(encoding="utf-8")


def test_redundant_deployment_logger_does_not_duplicate_handlers() -> None:
    logger = logging.getLogger("RedundantDeploymentModule")
    before = len(logger.handlers)

    module_a = RedundantDeploymentModule(log_level="INFO")
    mid = len(logging.getLogger("RedundantDeploymentModule").handlers)

    module_b = RedundantDeploymentModule(log_level="DEBUG")
    after = len(logging.getLogger("RedundantDeploymentModule").handlers)

    assert module_a.logger is module_b.logger
    assert module_a.logger.propagate is False
    assert mid >= before
    assert after == mid
