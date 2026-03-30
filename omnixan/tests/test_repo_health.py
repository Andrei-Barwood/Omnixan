from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_root_pyproject_exists_and_is_parseable() -> None:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert data["project"]["name"] == "omnixan"
    assert data["project"]["version"] == "0.2.0"


def test_core_entrypoints_import_cleanly() -> None:
    load_balancing = importlib.import_module(
        "omnixan.carbon_based_quantum_cloud.load_balancing_module"
    )
    redundant_deployment = importlib.import_module(
        "omnixan.carbon_based_quantum_cloud.redundant_deployment_module"
    )

    assert hasattr(load_balancing, "LoadBalancingModule")
    assert load_balancing.get_version() == "1.0.0"
    assert hasattr(redundant_deployment, "RedundantDeploymentModule")
    assert redundant_deployment.get_author() == "Kirtan Teg Singh"


def test_optional_quantum_modules_import_without_backends() -> None:
    module_names = [
        "omnixan.quantum_cloud_architecture.quantum_algorithm_module.module",
        "omnixan.quantum_cloud_architecture.quantum_circuit_optimizer_module.module",
        "omnixan.quantum_cloud_architecture.quantum_error_correction_module.module",
        "omnixan.quantum_cloud_architecture.quantum_ml_module.module",
        "omnixan.quantum_cloud_architecture.quantum_simulator_module.module",
    ]

    for module_name in module_names:
        module = importlib.import_module(module_name)
        assert module is not None


def test_load_balancing_cli_version() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "omnixan.carbon_based_quantum_cloud.load_balancing_module",
            "--version",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip().endswith("1.0.0")


def test_doctor_json_report_contains_expected_checks() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "omnixan.doctor", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)

    assert report["package"]["version"] == "0.2.0"
    assert report["module_imports"]["load_balancing"]["status"] == "ok"
    assert report["module_imports"]["redundant_deployment"]["status"] == "ok"
    assert report["module_imports"]["fog_computing"]["status"] == "ok"
    assert report["module_imports"]["cache_coherence"]["status"] == "ok"
    assert report["module_imports"]["non_blocking"]["status"] == "ok"
    assert report["module_imports"]["trillion_thread_parallel"]["status"] == "ok"
    assert report["module_imports"]["fault_mitigation"]["status"] == "ok"
    assert report["module_imports"]["quantum_algorithm"]["status"] == "ok"
    assert report["module_imports"]["quantum_circuit_optimizer"]["status"] == "ok"
    assert report["module_imports"]["quantum_error_correction"]["status"] == "ok"
    assert report["module_imports"]["quantum_ml"]["status"] == "ok"
    assert report["module_imports"]["quantum_simulator"]["status"] == "ok"
