"""
Workspace diagnostics for the OMNIXAN repo.

Run with:
    python -m omnixan.doctor
    python -m omnixan.doctor --json
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import platform
import sys
from typing import Any


OPTIONAL_DEPENDENCIES: dict[str, dict[str, str]] = {
    "numpy": {
        "purpose": "Numerical modules",
        "probe": "spec",
    },
    "pydantic": {
        "purpose": "Configuration and validation models",
        "probe": "spec",
    },
    "ray": {
        "purpose": "Distributed execution runtime",
        "probe": "spec",
    },
    "ray.data": {
        "purpose": "Ray dataset APIs",
        "probe": "spec",
    },
    "dask": {
        "purpose": "Distributed task graph runtime",
        "probe": "spec",
    },
    "dask.array": {
        "purpose": "Parallel array APIs",
        "probe": "spec",
    },
    "dask.distributed": {
        "purpose": "Distributed scheduler and workers",
        "probe": "import",
    },
    "qiskit": {
        "purpose": "Quantum backends",
        "probe": "spec",
    },
    "qiskit_aer": {
        "purpose": "Quantum simulator backend",
        "probe": "spec",
    },
    "cirq": {
        "purpose": "Quantum backends",
        "probe": "spec",
    },
    "pennylane": {
        "purpose": "Quantum machine learning backends",
        "probe": "spec",
    },
    "qutip": {
        "purpose": "Quantum analysis and simulation tooling",
        "probe": "spec",
    },
}

MODULE_CHECKS: dict[str, str] = {
    "package": "omnixan",
    "load_balancing": "omnixan.carbon_based_quantum_cloud.load_balancing_module",
    "redundant_deployment": "omnixan.carbon_based_quantum_cloud.redundant_deployment_module",
    "fog_computing": "omnixan.in_memory_computing_cloud.fog_computing_module.module",
    "cache_coherence": "omnixan.edge_computing_network.cache_coherence_module.module",
    "non_blocking": "omnixan.heterogenous_computing_group.non_blocking_module.module",
    "trillion_thread_parallel": "omnixan.heterogenous_computing_group.trillion_thread_parallel_module.module",
    "fault_mitigation": "omnixan.virtualized_cluster.fault_mitigation_module.module",
    "quantum_algorithm": "omnixan.quantum_cloud_architecture.quantum_algorithm_module.module",
    "quantum_circuit_optimizer": "omnixan.quantum_cloud_architecture.quantum_circuit_optimizer_module.module",
    "quantum_error_correction": "omnixan.quantum_cloud_architecture.quantum_error_correction_module.module",
    "quantum_ml": "omnixan.quantum_cloud_architecture.quantum_ml_module.module",
    "quantum_simulator": "omnixan.quantum_cloud_architecture.quantum_simulator_module.module",
}


def _probe_dependency(module_name: str) -> dict[str, Any]:
    """Check whether an optional dependency can be located."""
    check = OPTIONAL_DEPENDENCIES[module_name]
    purpose = check["purpose"]
    probe = check.get("probe", "spec")

    if probe == "import":
        try:
            importlib.import_module(module_name)
            return {
                "available": True,
                "purpose": purpose,
            }
        except Exception as exc:  # pragma: no cover - exercised via CLI
            return {
                "available": False,
                "purpose": purpose,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    try:
        available = importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        available = False

    return {
        "available": available,
        "purpose": purpose,
    }


def _probe_import(module_name: str) -> dict[str, Any]:
    """Try importing a module and capture any failure details."""
    try:
        module = importlib.import_module(module_name)
        return {
            "status": "ok",
            "module": module_name,
            "file": getattr(module, "__file__", None),
        }
    except Exception as exc:  # pragma: no cover - exercised via CLI
        return {
            "status": "error",
            "module": module_name,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def collect_report() -> dict[str, Any]:
    """Collect a structured repo health report."""
    package_status = _probe_import("omnixan")
    package_version = None
    if package_status["status"] == "ok":
        package = importlib.import_module("omnixan")
        package_version = getattr(package, "__version__", None)

    return {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "package": {
            "version": package_version,
            "status": package_status,
        },
        "dependencies": {
            name: _probe_dependency(name) for name in OPTIONAL_DEPENDENCIES
        },
        "module_imports": {
            name: _probe_import(module_name)
            for name, module_name in MODULE_CHECKS.items()
        },
    }


def _render_text(report: dict[str, Any]) -> str:
    """Render the report in a human-readable format."""
    lines = [
        "OMNIXAN doctor report",
        f"Python: {report['python']['version']} ({report['python']['platform']})",
        f"Executable: {report['python']['executable']}",
        f"Package version: {report['package']['version'] or 'unknown'}",
        "",
        "Optional dependencies:",
    ]

    for name, data in report["dependencies"].items():
        status = "ok" if data["available"] else "missing"
        line = f"  - {name}: {status} ({data['purpose']})"
        if not data["available"] and "error_type" in data:
            line += f" [{data['error_type']}: {data['error']}]"
        lines.append(line)

    lines.append("")
    lines.append("Module imports:")

    for name, data in report["module_imports"].items():
        if data["status"] == "ok":
            lines.append(f"  - {name}: ok")
        else:
            lines.append(
                f"  - {name}: error ({data['error_type']}: {data['error']})"
            )

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for repo diagnostics."""
    parser = argparse.ArgumentParser(description="Run OMNIXAN workspace diagnostics")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full report as JSON",
    )
    args = parser.parse_args(argv)

    report = collect_report()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
