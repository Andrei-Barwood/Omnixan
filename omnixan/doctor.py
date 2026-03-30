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


OPTIONAL_DEPENDENCIES: dict[str, str] = {
    "numpy": "Numerical modules",
    "pydantic": "Configuration and validation models",
    "ray": "Distributed execution modules",
    "dask": "Distributed execution modules",
    "qiskit": "Quantum backends",
    "qiskit_aer": "Quantum simulator backend",
    "cirq": "Quantum backends",
    "pennylane": "Quantum machine learning backends",
}

MODULE_CHECKS: dict[str, str] = {
    "package": "omnixan",
    "load_balancing": "omnixan.carbon_based_quantum_cloud.load_balancing_module",
    "redundant_deployment": "omnixan.carbon_based_quantum_cloud.redundant_deployment_module",
    "quantum_algorithm": "omnixan.quantum_cloud_architecture.quantum_algorithm_module.module",
    "quantum_simulator": "omnixan.quantum_cloud_architecture.quantum_simulator_module.module",
}


def _probe_dependency(module_name: str) -> dict[str, Any]:
    """Check whether an optional dependency can be located."""
    return {
        "available": importlib.util.find_spec(module_name) is not None,
        "purpose": OPTIONAL_DEPENDENCIES[module_name],
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
        lines.append(f"  - {name}: {status} ({data['purpose']})")

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
