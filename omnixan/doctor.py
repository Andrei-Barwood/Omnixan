"""
Workspace diagnostics for the OMNIXAN repo.

Run with:
    python -m omnixan doctor
    python -m omnixan.doctor
    omnixan-doctor
    python -m omnixan.doctor --json
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import platform
import subprocess
import sys
from typing import Any


DEPENDENCY_CHECKS: dict[str, dict[str, Any]] = {
    "numpy": {
        "purpose": "Numerical modules",
        "probe": "spec",
        "required": True,
        "stack": "core",
    },
    "pydantic": {
        "purpose": "Configuration and validation models",
        "probe": "spec",
        "required": True,
        "stack": "core",
    },
    "ray": {
        "purpose": "Distributed execution runtime",
        "probe": "spec",
        "required": False,
        "stack": "distributed",
    },
    "ray.data": {
        "purpose": "Ray dataset APIs",
        "probe": "spec",
        "required": False,
        "stack": "distributed",
    },
    "dask": {
        "purpose": "Distributed task graph runtime",
        "probe": "spec",
        "required": False,
        "stack": "distributed",
    },
    "dask.array": {
        "purpose": "Parallel array APIs",
        "probe": "spec",
        "required": False,
        "stack": "distributed",
    },
    "dask.distributed": {
        "purpose": "Distributed scheduler and workers",
        "probe": "import",
        "required": False,
        "stack": "distributed",
    },
    "cupy": {
        "purpose": "CUDA array backend",
        "probe": "spec",
        "required": False,
        "stack": "gpu",
    },
    "pycuda": {
        "purpose": "CUDA driver backend",
        "probe": "spec",
        "required": False,
        "stack": "gpu",
    },
    "tensorflow": {
        "purpose": "TensorFlow runtime",
        "probe": "spec",
        "required": False,
        "stack": "edge-ai",
    },
    "torch": {
        "purpose": "PyTorch runtime",
        "probe": "spec",
        "required": False,
        "stack": "edge-ai",
    },
    "tflite_runtime": {
        "purpose": "TensorFlow Lite runtime",
        "probe": "spec",
        "required": False,
        "stack": "edge-ai",
    },
    "tensorrt": {
        "purpose": "TensorRT runtime",
        "probe": "spec",
        "required": False,
        "stack": "edge-ai",
    },
    "openvino": {
        "purpose": "OpenVINO runtime",
        "probe": "spec",
        "required": False,
        "stack": "edge-ai",
    },
    "qiskit": {
        "purpose": "Quantum backends",
        "probe": "spec",
        "required": False,
        "stack": "quantum",
    },
    "qiskit_aer": {
        "purpose": "Quantum simulator backend",
        "probe": "spec",
        "required": False,
        "stack": "quantum",
    },
    "cirq": {
        "purpose": "Quantum backends",
        "probe": "spec",
        "required": False,
        "stack": "quantum",
    },
    "pennylane": {
        "purpose": "Quantum machine learning backends",
        "probe": "spec",
        "required": False,
        "stack": "quantum",
    },
    "qutip": {
        "purpose": "Quantum analysis and simulation tooling",
        "probe": "spec",
        "required": False,
        "stack": "quantum",
    },
}

MODULE_CHECKS: dict[str, str] = {
    "package": "omnixan",
    "load_balancing": "omnixan.carbon_based_quantum_cloud.load_balancing_module",
    "redundant_deployment": "omnixan.carbon_based_quantum_cloud.redundant_deployment_module",
    "cuda_acceleration": "omnixan.supercomputing_interconnect_cloud.cuda_acceleration_module.module",
    "edge_ai": "omnixan.in_memory_computing_cloud.edge_ai_module.module",
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

STACK_HEALTH_RULES: dict[str, dict[str, Any]] = {
    "core": {
        "requires_all": ("numpy", "pydantic"),
        "severity": "environment_error",
        "message": "Core Python dependencies are missing.",
    },
    "distributed": {
        "requires_all": ("ray", "dask.distributed"),
        "severity": "warning",
        "message": "Distributed runtime is not fully installed.",
    },
    "gpu": {
        "requires_any": (("cupy", "pycuda"),),
        "severity": "warning",
        "message": "GPU runtime is not installed.",
    },
    "edge-ai": {
        "requires_any": (
            ("tensorflow", "torch", "tflite_runtime", "tensorrt", "openvino"),
        ),
        "severity": "warning",
        "message": "Extended Edge AI runtimes are not installed.",
    },
    "quantum": {
        "requires_all": ("qiskit", "qiskit_aer"),
        "severity": "warning",
        "message": "Quantum runtime is not fully installed.",
    },
}

MODULE_HEALTH_RULES: dict[str, dict[str, Any]] = {
    "cuda_acceleration": {
        "stack": "gpu",
        "requires_any": (("cupy", "pycuda"),),
        "message": "CUDA acceleration is degraded because no GPU backend is available.",
    },
    "edge_ai": {
        "stack": "edge-ai",
        "capabilities": {
            "tensorflow_models": {
                "dependencies": ("tensorflow",),
                "message": "TensorFlow model support is unavailable.",
            },
            "pytorch_models": {
                "dependencies": ("torch",),
                "message": "PyTorch model support is unavailable.",
            },
            "tflite_models": {
                "dependencies": ("tflite_runtime", "tensorflow"),
                "message": "TFLite model support is unavailable.",
            },
            "tensorrt_models": {
                "dependencies": ("tensorrt",),
                "message": "TensorRT model support is unavailable.",
            },
            "openvino_models": {
                "dependencies": ("openvino",),
                "message": "OpenVINO model support is unavailable.",
            },
            "gpu_acceleration": {
                "dependencies": ("cupy", "torch", "tensorflow"),
                "message": "GPU acceleration for Edge AI is unavailable.",
            },
        },
        "message": "Edge AI optional runtimes are partially unavailable.",
    },
    "quantum_algorithm": {
        "stack": "quantum",
        "requires_all": ("qiskit",),
        "message": "Quantum algorithm execution is degraded because `qiskit` is missing.",
    },
    "quantum_circuit_optimizer": {
        "stack": "quantum",
        "requires_all": ("qiskit",),
        "message": "Quantum circuit optimization is degraded because `qiskit` is missing.",
    },
    "quantum_error_correction": {
        "stack": "quantum",
        "requires_all": ("qiskit",),
        "message": "Quantum error correction is degraded because `qiskit` is missing.",
    },
    "quantum_ml": {
        "stack": "quantum",
        "requires_any": (("pennylane", "qiskit"),),
        "message": "Quantum ML is degraded because no supported runtime is installed.",
    },
    "quantum_simulator": {
        "stack": "quantum",
        "requires_all": ("qiskit", "qiskit_aer"),
        "message": "Quantum simulation is degraded because `qiskit` or `qiskit_aer` is missing.",
    },
}


def _probe_dependency(module_name: str) -> dict[str, Any]:
    """Check whether a dependency can be located or imported."""
    check = DEPENDENCY_CHECKS[module_name]
    purpose = check["purpose"]
    probe = check.get("probe", "spec")

    base: dict[str, Any] = {
        "purpose": purpose,
        "required": bool(check.get("required", False)),
        "stack": check.get("stack", "misc"),
    }

    if probe == "import":
        try:
            importlib.import_module(module_name)
            return {
                **base,
                "available": True,
            }
        except Exception as exc:  # pragma: no cover - exercised via CLI
            return {
                **base,
                "available": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    try:
        available = importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        available = False

    return {
        **base,
        "available": available,
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
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via CLI
        return {
            "status": "error",
            "module": module_name,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "missing_module": exc.name,
        }
    except Exception as exc:  # pragma: no cover - exercised via CLI
        return {
            "status": "error",
            "module": module_name,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def run_package_check() -> dict[str, Any]:
    """Run ``pip check`` to detect broken requirements."""
    command = [sys.executable, "-m", "pip", "check"]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    output_lines = [
        line.strip()
        for line in (result.stdout.splitlines() + result.stderr.splitlines())
        if line.strip()
    ]
    conflicts = [
        line for line in output_lines if line != "No broken requirements found."
    ]
    return {
        "status": "ok" if result.returncode == 0 else "error",
        "command": command,
        "returncode": result.returncode,
        "conflicts": conflicts,
        "raw_output": output_lines,
    }


def _dependencies_available(
    dependency_results: dict[str, dict[str, Any]],
    dependency_names: tuple[str, ...],
) -> list[str]:
    """Return dependencies from the group that are currently available."""
    return [
        dependency_name
        for dependency_name in dependency_names
        if dependency_results[dependency_name]["available"]
    ]


def _evaluate_stack_health(
    dependency_results: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Evaluate high-level stack availability."""
    results: dict[str, dict[str, Any]] = {}

    for stack_name, rule in STACK_HEALTH_RULES.items():
        missing_dependencies: list[str] = []
        missing_dependency_groups: list[list[str]] = []

        for dependency_name in rule.get("requires_all", ()):
            if not dependency_results[dependency_name]["available"]:
                missing_dependencies.append(dependency_name)

        for group in rule.get("requires_any", ()):
            if _dependencies_available(dependency_results, group):
                continue
            missing_dependency_groups.append(list(group))
            missing_dependencies.extend(group)

        deduped_missing = sorted(set(missing_dependencies))
        status = "ok" if not deduped_missing else "degraded"
        results[stack_name] = {
            "status": status,
            "severity": rule["severity"],
            "message": rule["message"],
            "missing_dependencies": deduped_missing,
            "missing_dependency_groups": missing_dependency_groups,
        }

    return results


def _classify_import_failure(import_result: dict[str, Any]) -> str:
    """Classify whether an import failure points to code or environment issues."""
    missing_module = import_result.get("missing_module")
    if missing_module and not str(missing_module).startswith("omnixan"):
        return "environment_error"
    return "code_error"


def _evaluate_module_health(
    dependency_results: dict[str, dict[str, Any]],
    module_imports: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Assess whether each module is fully operational or degraded."""
    results: dict[str, dict[str, Any]] = {}

    for module_name, import_result in module_imports.items():
        if import_result["status"] != "ok":
            category = _classify_import_failure(import_result)
            results[module_name] = {
                "status": "error",
                "severity": category,
                "module": import_result["module"],
                "message": import_result["error"],
                "error_type": import_result["error_type"],
            }
            continue

        rule = MODULE_HEALTH_RULES.get(module_name)
        if not rule:
            results[module_name] = {
                "status": "ok",
                "severity": "ok",
                "module": import_result["module"],
            }
            continue

        missing_dependencies: list[str] = []
        missing_dependency_groups: list[list[str]] = []
        degraded_capabilities: list[str] = []
        capability_status: dict[str, dict[str, Any]] = {}

        for dependency_name in rule.get("requires_all", ()):
            if not dependency_results[dependency_name]["available"]:
                missing_dependencies.append(dependency_name)

        for group in rule.get("requires_any", ()):
            if _dependencies_available(dependency_results, group):
                continue
            missing_dependency_groups.append(list(group))
            missing_dependencies.extend(group)

        for capability_name, capability_rule in rule.get("capabilities", {}).items():
            available_dependencies = _dependencies_available(
                dependency_results,
                capability_rule["dependencies"],
            )
            available = bool(available_dependencies)
            capability_status[capability_name] = {
                "available": available,
                "dependencies": list(capability_rule["dependencies"]),
                "available_dependencies": available_dependencies,
                "message": capability_rule["message"],
            }
            if not available:
                degraded_capabilities.append(capability_name)
                missing_dependencies.extend(capability_rule["dependencies"])

        deduped_missing = sorted(set(missing_dependencies))
        degraded = bool(deduped_missing or degraded_capabilities)

        results[module_name] = {
            "status": "degraded" if degraded else "ok",
            "severity": "warning" if degraded else "ok",
            "module": import_result["module"],
            "stack": rule.get("stack"),
            "message": rule["message"] if degraded else None,
            "missing_dependencies": deduped_missing,
            "missing_dependency_groups": missing_dependency_groups,
            "degraded_capabilities": degraded_capabilities,
            "capabilities": capability_status,
        }

    return results


def _build_findings(
    dependency_results: dict[str, dict[str, Any]],
    module_imports: dict[str, dict[str, Any]],
    stack_health: dict[str, dict[str, Any]],
    module_health: dict[str, dict[str, Any]],
    package_conflicts: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split diagnostics into warnings, environment errors and code errors."""
    warnings: list[dict[str, Any]] = []
    environment_errors: list[dict[str, Any]] = []
    code_errors: list[dict[str, Any]] = []

    for dependency_name, result in dependency_results.items():
        if result["available"] or not result["required"]:
            continue
        environment_errors.append(
            {
                "kind": "missing_required_dependency",
                "name": dependency_name,
                "message": (
                    f"Required dependency `{dependency_name}` is missing "
                    f"for {result['purpose']}."
                ),
            }
        )

    if package_conflicts["status"] == "error":
        environment_errors.append(
            {
                "kind": "package_conflicts",
                "name": "pip-check",
                "message": "Package conflicts were detected by `pip check`.",
                "details": package_conflicts["conflicts"],
            }
        )

    for stack_name, result in stack_health.items():
        if result["status"] != "degraded":
            continue
        finding = {
            "kind": "stack_degraded",
            "name": stack_name,
            "message": result["message"],
            "missing_dependencies": result["missing_dependencies"],
        }
        if result["severity"] == "environment_error":
            environment_errors.append(finding)
        else:
            warnings.append(finding)

    for module_name, result in module_health.items():
        if result["status"] == "ok":
            continue
        finding = {
            "kind": "module_" + result["status"],
            "name": module_name,
            "message": result.get("message") or result.get("error_type", module_name),
        }
        if result["status"] == "degraded":
            finding["missing_dependencies"] = result.get("missing_dependencies", [])
            finding["degraded_capabilities"] = result.get(
                "degraded_capabilities",
                [],
            )
            warnings.append(finding)
            continue

        if result["severity"] == "environment_error":
            environment_errors.append(finding)
        else:
            code_errors.append(finding)

    for module_name, result in module_imports.items():
        if result["status"] != "error":
            continue
        category = _classify_import_failure(result)
        finding = {
            "kind": "module_import_error",
            "name": module_name,
            "message": f"{result['error_type']}: {result['error']}",
        }
        if category == "environment_error":
            environment_errors.append(finding)
        else:
            code_errors.append(finding)

    return warnings, environment_errors, code_errors


def collect_report(include_package_conflicts: bool = True) -> dict[str, Any]:
    """Collect a structured repo health report."""
    package_status = _probe_import("omnixan")
    package_version = None
    if package_status["status"] == "ok":
        package = importlib.import_module("omnixan")
        package_version = getattr(package, "__version__", None)

    dependency_results = {
        name: _probe_dependency(name) for name in DEPENDENCY_CHECKS
    }
    module_imports = {
        name: _probe_import(module_name)
        for name, module_name in MODULE_CHECKS.items()
    }
    stack_health = _evaluate_stack_health(dependency_results)
    module_health = _evaluate_module_health(dependency_results, module_imports)
    package_conflicts = (
        run_package_check()
        if include_package_conflicts
        else {
            "status": "skipped",
            "command": [sys.executable, "-m", "pip", "check"],
            "returncode": None,
            "conflicts": [],
            "raw_output": [],
        }
    )
    warnings, environment_errors, code_errors = _build_findings(
        dependency_results,
        module_imports,
        stack_health,
        module_health,
        package_conflicts,
    )

    summary_status = "ok"
    if environment_errors or code_errors:
        summary_status = "error"
    elif warnings:
        summary_status = "warning"

    return {
        "summary": {
            "status": summary_status,
            "warnings": len(warnings),
            "environment_errors": len(environment_errors),
            "code_errors": len(code_errors),
            "degraded_modules": sum(
                1 for result in module_health.values() if result["status"] == "degraded"
            ),
            "package_conflicts": len(package_conflicts["conflicts"]),
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "package": {
            "version": package_version,
            "status": package_status,
        },
        "dependencies": dependency_results,
        "stack_health": stack_health,
        "package_conflicts": package_conflicts,
        "module_imports": module_imports,
        "module_health": module_health,
        "warnings": warnings,
        "environment_errors": environment_errors,
        "code_errors": code_errors,
    }


def _render_findings(title: str, findings: list[dict[str, Any]]) -> list[str]:
    """Render a flat finding list for the text report."""
    lines = [title]
    if not findings:
        lines.append("  - none")
        return lines

    for finding in findings:
        line = f"  - {finding['name']}: {finding['message']}"
        missing = finding.get("missing_dependencies")
        if missing:
            line += f" (missing: {', '.join(missing)})"
        degraded_capabilities = finding.get("degraded_capabilities")
        if degraded_capabilities:
            line += (
                " (capabilities: " + ", ".join(degraded_capabilities) + ")"
            )
        lines.append(line)
    return lines


def _render_text(report: dict[str, Any]) -> str:
    """Render the report in a human-readable format."""
    summary = report["summary"]
    lines = [
        "OMNIXAN doctor report",
        f"Summary: {summary['status']}",
        f"Warnings: {summary['warnings']}",
        f"Environment errors: {summary['environment_errors']}",
        f"Code errors: {summary['code_errors']}",
        f"Degraded modules: {summary['degraded_modules']}",
        f"Package conflicts: {summary['package_conflicts']}",
        "",
        f"Python: {report['python']['version']} ({report['python']['platform']})",
        f"Executable: {report['python']['executable']}",
        f"Package version: {report['package']['version'] or 'unknown'}",
        "",
    ]

    lines.extend(_render_findings("Warnings:", report["warnings"]))
    lines.append("")
    lines.extend(
        _render_findings("Environment errors:", report["environment_errors"])
    )
    lines.append("")
    lines.extend(_render_findings("Code errors:", report["code_errors"]))
    lines.append("")

    lines.append("Stack health:")
    for stack_name, data in report["stack_health"].items():
        status = data["status"]
        line = f"  - {stack_name}: {status}"
        if data["missing_dependencies"]:
            line += f" (missing: {', '.join(data['missing_dependencies'])})"
        lines.append(line)

    lines.append("")
    lines.append("Package conflicts:")
    if report["package_conflicts"]["status"] == "ok":
        lines.append("  - none")
    elif report["package_conflicts"]["status"] == "skipped":
        lines.append("  - skipped")
    else:
        for conflict in report["package_conflicts"]["conflicts"]:
            lines.append(f"  - {conflict}")

    lines.append("")
    lines.append("Module health:")
    for module_name, data in report["module_health"].items():
        line = f"  - {module_name}: {data['status']}"
        if data.get("missing_dependencies"):
            line += f" (missing: {', '.join(data['missing_dependencies'])})"
        lines.append(line)

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for repo diagnostics."""
    parser = argparse.ArgumentParser(
        prog="omnixan-doctor",
        description="Run OMNIXAN workspace diagnostics",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full report as JSON",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when the report contains warnings or errors",
    )
    parser.add_argument(
        "--skip-pip-check",
        action="store_true",
        help="Skip `python -m pip check` to make the doctor run faster",
    )
    args = parser.parse_args(argv)

    report = collect_report(include_package_conflicts=not args.skip_pip_check)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_render_text(report))

    if args.strict and report["summary"]["status"] != "ok":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
