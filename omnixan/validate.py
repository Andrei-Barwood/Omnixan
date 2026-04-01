"""
Local CI-style validation command for OMNIXAN.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any

from .doctor import collect_report, run_package_check


MINIMAL_TEST_TARGETS = [
    "omnixan/tests/test_repo_health.py",
    "omnixan/tests/test_cli_entrypoints.py",
    "omnixan/tests/test_api_consistency.py",
    "omnixan/tests/test_core_block_smoke.py",
    "omnixan/tests/test_optional_backend_guards.py",
]

OPTIONAL_SMOKE_TARGETS = {
    "quantum": ["omnixan/tests/test_quantum_stack_smoke.py"],
    "distributed": ["omnixan/tests/test_distributed_stack_smoke.py"],
}


def _run_pytest(targets: list[str], pytest_args: list[str]) -> dict[str, Any]:
    """Run a specific pytest suite."""
    command = [sys.executable, "-m", "pytest"]
    command.extend(targets)
    command.extend(pytest_args)

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "status": "ok" if result.returncode == 0 else "error",
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _skipped_check(command: list[str]) -> dict[str, Any]:
    """Return a normalized skipped check payload."""
    return {
        "status": "skipped",
        "command": command,
        "returncode": None,
        "stdout": "",
        "stderr": "",
    }


def collect_validation_report(
    *,
    run_minimal_tests: bool = True,
    include_quantum_smoke: bool = False,
    include_distributed_smoke: bool = False,
    strict_environment: bool = False,
    include_package_conflicts: bool = True,
    pytest_args: list[str] | None = None,
) -> dict[str, Any]:
    """Collect a CI-style validation report."""
    pytest_args = pytest_args or []
    doctor_report = collect_report(include_package_conflicts=False)
    pip_check_report = (
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
    minimal_tests_report = (
        _run_pytest(MINIMAL_TEST_TARGETS, pytest_args)
        if run_minimal_tests
        else _skipped_check(
            [sys.executable, "-m", "pytest", *MINIMAL_TEST_TARGETS, *pytest_args]
        )
    )
    optional_smokes = {
        "quantum": (
            _run_pytest(OPTIONAL_SMOKE_TARGETS["quantum"], pytest_args)
            if include_quantum_smoke
            else _skipped_check(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    *OPTIONAL_SMOKE_TARGETS["quantum"],
                    *pytest_args,
                ]
            )
        ),
        "distributed": (
            _run_pytest(OPTIONAL_SMOKE_TARGETS["distributed"], pytest_args)
            if include_distributed_smoke
            else _skipped_check(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    *OPTIONAL_SMOKE_TARGETS["distributed"],
                    *pytest_args,
                ]
            )
        ),
    }

    warnings = list(doctor_report["warnings"])
    environment_errors = list(doctor_report["environment_errors"])
    code_errors = list(doctor_report["code_errors"])

    if pip_check_report["status"] == "error":
        environment_errors.append(
            {
                "kind": "package_conflicts",
                "name": "pip-check",
                "message": "Package conflicts were detected by `pip check`.",
                "details": pip_check_report["conflicts"],
            }
        )

    if minimal_tests_report["status"] == "error":
        code_errors.append(
            {
                "kind": "pytest_failure",
                "name": "minimal_tests",
                "message": (
                    f"Pytest exited with code {minimal_tests_report['returncode']}."
                ),
            }
        )

    for smoke_name, smoke_report in optional_smokes.items():
        if smoke_report["status"] != "error":
            continue
        code_errors.append(
            {
                "kind": "pytest_failure",
                "name": f"{smoke_name}_smoke",
                "message": (
                    f"Pytest exited with code {smoke_report['returncode']}."
                ),
            }
        )

    summary_status = "ok"
    if environment_errors or code_errors:
        summary_status = "error"
    elif warnings:
        summary_status = "warning"

    if strict_environment and warnings and summary_status == "warning":
        summary_status = "error"

    return {
        "summary": {
            "status": summary_status,
            "warnings": len(warnings),
            "environment_errors": len(environment_errors),
            "code_errors": len(code_errors),
            "tests": minimal_tests_report["status"],
            "quantum_smoke": optional_smokes["quantum"]["status"],
            "distributed_smoke": optional_smokes["distributed"]["status"],
            "strict_environment": strict_environment,
        },
        "doctor": doctor_report,
        "pip_check": pip_check_report,
        "checks": {
            "minimal_tests": minimal_tests_report,
            "optional_smokes": optional_smokes,
        },
        "tests": minimal_tests_report,
        "warnings": warnings,
        "environment_errors": environment_errors,
        "code_errors": code_errors,
    }


def _render_text(report: dict[str, Any]) -> str:
    """Render the validation report in a compact text format."""
    lines = [
        "OMNIXAN validation report",
        f"Summary: {report['summary']['status']}",
        f"Warnings: {report['summary']['warnings']}",
        f"Environment errors: {report['summary']['environment_errors']}",
        f"Code errors: {report['summary']['code_errors']}",
        f"Minimal tests: {report['summary']['tests']}",
        f"Quantum smoke: {report['summary']['quantum_smoke']}",
        f"Distributed smoke: {report['summary']['distributed_smoke']}",
        "",
        "Doctor summary:",
        f"  - status: {report['doctor']['summary']['status']}",
        f"  - degraded modules: {report['doctor']['summary']['degraded_modules']}",
        f"  - doctor package conflicts: {report['doctor']['summary']['package_conflicts']}",
        "",
        "Pip check:",
        f"  - status: {report['pip_check']['status']}",
        f"  - command: {' '.join(report['pip_check']['command'])}",
        "",
        "Minimal tests:",
        f"  - status: {report['checks']['minimal_tests']['status']}",
        f"  - command: {' '.join(report['checks']['minimal_tests']['command'])}",
        "",
        "Optional smokes:",
        f"  - quantum: {report['checks']['optional_smokes']['quantum']['status']}",
        f"  - distributed: {report['checks']['optional_smokes']['distributed']['status']}",
    ]

    if report["checks"]["minimal_tests"]["status"] == "error":
        lines.append(
            f"  - minimal returncode: {report['checks']['minimal_tests']['returncode']}"
        )

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for local CI-style validation."""
    parser = argparse.ArgumentParser(
        prog="omnixan-validate",
        description="Run OMNIXAN doctor + pytest in a local CI-style flow",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full report as JSON",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip minimal tests and optional smoke suites",
    )
    parser.add_argument(
        "--include-quantum-smoke",
        action="store_true",
        help="Run the optional quantum smoke suite",
    )
    parser.add_argument(
        "--include-distributed-smoke",
        action="store_true",
        help="Run the optional distributed smoke suite",
    )
    parser.add_argument(
        "--include-optional-smokes",
        action="store_true",
        help="Run both optional smoke suites",
    )
    parser.add_argument(
        "--skip-minimal-tests",
        action="store_true",
        help="Skip the reproducible minimal test suite",
    )
    parser.add_argument(
        "--strict-environment",
        action="store_true",
        help="Treat warnings such as degraded optional stacks as failures",
    )
    parser.add_argument(
        "--skip-pip-check",
        action="store_true",
        help="Skip `python -m pip check` inside doctor",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to pytest after `--`",
    )
    args = parser.parse_args(argv)

    pytest_args = list(args.pytest_args)
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]

    report = collect_validation_report(
        run_minimal_tests=not (args.skip_tests or args.skip_minimal_tests),
        include_quantum_smoke=(
            (args.include_optional_smokes or args.include_quantum_smoke)
            and not args.skip_tests
        ),
        include_distributed_smoke=(
            (args.include_optional_smokes or args.include_distributed_smoke)
            and not args.skip_tests
        ),
        strict_environment=args.strict_environment,
        include_package_conflicts=not args.skip_pip_check,
        pytest_args=pytest_args,
    )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_render_text(report))

    return 0 if report["summary"]["status"] != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())
