"""
Local CI-style validation command for OMNIXAN.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any

from .doctor import collect_report


def _run_pytest(pytest_args: list[str]) -> dict[str, Any]:
    """Run the repo test suite."""
    command = [sys.executable, "-m", "pytest"]
    if pytest_args:
        command.extend(pytest_args)
    else:
        command.append("omnixan/tests")

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


def collect_validation_report(
    *,
    run_tests: bool = True,
    strict_environment: bool = False,
    include_package_conflicts: bool = True,
    pytest_args: list[str] | None = None,
) -> dict[str, Any]:
    """Collect a CI-style validation report."""
    doctor_report = collect_report(
        include_package_conflicts=include_package_conflicts
    )
    test_report = (
        _run_pytest(pytest_args or [])
        if run_tests
        else {
            "status": "skipped",
            "command": [sys.executable, "-m", "pytest", "omnixan/tests"],
            "returncode": None,
            "stdout": "",
            "stderr": "",
        }
    )

    warnings = list(doctor_report["warnings"])
    environment_errors = list(doctor_report["environment_errors"])
    code_errors = list(doctor_report["code_errors"])

    if test_report["status"] == "error":
        code_errors.append(
            {
                "kind": "pytest_failure",
                "name": "pytest",
                "message": (
                    f"Pytest exited with code {test_report['returncode']}."
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
            "tests": test_report["status"],
            "strict_environment": strict_environment,
        },
        "doctor": doctor_report,
        "tests": test_report,
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
        f"Tests: {report['summary']['tests']}",
        "",
        "Doctor summary:",
        f"  - status: {report['doctor']['summary']['status']}",
        f"  - degraded modules: {report['doctor']['summary']['degraded_modules']}",
        f"  - package conflicts: {report['doctor']['summary']['package_conflicts']}",
        "",
        "Pytest:",
        f"  - status: {report['tests']['status']}",
        f"  - command: {' '.join(report['tests']['command'])}",
    ]

    if report["tests"]["status"] == "error":
        lines.append(f"  - returncode: {report['tests']['returncode']}")

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
        help="Only run doctor and skip pytest",
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
        run_tests=not args.skip_tests,
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
