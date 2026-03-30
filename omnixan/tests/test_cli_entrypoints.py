from __future__ import annotations

import json
import subprocess
import sys


def test_root_cli_help_lists_supported_commands() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "omnixan", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "doctor" in result.stdout
    assert "load-balancing" in result.stdout
    assert "redundant-deployment" in result.stdout


def test_root_cli_doctor_json_delegates() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "omnixan", "doctor", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    assert report["package"]["version"] == "0.2.0"


def test_load_balancing_cli_smoke_json() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "omnixan",
            "load-balancing",
            "--smoke",
            "--json",
            "--log-level",
            "WARNING",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["version"] == "1.0.0"
    assert payload["backends"] == 2
    assert payload["total_requests"] == 1


def test_redundant_deployment_cli_smoke_json() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "omnixan",
            "redundant-deployment",
            "--smoke",
            "--json",
            "--log-level",
            "WARNING",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["version"] == "1.0.0"
    assert len(payload["regions_deployed"]) == 2
    assert payload["healthy_regions"] >= 1


def test_redundant_deployment_package_cli_version() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "omnixan.carbon_based_quantum_cloud.redundant_deployment_module",
            "--version",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "omnixan-redundant-deployment 1.0.0"
