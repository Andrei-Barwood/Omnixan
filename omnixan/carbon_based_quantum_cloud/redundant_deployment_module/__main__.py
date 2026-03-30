"""
Command-line interface for OMNIXAN Redundant Deployment Module.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from typing import Any

try:
    from . import (
        DeploymentMode,
        RedundantDeploymentModule,
        RegionConfig,
        ServiceConfig,
        get_version,
    )
except ImportError:
    from omnixan.carbon_based_quantum_cloud.redundant_deployment_module import (
        DeploymentMode,
        RedundantDeploymentModule,
        RegionConfig,
        ServiceConfig,
        get_version,
    )


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration for the CLI."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _render_payload(payload: dict[str, Any], as_json: bool) -> None:
    """Render CLI results in a stable format."""
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    for key, value in payload.items():
        print(f"{key}: {value}")


async def run_smoke(log_level: str = "INFO") -> dict[str, Any]:
    """Run a short redundant deployment smoke check."""
    module = RedundantDeploymentModule(log_level=log_level)
    await module.initialize()

    try:
        service = ServiceConfig(
            name="omnixan-cli-smoke",
            version="0.0.0",
            regions=[
                RegionConfig(
                    region_id="us-east-1",
                    endpoint="https://us-east-1.omnixan.example",
                    priority=100,
                    latency_ms=10.0,
                    capacity=100,
                ),
                RegionConfig(
                    region_id="eu-west-1",
                    endpoint="https://eu-west-1.omnixan.example",
                    priority=90,
                    latency_ms=25.0,
                    capacity=80,
                ),
            ],
            deployment_mode=DeploymentMode.ACTIVE_PASSIVE,
            health_check_interval=60,
            min_healthy_instances=1,
        )
        deployment = await module.deploy_redundant(service=service, redundancy_level=2)
        status = await module.get_redundancy_status(service.service_id)
        return {
            "active_region": status.active_region,
            "healthy_regions": status.healthy_regions,
            "overall_health": status.overall_health.value,
            "regions_deployed": deployment.regions_deployed,
            "service_id": service.service_id,
            "version": get_version(),
        }
    finally:
        await module.shutdown()


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for redundant deployment helpers."""
    parser = argparse.ArgumentParser(
        prog="omnixan-redundant-deployment",
        description="OMNIXAN Redundant Deployment Module",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {get_version()}",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a short self-check and exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit smoke output as JSON when using --smoke",
    )
    args = parser.parse_args(argv)

    setup_logging(args.log_level)

    if not args.smoke:
        parser.print_help()
        return 0

    payload = asyncio.run(run_smoke(log_level=args.log_level))
    _render_payload(payload, as_json=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
