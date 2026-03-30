"""
Command-line interface for OMNIXAN Load Balancing Module
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError:
    yaml = None

try:
    from . import (
        LoadBalancingModule,
        LoadBalancingModuleConfig,
        LoadBalancingAlgorithm,
        LoadBalancingAlgorithmType,
        BackendConfig,
        HealthCheckConfig,
        Request,
        WorkloadType,
        get_version,
    )
except ImportError:
    from omnixan.carbon_based_quantum_cloud.load_balancing_module import (
        LoadBalancingModule,
        LoadBalancingModuleConfig,
        LoadBalancingAlgorithm,
        LoadBalancingAlgorithmType,
        BackendConfig,
        HealthCheckConfig,
        Request,
        WorkloadType,
        get_version,
    )


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stderr)
        ]
    )


def _render_payload(payload: dict[str, Any], as_json: bool) -> None:
    """Render CLI output in a stable format."""
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    for key, value in payload.items():
        print(f"{key}: {value}")


def load_config(config_path: Optional[Path]) -> LoadBalancingModuleConfig:
    """Load configuration from file"""
    if not config_path:
        return LoadBalancingModuleConfig()

    if not yaml:
        raise ImportError("PyYAML is required to load config files. Install with: pip install pyyaml")

    with open(config_path, encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    # Parse configuration (simplified - extend as needed)
    lb_config = config_data.get('load_balancing', {})

    algorithm_type = LoadBalancingAlgorithmType(
        lb_config.get('algorithm', {}).get('type', 'round_robin')
    )

    return LoadBalancingModuleConfig(
        algorithm=LoadBalancingAlgorithm(algorithm_type=algorithm_type),
        session_affinity=lb_config.get('session_affinity', False),
        metrics_enabled=lb_config.get('metrics_enabled', True),
    )


async def run_server(config: LoadBalancingModuleConfig, config_file: Optional[Path]) -> None:
    """Run the load balancing server"""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting OMNIXAN Load Balancing Module v{get_version()}")

    # Initialize module
    lb_module = LoadBalancingModule(config)
    await lb_module.initialize()

    try:
        # Load backends from config file if provided
        if config_file and yaml:
            with open(config_file, encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            for backend_data in config_data.get('backends', []):
                backend = BackendConfig(**backend_data)
                backend_id = await lb_module.add_backend(backend)
                logger.info(f"Added backend {backend_id}: {backend.host}:{backend.port}")

        logger.info("Load balancing module running. Press Ctrl+C to stop.")

        # Keep running
        while True:
            await asyncio.sleep(60)

            # Log status
            distribution = await lb_module.get_load_distribution()
            healthy_count = sum(
                1 for b in distribution.backends.values()
                if b['health_status'] == 'healthy'
            )
            logger.info(
                f"Status: {healthy_count}/{len(distribution.backends)} backends healthy, "
                f"{distribution.total_requests} total requests"
            )

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await lb_module.shutdown()
        logger.info("Shutdown complete")


async def run_smoke(
    algorithm_type: LoadBalancingAlgorithmType,
) -> dict[str, Any]:
    """Run a short, deterministic smoke check and exit."""
    config = LoadBalancingModuleConfig(
        algorithm=LoadBalancingAlgorithm(algorithm_type=algorithm_type),
        health_check=HealthCheckConfig(healthy_threshold=1),
        session_affinity=True,
        metrics_enabled=True,
    )
    lb_module = LoadBalancingModule(config)
    await lb_module.initialize()

    try:
        for index in range(2):
            backend = BackendConfig(
                host=f"smoke-backend-{index + 1}.omnixan.local",
                port=8080 + index,
                weight=1.0 + index,
                quantum_capable=index == 0,
                priority=10 - index,
            )
            await lb_module.add_backend(backend)

        result = await lb_module.route_request(
            Request(
                client_ip="127.0.0.1",
                workload_type=WorkloadType.QUANTUM_SIMULATION,
                session_id="omnixan-cli-smoke",
            )
        )
        distribution = await lb_module.get_load_distribution()
        return {
            "algorithm": distribution.algorithm.value,
            "backends": len(distribution.backends),
            "routed_backend": result.backend_id,
            "total_requests": distribution.total_requests,
            "version": get_version(),
        }
    finally:
        await lb_module.shutdown()


def main(argv: list[str] | None = None) -> int:
    """Main entry point"""
    parser = argparse.ArgumentParser(
        prog="omnixan-load-balancing",
        description="OMNIXAN Load Balancing Module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {get_version()}"
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file (YAML)"
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        choices=[algo.value for algo in LoadBalancingAlgorithmType],
        default="round_robin",
        help="Load balancing algorithm (default: round_robin)"
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

    # Setup logging
    setup_logging(args.log_level)

    if args.smoke:
        result = asyncio.run(
            run_smoke(LoadBalancingAlgorithmType(args.algorithm))
        )
        _render_payload(result, as_json=args.json)
        return 0

    # Load or create configuration
    try:
        if args.config:
            config = load_config(args.config)
        else:
            config = LoadBalancingModuleConfig(
                algorithm=LoadBalancingAlgorithm(
                    algorithm_type=LoadBalancingAlgorithmType(args.algorithm)
                )
            )
    except Exception as exc:
        parser.exit(status=2, message=f"error: {exc}\n")

    # Run server
    try:
        asyncio.run(run_server(config, args.config))
        return 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
