"""
Command-line interface for OMNIXAN Load Balancing Module
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None

from load_balancing_module import (
    LoadBalancingModule,
    LoadBalancingModuleConfig,
    LoadBalancingAlgorithm,
    LoadBalancingAlgorithmType,
    BackendConfig,
    get_version,
)


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_config(config_path: Optional[Path]) -> LoadBalancingModuleConfig:
    """Load configuration from file"""
    if not config_path:
        return LoadBalancingModuleConfig()

    if not yaml:
        raise ImportError("PyYAML is required to load config files. Install with: pip install pyyaml")

    with open(config_path) as f:
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
            with open(config_file) as f:
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


def main() -> None:
    """Main entry point"""
    parser = argparse.ArgumentParser(
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

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Load or create configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = LoadBalancingModuleConfig(
            algorithm=LoadBalancingAlgorithm(
                algorithm_type=LoadBalancingAlgorithmType(args.algorithm)
            )
        )

    # Run server
    try:
        asyncio.run(run_server(config, args.config))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
