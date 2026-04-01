"""
Official OMNIXAN CLI dispatcher.
"""

from __future__ import annotations

import argparse
import importlib
from typing import Callable

from . import __version__


COMMAND_TARGETS: dict[str, tuple[str, str]] = {
    "doctor": (
        "Run workspace diagnostics",
        "omnixan.doctor:main",
    ),
    "validate": (
        "Run local CI-style validation",
        "omnixan.validate:main",
    ),
    "load-balancing": (
        "Run or smoke-test the load balancing module",
        "omnixan.carbon_based_quantum_cloud.load_balancing_module.__main__:main",
    ),
    "redundant-deployment": (
        "Smoke-test the redundant deployment module",
        "omnixan.carbon_based_quantum_cloud.redundant_deployment_module.__main__:main",
    ),
}


def _load_callable(target: str) -> Callable[[list[str] | None], int]:
    """Resolve a ``module:function`` target lazily."""
    module_name, _, function_name = target.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level OMNIXAN parser."""
    command_help = "\n".join(
        f"  {name:<22} {summary}"
        for name, (summary, _) in COMMAND_TARGETS.items()
    )
    parser = argparse.ArgumentParser(
        prog="omnixan",
        description="Official OMNIXAN command-line interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Supported commands:\n{command_help}",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=sorted(COMMAND_TARGETS),
        help="Supported command to run",
    )
    parser.add_argument(
        "command_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the selected command",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch to an official OMNIXAN subcommand."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    command_args = list(args.command_args)
    if command_args and command_args[0] == "--":
        command_args = command_args[1:]

    _, target = COMMAND_TARGETS[args.command]
    handler = _load_callable(target)
    return int(handler(command_args) or 0)
