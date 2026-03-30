"""
OMNIXAN redundant deployment module exports.

This package is part of the ``omnixan`` workspace and re-exports the public
API from ``module.py`` for normal package imports.
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Kirtan Teg Singh"
__email__ = "kirtan@omnixan.io"
__license__ = "MIT"

from .module import RedundantDeploymentModule
from .module import ServiceConfig, RegionConfig, ReplicationConfig
from .module import DeploymentResult, FailoverResult, SyncResult, RedundancyStatus
from .module import DeploymentMode, HealthStatus, ReplicationStrategy
from .module import DeploymentError, ReplicationError, FailoverError, QuorumNotReachedError

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "RedundantDeploymentModule",
    "ServiceConfig",
    "RegionConfig",
    "ReplicationConfig",
    "DeploymentResult",
    "FailoverResult",
    "SyncResult",
    "RedundancyStatus",
    "DeploymentMode",
    "HealthStatus",
    "ReplicationStrategy",
    "DeploymentError",
    "ReplicationError",
    "FailoverError",
    "QuorumNotReachedError",
]


def get_version() -> str:
    """Return the exported module version."""
    return __version__


def get_author() -> str:
    """Return the module author."""
    return __author__
