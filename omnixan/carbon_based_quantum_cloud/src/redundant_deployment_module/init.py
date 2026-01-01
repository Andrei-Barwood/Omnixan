"""
OMNIXAN - Redundant Deployment Module
Carbon-Based Quantum Cloud Block

Author: Kirtan Teg Singh
License: MIT
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Kirtan Teg Singh"
__email__ = "kirtan@omnixan.io"
__license__ = "MIT"

# Re-export all public APIs
from .core import (
    RedundantDeploymentModule as RedundantDeploymentModule,
)

from .models import (
    # Configuration
    ServiceConfig as ServiceConfig,
    RegionConfig as RegionConfig,
    ReplicationConfig as ReplicationConfig,
    # Results
    DeploymentResult as DeploymentResult,
    FailoverResult as FailoverResult,
    SyncResult as SyncResult,
    RedundancyStatus as RedundancyStatus,
    # Enums
    DeploymentMode as DeploymentMode,
    HealthStatus as HealthStatus,
    ReplicationStrategy as ReplicationStrategy,
)

from .exceptions import (
    DeploymentError as DeploymentError,
    ReplicationError as ReplicationError,
    FailoverError as FailoverError,
    QuorumNotReachedError as QuorumNotReachedError,
)

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
