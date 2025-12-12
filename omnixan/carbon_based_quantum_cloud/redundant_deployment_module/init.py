"""
OMNIXAN - Redundant Deployment Module
Carbon-Based Quantum Cloud Block

A production-ready multi-region redundant deployment system with automatic
failover, state replication, and zero-downtime deployments for quantum cloud
computing infrastructure.

Author: Kirtan Teg Singh
License: MIT
Version: 1.0.0

Example:
    >>> from redundant_deployment_module import (
    ...     RedundantDeploymentModule,
    ...     ServiceConfig,
    ...     RegionConfig,
    ...     DeploymentMode
    ... )
    >>> module = RedundantDeploymentModule()
    >>> await module.initialize()
"""

from __future__ import annotations

# Version information
__version__ = "1.0.0"
__author__ = "Kirtan Teg Singh"
__email__ = "kirtan@omnixan.io"
__license__ = "MIT"
__copyright__ = "Copyright 2025, OMNIXAN Project"

# Import core module
from .redundant_deployment_module import RedundantDeploymentModule as RedundantDeploymentModule

# Import configuration models
from .redundant_deployment_module import ServiceConfig as ServiceConfig
from .redundant_deployment_module import RegionConfig as RegionConfig
from .redundant_deployment_module import ReplicationConfig as ReplicationConfig

# Import result models
from .redundant_deployment_module import DeploymentResult as DeploymentResult
from .redundant_deployment_module import FailoverResult as FailoverResult
from .redundant_deployment_module import SyncResult as SyncResult
from .redundant_deployment_module import RedundancyStatus as RedundancyStatus

# Import enums
from .redundant_deployment_module import DeploymentMode as DeploymentMode
from .redundant_deployment_module import HealthStatus as HealthStatus
from .redundant_deployment_module import ReplicationStrategy as ReplicationStrategy

# Import exceptions
from .redundant_deployment_module import DeploymentError as DeploymentError
from .redundant_deployment_module import ReplicationError as ReplicationError
from .redundant_deployment_module import FailoverError as FailoverError
from .redundant_deployment_module import QuorumNotReachedError as QuorumNotReachedError

# Public API - Controls what gets imported with "from redundant_deployment_module import *"
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    # Core module
    "RedundantDeploymentModule",
    # Configuration models
    "ServiceConfig",
    "RegionConfig",
    "ReplicationConfig",
    # Result models
    "DeploymentResult",
    "FailoverResult",
    "SyncResult",
    "RedundancyStatus",
    # Enumerations
    "DeploymentMode",
    "HealthStatus",
    "ReplicationStrategy",
    # Exceptions
    "DeploymentError",
    "ReplicationError",
    "FailoverError",
    "QuorumNotReachedError",
]


def get_version() -> str:
    """
    Get the current version of the module.
    
    Returns:
        Version string in semantic versioning format
        
    Example:
        >>> from redundant_deployment_module import get_version
        >>> print(get_version())
        1.0.0
    """
    return __version__


def get_author() -> str:
    """
    Get the author of the module.
    
    Returns:
        Author name
        
    Example:
        >>> from redundant_deployment_module import get_author
        >>> print(get_author())
        Kirtan Teg Singh
    """
    return __author__


# Module initialization logging (optional - can be removed for production)
import logging

_logger = logging.getLogger(__name__)
_logger.debug(f"Loaded redundant_deployment_module v{__version__} by {__author__}")
