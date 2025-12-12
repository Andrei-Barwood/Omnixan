# Style 1: Import everything needed
from redundant_deployment_module import (
    RedundantDeploymentModule,
    ServiceConfig,
    RegionConfig,
    DeploymentMode,
)

# Style 2: Import module and access via dot notation
import redundant_deployment_module as rdm
module = rdm.RedundantDeploymentModule()

# Style 3: Import specific items
from redundant_deployment_module import RedundantDeploymentModule

# Style 4: Check version
from redundant_deployment_module import __version__, __author__
print(f"Version {__version__} by {__author__}")

# Style 5: Wildcard import (controlled by __all__)
from redundant_deployment_module import *
