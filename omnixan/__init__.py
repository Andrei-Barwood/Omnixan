"""
🌌 OMNIXAN - Kaamo Station Technologies Implementation
⚛️  WITH QUANTUM COMPUTING SUPPORT

Implementación de conceptos tecnológicos avanzados inspirados en EVE Online,
aplicados a arquitecturas de computación real en la Tierra.

Bloques principales:
  🌐 Carbon-Based Quantum Cloud
  🚀 Supercomputing Interconnect Cloud
  🌍 Edge Computing Network
  ⚡ In-Memory Computing Cloud
  🔌 Heterogenous Computing Group
  🖥️ Virtualized Cluster
  ⚛️  Quantum Cloud Architecture

Versión: 0.2.0 - QUANTUM EDITION
Licencia: MIT
"""

__version__ = "0.2.0"
__author__ = "Kirtan Teg Singh"
__license__ = "MIT"
__quantum_support__ = True

# Importar bloques principales
try:
    from . import carbon_based_quantum_cloud
except ImportError:
    pass

try:
    from . import supercomputing_interconnect_cloud
except ImportError:
    pass

try:
    from . import edge_computing_network
except ImportError:
    pass

try:
    from . import in_memory_computing_cloud
except ImportError:
    pass

try:
    from . import heterogenous_computing_group
except ImportError:
    pass

try:
    from . import virtualized_cluster
except ImportError:
    pass

# ⚛️ NUEVO: Quantum Cloud
try:
    from . import quantum_cloud_architecture
except ImportError:
    pass

try:
    from . import data_model
except ImportError:
    pass

__all__ = [
    "carbon_based_quantum_cloud",
    "supercomputing_interconnect_cloud",
    "edge_computing_network",
    "in_memory_computing_cloud",
    "heterogenous_computing_group",
    "virtualized_cluster",
    "quantum_cloud_architecture",
    "data_model",
]
