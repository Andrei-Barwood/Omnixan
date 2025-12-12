
Package Structure
=================

omnixan-load-balancing/
├── load_balancing_module/
│   ├── __init__.py              # Package exports and convenience functions
│   ├── __main__.py              # CLI entry point
│   ├── load_balancing_module.py # Core implementation
│   └── py.typed                 # Type checking marker
├── tests/
│   └── test_load_balancing_module.py
├── docs/
│   └── (documentation files)
├── README.md
├── requirements.txt
├── setup.py                     # Legacy setup file
├── pyproject.toml              # Modern Python packaging
├── MANIFEST.in                 # Package data manifest
├── config.example.yaml
├── docker-compose.yml
└── .gitignore

Installation Methods
====================

1. Development Installation:
   pip install -e .

2. Production Installation:
   pip install omnixan-load-balancing

3. With development dependencies:
   pip install -e ".[dev]"

4. With monitoring dependencies:
   pip install omnixan-load-balancing[monitoring]

Import Methods
==============

# Standard import
from load_balancing_module import LoadBalancingModule, BackendConfig, Request

# Convenience functions
from load_balancing_module import create_quantum_aware_module

# Constants
from load_balancing_module import ALGORITHM_TYPES, WORKLOAD_TYPES

# Advanced usage
from load_balancing_module import CircuitBreaker, RateLimiter

Command-Line Usage
==================

# Run with default configuration
python -m load_balancing_module

# Run with config file
python -m load_balancing_module --config config.yaml

# Run with specific algorithm
python -m load_balancing_module --algorithm quantum_aware --log-level DEBUG

# Show version
python -m load_balancing_module --version

Type Checking Support
====================

The package includes a py.typed marker and full type hints:

# mypy
mypy your_code.py

# pyright
pyright your_code.py
