"""
Compatibility wrapper for the historical load balancing module path.

The canonical implementation lives in ``module.py``. This shim keeps older
imports working while the repo uses the package as part of ``omnixan``.
"""

from .module import *  # noqa: F401,F403
