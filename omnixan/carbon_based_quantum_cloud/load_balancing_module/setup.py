"""
Setup configuration for OMNIXAN Load Balancing Module
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="omnixan-load-balancing",
    version="1.0.0",
    author="OMNIXAN Project",
    author_email="contact@omnixan.local",
    description="Production-ready load balancing module for carbon_based_quantum_cloud",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Andrei-Barwood/Omnixan",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Typing :: Typed",
    ],
    python_requires=">=3.10",
    install_requires=[
        "pydantic>=2.0.0,<3.0.0",
        "pydantic-core>=2.10.0",
        "aiohttp>=3.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "mypy>=1.5.0",
            "ruff>=0.1.0",
            "black>=23.0.0",
        ],
        "monitoring": [
            "prometheus-client>=0.18.0",
            "grafana-api>=1.0.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "omnixan-lb=load_balancing_module.__main__:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
