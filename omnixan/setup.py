from setuptools import setup, find_packages

setup(
    name="omnixan",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.13",
    install_requires=[
        "numpy>=1.26.0",
        "scipy>=1.11.0",
        "pandas>=2.1.0",
        "scikit-learn>=1.3.0",
        "ray>=2.8.0",
        "dask>=2023.11.0",
        "qiskit>=1.0.0",
        "cirq>=1.4.0",
        "pennylane>=0.33.0",
        "pydantic>=2.5.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "gpu": ["tensorflow>=2.14.0", "torch>=2.1.0", "cupy-cuda12x>=12.0.0"],
        "quantum": [
            "qiskit>=1.0.0",
            "qiskit-aer>=0.13.0",
            "cirq>=1.4.0",
            "pennylane>=0.33.0",
            "qutip>=4.7.0",
            "tensorflow-quantum>=0.7.0",
        ],
        "dev": ["pytest>=7.4.0", "black>=23.11.0", "flake8>=6.1.0", "mypy>=1.7.0"],
        "docs": ["sphinx>=7.2.0", "sphinx-rtd-theme>=2.0.0"],
        "jupyter": ["jupyter>=1.0.0", "notebook>=7.0.0", "ipython>=8.17.0"],
    },
)
