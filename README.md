# OMNIXAN

OMNIXAN es un workspace Python orientado a experimentación modular en cloud,
edge, supercomputación y computación cuántica. El repositorio contiene bloques
independientes que se pueden importar por separado y un conjunto de utilidades
para verificar el estado del entorno antes de ejecutar módulos opcionales.

## Estado actual

- El paquete principal vive en [`omnixan/`](/Users/kirtantegsingh/Public/omnixan/Omnixan/omnixan).
- Los backends cuánticos y distribuidos son opcionales: el código puede
  importarse sin ellos, pero las funciones específicas requieren instalar sus
  dependencias.
- La raíz del repo ahora incluye un `pyproject.toml` para dar un punto de
  entrada consistente a instalación, testing y tooling.

## Estructura

```text
omnixan/
├── carbon_based_quantum_cloud/
├── edge_computing_network/
├── heterogenous_computing_group/
├── in_memory_computing_cloud/
├── quantum_cloud_architecture/
├── supercomputing_interconnect_cloud/
├── virtualized_cluster/
├── docs/
├── tests/
└── doctor.py
```

## Verificación rápida

```bash
PYENV_VERSION=hokkaido python -m omnixan.doctor
PYENV_VERSION=hokkaido python -m pytest omnixan/tests
```

Smoke suite cuántica opcional:

```bash
python -m pip install -e '.[quantum,dev]'
python -m pytest omnixan/tests -k quantum_stack_smoke
```

Smoke suite distribuida opcional:

```bash
python -m pip install -e '.[distributed,dev]'
python -m pytest omnixan/tests -k distributed_stack_smoke
python -m omnixan.doctor --json
```

## Instalación

Instalación editable mínima:

```bash
PYENV_VERSION=hokkaido python -m pip install -e .
```

Instalación con extras útiles:

```bash
PYENV_VERSION=hokkaido python -m pip install -e '.[dev]'
PYENV_VERSION=hokkaido python -m pip install -e '.[quantum]'
PYENV_VERSION=hokkaido python -m pip install -e '.[distributed]'
```

Para una validación cuántica profunda en un solo entorno:

```bash
python -m pip install -e '.[quantum,dev]'
python -m omnixan.doctor --json
python -m pytest omnixan/tests
```

## Documentación

- Arquitectura: [`omnixan/docs/ARCHITECTURE.md`](/Users/kirtantegsingh/Public/omnixan/Omnixan/omnixan/docs/ARCHITECTURE.md)
- Desarrollo y validación: [`omnixan/docs/DEVELOPMENT.md`](/Users/kirtantegsingh/Public/omnixan/Omnixan/omnixan/docs/DEVELOPMENT.md)
- Estado del repo: [`omnixan/docs/REPO_STATUS.md`](/Users/kirtantegsingh/Public/omnixan/Omnixan/omnixan/docs/REPO_STATUS.md)
- Notas cuánticas adicionales: [`omnixan/QUANTUM_SETUP.md`](/Users/kirtantegsingh/Public/omnixan/Omnixan/omnixan/QUANTUM_SETUP.md)

## Ejemplo

```python
from omnixan.carbon_based_quantum_cloud.load_balancing_module import (
    BackendConfig,
    LoadBalancingModule,
)

module = LoadBalancingModule()
backend = BackendConfig(host="127.0.0.1", port=8080)
```
