# OMNIXAN

OMNIXAN es un workspace Python orientado a experimentacion modular en cloud,
edge, supercomputacion y computacion cuantica. El repo ya tiene una base
estable para importar paquetes, ejecutar smokes y diagnosticar dependencias
opcionales sin romper imports.

## Estado actual

- Fuente de verdad de packaging: `pyproject.toml`
- CLI oficial: `python -m omnixan`
- Diagnostico oficial: `python -m omnixan doctor --json`
- La API publica comun en modulos core usa `initialize()`, `execute()`,
  `shutdown()`, `get_status()` y `get_metrics()`
- Los backends cuanticos, distribuidos y GPU siguen siendo opcionales

## Instalacion

Instalacion minima:

```bash
PYENV_VERSION=hokkaido python -m pip install -e .
```

Extras utiles:

```bash
PYENV_VERSION=hokkaido python -m pip install -e '.[dev]'
PYENV_VERSION=hokkaido python -m pip install -e '.[quantum]'
PYENV_VERSION=hokkaido python -m pip install -e '.[distributed]'
PYENV_VERSION=hokkaido python -m pip install -e '.[cloud]'
```

## Verificacion rapida

```bash
PYENV_VERSION=hokkaido python -m omnixan doctor --json
PYENV_VERSION=hokkaido python -m pytest omnixan/tests
```

Smokes por bloque:

```bash
PYENV_VERSION=hokkaido python -m pytest omnixan/tests -k core_block_smoke
```

Smokes opcionales:

```bash
PYENV_VERSION=hokkaido python -m pytest omnixan/tests -k quantum_stack_smoke
PYENV_VERSION=hokkaido python -m pytest omnixan/tests -k distributed_stack_smoke
```

## CLI oficial

```bash
python -m omnixan --help
python -m omnixan doctor --json
python -m omnixan load-balancing --smoke --json
python -m omnixan redundant-deployment --smoke --json
```

## Documentacion operativa por bloque

- [`omnixan/carbon_based_quantum_cloud/README.md`](./omnixan/carbon_based_quantum_cloud/README.md)
- [`omnixan/edge_computing_network/README.md`](./omnixan/edge_computing_network/README.md)
- [`omnixan/heterogenous_computing_group/README.md`](./omnixan/heterogenous_computing_group/README.md)
- [`omnixan/in_memory_computing_cloud/README.md`](./omnixan/in_memory_computing_cloud/README.md)
- [`omnixan/quantum_cloud_architecture/README.md`](./omnixan/quantum_cloud_architecture/README.md)
- [`omnixan/supercomputing_interconnect_cloud/README.md`](./omnixan/supercomputing_interconnect_cloud/README.md)
- [`omnixan/virtualized_cluster/README.md`](./omnixan/virtualized_cluster/README.md)

## Documentacion adicional

- [`omnixan/docs/ARCHITECTURE.md`](./omnixan/docs/ARCHITECTURE.md)
- [`omnixan/docs/CLI.md`](./omnixan/docs/CLI.md)
- [`omnixan/docs/DEVELOPMENT.md`](./omnixan/docs/DEVELOPMENT.md)
- [`omnixan/docs/PACKAGING.md`](./omnixan/docs/PACKAGING.md)
- [`omnixan/docs/REPO_STATUS.md`](./omnixan/docs/REPO_STATUS.md)

## Ejemplo minimo

```python
import asyncio

from omnixan.carbon_based_quantum_cloud.load_balancing_module.module import (
    BackendConfig,
    HealthCheckConfig,
    LoadBalancingAlgorithm,
    LoadBalancingAlgorithmType,
    LoadBalancingModule,
    LoadBalancingModuleConfig,
    Request,
)


async def main() -> None:
    module = LoadBalancingModule(
        LoadBalancingModuleConfig(
            algorithm=LoadBalancingAlgorithm(
                algorithm_type=LoadBalancingAlgorithmType.ROUND_ROBIN
            ),
            health_check=HealthCheckConfig(healthy_threshold=1),
        )
    )
    await module.initialize()
    try:
        await module.add_backend(BackendConfig(host="127.0.0.1", port=8080))
        result = await module.route_request(Request(client_ip="127.0.0.1"))
        print(result.backend_id)
        print(module.get_status())
    finally:
        await module.shutdown()


asyncio.run(main())
```
