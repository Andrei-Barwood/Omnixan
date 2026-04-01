# carbon_based_quantum_cloud

Guia operativa del bloque que concentra modulos de orquestacion clasica para
workloads hibridos dentro de OMNIXAN.

## Estado actual

- Validados hoy: `load_balancing_module`, `redundant_deployment_module`
- Con documentacion historica o sin smoke reciente:
  `auto_scaling_module`, `cold_migration_module`, `containerized_module`
- CLIs oficiales del bloque:
  `python -m omnixan load-balancing --smoke --json`
  `python -m omnixan redundant-deployment --smoke --json`

## Modulos

| Modulo | Estado actual | Dependencias opcionales conocidas | Ruta feliz documentada |
| --- | --- | --- | --- |
| `load_balancing_module` | Validado con smoke, CLI y suite de consistencia | Ninguna | `initialize()` -> `add_backend()` -> `route_request()` -> `get_load_distribution()` |
| `redundant_deployment_module` | Validado con smoke, CLI y suite de consistencia | Ninguna | `initialize()` -> `execute({"operation": "get_status"})` -> `get_metrics()` |
| `auto_scaling_module` | Historico, sin smoke reciente | Sin extras Python verificados en esta revision | Revisar README del modulo antes de integrarlo |
| `cold_migration_module` | Historico, sin smoke reciente | Sin extras Python verificados en esta revision | Revisar README del modulo antes de integrarlo |
| `containerized_module` | Historico, sin smoke reciente | Runtime externo de contenedores segun despliegue | Revisar README del modulo antes de integrarlo |

## Ejemplo minimo ejecutable

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
        distribution = await module.get_load_distribution()
        print(result.backend_id)
        print(distribution.total_requests)
    finally:
        await module.shutdown()


asyncio.run(main())
```

## Verificacion rapida

```bash
PYENV_VERSION=hokkaido python -m pytest omnixan/tests -k carbon_based_quantum_cloud_smoke
PYENV_VERSION=hokkaido python -m omnixan load-balancing --smoke --json
PYENV_VERSION=hokkaido python -m omnixan redundant-deployment --smoke --json
```

## Modulos recomendados para arrancar

- [`load_balancing_module/README.md`](./load_balancing_module/README.md)
- [`redundant_deployment_module/README.md`](./redundant_deployment_module/README.md)
