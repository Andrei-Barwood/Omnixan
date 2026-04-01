# Load Balancing Module

Modulo operativo para balanceo de trafico dentro de
`carbon_based_quantum_cloud`.

## Estado actual

- Validado con smoke, CLI y suite de consistencia
- No requiere extras opcionales para la ruta feliz documentada
- Expone la API publica comun:
  `initialize()`, `execute()`, `shutdown()`, `get_status()`, `get_metrics()`

## Ruta feliz

1. Crear `LoadBalancingModuleConfig`
2. Inicializar el modulo
3. Registrar al menos un backend con `add_backend()`
4. Enrutar una request con `route_request()`
5. Consultar `get_load_distribution()` o `get_metrics()`
6. Cerrar con `shutdown()`

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
        backend_id = await module.add_backend(
            BackendConfig(host="127.0.0.1", port=8080)
        )
        result = await module.route_request(Request(client_ip="127.0.0.1"))
        print(backend_id == result.backend_id)
        print(module.get_status())
        print(module.get_metrics())
    finally:
        await module.shutdown()


asyncio.run(main())
```

## API publica

- `await initialize()`
- `await execute({"operation": ...})`
- `get_status()`
- `get_metrics()`
- `await add_backend(backend)`
- `await route_request(request)`
- `await get_load_distribution()`
- `await shutdown()`

## CLI oficial

```bash
python -m omnixan load-balancing --smoke --json
python -m omnixan-load-balancing --version
```

## Dependencias opcionales

- Ninguna para la ruta feliz del modulo
