# Redundant Deployment Module

Modulo operativo para despliegue redundante y failover dentro de
`carbon_based_quantum_cloud`.

## Estado actual

- Validado con smoke, CLI y suite de consistencia
- No requiere extras opcionales para la ruta feliz documentada
- Expone la API publica comun:
  `initialize()`, `execute()`, `shutdown()`, `get_status()`, `get_metrics()`

## Ruta feliz

1. Crear `RedundantDeploymentModule()`
2. Llamar `initialize()`
3. Consultar `execute({"operation": "get_status"})`
4. Consultar `get_metrics()`
5. Integrar servicios y regiones solo cuando ya tengas configuracion propia
6. Cerrar con `shutdown()`

## Ejemplo minimo ejecutable

```python
import asyncio

from omnixan.carbon_based_quantum_cloud.redundant_deployment_module.module import (
    RedundantDeploymentModule,
)


async def main() -> None:
    module = RedundantDeploymentModule()
    await module.initialize()
    try:
        print(await module.execute({"operation": "get_status"}))
        print(await module.execute({"operation": "get_metrics"}))
    finally:
        await module.shutdown()


asyncio.run(main())
```

## API publica

- `await initialize()`
- `await execute({"operation": "deploy" | "sync" | "failover" | "status" | "get_status" | "get_metrics"})`
- `get_status()`
- `get_metrics()`
- `await shutdown()`

## CLI oficial

```bash
python -m omnixan redundant-deployment --smoke --json
python -m omnixan-redundant-deployment --version
```

## Dependencias opcionales

- Ninguna para la ruta feliz del modulo
