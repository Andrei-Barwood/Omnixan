# Fog Computing Module

Modulo operativo para scheduling fog y offloading basico dentro de
`in_memory_computing_cloud`.

## Estado actual

- Validado con smoke y suite de consistencia
- No requiere extras opcionales para la ruta feliz documentada
- Expone la API publica comun:
  `initialize()`, `execute()`, `shutdown()`, `get_status()`, `get_metrics()`

## Nota de consistencia

- `execute()` usa envelope comun
- El estado especifico de una tarea se expone como `task_status` dentro del
  payload cuando corresponde

## Ejemplo minimo ejecutable

```python
import asyncio

from omnixan.in_memory_computing_cloud.fog_computing_module.module import (
    FogComputingModule,
    FogConfig,
    NodeType,
)


async def main() -> None:
    module = FogComputingModule(FogConfig(resource_check_interval=60.0))
    await module.initialize()
    try:
        await module.register_node(
            name="edge-1",
            node_type=NodeType.EDGE,
            location=(0.0, 0.0),
            cpu_cores=4,
            memory_mb=4096,
            bandwidth_mbps=100.0,
            latency_ms=5.0,
        )
        await module.submit_task(name="doc-task", compute_units=1, memory_mb=128)
        await asyncio.sleep(0.2)
        print(module.get_status())
        print(module.get_metrics())
    finally:
        await module.shutdown()


asyncio.run(main())
```

## Dependencias opcionales

- Ninguna para la ruta feliz del modulo
