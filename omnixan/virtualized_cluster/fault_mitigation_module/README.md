# Fault Mitigation Module

Modulo operativo para deteccion de fallos y checkpoints dentro de
`virtualized_cluster`.

## Estado actual

- Validado con smoke y suite de consistencia
- No requiere extras opcionales para la ruta feliz documentada
- Expone la API publica comun:
  `initialize()`, `execute()`, `shutdown()`, `get_status()`, `get_metrics()`

## Ruta feliz

1. Inicializar el modulo
2. Registrar un componente con `register_component()`
3. Emitir `heartbeat()` cuando corresponda
4. Crear checkpoint con `create_checkpoint()`
5. Restaurar estado con `restore_checkpoint()`

## Ejemplo minimo ejecutable

```python
import asyncio

from omnixan.virtualized_cluster.fault_mitigation_module.module import (
    FaultMitigationModule,
)


async def main() -> None:
    module = FaultMitigationModule()
    await module.initialize()
    try:
        component = await module.register_component("worker-a")
        checkpoint = await module.create_checkpoint(
            component.component_id,
            {"mode": "warm", "sequence": 1},
        )
        restored = await module.restore_checkpoint(
            component.component_id,
            checkpoint.checkpoint_id,
        )
        print(restored)
        print(module.get_status())
        print(module.get_metrics())
    finally:
        await module.shutdown()


asyncio.run(main())
```

## Dependencias opcionales

- Ninguna para la ruta feliz del modulo
