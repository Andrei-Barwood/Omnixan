# Non-Blocking Module

Modulo operativo para ejecucion asincrona y estructuras non-blocking dentro de
`heterogenous_computing_group`.

## Estado actual

- Validado con smoke y suite de consistencia
- No requiere extras opcionales para la ruta feliz documentada
- Expone la API publica comun:
  `initialize()`, `execute()`, `shutdown()`, `get_status()`, `get_metrics()`

## Nota de consistencia

- `execute()` devuelve un envelope comun con `status`, `operation` y payload
- El estado especifico de una operacion se expone como `op_status` dentro del
  payload cuando corresponde

## Ejemplo minimo ejecutable

```python
import asyncio

from omnixan.heterogenous_computing_group.non_blocking_module.module import (
    NonBlockingModule,
    OperationType,
)


async def main() -> None:
    module = NonBlockingModule()
    await module.initialize()
    try:
        op = await module.submit(OperationType.COMPUTE, data={"payload": "hello"})
        result = await module.wait(op.op_id, timeout=1.0)
        print(result.result)
        print(module.get_status())
        print(module.get_metrics())
    finally:
        await module.shutdown()


asyncio.run(main())
```

## Dependencias opcionales

- Ninguna para la ruta feliz del modulo
