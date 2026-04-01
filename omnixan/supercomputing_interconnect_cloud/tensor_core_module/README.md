# Tensor Core Module

Modulo operativo para operaciones matriciales dentro de
`supercomputing_interconnect_cloud`.

## Estado actual

- Validado con smoke y suite de consistencia
- No requiere extras opcionales para la ruta feliz documentada
- Expone la API publica comun:
  `initialize()`, `execute()`, `shutdown()`, `get_status()`, `get_metrics()`

## Ruta feliz

1. Inicializar `TensorCoreModule`
2. Ejecutar GEMM por `execute()` o `gemm()`
3. Consultar `get_status()` y `get_metrics()`
4. Cerrar con `shutdown()`

## Ejemplo minimo ejecutable

```python
import asyncio

from omnixan.supercomputing_interconnect_cloud.tensor_core_module.module import (
    TensorCoreModule,
)


async def main() -> None:
    module = TensorCoreModule()
    await module.initialize()
    try:
        result = await module.execute(
            {
                "operation": "gemm",
                "A": [[1, 2], [3, 4]],
                "B": [[5, 6], [7, 8]],
            }
        )
        print(result["result"])
        print(module.get_metrics())
    finally:
        await module.shutdown()


asyncio.run(main())
```

## Dependencias opcionales

- Ninguna para la ruta feliz del modulo
