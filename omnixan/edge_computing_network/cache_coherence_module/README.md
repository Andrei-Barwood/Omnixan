# Cache Coherence Module

Modulo operativo de coherencia de cache distribuida dentro de
`edge_computing_network`.

## Estado actual

- Validado con smoke y suite de consistencia
- No requiere extras opcionales para la ruta feliz documentada
- Expone la API publica comun:
  `initialize()`, `execute()`, `shutdown()`, `get_status()`, `get_metrics()`

## Ruta feliz

1. Inicializar el modulo
2. Registrar nodos con `register_node()`
3. Escribir un valor compartido con `write()`
4. Leerlo desde otro nodo con `read()`
5. Consultar `get_status()` y `get_metrics()`

## Ejemplo minimo ejecutable

```python
import asyncio

from omnixan.edge_computing_network.cache_coherence_module.module import (
    CacheCoherenceModule,
)


async def main() -> None:
    module = CacheCoherenceModule()
    await module.initialize()
    try:
        module.register_node("node-a")
        module.register_node("node-b")
        await module.write("node-a", "shared-key", {"value": 42})
        value, hit = await module.read("node-b", "shared-key")
        print(value, hit)
        print(module.get_status())
        print(module.get_metrics())
    finally:
        await module.shutdown()


asyncio.run(main())
```

## Dependencias opcionales

- Ninguna para la ruta feliz del modulo
