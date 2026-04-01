# edge_computing_network

Guia operativa del bloque orientado a coherencia de datos y procesamiento
distribuido cerca de la fuente.

## Estado actual

- Validado hoy: `cache_coherence_module`
- Con documentacion historica o sin smoke reciente:
  `columnar_storage_module`, `memory_pooling_module`,
  `near_data_processing_module`, `persistent_memory_module`

## Modulos

| Modulo | Estado actual | Dependencias opcionales conocidas | Ruta feliz documentada |
| --- | --- | --- | --- |
| `cache_coherence_module` | Validado con smoke y suite de consistencia | Ninguna | `initialize()` -> `register_node()` -> `write()` -> `read()` |
| `columnar_storage_module` | Historico, sin smoke reciente | Sin extras Python verificados en esta revision | Revisar README del modulo antes de integrarlo |
| `memory_pooling_module` | Historico, sin smoke reciente | Sin extras Python verificados en esta revision | Revisar README del modulo antes de integrarlo |
| `near_data_processing_module` | Historico, sin smoke reciente | Sin extras Python verificados en esta revision | Revisar README del modulo antes de integrarlo |
| `persistent_memory_module` | Historico, sin smoke reciente | Hardware o runtimes de persistencia segun despliegue | Revisar README del modulo antes de integrarlo |

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
    finally:
        await module.shutdown()


asyncio.run(main())
```

## Verificacion rapida

```bash
PYENV_VERSION=hokkaido python -m pytest omnixan/tests -k edge_computing_network_smoke
```

## Modulo recomendado para arrancar

- [`cache_coherence_module/README.md`](./cache_coherence_module/README.md)
