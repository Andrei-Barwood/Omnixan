# 🔄 Cache Coherence Module

## 📖 Descripción

Módulo de coherencia de caché distribuida para OMNIXAN que implementa protocolos MESI/MOESI con invalidación automática, sincronización y resolución de conflictos.

## 🎯 Características

- 🔄 Protocolos MESI y MOESI
- 📡 Operaciones de bus (read, write, invalidate)
- 🗂️ Directory-based coherence tracking
- 📊 Métricas detalladas (hit rate, invalidations)
- ⚡ Sincronización automática

## 🏗️ Estados de Cache Line

| Estado | Descripción |
|--------|-------------|
| M (Modified) | Modificado, único propietario |
| O (Owned) | Modificado pero compartido (MOESI) |
| E (Exclusive) | Limpio, único propietario |
| S (Shared) | Limpio, múltiples copias |
| I (Invalid) | Inválido |

## 💡 Uso Rápido

```python
import asyncio
from omnixan.edge_computing_network.cache_coherence_module.module import (
    CacheCoherenceModule,
    CacheCoherenceConfig,
    CoherenceProtocol
)

async def main():
    config = CacheCoherenceConfig(
        protocol=CoherenceProtocol.MESI,
        cache_size=1000
    )
    
    module = CacheCoherenceModule(config)
    await module.initialize()
    
    # Registrar nodos
    module.register_node("node1")
    module.register_node("node2")
    
    # Escritura
    await module.write("node1", "key1", "value1")
    
    # Lectura (coherente)
    value, hit = await module.read("node2", "key1")
    
    # Métricas
    metrics = module.get_metrics()
    print(f"Hit rate: {metrics['overall_hit_rate']:.2%}")
    
    await module.shutdown()

asyncio.run(main())
```

## 📦 Dependencias

- `pydantic>=2.5.0`
- `asyncio`

---
**Status:** ✅ Implementado | **Última actualización:** 2025-01-XX
