# heterogenous_computing_group

Guia operativa del bloque de computacion heterogenea de OMNIXAN.

## Estado actual

- Validado hoy: `non_blocking_module`
- Con documentacion historica o sin smoke reciente: `infiniband_module`, `rdma_acceleration_module`, `liquid_cooling_module`, `trillion_thread_parallel_module`

## Modulos

| Modulo | Estado actual | Dependencias opcionales conocidas | Ruta feliz documentada |
| --- | --- | --- | --- |
| `non_blocking_module` | Validado con smoke y suite de consistencia | Ninguna | `initialize()` -> `submit()` -> `wait()` -> `get_metrics()` |
| `infiniband_module` | Sin smoke reciente | Stack InfiniBand fuera del entorno Python | Revisar README del modulo antes de integrarlo |
| `rdma_acceleration_module` | Sin smoke reciente | Stack RDMA fuera del entorno Python | Revisar README del modulo antes de integrarlo |
| `liquid_cooling_module` | Sin smoke reciente | Telemetria y control de hardware segun despliegue | Revisar README del modulo antes de integrarlo |
| `trillion_thread_parallel_module` | Sin smoke reciente | Sin extras Python verificados en esta revision | Revisar README del modulo antes de integrarlo |

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
        print(module.get_metrics())
    finally:
        await module.shutdown()


asyncio.run(main())
```

## Verificacion rapida

```bash
PYENV_VERSION=hokkaido python -m pytest omnixan/tests -k heterogenous_computing_group_smoke
PYENV_VERSION=hokkaido python -m pytest omnixan/tests -k api_consistency
```

## Modulo recomendado para arrancar

- [`non_blocking_module/README.md`](./non_blocking_module/README.md)
