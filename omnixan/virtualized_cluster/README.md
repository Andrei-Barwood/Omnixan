# virtualized_cluster

Guia operativa del bloque orientado a tolerancia a fallos y coordinacion de
componentes virtualizados.

## Estado actual

- Validado hoy: `fault_mitigation_module`
- Con documentacion historica o sin smoke reciente:
  `cryogenic_control_module`, `hybrid_algorithm_module`,
  `quantum_interface_module`, `error_correcting_code_module`

## Modulos

| Modulo | Estado actual | Dependencias opcionales conocidas | Ruta feliz documentada |
| --- | --- | --- | --- |
| `fault_mitigation_module` | Validado con smoke y suite de consistencia | Ninguna | `initialize()` -> `register_component()` -> `create_checkpoint()` -> `restore_checkpoint()` |
| `cryogenic_control_module` | Historico, sin smoke reciente | Hardware y telemetria segun despliegue | Revisar README del modulo antes de integrarlo |
| `hybrid_algorithm_module` | Historico, sin smoke reciente | Sin extras Python verificados en esta revision | Revisar README del modulo antes de integrarlo |
| `quantum_interface_module` | Historico, sin smoke reciente | Integraciones cuanticas segun despliegue | Revisar README del modulo antes de integrarlo |
| `error_correcting_code_module` | Historico, sin smoke reciente | Sin extras Python verificados en esta revision | Revisar README del modulo antes de integrarlo |

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
    finally:
        await module.shutdown()


asyncio.run(main())
```

## Verificacion rapida

```bash
PYENV_VERSION=hokkaido python -m pytest omnixan/tests -k virtualized_cluster_smoke
```

## Modulo recomendado para arrancar

- [`fault_mitigation_module/README.md`](./fault_mitigation_module/README.md)
