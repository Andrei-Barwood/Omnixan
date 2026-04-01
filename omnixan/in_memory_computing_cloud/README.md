# in_memory_computing_cloud

Guia operativa del bloque que concentra modulos de orquestacion en memoria,
fog y edge AI.

## Estado actual

- Validados hoy: `fog_computing_module`, `edge_ai_module`
- Con documentacion historica o sin smoke reciente:
  `base_station_deployment_module`, `local_traffic_shunting_module`,
  `low_latency_routing_module`

## Modulos

| Modulo | Estado actual | Dependencias opcionales conocidas | Ruta feliz documentada |
| --- | --- | --- | --- |
| `fog_computing_module` | Validado con smoke y suite de consistencia | Ninguna | `initialize()` -> `register_node()` -> `submit_task()` -> `get_metrics()` |
| `edge_ai_module` | Validado con suite de consistencia y guards opcionales | `tensorflow`, `torch`, `tflite_runtime`, `tensorrt`, `openvino`; aceleracion GPU opcional | `initialize()` -> `deploy_model()` en CPU/ONNX -> `infer()` |
| `base_station_deployment_module` | Historico, sin smoke reciente | Sin extras Python verificados en esta revision | Revisar README del modulo antes de integrarlo |
| `local_traffic_shunting_module` | Historico, sin smoke reciente | Sin extras Python verificados en esta revision | Revisar README del modulo antes de integrarlo |
| `low_latency_routing_module` | Historico, sin smoke reciente | Sin extras Python verificados en esta revision | Revisar README del modulo antes de integrarlo |

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

## Verificacion rapida

```bash
PYENV_VERSION=hokkaido python -m pytest omnixan/tests -k in_memory_computing_cloud_smoke
PYENV_VERSION=hokkaido python -m pytest omnixan/tests -k optional_backend_guards
```

## Modulos recomendados para arrancar

- [`fog_computing_module/README.md`](./fog_computing_module/README.md)
- [`edge_ai_module/README.md`](./edge_ai_module/README.md)
