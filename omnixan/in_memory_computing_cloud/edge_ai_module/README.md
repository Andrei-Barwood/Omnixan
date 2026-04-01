# Edge AI Module

Modulo operativo de inferencia y despliegue AI en edge dentro de
`in_memory_computing_cloud`.

## Estado actual

- Validado con suite de consistencia y guards de dependencias opcionales
- La ruta feliz documentada funciona con CPU y formato `ONNX`
- Expone la API publica comun:
  `initialize()`, `execute()`, `shutdown()`, `get_status()`, `get_metrics()`

## Ruta feliz

1. Crear `EdgeAIConfig` con `AcceleratorType.CPU`
2. Inicializar el modulo
3. Desplegar un modelo `ModelFormat.ONNX`
4. Ejecutar `infer()`
5. Consultar `get_runtime_status()`, `get_status()` y `get_metrics()`

## Ejemplo minimo ejecutable

```python
import asyncio

import numpy as np

from omnixan.in_memory_computing_cloud.edge_ai_module.module import (
    AcceleratorType,
    EdgeAIConfig,
    EdgeAIModule,
    ModelFormat,
)


async def main() -> None:
    module = EdgeAIModule(
        EdgeAIConfig(default_accelerator=AcceleratorType.CPU, max_models=2)
    )
    await module.initialize()
    try:
        weights = np.ones((4, 4), dtype=np.float32)
        model = await module.deploy_model(
            name="demo",
            version="1.0",
            format=ModelFormat.ONNX,
            input_shape=[1, 4],
            output_shape=[1, 4],
            weights=weights,
        )
        result = await module.infer(model.model_id, np.ones(4, dtype=np.float32))
        print(result.success, result.output.shape)
        print(module.get_runtime_status())
    finally:
        await module.shutdown()


asyncio.run(main())
```

## Dependencias opcionales

- `tensorflow`: solo para `ModelFormat.TENSORFLOW`
- `torch`: solo para `ModelFormat.PYTORCH`
- `tflite_runtime`: solo para `ModelFormat.TFLITE`
- `tensorrt`: solo para `ModelFormat.TENSORRT`
- `openvino`: solo para `ModelFormat.OPENVINO`
- GPU: opcional; si falta runtime GPU el modulo falla con `EdgeAIError` al
  inicializar esa ruta, no al importar el paquete
