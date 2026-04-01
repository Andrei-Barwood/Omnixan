# CUDA Acceleration Module

Modulo de aceleracion GPU dentro de `supercomputing_interconnect_cloud`.

## Estado actual

- Import seguro validado en entornos sin GPU
- Las dependencias GPU son opcionales y se cargan de forma lazy
- Si faltan `cupy` y `pycuda`, el modulo falla recien al instanciar o usar la
  ruta GPU, con un mensaje claro

## Ruta feliz

1. Importar el modulo
2. Consultar `get_optional_backend_status()`
3. Instanciar `CUDAAccelerationModule()` solo si hay backend disponible
4. Ejecutar operaciones GPU reales en entornos con `cupy` o `pycuda`

## Ejemplo minimo ejecutable

```python
from omnixan.supercomputing_interconnect_cloud.cuda_acceleration_module.module import (
    get_optional_backend_status,
)

status = get_optional_backend_status()
print(status)
```

## Dependencias opcionales

- `cupy`: backend recomendado para la mayoria de operaciones
- `pycuda`: backend alternativo
- Si ambos faltan, el import del paquete sigue funcionando y el error aparece
  solo al usar el runtime GPU
