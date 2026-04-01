# Clasificación Operativa de Módulos

## Propósito

Este documento clasifica los módulos del repo como `core`, `experimental`,
`historical` o `placeholder`. La meta es que una persona nueva pueda distinguir
rápidamente qué se puede usar hoy, qué requiere cautela y qué conviene tratar
como deuda o herencia.

## Criterios de clasificación

### `core`

- validado recientemente con smoke, tests o ruta feliz documentada
- documentado con una API pública reconocible
- utilizable hoy sin depender de revalidación manual previa

### `experimental`

- funcional o prometedor, pero con caveats operativos
- depende de extras opcionales, de integración futura o de una validación más profunda
- puede usarse, pero no forma parte del baseline más estable del repo

### `historical`

- parece implementado o tiene documentación rica, pero no se validó en esta ronda
- no forma parte de la superficie actualmente mantenida del producto
- no se recomienda usarlo hoy sin una revalidación específica

### `placeholder`

- scaffold genérico o documentación de “pendiente”
- no debe tratarse como implementación disponible

## Cómo leer la matriz

- `Usar hoy`: `si`, `si con extras`, `no sin revalidar` o `no`
- `Razón corta`: resume por qué quedó en esa categoría

## Matriz por bloque

### `carbon_based_quantum_cloud`

| Módulo | Clasificación | Usar hoy | Razón corta |
| --- | --- | --- | --- |
| `load_balancing_module` | `core` | `si` | Validado con CLI, smoke y API consistente |
| `redundant_deployment_module` | `core` | `si` | Validado con CLI, smoke y continuidad imperial clara |
| `auto_scaling_module` | `historical` | `no sin revalidar` | Parece rico, pero no fue revalidado ni entra al canon actual |
| `cold_migration_module` | `placeholder` | `no` | README genérico en estado pendiente |
| `containerized_module` | `placeholder` | `no` | README genérico en estado pendiente |

### `edge_computing_network`

| Módulo | Clasificación | Usar hoy | Razón corta |
| --- | --- | --- | --- |
| `cache_coherence_module` | `core` | `si` | Validado y documentado con ruta feliz simple |
| `memory_pooling_module` | `historical` | `no sin revalidar` | Implementado en apariencia, pero fuera de la superficie mantenida hoy |
| `near_data_processing_module` | `historical` | `no sin revalidar` | Prometedor, pero no validado en esta campaña |
| `persistent_memory_module` | `historical` | `no sin revalidar` | Buen alcance técnico, sin smoke ni soporte actual |
| `columnar_storage_module` | `placeholder` | `no` | README genérico en estado pendiente |

### `heterogenous_computing_group`

| Módulo | Clasificación | Usar hoy | Razón corta |
| --- | --- | --- | --- |
| `non_blocking_module` | `core` | `si` | Validado y útil como superficie asincrónica estable |
| `infiniband_module` | `historical` | `no sin revalidar` | Hardware-heavy y fuera del canon actual |
| `rdma_acceleration_module` | `historical` | `no sin revalidar` | Parece implementado, pero sin validación reciente |
| `liquid_cooling_module` | `historical` | `no sin revalidar` | Más cercano a infraestructura histórica que a la ruta feliz |
| `trillion_thread_parallel_module` | `historical` | `no sin revalidar` | Import-safe, pero no mantenido como parte del producto actual |

### `in_memory_computing_cloud`

| Módulo | Clasificación | Usar hoy | Razón corta |
| --- | --- | --- | --- |
| `fog_computing_module` | `core` | `si` | Validado y con ruta feliz clara |
| `edge_ai_module` | `experimental` | `si con extras` | Import-safe y usable, pero depende de runtimes opcionales |
| `base_station_deployment_module` | `historical` | `no sin revalidar` | Telecom-rich, pero fuera del canon actual |
| `local_traffic_shunting_module` | `historical` | `no sin revalidar` | Potencialmente útil, sin validación reciente |
| `low_latency_routing_module` | `historical` | `no sin revalidar` | Buen candidato futuro, no mantenido hoy |

### `quantum_cloud_architecture`

| Módulo | Clasificación | Usar hoy | Razón corta |
| --- | --- | --- | --- |
| `quantum_algorithm_module` | `experimental` | `si con extras` | Estratégico para el producto, pero depende de stack cuántico y aún no vive en pipeline canónica |
| `quantum_circuit_optimizer_module` | `experimental` | `si con extras` | Validado con stack cuántico, aún no integrado al canon completo |
| `quantum_error_correction_module` | `experimental` | `si con extras` | Pieza central a futuro, todavía fuera del baseline mínimo |
| `quantum_ml_module` | `experimental` | `si con extras` | Capacidad real con dependencias opcionales y acople parcial |
| `quantum_simulator_module` | `experimental` | `si con extras` | Útil y validado con extras, pero no es baseline estable |

### `supercomputing_interconnect_cloud`

| Módulo | Clasificación | Usar hoy | Razón corta |
| --- | --- | --- | --- |
| `tensor_core_module` | `core` | `si` | Validado, sin extras y con operación simple clara |
| `cuda_acceleration_module` | `experimental` | `si con extras` | Import seguro, pero valor real solo con backend GPU disponible |
| `compute_storage_integrated_module` | `historical` | `no sin revalidar` | Parece implementado, fuera de la ruta feliz actual |
| `ray_tracing_unit_module` | `historical` | `no sin revalidar` | Interesante, pero alejado del producto actual |
| `tensor_slicing_module` | `historical` | `no sin revalidar` | Posible backend futuro, no mantenido hoy |

### `virtualized_cluster`

| Módulo | Clasificación | Usar hoy | Razón corta |
| --- | --- | --- | --- |
| `fault_mitigation_module` | `core` | `si` | Validado y alineado con recuperación e integridad operativa |
| `hybrid_algorithm_module` | `experimental` | `no sin revalidar` | Estratégicamente interesante, pero sin validación reciente |
| `quantum_interface_module` | `experimental` | `no sin revalidar` | Muy alineado al futuro producto, pero no mantenido aún |
| `cryogenic_control_module` | `historical` | `no sin revalidar` | Hardware-heavy y fuera de la ruta feliz |
| `error_correcting_code_module` | `historical` | `no sin revalidar` | Implementado en apariencia, pero periférico al canon vigente |

## Resumen rápido

### Módulos `core`

- `load_balancing_module`
- `redundant_deployment_module`
- `cache_coherence_module`
- `non_blocking_module`
- `fog_computing_module`
- `tensor_core_module`
- `fault_mitigation_module`

### Módulos `experimental`

- `edge_ai_module`
- `quantum_algorithm_module`
- `quantum_circuit_optimizer_module`
- `quantum_error_correction_module`
- `quantum_ml_module`
- `quantum_simulator_module`
- `cuda_acceleration_module`
- `hybrid_algorithm_module`
- `quantum_interface_module`

### Módulos `historical`

- `auto_scaling_module`
- `memory_pooling_module`
- `near_data_processing_module`
- `persistent_memory_module`
- `infiniband_module`
- `rdma_acceleration_module`
- `liquid_cooling_module`
- `trillion_thread_parallel_module`
- `base_station_deployment_module`
- `local_traffic_shunting_module`
- `low_latency_routing_module`
- `compute_storage_integrated_module`
- `ray_tracing_unit_module`
- `tensor_slicing_module`
- `cryogenic_control_module`
- `error_correcting_code_module`

### Módulos `placeholder`

- `cold_migration_module`
- `containerized_module`
- `columnar_storage_module`

## Reglas de uso recomendadas

- Si es `core`: puede usarse hoy con la ruta documentada.
- Si es `experimental`: usar solo con intención explícita y validación local.
- Si es `historical`: no integrarlo a nuevas rutas felices sin una revisión dedicada.
- Si es `placeholder`: tratarlo como deuda o espacio reservado, no como implementación.

## Criterio de cierre del Día 15

Una persona nueva debe poder saber en pocos minutos:

- qué módulos conviene usar hoy
- qué módulos requieren entorno o cautela especial
- qué módulos son herencia no mantenida
- qué módulos todavía no existen de forma real
