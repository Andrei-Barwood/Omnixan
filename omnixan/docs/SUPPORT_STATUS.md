# Estado de Soporte

## Convenciones

- `supported`: validado en baseline actual con import, inicializacion y operacion simple.
- `degraded`: import seguro, pero requiere extras opcionales para funcionalidad completa.
- `historical`: presente en el repo, sin validacion profunda en esta ronda.

## Soporte por bloque

| Bloque | Estado | Modulos validados | Notas |
| --- | --- | --- | --- |
| `carbon_based_quantum_cloud` | supported | `load_balancing_module`, `redundant_deployment_module` | CLI soportada y smoke estable. |
| `edge_computing_network` | supported | `cache_coherence_module` | Baseline sin stacks opcionales pesados. |
| `heterogenous_computing_group` | supported | `non_blocking_module` | `trillion_thread_parallel_module` queda import-safe, no profundizado. |
| `in_memory_computing_cloud` | supported | `fog_computing_module` | `edge_ai_module` queda degraded si faltan runtimes pesados. |
| `supercomputing_interconnect_cloud` | supported | `tensor_core_module` | `cuda_acceleration_module` queda degraded si faltan backends GPU. |
| `virtualized_cluster` | supported | `fault_mitigation_module` | Cobertura baseline y smoke core. |
| `quantum_cloud_architecture` | degraded | `quantum_algorithm_module`, `quantum_circuit_optimizer_module`, `quantum_error_correction_module`, `quantum_ml_module`, `quantum_simulator_module` | Import seguro sin extras; validacion profunda solo con stack cuantico instalado. |

## Stacks opcionales

- `distributed`: validado cuando el entorno instala `ray` y `dask[distributed]`.
- `quantum`: validado cuando el entorno instala `qiskit`, `qiskit-aer`, `cirq` y `pennylane`.
- `gpu`: soporte preparado para degradar con mensajes claros; no forma parte del baseline minimo.

## Uso recomendado

- Para CI y desarrollo diario: usar el baseline soportado.
- Para exploracion cuantica o distribuida: instalar extras tematicos y ejecutar las smoke suites opcionales.
- Para GPU y runtimes Edge AI: tratar el soporte actual como opt-in por entorno.
