# Auditoria de Huecos Cuanticos

## Proposito

Este documento cierra el Dia 17. Su objetivo es responder, con evidencia
concreta, que partes de la cadena cuantica canonica ya existen en OMNIXAN, que
partes siguen fragmentadas y que partes aun no existen como servicio real.

## Evidencia usada

- La cadena oficial definida en `QUANTUM_PIPELINE.md`.
- Los contratos de `omnixan.quantum_pipeline`.
- La implementacion real de `module.py` en los cinco modulos del bloque
  `quantum_cloud_architecture`.
- La smoke suite cuantica en `omnixan/tests/test_quantum_stack_smoke.py`.
- La prueba de imports seguros en `omnixan/tests/test_repo_health.py`.

## Hallazgo principal

OMNIXAN ya tiene piezas cuanticas reales y utilizables. Los cinco modulos del
bloque `quantum_cloud_architecture` pasaron la smoke suite en el `venv` cuantico
del repo el 1 de abril de 2026. El problema mayor no es "no hay modulo", sino
"las piezas no estan unidas por una pipeline ejecutable de extremo a extremo".

## Resumen ejecutivo

- No hay placeholders funcionales dentro de `quantum_cloud_architecture`.
- Si existen piezas reales para diseno, optimizacion, simulacion, mitigacion y
  QML.
- La fragmentacion principal esta entre los contratos canonicos y las
  superficies propias de cada modulo.
- `QuantumMission`, `QuantumExecutionPlan` y `QuantumPipelineReport` ya existen
  como contratos, pero todavia no existe un servicio que los recorra y los
  llene de punta a punta.

## Tabla de cobertura cuantica

| Etapa canonica | Estado real | Piezas actuales | Utilizable hoy | Vacio principal |
| --- | --- | --- | --- | --- |
| `mission` | ausente como servicio | `QuantumMission` en `omnixan.quantum_pipeline` | `no` | no existe un servicio de mision que valide objetivos, restricciones y capacidades |
| `planning` | ausente como servicio | `QuantumExecutionPlan` y `build_baseline_quantum_plan()` | `no` | no existe seleccion real de backend, politica o ruta a partir de capacidades disponibles |
| `circuit_design` | fragmentado pero real | `quantum_algorithm_module`, `quantum_ml_module` | `si con extras` | no emiten `QuantumCircuitArtifact` ni separan claramente diseno de ejecucion |
| `optimization` | real pero aislado | `quantum_circuit_optimizer_module` | `si con extras` | trabaja sobre `QuantumCircuit` crudo y no sobre artefactos canonicos |
| `execution` | real pero aislado | `quantum_simulator_module`, ejecucion interna en `quantum_algorithm_module` | `si con extras` | no produce `QuantumExecutionRecord`, no conserva contexto de mision y no unifica simulacion con ejecucion externa |
| `mitigation` | real pero aislado | `quantum_error_correction_module` | `si con extras` | no produce `QuantumMitigationRecord` ni se enchufa al resultado de ejecucion canonico |
| `reporting` | soportado a nivel repo, no a nivel mision cuantica | `omnixan.doctor`, `omnixan.validate`, `QuantumPipelineReport` | `parcial` | falta un report builder de mision cuantica que combine plan, circuito, ejecucion y mitigacion |

## Auditoria modulo por modulo

| Modulo | Rol real en la pipeline | Evidencia | Estado operativo | Hueco principal |
| --- | --- | --- | --- | --- |
| `quantum_algorithm_module` | diseno de circuito y ejecucion acoplada | import seguro sin extras, smoke `ok` con Qiskit | `experimental`, utilizable con extras | mezcla construccion de circuito, ejecucion y procesamiento; no entrega `QuantumCircuitArtifact` |
| `quantum_circuit_optimizer_module` | optimizacion de circuito | import seguro sin extras, smoke `ok` con Qiskit | `experimental`, utilizable con extras | no recibe plan canonico ni devuelve un artefacto optimizado estandar |
| `quantum_simulator_module` | simulacion y ejecucion local | import seguro sin extras, smoke `ok` con Qiskit | `experimental`, utilizable con extras | devuelve `SimulationResult`, no `QuantumExecutionRecord`; no existe puente oficial con la mision |
| `quantum_error_correction_module` | correccion y mitigacion | import seguro sin extras, smoke `ok` con Qiskit Aer | `experimental`, utilizable con extras | la salida es propia del modulo y no queda ligada al reporte canonico |
| `quantum_ml_module` | diseno cuantico-hibrido y entrenamiento | import seguro sin extras, smoke `ok` con Qiskit | `experimental`, utilizable con extras | su lugar en la pipeline oficial sigue ambiguo: hoy es una capacidad lateral, no una etapa canonica clara |

## Distincion importante

### Piezas realmente utilizables

- `quantum_algorithm_module`
- `quantum_circuit_optimizer_module`
- `quantum_simulator_module`
- `quantum_error_correction_module`
- `quantum_ml_module`

Estas piezas no solo importan: tambien pasaron la smoke suite cuantica del
repo cuando el entorno tenia `qiskit`, `qiskit-aer`, `cirq` y `pennylane`
instalados.

### Piezas fragmentadas

- La mision y el plan existen solo como contratos.
- No existe una funcion orquestadora que una las etapas.
- Cada modulo devuelve modelos o diccionarios propios en vez de usar los
  contratos canonicos de pipeline.

### Piezas ausentes

- Servicio de mision cuantica real.
- Servicio de decreto operativo real.
- Construccion automatica de `QuantumPipelineReport`.
- Smoke end-to-end de la cadena completa.

## Backlog cuantico priorizado

| Prioridad | Vacio real | Impacto en la pipeline | Accion recomendada |
| --- | --- | --- | --- |
| `P0` | Falta un orquestador canonico de extremo a extremo | sin esto no existe ruta feliz cuantica oficial ejecutable | crear `run_quantum_mission()` o equivalente sobre `omnixan.quantum_pipeline` |
| `P0` | Faltan adaptadores desde modulos reales a contratos canonicos | las piezas siguen aisladas aunque funcionen | mapear salidas de diseno, ejecucion y mitigacion a `QuantumCircuitArtifact`, `QuantumExecutionRecord` y `QuantumMitigationRecord` |
| `P0` | Falta una smoke suite end-to-end de la cadena cuantica | hoy se prueban modulos, no la historia completa | anadir test de mision -> plan -> circuito -> optimizacion -> simulacion -> mitigacion -> reporte |
| `P1` | `quantum_algorithm_module` mezcla diseno y ejecucion | borra fronteras entre etapas de la pipeline | introducir un modo "solo construir artefacto" o un adaptador de circuito |
| `P1` | `quantum_simulator_module` no conserva contexto de mision | dificulta reportes y observabilidad de pipeline | agregar capa de adaptacion a `QuantumExecutionRecord` con `mission_id`, `backend_mode` y `execution_path` |
| `P1` | `quantum_error_correction_module` no se integra con el reporte canonico | mitigacion queda aislada del resto del flujo | agregar adaptador a `QuantumMitigationRecord` con fidelidad estimada y notas |
| `P1` | Falta servicio de planeamiento real | la seleccion de backend y estrategia sigue fija o manual | convertir `build_baseline_quantum_plan()` en un planificador minimo consciente del entorno |
| `P2` | `quantum_ml_module` no tiene lugar canonico claro | agrega valor, pero hoy no entra naturalmente en la ruta feliz | decidir si vive en `circuit_design`, `hybrid_runtime` o como servicio aparte |
| `P2` | Falta separar simulacion local, ruidosa y backend externo con adaptadores comunes | la pipeline canonica ya distingue modos, el codigo aun no | normalizar modos `simulator_local`, `simulator_noisy` y `external_backend` en la capa orquestadora |
| `P3` | Documentacion del bloque cuantico estaba demasiado generica | aumenta confusion para contribuidores nuevos | mantener el README del bloque alineado con este documento y con `QUANTUM_PIPELINE.md` |

## Orden recomendado despues del Dia 17

1. Implementar el orquestador cuantico minimo.
2. Normalizar adaptadores a contratos canonicos.
3. Agregar la smoke suite end-to-end.
4. Bajar el planificador de mision a backend.
5. Resolver la posicion canonica de QML.

## Criterio de cierre del Dia 17

El backlog cuantico ya no es una intuicion difusa. Quedo priorizado por impacto
directo en la cadena cuantica oficial y distingue con claridad:

- que piezas son reales hoy
- que piezas siguen fragmentadas
- que piezas aun no existen
