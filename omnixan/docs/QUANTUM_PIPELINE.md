# Cadena Cuántica Canónica

## Propósito

Este documento define la pipeline cuántica oficial de OMNIXAN. Su meta es
dejar un flujo continuo entre problema, circuito, optimización, simulación,
corrección y reporte, aunque algunas etapas sigan parcialmente implementadas.

## Historia operativa

La historia técnica oficial de OMNIXAN ahora es esta:

1. Una persona o servicio crea una `Misión Cuántica`.
2. Esa misión se transforma en un `Decreto Operativo`.
3. El sistema diseña o selecciona un `Artefacto de Circuito`.
4. El circuito se optimiza para el backend o simulador elegido.
5. La misión se simula o ejecuta.
6. La ejecución pasa por corrección o mitigación.
7. El sistema emite un reporte con estado, fidelidad y degradación.

## Flujo canónico

| Etapa | Servicio Amarr | Alias técnico | Salida mínima | Estado actual |
| --- | --- | --- | --- | --- |
| Misión | `Servicio de Misión Cuántica` | `mission-service` | `QuantumMission` | conceptual |
| Planeamiento | `Servicio de Decreto Operativo` | `planning-service` | `QuantumExecutionPlan` | conceptual |
| Diseño | `Servicio de Diseño de Circuito` | `circuit-design-service` | `QuantumCircuitArtifact` | partial |
| Optimización | `Servicio de Optimización Imperial` | `optimization-service` | circuito adaptado | partial |
| Ejecución | `Servicio del Trono de Ejecución` | `execution-service` | `QuantumExecutionRecord` | partial |
| Mitigación | `Servicio de Corrección y Mitigación` | `mitigation-service` | `QuantumMitigationRecord` | partial |
| Reporte | `Servicio de Juicio y Observación` | `observation-service` | `QuantumPipelineReport` | supported |

## Interfaces mínimas

La interfaz mínima de la pipeline vive en
[`omnixan/quantum_pipeline.py`](/Users/kirtantegsingh/Public/omnixan/Omnixan/omnixan/quantum_pipeline.py).
El lenguaje de datos público compartido por la pipeline vive en
[`omnixan/data_model.py`](/Users/kirtantegsingh/Public/omnixan/Omnixan/omnixan/data_model.py).

Los contratos base son:

- `QuantumRequest`
- `QuantumMission`
- `QuantumBackendProfile`
- `QuantumExecutionPolicy`
- `QuantumExecutionPlan`
- `QuantumCircuitArtifact`
- `QuantumJob`
- `QuantumResultSummary`
- `QuantumMetricRecord`
- `QuantumExecutionRecord`
- `QuantumMitigationRecord`
- `QuantumPipelineReport`

## Mapeo al repo actual

### Misión

- Contrato mínimo disponible: `QuantumMission`
- Implementación de servicio: todavía conceptual
- Observación: hoy no existía una entidad pública única; esta interfaz fija ese hueco

### Planeamiento

- Contrato mínimo disponible: `QuantumExecutionPlan`
- Implementación de servicio: todavía conceptual
- Observación: `build_baseline_quantum_plan()` fija la ruta mínima para el canon

### Diseño de circuito

- Módulos candidatos:
  - `omnixan.quantum_cloud_architecture.quantum_algorithm_module`
  - `omnixan.quantum_cloud_architecture.quantum_ml_module`
- Estado: parcial

### Optimización

- Módulo candidato:
  - `omnixan.quantum_cloud_architecture.quantum_circuit_optimizer_module`
- Estado: parcial

### Ejecución

- Módulo candidato:
  - `omnixan.quantum_cloud_architecture.quantum_simulator_module`
- Estado: parcial
- Baseline actual: simulación local o degradación honesta según entorno

### Mitigación

- Módulos candidatos:
  - `omnixan.quantum_cloud_architecture.quantum_error_correction_module`
  - `omnixan.virtualized_cluster.fault_mitigation_module`
- Estado: parcial

### Reporte

- Módulos candidatos:
  - `omnixan.doctor`
  - `omnixan.validate`
- Estado: soportado

## Ruta feliz baseline

La ruta feliz oficial, incluso en un entorno sin stack cuántico completo, es:

1. Crear `QuantumMission`.
2. Generar `QuantumExecutionPlan` con `build_baseline_quantum_plan()`.
3. Usar el contrato de pipeline para saber qué servicios y módulos deberían
   participar.
4. Si faltan backends cuánticos, degradar con claridad en la etapa de ejecución.
5. Emitir un reporte final con estado técnico y narrativo.

## Modos de ejecución

- `simulator_local`: baseline recomendado para la ruta feliz canónica.
- `simulator_noisy`: extensión del baseline para simulación con ruido.
- `hybrid_runtime`: modo futuro para integrar capas clásicas y cuánticas.
- `external_backend`: modo futuro para hardware o proveedores externos.

## Decisión importante del Día 16

OMNIXAN ya no describe la computación cuántica como módulos sueltos. A partir de
esta fase, la unidad oficial del producto es una misión que atraviesa una
cadena canónica de servicios.

## Criterio de cierre

La pipeline ya puede contarse como historia y como arquitectura:

- historia: misión -> decreto -> circuito -> optimización -> ejecución ->
  mitigación -> juicio
- arquitectura: contratos mínimos + servicios + módulos candidatos + estados
