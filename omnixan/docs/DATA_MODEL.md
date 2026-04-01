# Modelo de Datos Canónico

## Proposito

Este documento cierra el Dia 19. Su objetivo es unificar las entidades publicas
del producto para que la ruta feliz oficial use un mismo lenguaje de datos,
independientemente de los nombres locales que existan dentro de cada modulo.

## Fuente de verdad

La fuente de verdad del modelo de datos canonico vive en
`omnixan.data_model`. `omnixan.quantum_pipeline` reutiliza y reexporta esas
entidades para el flujo oficial.

## Entidades canonicas

| Concepto | Entidad canónica | Rol |
| --- | --- | --- |
| solicitud | `QuantumRequest` | entrada externa del producto |
| misión | `QuantumMission` | solicitud aceptada por la cadena oficial |
| backend | `QuantumBackendProfile` | descripción pública del runtime elegido |
| política | `QuantumExecutionPolicy` | reglas de optimización, mitigación y ejecución |
| plan | `QuantumExecutionPlan` | decisión validada de servicios, backend y módulos |
| artefacto de circuito | `QuantumCircuitArtifact` | circuito compartido entre diseño, optimización y ejecución |
| job cuántico | `QuantumJob` | unidad rastreable de ejecución dentro de la misión |
| resultado | `QuantumResultSummary` | resultado público resumido de la misión o job |
| métrica | `QuantumMetricRecord` | métrica canónica para la superficie pública del producto |
| ejecución | `QuantumExecutionRecord` | resultado completo de la etapa de ejecución |
| mitigación | `QuantumMitigationRecord` | salida de la etapa de corrección o mitigación |
| reporte | `QuantumPipelineReport` | visión consolidada del flujo oficial |

## Principio de unificación

Una entidad local de modulo puede seguir existiendo, pero si no forma parte de
la superficie pública del producto, ya no se considera semántica canónica.

## Duplicados semánticos detectados y resolución

| Semántica | Duplicados actuales | Resolución canónica |
| --- | --- | --- |
| `QuantumJob` | `virtualized_cluster.hybrid_algorithm_module.QuantumJob`, `virtualized_cluster.quantum_interface_module.QuantumJob` | el nombre público canónico pasa a ser `omnixan.data_model.QuantumJob`; los demás quedan locales a su módulo |
| backend público | `BackendType`, `SimulatorBackend`, `BackendInfo` y variantes por módulo | la entidad pública canónica es `QuantumBackendProfile` junto con `QuantumBackendMode` |
| política de ejecución | `OptimizerConfig`, `SimulatorConfig`, `ErrorCorrectionConfig` y configs locales | la política pública canónica es `QuantumExecutionPolicy`; las configs específicas siguen siendo internas a cada módulo |
| resultado público | `AlgorithmResult`, `SimulationResult`, `CorrectionResult`, `OptimizationResult` | el resumen público canónico es `QuantumResultSummary`; los demás quedan como resultados de etapa o de módulo |
| métricas públicas | `AlgorithmMetrics`, `CircuitMetrics`, `SimulationMetrics`, `TrainingMetrics`, `ErrorCorrectionMetrics` | la métrica pública canónica es `QuantumMetricRecord`; las demás siguen siendo métricas locales de implementación |

## Reglas de uso

- Si una API pública del producto necesita hablar de solicitud, job, backend,
  política, resultado o métrica, debe usar estas entidades canónicas.
- Si un módulo necesita modelos más ricos o específicos, puede mantenerlos
  localmente, pero no deben desplazar el lenguaje de datos oficial.
- La conversión entre modelos locales y canónicos se considera responsabilidad
  de adaptadores u orquestadores del flujo oficial.

## Relación con la pipeline

- `QuantumRequest` y `QuantumMission` cubren entrada y aceptación.
- `QuantumExecutionPlan`, `QuantumBackendProfile` y `QuantumExecutionPolicy`
  cubren planeamiento.
- `QuantumCircuitArtifact` cubre diseño y optimización.
- `QuantumJob`, `QuantumExecutionRecord` y `QuantumResultSummary` cubren la
  ejecución.
- `QuantumMitigationRecord` cubre integridad.
- `QuantumMetricRecord` y `QuantumPipelineReport` cubren observabilidad.

## Decisión del Dia 19

La unidad pública del producto ya no es un diccionario ad hoc por módulo. A
partir de esta fase, la ruta feliz oficial habla un lenguaje de datos común.

## Criterio de cierre del Dia 19

Existe una capa canónica de entidades públicas que evita duplicados semánticos
en el flujo oficial y deja claro qué modelos son de producto y cuáles siguen
siendo detalles de implementación.
