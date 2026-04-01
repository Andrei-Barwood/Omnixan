# Mapa Oficial de Servicios Amarr

## Proposito

Este documento cierra el Dia 18. Su objetivo es convertir la pipeline cuantica
y la plataforma de soporte de OMNIXAN en un mapa oficial de servicios con
fronteras, contratos de alto nivel y politica de exposicion publica.

## Regla de clasificacion

### Servicios nucleares

- Definen la ruta feliz oficial del producto.
- Deben poder narrarse de punta a punta como una mision cuantica.
- Son candidatos a superficie publica de arquitectura, docs y CLI.

### Servicios auxiliares

- No son el producto central, pero sostienen continuidad, resiliencia u
  observabilidad.
- Pueden ser publicos si ayudan a operar el sistema sin confundir la narrativa
  principal.

### Servicios internos

- Existen para adaptar contratos, resolver capacidades o conectar runtimes.
- No deben exponerse publicamente mientras no tengan semantica estable de
  producto.

## Mapa oficial

| Servicio canónico | Alias técnico | Clase | Exposición | Input de alto nivel | Output de alto nivel | Estado | Dependencias principales | Implementación actual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Servicio de Misión Cuántica` | `mission-service` | `nuclear` | `publico futuro` | solicitud narrativa o técnica de una misión | `QuantumMission` | `conceptual` | `omnixan.quantum_pipeline` | contrato disponible, servicio ausente |
| `Servicio de Decreto Operativo` | `planning-service` | `nuclear` | `publico futuro` | `QuantumMission` y capacidades del entorno | `QuantumExecutionPlan` | `conceptual` | `omnixan.quantum_pipeline`, `omnixan.doctor` | `build_baseline_quantum_plan()` como baseline mínima |
| `Servicio de Diseño de Circuito` | `circuit-design-service` | `nuclear` | `publico futuro` | `QuantumMission`, `QuantumExecutionPlan` | `QuantumCircuitArtifact` | `partial` | `quantum_algorithm_module`, `quantum_ml_module`, stack cuántico opcional | piezas reales, sin adaptador canónico |
| `Servicio de Optimización Imperial` | `optimization-service` | `nuclear` | `publico futuro` | `QuantumCircuitArtifact`, `QuantumExecutionPlan` | `QuantumCircuitArtifact` optimizado | `partial` | `quantum_circuit_optimizer_module`, `qiskit` | pieza real, sin contrato canonizado |
| `Servicio del Trono de Ejecución` | `execution-service` | `nuclear` | `publico futuro` | artefacto optimizado y plan | `QuantumExecutionRecord` | `partial` | `quantum_simulator_module`, `qiskit`, `qiskit_aer`, `cirq`, `pennylane` | simulación real, sin record canónico |
| `Servicio de Corrección y Mitigación` | `mitigation-service` | `nuclear` | `publico futuro` | artefacto, plan y resultado de ejecución | `QuantumMitigationRecord` | `partial` | `quantum_error_correction_module`, `fault_mitigation_module`, `qiskit` | mitigación real, integración parcial |
| `Servicio de Juicio y Observación` | `observation-service` | `nuclear` | `publico ahora` | estado del entorno, estado de módulos, resultados de pipeline | `QuantumPipelineReport` o diagnóstico operativo | `supported` | `omnixan.doctor`, `omnixan.validate` | soporte real y CLI actual |
| `Servicio de Continuidad Imperial` | `continuity-service` | `auxiliar` | `publico ahora` | intención de despliegue, topología, salud de nodos | estado de continuidad, balanceo y redundancia | `partial` | `load_balancing_module`, `redundant_deployment_module` | soporte real, aún no unido a misión cuántica |
| `Servicio de Resolución de Capacidades` | `capability-service` | `interno` | `no publico` | entorno, dependencias instaladas, restricciones de misión | capacidades efectivas y degradación permitida | `conceptual` | `omnixan.doctor`, `omnixan.quantum_pipeline` | rol implícito, no formalizado |
| `Servicio de Adaptación de Contratos` | `contract-adapter-service` | `interno` | `no publico` | resultados propios de módulos cuánticos | contratos canonicos de pipeline | `conceptual` | `omnixan.quantum_pipeline`, módulos cuánticos reales | vacío identificado en el Día 17 |

## Contratos de alto nivel

### `mission-service`

- Input: solicitud de misión con objetivo, restricciones y backend preferido.
- Output: `QuantumMission`.
- Estado: `conceptual`.
- Nota: debe ser la puerta de entrada oficial del producto.

### `planning-service`

- Input: `QuantumMission` y capacidades del entorno.
- Output: `QuantumExecutionPlan`.
- Estado: `conceptual`.
- Nota: hoy solo existe como baseline fija; todavía no decide realmente según el entorno.

### `circuit-design-service`

- Input: `QuantumMission`, `QuantumExecutionPlan`.
- Output: `QuantumCircuitArtifact`.
- Estado: `partial`.
- Nota: hoy el diseño vive fragmentado entre algoritmos y QML.

### `optimization-service`

- Input: `QuantumCircuitArtifact`, `QuantumExecutionPlan`.
- Output: `QuantumCircuitArtifact` optimizado.
- Estado: `partial`.
- Nota: ya existe lógica útil, pero no recibe ni devuelve el contrato canónico.

### `execution-service`

- Input: artefacto optimizado, plan y modo de backend.
- Output: `QuantumExecutionRecord`.
- Estado: `partial`.
- Nota: la simulación local es real; la formalización del record aún falta.

### `mitigation-service`

- Input: plan, artefacto y resultado de ejecución.
- Output: `QuantumMitigationRecord`.
- Estado: `partial`.
- Nota: la pieza técnica existe, pero la interfaz de pipeline aún no.

### `observation-service`

- Input: estado del entorno, salud de módulos y resultados de pipeline.
- Output: reporte operativo o `QuantumPipelineReport`.
- Estado: `supported`.
- Nota: es la superficie pública más madura hoy.

### `continuity-service`

- Input: configuración de despliegue, backends, regiones y salud.
- Output: balanceo, redundancia y continuidad.
- Estado: `partial`.
- Nota: es público hoy, pero como servicio auxiliar de plataforma.

## Politica de exposicion publica

### Publicos ahora

- `observation-service`
- `continuity-service`

Son publicos porque ya tienen entrypoints estables o capacidades operativas
claras sin exigir que toda la misión cuántica exista de punta a punta.

### Publicos despues

- `mission-service`
- `planning-service`
- `circuit-design-service`
- `optimization-service`
- `execution-service`
- `mitigation-service`

No conviene exponerlos todavia como CLI oficial independiente hasta que exista
una mision cuantica ejecutable con contratos canonicos de extremo a extremo.

### No publicos

- `capability-service`
- `contract-adapter-service`

Estos servicios son arquitectura interna. Deben existir para hacer funcionar el
producto, pero exponerlos antes de tiempo haria mas confusa la superficie
publica.

## Mapeo a CLI y documentacion

| Servicio | Superficie CLI actual | Rol en documentación |
| --- | --- | --- |
| `observation-service` | `python -m omnixan doctor`, `python -m omnixan validate` | diagnóstico, soporte y salud del repo |
| `continuity-service` | `python -m omnixan load-balancing`, `python -m omnixan redundant-deployment` | continuidad imperial y soporte de plataforma |
| servicios nucleares cuanticos | sin CLI estable todavía | arquitectura del producto y ruta feliz futura |
| servicios internos | sin CLI y no documentados como entrada pública | arquitectura interna y backlog técnico |

## Decisiones del Dia 18

- La ruta feliz del producto sigue siendo nuclearmente cuantica.
- La observabilidad y la continuidad quedan como servicios publicos reales desde
  ya, porque son las capas mas maduras del repo.
- Los servicios cuanticos nucleares quedan definidos, pero todavia no se
  exponen como comandos estables separados.
- La capa de capacidades y adaptadores se reconoce como arquitectura interna y
  no como cara publica del producto.

## Uso del service map

Este mapa ya puede usarse para:

- arquitectura: decidir fronteras y responsabilidades
- CLI: decidir que se expone hoy y que se reserva
- documentación: explicar el producto sin mezclar modulos con servicios

## Criterio de cierre del Dia 18

OMNIXAN ya tiene un service map oficial que distingue:

- que servicios son el producto nuclear
- que servicios sostienen la plataforma
- que servicios deben permanecer internos por ahora
