# Lenguaje de Servicios

## Propósito

Este documento define el lenguaje oficial de servicios Amarr en OMNIXAN. Su
meta es evitar naming ornamental y convertir la narrativa en una taxonomía de
servicios con límites, autoridad y responsabilidad claras.

## Qué cuenta como servicio

Un servicio Amarr debe cumplir estas condiciones:

- tener una misión técnica explícita
- aceptar una entrada reconocible
- producir una salida o estado verificable
- exponer una frontera clara con otros servicios
- poder reportar salud, degradación y soporte real

Un módulo no se vuelve servicio solo por existir en el repo. Si no tiene un rol
coherente dentro de la cadena cuántica o de la continuidad de servicio, queda
fuera del canon público.

## Gramática de naming

### Nombre canónico

- Formato recomendado: `Servicio de <Rol Imperial>`.
- Debe sonar estable, no fantasioso.
- Debe sugerir una responsabilidad operativa concreta.

### Alias técnico

- Cada servicio debe tener además un alias técnico corto para docs, CLI y
  código futuro.
- El alias técnico debe ser más directo que el nombre narrativo.

### Reglas

- Si un nombre narrativo complica la comprensión, gana el alias técnico.
- Los nombres deben reflejar frontera y autoridad, no solo estilo.
- Un servicio no debe mezclar más de una responsabilidad principal.
- Los nombres deben sobrevivir fuera del lore.

## Fronteras de servicio

### Servicios de misión

- Traducen intención a una solicitud ejecutable.
- No ejecutan circuitos ni optimizan backends por sí mismos.

### Servicios de construcción

- Preparan artefactos cuánticos o planes derivados.
- No deben asumir control de simulación o continuidad.

### Servicios de ejecución

- Corren o simulan la misión sobre un backend o runtime.
- No deberían decidir toda la política del sistema.

### Servicios de continuidad

- Mantienen disponibilidad, redundancia, balanceo y failover.
- No reemplazan la lógica cuántica principal.

### Servicios de juicio

- Observan salud, métricas, soporte y degradación.
- No maquillan la realidad operativa.

## Catálogo inicial de servicios Amarr

| Servicio canónico | Alias técnico | Misión técnica | Frontera principal | Estado actual |
| --- | --- | --- | --- | --- |
| `Servicio de Misión Cuántica` | `mission-service` | Recibir la solicitud de negocio o experimento y convertirla en una misión formal | Entrada al flujo oficial | conceptual |
| `Servicio de Decreto Operativo` | `planning-service` | Traducir la misión a un plan validado, política y restricciones | Planeación y validación previa | conceptual |
| `Servicio de Diseño de Circuito` | `circuit-design-service` | Construir o seleccionar el artefacto cuántico a ejecutar | Generación o selección de circuito | partial |
| `Servicio de Optimización Imperial` | `optimization-service` | Ajustar el circuito al backend o simulador objetivo | Optimización y reducción de coste | partial |
| `Servicio del Trono de Ejecución` | `execution-service` | Ejecutar o simular la misión sobre runtime disponible | Simulación o ejecución efectiva | partial |
| `Servicio de Corrección y Mitigación` | `mitigation-service` | Preservar fidelidad, corregir o mitigar errores | Integridad cuántica | partial |
| `Servicio de Continuidad Imperial` | `continuity-service` | Mantener balanceo, redundancia y continuidad operativa | Resiliencia y failover | partial |
| `Servicio de Juicio y Observación` | `observation-service` | Exponer métricas, soporte, warnings y degradación | Diagnóstico y observabilidad | supported |

## Mapeo inicial al repo actual

### Servicio de Misión Cuántica

- Estado: conceptual.
- Rol esperado: recibir una solicitud cuántica canónica.
- Hueco actual: no existe todavía una entidad pública única de misión.

### Servicio de Decreto Operativo

- Estado: conceptual.
- Rol esperado: traducir misión a plan, políticas y selección de ruta.
- Hueco actual: no existe un planificador unificado entre bloques.

### Servicio de Diseño de Circuito

- Estado: partial.
- Módulos candidatos:
  - `omnixan.quantum_cloud_architecture.quantum_algorithm_module`
  - `omnixan.quantum_cloud_architecture.quantum_ml_module`
- Observación: hay lógica útil, pero todavía no hay una interfaz canónica de
  diseño separada del resto del pipeline.

### Servicio de Optimización Imperial

- Estado: partial.
- Módulos candidatos:
  - `omnixan.quantum_cloud_architecture.quantum_circuit_optimizer_module`
- Observación: existe capacidad real, pero aún no está claramente encajada en
  una cadena de servicios formal.

### Servicio del Trono de Ejecución

- Estado: partial.
- Módulos candidatos:
  - `omnixan.quantum_cloud_architecture.quantum_simulator_module`
- Observación: hoy mezcla simulación y ejecución posible según el entorno; falta
  separar mejor backend, simulación y políticas.

### Servicio de Corrección y Mitigación

- Estado: partial.
- Módulos candidatos:
  - `omnixan.quantum_cloud_architecture.quantum_error_correction_module`
  - `omnixan.virtualized_cluster.fault_mitigation_module`
- Observación: hay piezas valiosas, pero todavía no existe una narrativa única
  de integridad entre capa cuántica y continuidad del sistema.

### Servicio de Continuidad Imperial

- Estado: partial.
- Módulos candidatos:
  - `omnixan.carbon_based_quantum_cloud.load_balancing_module`
  - `omnixan.carbon_based_quantum_cloud.redundant_deployment_module`
- Observación: es el servicio no cuántico más tangible del canon actual.

### Servicio de Juicio y Observación

- Estado: supported.
- Módulos candidatos:
  - `omnixan.doctor`
  - `omnixan.validate`
- Observación: es la capa más madura del producto interno porque ya distingue
  baseline, degradación, soporte y errores reales.

## Qué no entra todavía al canon público

- Módulos que solo puedan justificarse como experimento aislado.
- Bloques que no tengan una relación clara con la misión cuántica o con la
  continuidad imperial.
- Términos Amarr que no puedan mapearse a contrato, métrica o responsabilidad.

## Reglas para los siguientes días

- Día 14 debe usar este catálogo para mapear bloques contra visión.
- Día 16 debe convertir estos servicios en una cadena cuántica oficial.
- Día 18 debe derivar de aquí un service map con contratos más explícitos.

## Criterio de cierre del Día 13

OMNIXAN debe poder nombrar sus servicios principales sin caer en ambigüedad,
y cada nombre debe apuntar a una responsabilidad técnica real o a un hueco
claramente reconocido.
