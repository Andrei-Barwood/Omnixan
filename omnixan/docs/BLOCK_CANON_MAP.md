# Mapa Canónico de Bloques

## Propósito

Este documento responde la pregunta del Día 14: qué papel cumple cada bloque
del repo dentro de la visión canónica de OMNIXAN, qué tan cerca está de la ruta
feliz oficial y qué decisión de producto conviene tomar hoy.

## Criterios de evaluación

- alineación con la cadena cuántica oficial
- relación con los servicios Amarr ya definidos
- grado de validación real en baseline
- cercanía a la ruta feliz del producto
- riesgo de dispersión si el bloque se mantiene sin recorte

## Hallazgos principales

- El bloque más cercano al corazón del producto es
  `quantum_cloud_architecture`, pero todavía depende de extras opcionales para
  expresar completamente la promesa del repo.
- Las capas más maduras hoy no son la misión cuántica ni el planeamiento, sino
  la continuidad de servicio y el juicio operativo.
- `carbon_based_quantum_cloud` ya actúa como columna de continuidad imperial.
- Varios bloques laterales tienen valor, pero todavía no justifican pertenecer
  al canon público de la ruta feliz.

## Matriz bloque por bloque

| Bloque | Rol en la visión | Servicios Amarr relacionados | Estado validado hoy | Cercanía a la ruta feliz | Decisión actual | Próxima acción |
| --- | --- | --- | --- | --- | --- | --- |
| `quantum_cloud_architecture` | Núcleo del producto | `circuit-design-service`, `optimization-service`, `execution-service`, `mitigation-service` | degraded | directa | `core` | Priorizar integración de pipeline en Días 16 y 17 |
| `carbon_based_quantum_cloud` | Continuidad imperial del sistema | `continuity-service` | supported | directa como soporte | `core` | Mantener y alinear con la misión cuántica oficial |
| `virtualized_cluster` | Recuperación, checkpoints y mitigación sistémica | `mitigation-service`, `continuity-service` | supported | media | `auxiliary` | Acotarlo a integridad y recuperación, sin expansión lateral |
| `in_memory_computing_cloud` | Soporte híbrido para edge, fog y AI cerca del runtime | `execution-service`, soporte futuro a misión | supported/degraded | indirecta | `experimental` | Mantener fuera del canon principal hasta definir orquestación híbrida |
| `supercomputing_interconnect_cloud` | Aceleración numérica y GPU para ejecución | `execution-service` | supported/degraded | indirecta | `experimental` | Mantener como capa backend, no como cara pública del producto |
| `edge_computing_network` | Coherencia y proximidad de datos | soporte futuro a planeamiento y ejecución distribuida | supported | baja | `auxiliary-future` | No promoverlo aún como bloque central del canon |
| `heterogenous_computing_group` | Paralelismo e infraestructura heterogénea de bajo nivel | soporte futuro a ejecución híbrida | supported/historical | baja | `historical` | Congelar su exposición pública hasta el Día 21 |

## Lectura por bloque

### `quantum_cloud_architecture`

- Es el bloque que mejor encarna la promesa de OMNIXAN.
- Ya contiene piezas reales para diseño, optimización, simulación y corrección.
- Su problema no es de identidad sino de integración y de dependencia de stack.
- Decisión: tratarlo como centro del producto, aunque el baseline siga
  degradado cuando faltan extras cuánticos.

### `carbon_based_quantum_cloud`

- Hoy es el bloque más operativo del canon público junto con la observación.
- Balanceo y despliegue redundante ya sostienen la identidad de continuidad
  imperial.
- Decisión: mantenerlo como bloque `core`, pero como soporte del flujo cuántico,
  no como producto separado.

### `virtualized_cluster`

- Tiene sentido como bloque de mitigación y recuperación.
- `fault_mitigation_module` encaja bien con la idea de preservar integridad
  operacional.
- Decisión: conservarlo como auxiliar canónico y evitar dispersarlo hacia
  interfaces o hardware poco integrados.

### `in_memory_computing_cloud`

- Aporta una idea interesante de orquestación híbrida cercana al runtime.
- Fog y Edge AI son útiles, pero todavía no son parte de la ruta feliz oficial.
- Decisión: mantenerlo como experimental hasta definir claramente el papel de
  edge y AI en el producto.

### `supercomputing_interconnect_cloud`

- Tiene valor como capa de aceleración, no como narrativa principal del producto.
- `tensor_core_module` y los guards de CUDA ya están bien parados técnicamente.
- Decisión: mantenerlo como bloque experimental de backend.

### `edge_computing_network`

- Su valor es más de soporte futuro que de corazón del canon actual.
- La coherencia de caché puede ser relevante si la misión cuántica necesita
  coordinación distribuida más rica.
- Decisión: mantenerlo visible, pero fuera del centro narrativo por ahora.

### `heterogenous_computing_group`

- Contiene ideas de infraestructura útiles, pero aún no justificadas por la
  ruta feliz del producto.
- `non_blocking_module` está sano, pero no basta para darle centralidad.
- Decisión: dejarlo en estado `historical` hasta reintroducirlo con una
  estrategia híbrida explícita.

## Canon de producto resultante

### Bloques `core`

- `quantum_cloud_architecture`
- `carbon_based_quantum_cloud`

### Bloques `auxiliary`

- `virtualized_cluster`
- `edge_computing_network` como auxiliar futuro, no central

### Bloques `experimental`

- `in_memory_computing_cloud`
- `supercomputing_interconnect_cloud`

### Bloques `historical`

- `heterogenous_computing_group`

## Implicaciones para los siguientes días

- Día 15 debe usar esta matriz para clasificar módulos individuales.
- Día 16 debe concentrarse en `quantum_cloud_architecture` como centro del
  flujo canónico.
- Día 21 decidirá si `in_memory_computing_cloud`,
  `supercomputing_interconnect_cloud` y `edge_computing_network` vuelven al
  centro o permanecen como capas de soporte.

## Criterio de cierre del Día 14

Cada bloque del repo ya tiene una justificación explícita dentro del producto o
una decisión clara de contención.
