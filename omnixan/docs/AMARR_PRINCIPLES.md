# Principios Amarr

## Propósito

Este documento traduce la inspiración Amarr de OMNIXAN a decisiones técnicas y
de diseño concretas. La narrativa no se usa como adorno: debe orientar naming,
arquitectura, observabilidad, jerarquía de servicios y degradación operativa.

## Supuesto de trabajo

OMNIXAN toma la inspiración Amarr como un lenguaje de infraestructura imperial:
ordenada, jerárquica, resiliente, ceremonial y sostenida por continuidad de
servicio. Ese imaginario guía el producto aunque la implementación siga siendo
una plataforma Python verificable y no una reproducción literal de un universo
externo.

## Principios técnicos canonicos

### 1. Soberanía de servicio

- Cada servicio debe tener una autoridad clara y una frontera definida.
- No se deben exponer módulos ambiguos que hagan "de todo un poco".
- Las rutas felices y los contratos públicos deben tener un responsable técnico
  identificable.

### 2. Liturgia operativa

- Las operaciones importantes deben seguir secuencias canónicas y repetibles.
- Inicialización, ejecución, shutdown y diagnóstico forman parte de un rito
  técnico estable, no de scripts improvisados.
- La CLI, la documentación y los tests deben narrar el mismo flujo.

### 3. Continuidad imperial

- Un servicio noble no colapsa por la ausencia de un backend opcional.
- La degradación debe ser explícita, digna y observable.
- Redundancia, failover y balanceo no son accesorios; son rasgos de identidad.

### 4. Jerarquía legible

- La arquitectura debe dejar claro qué capa decide, qué capa ejecuta y qué capa
  observa.
- El repositorio debe reflejar niveles de autoridad: visión, servicios,
  contratos, módulos y runtimes.
- La documentación debe distinguir soporte oficial, experimental y herencia
  histórica.

### 5. Belleza operativa

- La narrativa Amarr debe aumentar comprensión, no ocultar problemas reales.
- Cada término narrativo debe tener un equivalente técnico explícito.
- Los reportes pueden tener tono imperial, pero nunca deben maquillar errores.

### 6. Fidelidad antes que amplitud

- Es preferible una cadena cuántica canónica pequeña y verdadera antes que una
  colección extensa de módulos sin integración.
- El baseline soportado manda sobre la promesa teórica.
- Cada nuevo bloque debe justificar cómo sirve a la misión cuántica oficial.

### 7. Juicio verificable

- Toda afirmación de salud o soporte debe poder validarse con `doctor`, tests o
  smoke suites.
- El lenguaje narrativo de observabilidad debe mapear a métricas técnicas.
- No hay "estado imperial" válido si no existe un estado técnico equivalente.

## Decisiones de arquitectura derivadas

- Los servicios oficiales deberán tener nombre, rol, input, output, métricas,
  modo degradado y dependencia principal.
- La cadena cuántica debe tratarse como una misión formal con etapas
  reconocibles, no como una suma casual de módulos.
- Los bloques auxiliares como redundancia, edge, distribuido y GPU deben
  definirse como soportes del flujo principal o quedar expresamente fuera del
  canon.
- La clasificación `core`, `experimental`, `historical` y `degraded` forma
  parte de la disciplina Amarr porque establece jerarquía y verdad operativa.

## Reglas de naming

- Usar términos narrativos solo cuando puedan sostener una función técnica.
- Preferir nombres de servicio que sugieran autoridad, estabilidad o misión.
- Evitar nombres místicos sin semántica operativa.
- Cuando exista duda, el nombre técnico gana y la capa narrativa se agrega como
  alias documental o de UX.

## Reglas de observabilidad

- Todo estado narrativo debe incluir o derivarse de un estado técnico.
- Los warnings deben describir degradación real y dependencias faltantes.
- La capa Amarr podrá expresar conceptos como juicio, pureza, continuidad o
  desviación, pero siempre anclados a datos verificables.

## Glosario operativo inicial

| Término Amarr | Significado técnico | Uso recomendado |
| --- | --- | --- |
| `Misión` | Solicitud cuántica u objetivo de ejecución | Entrada principal del flujo oficial |
| `Decreto` | Plan validado o política de ejecución | Decisión formal antes de ejecutar |
| `Liturgia` | Secuencia canónica de pasos de servicio | Flujo estable entre servicios |
| `Trono de Ejecución` | Backend, simulador o runtime seleccionado | Capa donde corre la misión |
| `Optimización Imperial` | Adaptación del circuito y reducción de coste | Servicio de ajuste antes de ejecutar |
| `Continuidad Imperial` | Redundancia, balanceo, failover y resiliencia | Servicios que sostienen disponibilidad |
| `Juicio` | Diagnóstico, validación y observabilidad | Reportes, doctor y métricas |
| `Canon` | Ruta feliz oficialmente soportada | Documentación, CLI y soporte real |
| `Desviación` | Estado degradado o fuera de baseline | Warnings y modos parciales |
| `Jerarquía` | Nivel de autoridad entre capas del sistema | Arquitectura, ownership y soporte |

## Criterios para aceptar nuevos términos

- Deben describir un comportamiento técnico reconocible.
- Deben poder documentarse sin depender de conocer el lore.
- Deben mejorar la experiencia de uso o la legibilidad del sistema.
- Si generan más confusión que claridad, no entran al canon.

## Salida esperada para la siguiente fase

Los Dias 13 a 18 deberán reutilizar estos principios para construir:

- el lenguaje oficial de servicios
- el mapa de servicios Amarr
- la cadena cuántica canónica
- el modelo de datos y los contratos que sostienen ese canon
