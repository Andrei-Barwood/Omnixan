# Tareas Diarias

Esta hoja de ruta ya no cubre solo saneamiento tecnico. A partir de esta ronda
se usa como documento operativo para llevar OMNIXAN desde "repo verificable" a
"plataforma con identidad, narrativa y cadena cuantica coherente".

## Principios operativos

- Cada dia debe cerrar con un artefacto concreto: documento, prueba, demo,
  refactor o modulo alineado con la vision.
- Cada cambio debe dejar claro si algo queda `core`, `experimental`,
  `historical` o `degraded`.
- La narrativa Amarr debe traducirse a arquitectura y servicios reales, no
  solo a nombres esteticos.
- La ruta feliz oficial del repo debe mantenerse ejecutable durante toda la
  campaña.
- Cada cierre diario debe validar al menos `doctor`, baseline local y el area
  tocada ese dia.

## Fase 1 cerrada: saneamiento base

### Día 1: Stack distribuido mínimo

- Objetivo: validar `ray` y `dask[distributed]` y dejar el stack distribuido
  como extra opcional soportado.
- Estado: completado.

### Día 2: Consolidación de packaging

- Objetivo: consolidar `pyproject.toml` como fuente de verdad de packaging.
- Estado: completado.

### Día 3: Entry points y CLIs

- Objetivo: definir una CLI oficial minima y corregir entry points rotos.
- Estado: completado.

### Día 4: Cobertura de bloques core

- Objetivo: cubrir import, inicializacion y una operacion simple por bloque
  principal no cuantico.
- Estado: completado.

### Día 5: Módulos con dependencias pesadas

- Objetivo: volver opcionales los backends GPU y runtimes pesados donde
  corresponda.
- Estado: completado.

### Día 6: Calidad de API y consistencia

- Objetivo: alinear la superficie publica de los modulos core.
- Estado: completado.

### Día 7: Documentación operativa

- Objetivo: actualizar READMEs por bloque y rutas felices reales.
- Estado: completado.

### Día 8: Observabilidad y diagnósticos

- Objetivo: reforzar `omnixan.doctor` y separar errores de entorno de errores
  de codigo.
- Estado: completado.

### Día 9: Automatización de validación

- Objetivo: dejar una rutina reproducible de checks y base de CI.
- Estado: completado.

### Día 10: Endurecimiento final

- Objetivo: limpiar warnings evitables y preparar release interna.
- Estado: completado.

## Fase 2 activa: propósito, dominio y experiencia

### Día 11: Canon del producto

- Objetivo: definir que es OMNIXAN en una frase, para quien existe y que
  problema resuelve.
- Diagnóstico: revisar si el repo actual se comporta como plataforma,
  middleware, sandbox de investigacion o mezcla desordenada.
- Entregables: `VISION.md` con definicion oficial, promesa de valor y no
  objetivos explicitos.
- Validación: coherencia entre `VISION.md`, `README.md` y estado real del repo.
- Criterio de cierre: poder explicar OMNIXAN en menos de 60 segundos sin
  contradicciones.
- Estado: completado en `VISION.md`.

### Día 12: Traducción Amarr

- Objetivo: convertir la narrativa Amarr en principios tecnicos y de diseño.
- Diagnóstico: enumerar conceptos narrativos como orden, liturgia, autoridad,
  resiliencia y servicio imperial, y mapearlos a decisiones concretas.
- Entregables: documento de principios Amarr y glosario operativo inicial.
- Validación: cada principio debe poder traducirse a comportamiento de sistema,
  no solo a estilo visual.
- Criterio de cierre: lista canonica de principios que guien arquitectura,
  naming y observabilidad.
- Estado: completado en `AMARR_PRINCIPLES.md`.

### Día 13: Lenguaje y catálogo de servicios

- Objetivo: definir el lenguaje oficial de "servicios Amarr".
- Diagnóstico: revisar naming inconsistente en bloques y modulos actuales.
- Entregables: primer catalogo de servicios con nombres, rol, fronteras y tono.
- Validación: evitar nombres intercambiables o demasiado genericos.
- Criterio de cierre: cada servicio debe tener una responsabilidad unica y
  reconocible.
- Estado: iniciado en `SERVICE_LANGUAGE.md`.

### Día 14: Mapa del repo contra la visión

- Objetivo: revisar bloque por bloque y responder si aporta a la vision final.
- Diagnóstico: relacionar cada bloque con un rol del producto canónico.
- Entregables: matriz bloque -> funcion -> estado -> decision.
- Validación: no dejar bloques "presentes sin razon".
- Criterio de cierre: poder justificar por que cada bloque sigue existiendo.

### Día 15: Clasificación operativa del código

- Objetivo: etiquetar modulos como `core`, `experimental`, `historical` o
  `placeholder`.
- Diagnóstico: identificar modulos que hoy solo agregan ruido o deuda.
- Entregables: clasificacion documentada y marcada en docs.
- Validación: al menos un criterio claro por categoria.
- Criterio de cierre: cualquier persona nueva puede distinguir que se puede usar
  hoy y que no.

### Día 16: Cadena cuántica canónica

- Objetivo: diseñar la pipeline cuantica oficial de OMNIXAN.
- Diagnóstico: verificar si ya existe un camino continuo entre problema,
  circuito, optimizacion, simulacion, correccion y reporte.
- Entregables: flujo canonico de extremo a extremo y sus interfaces minimas.
- Validación: la pipeline debe poder contarse como historia y como arquitectura.
- Criterio de cierre: existe un flujo cuantico oficial, aunque sea baseline.

### Día 17: Auditoría de huecos cuánticos

- Objetivo: detectar que partes de la pipeline cuantica ya existen y cuales
  siguen fragmentadas o ausentes.
- Diagnóstico: inspeccion modulo por modulo del bloque cuantico contra el flujo
  canonico.
- Entregables: tabla de cobertura cuantica y backlog de vacios reales.
- Validación: distinguir placeholders de piezas realmente utilizables.
- Criterio de cierre: backlog cuantico priorizado por impacto en la pipeline.

### Día 18: Catálogo oficial de servicios Amarr

- Objetivo: convertir la pipeline y la plataforma en servicios explicitos.
- Diagnóstico: decidir que servicios son nucleares, cuales son auxiliares y
  cuales no deben exponerse publicamente.
- Entregables: mapa oficial de servicios Amarr con contratos de alto nivel.
- Validación: cada servicio debe tener input, output, estado y dependencias.
- Criterio de cierre: existe un service map que pueda usarse para arquitectura,
  CLI y documentacion.

### Día 19: Modelo de datos canónico

- Objetivo: unificar entidades publicas del producto.
- Diagnóstico: revisar modelos genericos o ambiguos entre modulos.
- Entregables: entidades canonicas como solicitud, job cuantico, plan,
  backend, politica, resultado y metrica.
- Validación: evitar duplicados semanticos entre bloques.
- Criterio de cierre: existe un lenguaje de datos comun para el flujo oficial.

### Día 20: Contratos de API

- Objetivo: formalizar requests, responses, errores y estados entre servicios.
- Diagnóstico: revisar envelopes inconsistentes y errores sin semantica de
  negocio.
- Entregables: contrato API comun y lista de adaptaciones necesarias.
- Validación: los modulos core deben poder integrarse sin glue ambiguo.
- Criterio de cierre: contrato publico minimo aceptado para la ruta feliz.

### Día 21: Orquestación híbrida

- Objetivo: decidir el papel real de CPU, edge, distribuido y GPU dentro de
  OMNIXAN.
- Diagnóstico: determinar si son soportes del flujo cuantico o productos
  paralelos sin integracion.
- Entregables: modelo de orquestacion hibrida y jerarquia de ejecucion.
- Validación: evitar que "hibrido" signifique simplemente "todo cabe".
- Criterio de cierre: el papel de cada stack queda acotado y documentado.

### Día 22: Backends cuánticos reales

- Objetivo: separar oficialmente simulacion local, simulacion con ruido,
  optimizacion hibrida y ejecucion sobre backend externo.
- Diagnóstico: auditar que partes del stack cuantico son reales hoy y cuales
  son wrappers incompletos.
- Entregables: matriz backend -> capacidad -> estado -> dependencia.
- Validación: cada modo de ejecucion debe tener condiciones de uso claras.
- Criterio de cierre: el repo deja de mezclar "simulacion" y "ejecucion real".

### Día 23: Observabilidad técnica

- Objetivo: definir metricas tecnicas canonicas de la plataforma.
- Diagnóstico: inventariar que metrica existe hoy y cual falta para el flujo
  oficial.
- Entregables: esquema de metricas tecnicas para latencia, disponibilidad,
  fidelidad, coste, degradacion y exito de job.
- Validación: cada metrica debe estar ligada a una decision operativa.
- Criterio de cierre: existe una base comun para reportes y diagnosticos.

### Día 24: Observabilidad narrativa

- Objetivo: traducir las metricas tecnicas a una capa Amarr util.
- Diagnóstico: decidir que terminologia narrativa aporta claridad y cual solo
  agrega ruido.
- Entregables: mapeo entre estados tecnicos y estados narrativos.
- Validación: el reporte narrativo no debe ocultar el estado tecnico real.
- Criterio de cierre: observabilidad dual, tecnica y narrativa, sin ambigüedad.

### Día 25: Demo canónica

- Objetivo: diseñar una demo que cuente la historia correcta del producto y que
  pueda ejecutarse hoy.
- Diagnóstico: elegir un caso de uso pequeño pero representativo.
- Entregables: demo oficial, guion de ejecucion y resultado esperado.
- Validación: cualquier colaborador deberia poder correrla de principio a fin.
- Criterio de cierre: existe una demo oficial de OMNIXAN.

### Día 26: CLI y experiencia de uso

- Objetivo: convertir la demo y los servicios oficiales en una experiencia CLI
  coherente.
- Diagnóstico: revisar fricciones entre comandos, salidas y nombres.
- Entregables: comandos alineados con la vision y documentacion de ruta feliz.
- Validación: el entrypoint principal debe contar la historia correcta.
- Criterio de cierre: `python -m omnixan` refleja el producto, no solo el repo.

### Día 27: Documentación maestra

- Objetivo: consolidar vision, service map, flujo principal y limites actuales.
- Diagnóstico: detectar dispersion entre README, docs tecnicos y estado real.
- Entregables: documentacion maestra navegable y consistente.
- Validación: un lector nuevo debe poder entender producto, arquitectura y
  soporte real sin leer todo el repo.
- Criterio de cierre: existe un paquete documental canonico del proyecto.

### Día 28: Riesgos y recorte de alcance

- Objetivo: priorizar regresiones reales y recortar lo que no aporta al alpha.
- Diagnóstico: revisar deuda conceptual, amplitud excesiva y modulos
  irrelevantes para la ruta feliz.
- Entregables: lista priorizada de riesgos y decisiones de congelamiento.
- Validación: cada riesgo debe tener impacto y accion recomendada.
- Criterio de cierre: el backlog deja de ser una lista plana y pasa a ser una
  lista priorizada.

### Día 29: Soporte por bloque y servicio

- Objetivo: publicar estado de soporte por bloque, servicio y dependencia.
- Diagnóstico: comprobar que el soporte documentado coincide con la realidad.
- Entregables: matriz de soporte revisada para producto y plataforma.
- Validación: no prometer como soportado nada que solo importe o simule.
- Criterio de cierre: soporte real y expectativas quedan alineados.

### Día 30: Alpha interna

- Objetivo: cerrar una version interna con identidad, ruta feliz y backlog
  siguiente.
- Diagnóstico: revisar si ya existe un producto reconocible y no solo un repo
  saneado.
- Entregables: release alpha interna, changelog, demo oficial y plan de ronda
  siguiente.
- Validación: criterio go/no-go para seguir profundizando stacks opcionales.
- Criterio de cierre: OMNIXAN tiene una primera forma de producto interno.

## Rutina diaria recomendada

- Revisar el entregable pendiente del dia anterior.
- Ejecutar `python -m omnixan doctor --json`.
- Ejecutar `./scripts/ci_local.sh baseline`.
- Tocar solo el frente del dia, salvo regressiones directas.
- Documentar decisiones y cambios de clasificacion en la misma jornada.
- Cerrar con un resumen de estado, riesgos y siguiente paso.

## Dependencias de la campaña

- Los Dias 11 al 15 definen lenguaje, vision y recorte de alcance.
- Los Dias 16 al 20 convierten esa vision en pipeline, datos y APIs.
- Los Dias 21 al 26 aterrizan orquestacion, observabilidad y experiencia real.
- Los Dias 27 al 30 consolidan documentacion, soporte y alpha interna.
