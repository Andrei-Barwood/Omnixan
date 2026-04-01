# Visión de Producto

## Definición canónica

OMNIXAN es una plataforma de orquestación cuántica inspirada en una estética de
infraestructura imperial: coordina servicios híbridos de simulación,
optimización, corrección, ejecución y resiliencia alrededor de una cadena
cuántica principal, con narrativa Amarr y operación verificable.

## Problema que resuelve

El repo actual contiene muchas capacidades prometedoras, pero dispersas. La
vision de producto busca convertir ese conjunto de bloques en un sistema con:

- una ruta feliz oficial
- servicios reconocibles y con responsabilidad clara
- observabilidad técnica y narrativa
- una separación honesta entre baseline soportado y stacks opcionales

## Para quién existe

- Para quienes quieran experimentar con una plataforma de servicios cuánticos
  con fuerte identidad narrativa.
- Para desarrollo interno de una arquitectura híbrida que mezcle simulación,
  optimización, resiliencia y orquestación.
- Para demos y prototipos donde la computación cuántica no sea solo un módulo
  aislado, sino una cadena de servicios coordinados.

## No es, por ahora

- Un backend productivo de hardware cuántico en producción.
- Un framework generalista que intente cubrir cualquier necesidad científica.
- Un simple repositorio de módulos independientes sin relación de producto.

## Principios canónicos iniciales

- Jerarquía clara: cada servicio debe tener rol, frontera y autoridad definida.
- Resiliencia ceremonial: degradar con dignidad y mensajes claros, no romper al
  importar o al arrancar.
- Belleza operativa: la narrativa Amarr debe sumar claridad, no ocultar estado.
- Ruta feliz primero: la experiencia oficial importa más que la amplitud teórica.
- Imperio de servicios: los módulos deben converger a servicios coordinados, no
  a demos aisladas.

El desarrollo detallado de estos principios quedó aterrizado en
`AMARR_PRINCIPLES.md`.

## Cadena cuántica oficial propuesta

1. Definir una misión o solicitud cuántica.
2. Transformarla en un plan de ejecución.
3. Construir o seleccionar un circuito.
4. Optimizar el circuito para un backend o simulador.
5. Simular, corregir o ejecutar según el entorno disponible.
6. Reportar resultado, estado del servicio y métricas.

La definición canónica y los contratos mínimos de esta cadena viven ahora en
`QUANTUM_PIPELINE.md` y `omnixan.quantum_pipeline`.

## Servicios Amarr iniciales

- Servicio de Misión Cuántica: recibe la solicitud y la traduce a objetivo.
- Servicio de Diseño de Circuito: prepara el artefacto cuántico ejecutable.
- Servicio de Optimización Imperial: reduce coste y adapta al backend.
- Servicio de Simulación o Trono de Ejecución: ejecuta o simula la misión.
- Servicio de Corrección y Mitigación: protege integridad y fidelidad.
- Servicio de Continuidad Imperial: aporta redundancia, balanceo y failover.
- Servicio de Juicio y Observación: expone métricas, estado y degradación.

El lenguaje formal y el primer catálogo de estos servicios quedaron definidos en
`SERVICE_LANGUAGE.md`. El mapa oficial con clasificación nuclear, auxiliar e
interna quedó fijado en `SERVICE_MAP.md`.

## Estado real al inicio de esta fase

- El baseline del repo ya es import-safe, testeable y documentado.
- Existe una base de CLI, doctor, validación local y soporte por bloque.
- La identidad de producto todavía es parcial: los bloques existen, pero aún no
  convergen del todo a un único flujo cuantico canónico.

## Criterio de éxito de esta fase

Al final de la campaña, una persona nueva debe poder responder tres preguntas
sin leer todo el código:

- qué es OMNIXAN
- cuál es su ruta feliz oficial
- qué servicios Amarr están realmente soportados hoy
