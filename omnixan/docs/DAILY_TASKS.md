# Tareas Diarias

Esta hoja de ruta prioriza llevar OMNIXAN desde "repo utilizable" a
"workspace reproducible y verificable" sin abrir demasiados frentes a la vez.

## Día 1: Stack distribuido mínimo

- Instalar y validar `ray` y `dask` en un entorno limpio.
- Ampliar `omnixan.doctor` con chequeos de imports distribuidos clave.
- Añadir una smoke suite opcional para módulos distribuidos comparables a la cuántica.

## Día 2: Consolidación de packaging

- Revisar y reconciliar `pyproject.toml`, `omnixan/setup.py` y `omnixan/requirements.txt`.
- Marcar claramente qué archivos son históricos y cuál es la fuente de verdad.
- Evitar versiones incompatibles o dependencias declaradas pero no usadas.

## Día 3: Entry points y CLIs

- Detectar módulos con `__main__.py`, `main()` o scripts incompletos.
- Corregir imports relativos rotos y salidas inconsistentes.
- Documentar qué comandos están soportados oficialmente.

## Día 4: Cobertura de bloques core

- Añadir tests mínimos por bloque:
  `carbon_based_quantum_cloud`, `edge_computing_network`,
  `heterogenous_computing_group`, `in_memory_computing_cloud`,
  `supercomputing_interconnect_cloud` y `virtualized_cluster`.
- Verificar al menos import, inicialización y una operación simple por módulo crítico.

## Día 5: Módulos con dependencias pesadas

- Auditar módulos CUDA, GPU, TensorFlow, PyTorch y backends opcionales.
- Convertir imports duros en imports opcionales cuando corresponda.
- Hacer que fallen con mensajes claros y no al importar el paquete.

## Día 6: Calidad de API y consistencia

- Unificar nombres de métodos como `initialize`, `execute`, `shutdown`, `get_metrics`.
- Corregir respuestas inconsistentes entre módulos.
- Añadir tipos y validaciones donde la interfaz pública sea ambigua.

## Día 7: Documentación operativa

- Completar READMEs faltantes o desactualizados por bloque.
- Añadir ejemplos pequeños que sí corran con el estado actual del repo.
- Documentar rutas felices y dependencias opcionales por módulo.

## Día 8: Observabilidad y diagnósticos

- Mejorar `omnixan.doctor` para reportar warnings, conflictos de paquetes y módulos degradados.
- Añadir un comando de validación integral para CI local.
- Separar claramente errores de entorno de errores de código.

## Día 9: Automatización de validación

- Crear una rutina de checks reproducible:
  `doctor`, `pip check`, tests mínimos y smokes opcionales.
- Preparar una base de CI para Python 3.10 y una versión más nueva.

## Día 10: Endurecimiento final

- Revisar riesgos abiertos y priorizar regresiones reales.
- Limpiar warnings evitables.
- Preparar una release interna con changelog y estado de soporte por bloque.

## Orden recomendado

1. Día 1: stack distribuido mínimo.
2. Día 2: consolidación de packaging.
3. Día 4: cobertura de bloques core.
4. Día 3: entry points y CLIs.
5. Día 5 en adelante según lo que encontremos.
