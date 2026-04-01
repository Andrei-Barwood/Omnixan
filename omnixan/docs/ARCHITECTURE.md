# Arquitectura

OMNIXAN está organizado como un paquete Python con bloques temáticos. Cada
bloque agrupa módulos autocontenidos, pero la dirección actual del proyecto es
convergerlos hacia una plataforma de servicios híbridos con cadena cuántica
canónica y narrativa Amarr.

La visión operativa de esa convergencia quedó documentada en `VISION.md`, y la
campaña diaria que guía esa transición vive en `DAILY_TASKS.md`. Los principios
que traducen la narrativa Amarr a decisiones de arquitectura y operación están
en `AMARR_PRINCIPLES.md`. El lenguaje oficial y el catálogo inicial de
servicios viven en `SERVICE_LANGUAGE.md`. La decisión actual bloque por bloque
respecto del canon del producto quedó en `BLOCK_CANON_MAP.md`. La clasificación
operativa módulo por módulo vive en `MODULE_CLASSIFICATION.md`. La cadena
cuántica canónica y sus contratos mínimos viven en `QUANTUM_PIPELINE.md` y
`omnixan.quantum_pipeline`. La auditoría de huecos cuánticos y el backlog
priorizado de integración viven en `QUANTUM_GAP_AUDIT.md`.

## Bloques principales

- `carbon_based_quantum_cloud`: balanceo, despliegue redundante, migración y autoescalado.
- `edge_computing_network`: coherencia de caché, memoria persistente y procesamiento cercano a datos.
- `heterogenous_computing_group`: aceleración RDMA, Infiniband, cooling y paralelismo masivo.
- `in_memory_computing_cloud`: edge AI, routing de baja latencia y fog computing.
- `quantum_cloud_architecture`: algoritmos cuánticos, simulación, optimización y corrección de errores.
- `supercomputing_interconnect_cloud`: módulos de tensor core, CUDA y cómputo integrado.
- `virtualized_cluster`: interfaz cuántica, mitigación de fallos y control criogénico.

## Convenciones útiles

- La implementación real de la mayoría de módulos vive en `module.py`.
- Los imports de alto nivel se hacen desde el paquete del bloque, por ejemplo:
  `omnixan.carbon_based_quantum_cloud.load_balancing_module`.
- Las dependencias pesadas son opcionales. El repo intenta permitir imports
  seguros aun cuando `qiskit`, `cirq`, `pennylane`, `ray` o `dask` no estén instalados.

## Diagnóstico

El comando [`omnixan/doctor.py`](/Users/kirtantegsingh/Public/omnixan/Omnixan/omnixan/doctor.py)
resume:

- versión de Python y ejecutable activo
- disponibilidad de dependencias opcionales
- estado de imports clave del paquete
