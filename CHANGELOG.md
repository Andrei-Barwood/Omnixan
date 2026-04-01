# Changelog

## 0.2.0 - 2026-04-01

Release interna de endurecimiento y saneamiento del workspace.

### Added

- CLI raiz oficial con `python -m omnixan` y scripts soportados.
- `omnixan.doctor` con reportes de warnings, conflictos de paquetes y modulos degradados.
- `python -m omnixan validate` y `scripts/ci_local.sh` como rutina reproducible de validacion.
- Base de GitHub Actions para Python 3.10 y 3.13.
- Smoke suites para bloques core, stack cuantico opcional y stack distribuido opcional.
- Documentacion operativa por bloque, changelog, release interna y matriz de soporte.

### Changed

- `pyproject.toml` en la raiz quedo como unica fuente de verdad de packaging.
- La API publica comun de modulos core se alineo en `initialize()`, `execute()`,
  `shutdown()`, `get_status()` y `get_metrics()`.
- Los modulos con dependencias pesadas ahora degradan en runtime con mensajes
  claros en lugar de romper al importar.
- Los imports de logging ya no configuran `basicConfig()` al cargar paquetes.

### Fixed

- Imports relativos y entrypoints rotos en modulos de balanceo y despliegue redundante.
- Compatibilidad con Qiskit y Aer actuales en modulos cuanticos validados.
- Extra distribuido para instalar `dask[distributed]` junto con `ray`.
- Duplicacion de handlers en el logger de `RedundantDeploymentModule`.

### Known limits

- GPU, CUDA y varios backends cuanticos siguen siendo opcionales y no forman parte
  del baseline minimo.
- Muchos modulos historicos fuera del camino validado siguen con cobertura de
  comportamiento superficial.
