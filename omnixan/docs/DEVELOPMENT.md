# Desarrollo

## Entorno recomendado

- Python 3.10 o superior
- Un intérprete con `pytest` disponible para validaciones locales
- Dependencias opcionales solo cuando se vayan a ejecutar módulos cuánticos o distribuidos

## Comandos de validación

Diagnóstico rápido del workspace:

```bash
PYENV_VERSION=hokkaido python -m omnixan doctor
```

Pruebas mínimas del repo:

```bash
PYENV_VERSION=hokkaido python -m pytest omnixan/tests
```

Smoke suite core por bloque:

```bash
PYENV_VERSION=hokkaido python -m pytest omnixan/tests -k core_block_smoke
```

Preparación del stack cuántico opcional:

```bash
python -m pip install -e '.[quantum,dev]'
```

Preparación del stack distribuido opcional:

```bash
python -m pip install -e '.[distributed,dev]'
```

Validación cuántica profunda:

```bash
python -m omnixan doctor --json
python -m pytest omnixan/tests -k quantum_stack_smoke
```

Validación distribuida profunda:

```bash
python -m omnixan doctor --json
python -m pytest omnixan/tests -k distributed_stack_smoke
python -m pip check
```

Verificación del entrypoint de balanceo:

```bash
PYENV_VERSION=hokkaido python -m omnixan load-balancing --version
```

## Notas del repo

- El `pyproject.toml` de la raíz es la referencia actual para instalar y testear el proyecto.
- El repo contiene un `venv/` dentro del árbol histórico; no debe tomarse como fuente de verdad de packaging.
- Los archivos `omnixan/setup.py`, `omnixan/requirements.txt` y `omnixan/pyproject.toml` se conservan solo por compatibilidad y documentación local.
- Los módulos cuánticos están preparados para importarse aunque falten backends, pero sus funciones concretas sí requieren instalar los extras adecuados.
- El stack distribuido mínimo validado en esta revisión usa `ray` y `dask[distributed]`.
- La smoke suite cuántica se salta automáticamente cuando `qiskit` y `qiskit-aer` no están instalados.
- La smoke suite distribuida se salta automáticamente cuando faltan `ray` o `dask[distributed]`.
- La smoke suite core cubre un módulo crítico por bloque principal sin depender de stacks opcionales pesados.
- Los módulos con CUDA, GPU o runtimes pesados ahora deben fallar al usar el backend ausente, no al importar el paquete.
- Los módulos core alineados en esta revisión exponen `initialize()`, `execute()`, `shutdown()`, `get_status()` y `get_metrics()` como superficie pública común.
- Los README de bloque y de los módulos core validados se actualizaron con rutas felices, comandos oficiales y ejemplos mínimos ejecutables contra el estado actual del repo.
- En esta revisión, la validación módulo por módulo del stack cuántico se ejecutó sobre Python 3.10.

## Packaging

- Fuente de verdad: [`/pyproject.toml`](/Users/kirtantegsingh/Public/omnixan/Omnixan/pyproject.toml)
- Referencia operativa: [`PACKAGING.md`](/Users/kirtantegsingh/Public/omnixan/Omnixan/omnixan/docs/PACKAGING.md)

## CLI y Entry Points

- La CLI oficial está documentada en [`CLI.md`](/Users/kirtantegsingh/Public/omnixan/Omnixan/omnixan/docs/CLI.md).
- `python -m omnixan` es ahora el entrypoint raíz recomendado para comandos soportados.
- Los `main()` dentro de `module.py` siguen existiendo en varios bloques, pero se consideran demos o smoke helpers internos y no comandos estables.
