# Desarrollo

## Entorno recomendado

- Python 3.10 o superior
- Un intérprete con `pytest` disponible para validaciones locales
- Dependencias opcionales solo cuando se vayan a ejecutar módulos cuánticos o distribuidos

## Comandos de validación

Diagnóstico rápido del workspace:

```bash
PYENV_VERSION=hokkaido python -m omnixan.doctor
```

Pruebas mínimas del repo:

```bash
PYENV_VERSION=hokkaido python -m pytest omnixan/tests
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
python -m omnixan.doctor --json
python -m pytest omnixan/tests -k quantum_stack_smoke
```

Validación distribuida profunda:

```bash
python -m omnixan.doctor --json
python -m pytest omnixan/tests -k distributed_stack_smoke
python -m pip check
```

Verificación del entrypoint de balanceo:

```bash
PYENV_VERSION=hokkaido python -m omnixan.carbon_based_quantum_cloud.load_balancing_module --version
```

## Notas del repo

- El `pyproject.toml` de la raíz es la referencia actual para instalar y testear el proyecto.
- El repo contiene un `venv/` dentro del árbol histórico; no debe tomarse como fuente de verdad de packaging.
- Los archivos `omnixan/setup.py`, `omnixan/requirements.txt` y `omnixan/pyproject.toml` se conservan solo por compatibilidad y documentación local.
- Los módulos cuánticos están preparados para importarse aunque falten backends, pero sus funciones concretas sí requieren instalar los extras adecuados.
- El stack distribuido mínimo validado en esta revisión usa `ray` y `dask[distributed]`.
- La smoke suite cuántica se salta automáticamente cuando `qiskit` y `qiskit-aer` no están instalados.
- La smoke suite distribuida se salta automáticamente cuando faltan `ray` o `dask[distributed]`.
- En esta revisión, la validación módulo por módulo del stack cuántico se ejecutó sobre Python 3.10.

## Packaging

- Fuente de verdad: [`/pyproject.toml`](/Users/kirtantegsingh/Public/omnixan/Omnixan/pyproject.toml)
- Referencia operativa: [`PACKAGING.md`](/Users/kirtantegsingh/Public/omnixan/Omnixan/omnixan/docs/PACKAGING.md)
