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

Validación cuántica profunda:

```bash
python -m omnixan.doctor --json
python -m pytest omnixan/tests -k quantum_stack_smoke
```

Verificación del entrypoint de balanceo:

```bash
PYENV_VERSION=hokkaido python -m omnixan.carbon_based_quantum_cloud.load_balancing_module --version
```

## Notas del repo

- El `pyproject.toml` de la raíz es la referencia actual para instalar y testear el proyecto.
- El repo contiene un `venv/` dentro del árbol histórico; no debe tomarse como fuente de verdad de packaging.
- Los módulos cuánticos están preparados para importarse aunque falten backends, pero sus funciones concretas sí requieren instalar los extras adecuados.
- La smoke suite cuántica se salta automáticamente cuando `qiskit` y `qiskit-aer` no están instalados.
- En esta revisión, la validación módulo por módulo del stack cuántico se ejecutó sobre Python 3.10.
