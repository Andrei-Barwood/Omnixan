# Packaging

## Fuente de verdad

El archivo autoritativo de packaging del repositorio es:

- [`/pyproject.toml`](/Users/kirtantegsingh/Public/omnixan/Omnixan/pyproject.toml)

Toda instalación, validación y mantenimiento de dependencias debe partir desde
ese archivo.

## Archivos históricos

Los siguientes archivos se conservan por compatibilidad o contexto local, pero
no son la fuente de verdad:

- [`omnixan/setup.py`](/Users/kirtantegsingh/Public/omnixan/Omnixan/omnixan/setup.py)
  : shim histórico que redirige al `pyproject.toml` raíz.
- [`omnixan/requirements.txt`](/Users/kirtantegsingh/Public/omnixan/Omnixan/omnixan/requirements.txt)
  : snapshot plano y legible de extras relevantes.
- [`omnixan/pyproject.toml`](/Users/kirtantegsingh/Public/omnixan/Omnixan/omnixan/pyproject.toml)
  : configuración local de herramientas dentro del subdirectorio `omnixan/`.

Si alguno de estos archivos entra en conflicto con la raíz, gana la raíz.

## Extras actuales

- `data`: utilidades de datos hoy usadas de forma explícita (`pandas`).
- `cloud`: dependencias opcionales de módulos de autoescalado, migración y contenedores.
- `distributed`: `ray` y `dask[distributed]`.
- `quantum`: backends y librerías cuánticas validadas.
- `dev`: tooling de tests y formateo.
- `docs`: tooling mínimo de documentación.

## Comandos recomendados

Instalación mínima:

```bash
python -m pip install -e .
```

Instalación para cloud/distribuido/cuántico:

```bash
python -m pip install -e '.[cloud,distributed,quantum]'
```

Instalación para desarrollo completo:

```bash
python -m pip install -e '.[cloud,distributed,quantum,dev,docs]'
```

## Criterios de mantenimiento

- No duplicar metadata funcional en más de un lugar.
- No declarar dependencias fuertes que el código no importe realmente.
- Agrupar dependencias opcionales por bloque o caso de uso.
- Validar cambios de packaging con `pytest`, `omnixan.doctor` y `pip check`.
