# CLI

## Comandos soportados oficialmente

La superficie oficial de línea de comandos de OMNIXAN quedó reducida a unos
pocos entrypoints estables y testeados:

- `python -m omnixan --help`
- `python -m omnixan doctor [--json]`
- `python -m omnixan validate [--json|--skip-tests|--strict-environment]`
- `python -m omnixan load-balancing [--version|--smoke|--json|--config ...]`
- `python -m omnixan redundant-deployment [--version|--smoke|--json]`

Los mismos comandos quedan expuestos además como console scripts al instalar el
repo desde la raíz:

- `omnixan`
- `omnixan-doctor`
- `omnixan-validate`
- `omnixan-load-balancing`
- `omnixan-redundant-deployment`

## Mapeo actual de servicios a CLI

La política oficial de exposición pública quedó definida en `SERVICE_MAP.md`.
Hoy la CLI soportada representa solo los servicios maduros del producto:

- `observation-service` -> `python -m omnixan doctor`, `python -m omnixan validate`
- `continuity-service` -> `python -m omnixan load-balancing`, `python -m omnixan redundant-deployment`

Los servicios nucleares cuánticos siguen definidos a nivel de arquitectura, pero
todavía no se exponen como comandos estables separados porque falta una misión
cuántica canónica ejecutable de extremo a extremo.

## Qué quedó fuera de soporte oficial

El repositorio todavía contiene decenas de bloques con `async def main()` o
`if __name__ == "__main__"` dentro de `module.py`. Esos bloques se consideran
demos locales o ejemplos manuales, no una API CLI estable.

En esta revisión no se prometen como comandos soportados:

- ejecuciones directas de `python -m ...module`
- snippets de README con `main()` embebido
- scripts de ejemplo históricos fuera de los entrypoints listados arriba

## Comandos recomendados

Diagnóstico general:

```bash
python -m omnixan doctor
python -m omnixan doctor --json
python -m omnixan validate --json --skip-tests
python -m omnixan validate --json
```

Rutina reproducible local:

```bash
./scripts/ci_local.sh baseline
./scripts/ci_local.sh optional-smokes
```

Smoke rápido de load balancing:

```bash
python -m omnixan load-balancing --smoke --json
```

Smoke rápido de redundant deployment:

```bash
python -m omnixan redundant-deployment --smoke --json
```

Ejecución del servidor de load balancing:

```bash
python -m omnixan load-balancing --algorithm round_robin
```

## Reglas de mantenimiento

- Si se agrega un nuevo comando oficial, debe tener `__main__.py` o script declarado en la raíz.
- Todo comando oficial debe mostrar `--help` y una salida consistente de `--version` o `--json`.
- Los `main()` de ejemplo dentro de módulos no reemplazan un entrypoint oficial.
