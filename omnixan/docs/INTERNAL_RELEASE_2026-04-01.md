# Release Interna 2026-04-01

## Resumen

Esta release interna consolida los Dias 1 a 10 del plan de saneamiento del repo.
El objetivo fue dejar un baseline reproducible, import-safe y documentado para el
workspace OMNIXAN sin exigir stacks opcionales por defecto.

## Validacion base esperada

```bash
PYENV_VERSION=hokkaido python -m omnixan doctor --json
PYENV_VERSION=hokkaido python -m omnixan validate --json
PYENV_VERSION=hokkaido python -m pip check
PYENV_VERSION=hokkaido python -m pytest omnixan/tests
./scripts/ci_local.sh baseline
```

## Cambios de endurecimiento finales

- Se elimino `logging.basicConfig()` en imports de modulos validados para evitar
  efectos globales al importar paquetes.
- El logger de `RedundantDeploymentModule` ya no duplica handlers entre instancias.
- Se priorizaron riesgos reales del baseline y se separaron claramente de faltantes
  de entorno opcional.

## Riesgos abiertos priorizados

1. Muchos modulos historicos siguen con cobertura funcional superficial fuera del
   camino baseline soportado.
2. Los stacks cuantico, distribuido y GPU dependen del entorno y deben revisarse
   en una ronda posterior especifica.
3. El `venv` historico dentro del repo puede arrastrar paquetes ajenos si se
   reutiliza sin limpieza.

## Go / No-Go

- Estado recomendado: `go` para uso interno, desarrollo diario, CI baseline y
  smoke suites core.
- Estado no cubierto por baseline: backends GPU, cuanticos completos y variantes
  distribuidas fuera del extra oficial.

## Referencias

- Ver `CHANGELOG.md` para el resumen de cambios.
- Ver `omnixan/docs/SUPPORT_STATUS.md` para el estado de soporte por bloque.
- Ver `omnixan/docs/REPO_STATUS.md` para riesgos abiertos y contexto tecnico.
