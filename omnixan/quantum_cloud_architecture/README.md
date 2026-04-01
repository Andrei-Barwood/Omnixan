# quantum_cloud_architecture

## Proposito

Este bloque es el nucleo cuantico de OMNIXAN. Aqui viven las piezas que hoy
cubren diseno de circuitos, optimizacion, simulacion, mitigacion y
aprendizaje cuantico.

## Estado actual

- Estado general: `experimental`, pero con piezas reales y smoke tests
  ejecutables cuando el stack cuantico esta instalado.
- Canon oficial: `omnixan.quantum_pipeline`
- Pipeline documentada: `omnixan/docs/QUANTUM_PIPELINE.md`
- Auditoria de huecos: `omnixan/docs/QUANTUM_GAP_AUDIT.md`

## Modulos del bloque

| Modulo | Rol principal | Usar hoy | Nota corta |
| --- | --- | --- | --- |
| `quantum_algorithm_module` | diseno de circuito y ejecucion acoplada | `si con extras` | util, pero todavia mezcla etapas canonicas |
| `quantum_circuit_optimizer_module` | optimizacion de circuitos | `si con extras` | pieza real para la etapa de optimizacion |
| `quantum_simulator_module` | simulacion local y backend unificado | `si con extras` | cubre ejecucion baseline, no reporte canonico |
| `quantum_error_correction_module` | correccion y mitigacion | `si con extras` | cubre mitigacion, aun sin adaptador canonico |
| `quantum_ml_module` | QML e hibridacion | `si con extras` | funcional, pero fuera de la ruta feliz principal |

## Ruta feliz actual del bloque

1. Definir una `QuantumMission` y un `QuantumExecutionPlan`.
2. Construir o seleccionar un circuito con `quantum_algorithm_module`.
3. Optimizarlo con `quantum_circuit_optimizer_module`.
4. Simularlo con `quantum_simulator_module`.
5. Aplicar mitigacion con `quantum_error_correction_module`.
6. Reportar estado con la capa de observabilidad del repo.

## Limite actual

La ruta de arriba todavia no esta unida por un orquestador canonico de extremo
a extremo. Las piezas existen, pero siguen conectandose mediante adapters
implicitos o manuales.

## Validacion

Baseline sin extras:

```bash
PYENV_VERSION=hokkaido python -m pytest omnixan/tests/test_repo_health.py
```

Smokes cuanticos con stack instalado:

```bash
omnixan/venv/bin/python -m pytest omnixan/tests/test_quantum_stack_smoke.py
```
