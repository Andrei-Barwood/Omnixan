# ⚡ Quantum Circuit Optimizer Module

## 📖 Descripción

Módulo de optimización de circuitos cuánticos para OMNIXAN que implementa múltiples técnicas de optimización: fusión de puertas, cancelación, conmutación, descomposición y optimización de layout.

## 🎯 Características

- 🔄 Cancelación de puertas inversas (HH=I, XX=I)
- 🔗 Fusión de puertas de un qubit
- 📐 Reordenamiento por conmutación
- 🎯 Descomposición a conjuntos de puertas objetivo
- 🗺️ Optimización de layout para hardware
- 📊 Métricas detalladas de mejora

## 🏗️ Niveles de Optimización

| Nivel | Descripción | Passes |
|-------|-------------|--------|
| NONE | Sin optimización | - |
| LIGHT | Básico | Gate cancellation |
| MEDIUM | Moderado | + Gate fusion, commutation |
| HEAVY | Intensivo | + Two-qubit opt, layout |
| AGGRESSIVE | Máximo | + Decomposition |

## 💡 Uso Rápido

```python
import asyncio
from qiskit import QuantumCircuit
from omnixan.quantum_cloud_architecture.quantum_circuit_optimizer_module.module import (
    QuantumCircuitOptimizerModule,
    OptimizerConfig,
    OptimizationLevel,
    OptimizationGoal
)

async def main():
    config = OptimizerConfig(
        optimization_level=OptimizationLevel.HEAVY,
        optimization_goal=OptimizationGoal.BALANCED
    )
    
    module = QuantumCircuitOptimizerModule(config)
    await module.initialize()
    
    # Crear circuito con puertas redundantes
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.h(0)  # Se cancela
    qc.cx(0, 1)
    qc.cx(0, 1)  # Se cancela
    
    # Optimizar
    result = await module.optimize_circuit(qc)
    
    print(f"Profundidad: {result.original_metrics.depth} -> {result.optimized_metrics.depth}")
    print(f"Mejora: {result.improvement['depth']:.1f}%")
    
    await module.shutdown()

asyncio.run(main())
```

## 🔧 Configuración

```python
class OptimizerConfig:
    optimization_level: OptimizationLevel = MEDIUM
    optimization_goal: OptimizationGoal = BALANCED
    target_basis_gates: List[str] = ["cx", "u1", "u2", "u3"]
    coupling_map: Optional[List[List[int]]] = None
    max_optimization_passes: int = 10
    enable_approximation: bool = False
```

## 📦 Dependencias

- `qiskit>=1.0.0`
- `numpy>=1.26.0`
- `pydantic>=2.5.0`

---
**Status:** ✅ Implementado | **Última actualización:** 2025-01-XX
