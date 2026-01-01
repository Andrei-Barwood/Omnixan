# ⚛️ Quantum Simulator Module

## 📖 Descripción

Módulo de simulador cuántico unificado para OMNIXAN que proporciona una interfaz común para múltiples backends de simulación cuántica (Qiskit Aer, Cirq, PennyLane) con varios métodos de simulación (statevector, density matrix, stabilizer, etc.).

## 🎯 Características

- ✨ Interfaz unificada para múltiples backends cuánticos
- ⚡ Soporte para múltiples métodos de simulación
- 🚀 Optimización de rendimiento con GPU opcional
- 📊 Métricas de rendimiento integradas
- 🔧 Modelos de ruido configurables
- 💾 Soporte para simulaciones statevector, density matrix, stabilizer y más

## 🏗️ Interfaz Principal

```python
from omnixan.quantum_cloud_architecture.quantum_simulator_module.module import (
    QuantumSimulatorModule,
    SimulatorConfig,
    SimulatorBackend,
    SimulationMethod
)

# Configurar módulo
config = SimulatorConfig(
    backend=SimulatorBackend.QISKIT,
    method=SimulationMethod.STATEVECTOR,
    shots=1024,
    enable_gpu=False
)

# Inicializar
module = QuantumSimulatorModule(config)
await module.initialize()

# Simular circuito
result = await module.simulate_circuit(
    circuit=quantum_circuit,
    shots=2048
)

# Obtener métricas
summary = module.get_metrics_summary()
```

## 💡 Uso Rápido

### Ejemplo con Qiskit

```python
import asyncio
from qiskit import QuantumCircuit
from omnixan.quantum_cloud_architecture.quantum_simulator_module.module import (
    QuantumSimulatorModule,
    SimulatorConfig,
    SimulatorBackend,
    SimulationMethod
)

async def main():
    # Configurar simulador
    config = SimulatorConfig(
        backend=SimulatorBackend.QISKIT,
        method=SimulationMethod.STATEVECTOR,
        shots=1024
    )
    
    module = QuantumSimulatorModule(config)
    await module.initialize()
    
    try:
        # Crear circuito Bell State
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        # Simular
        result = await module.simulate_circuit(qc, shots=2048)
        
        print(f"Counts: {result.counts}")
        print(f"Execution Time: {result.metrics.execution_time:.3f}s")
        
    finally:
        await module.shutdown()

asyncio.run(main())
```

### Usando el método execute()

```python
result = await module.execute({
    "operation": "simulate",
    "circuit": quantum_circuit,
    "backend": "qiskit",
    "method": "statevector",
    "shots": 1024
})
```

## 📊 Métodos de Simulación Soportados

- **STATEVECTOR**: Simulación completa del vector de estado (máximo ~30 qubits)
- **DENSITY_MATRIX**: Simulación de matriz de densidad (para estados mixtos)
- **STABILIZER**: Simulación estabilizadora (eficiente para circuitos Clifford)
- **MATRIX_PRODUCT_STATE**: Simulación MPS (para circuitos 1D)
- **EXTENDED_STABILIZER**: Estabilizador extendido
- **PAULI_TWIRL**: Pauli twirling para ruido

## 🔧 Configuración

```python
class SimulatorConfig:
    backend: SimulatorBackend = SimulatorBackend.QISKIT
    method: SimulationMethod = SimulationMethod.STATEVECTOR
    shots: int = 1024
    max_qubits: int = 30
    precision: str = "double"  # "single" or "double"
    enable_noise: bool = False
    enable_gpu: bool = False  # Requiere CUDA
    max_workers: int = 4
```

## 📦 Dependencias

- `qiskit>=1.0.0` (opcional, para backend Qiskit)
- `qiskit-aer>=0.13.0` (opcional, simuladores Qiskit)
- `cirq>=1.4.0` (opcional, para backend Cirq)
- `pennylane>=0.33.0` (opcional, para backend PennyLane)
- `pydantic>=2.5.0`
- `numpy>=1.26.0`

## 🔗 Módulos Relacionados

- `quantum_algorithm_module` - Ejecución de algoritmos cuánticos
- `quantum_circuit_optimizer_module` - Optimización de circuitos
- `quantum_error_correction_module` - Corrección de errores

## 🐛 Estado

- ✅ Estructura creada
- ✅ Implementación: Completada
- 📝 Documentación: 90%
- 🧪 Tests: Pendiente

---
**Status:** ✅ Implementado | **Última actualización:** 2025-01-XX
