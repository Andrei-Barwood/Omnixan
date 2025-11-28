# ⚛️ OMNIXAN Quantum Computing Module

## 🔬 Librerías Cuánticas Instaladas

Este proyecto incluye soporte completo para computación cuántica:

### 1. **Qiskit** (IBM)
```python
from qiskit import QuantumCircuit, QuantumSimulator

# Crear circuito cuántico
circuit = QuantumCircuit(2)
circuit.h(0)  # Hadamard
circuit.cx(0, 1)  # CNOT
```

### 2. **Cirq** (Google)
```python
import cirq

# Dispositivo y circuito
q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1)
)
```

### 3. **PennyLane** (Quantum ML)
```python
import pennylane as qml

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def quantum_circuit(x):
    qml.RX(x[0], wires=0)
    qml.RY(x[1], wires=1)
    return qml.expval(qml.Z(0))
```

### 4. **QuTiP** (Open Quantum Systems)
```python
import qutip as qt

# Operadores de Pauli
sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()
```

### 5. **Strawberry Fields** (Fotónica)
```python
import strawberryfields as sf

prog = sf.Program(4)
with prog.context as q:
    sf.ops.Sgate(0.5) | q[0]
    sf.ops.BSgate() | (q[0], q[1])
```

### 6. **ProjectQ** (Compilador Cuántico)
```python
from projectq import MainEngine
from projectq.ops import All, Measure, H

engine = MainEngine()
qubit = engine.allocate_qureg(2)
All(H) | qubit
```

---

## 📚 Estructura de Quantum Workspace

Se crea un directorio especial para experimentos cuánticos:

```
omnixan/quantum_workspace/
├── circuits/          # Tus circuitos cuánticos
├── simulations/       # Resultados de simulaciones
├── algorithms/        # Algoritmos cuánticos (Shor, Grover, etc)
├── qasm/             # Quantum Assembly Language files
├── results/          # Resultados de ejecuciones
└── notebooks/        # Jupyter notebooks para experimentación
```

---

## 🚀 Quick Start Quantum

### 1. Instalar con soporte Quantum
```bash
pip install -r requirements.txt
# O específicamente
pip install qiskit cirq pennylane qutip
```

### 2. Crear tu primer circuito Qiskit
```python
from qiskit import QuantumCircuit, QuantumSimulator
from qiskit.visualization import plot_histogram

# Bell State
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

# Simular
simulator = QuantumSimulator()
job = simulator.run(qc, shots=1024)
result = job.result()
counts = result.get_counts(qc)
print(counts)
```

### 3. Usar PennyLane para Quantum ML
```python
import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def quantum_nn(params):
    for i in range(3):
        qml.RX(params[i], wires=i)
    for i in range(2):
        qml.CNOT(wires=[i, i+1])
    return qml.expval(qml.Z(0))

# Entrenar con descenso de gradientes
params = np.array([0.1, 0.2, 0.3], requires_grad=True)
opt = qml.GradientDescentOptimizer(stepsize=0.01)

for step in range(100):
    params = opt.step(quantum_nn, params)
```

---

## 🎯 Algoritmos Cuánticos Disponibles

Con estas librerías puedes implementar:

- **Shor's Algorithm** - Factorización (Qiskit, Cirq)
- **Grover's Algorithm** - Búsqueda cuántica (PennyLane)
- **VQE** - Variational Quantum Eigensolver (Qiskit, PennyLane)
- **QAOA** - Quantum Approximate Optimization (Cirq, PennyLane)
- **Quantum Walk** - Paseos cuánticos (QuTiP)
- **Quantum Simulation** - Simulación de sistemas (QuTiP)

---

## 💻 Ejecutar Simulaciones

### Local Simulator
```bash
# Qiskit
python -c "from qiskit_aer import AerSimulator; print('Qiskit Ready')"

# Cirq
python -c "import cirq; print('Cirq Ready')"
```

### Notebook Interactivo
```bash
jupyter notebook
# En quantum_workspace/notebooks/
# Crea: quantum_experiments.ipynb
```

---

## 🔗 Integración con OMNIXAN

Integra módulos cuánticos en tus bloques:

```python
# omnixan/quantum_based_cloud/quantum_processor/module.py

from qiskit import QuantumCircuit, QuantumSimulator

class QuantumProcessorModule:
    def execute_circuit(self, circuit_definition):
        qc = QuantumCircuit.from_qasm_str(circuit_definition)
        simulator = QuantumSimulator()
        job = simulator.run(qc)
        return job.result()
```

---

## 📊 Performance

| Librería | Qubits Máx | Simulador | Hardware |
|----------|-----------|-----------|----------|
| Qiskit | 25+ | ✅ Aer | ✅ IBM Hardware |
| Cirq | 30+ | ✅ Simulator | ✅ Google Sycamore |
| PennyLane | 20+ | ✅ Multiple | ✅ Múltiples |
| QuTiP | Ilimitado | ✅ Exact | - |
| ProjectQ | 25+ | ✅ Simulator | ✅ Multiple |

---

## 🐛 Troubleshooting

### Error: "Qiskit not found"
```bash
pip install qiskit qiskit-aer --upgrade
```

### Error: "CUDA not available for Qiskit"
```bash
# Usa simulador CPU
from qiskit_aer import AerSimulator
simulator = AerSimulator(method='statevector')
```

### Performance lento
```bash
# Usa GPU si está disponible
from qiskit_aer import AerSimulator
simulator = AerSimulator(device='GPU')
```

---

## 📚 Recursos

- [Qiskit Documentation](https://qiskit.org/documentation/)
- [Cirq Documentation](https://quantumai.google/cirq)
- [PennyLane Documentation](https://pennylane.ai/)
- [QuTiP Documentation](http://qutip.org/)

---

**¡Listo para explorar el mundo cuántico! ⚛️🚀**

