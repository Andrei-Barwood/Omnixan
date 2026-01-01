# 🧠 Quantum Machine Learning Module

## 📖 Descripción

Módulo de Machine Learning Cuántico para OMNIXAN que implementa clasificadores cuánticos variacionales (VQC), redes neuronales cuánticas (QNN) y kernels cuánticos con optimización híbrida cuántico-clásica.

## 🎯 Características

- 🔄 Variational Quantum Classifiers (VQC)
- 🧠 Quantum Neural Networks
- 🎯 Quantum Kernel Methods
- 📊 Feature maps configurables (Z, ZZ, Amplitude)
- 🏗️ Ansatz variacionales (RealAmplitudes, EfficientSU2, StronglyEntangling)
- 📈 Historial de entrenamiento y métricas

## 🏗️ Modelos Soportados

| Modelo | Descripción | Uso |
|--------|-------------|-----|
| VQC | Clasificador cuántico variacional | Clasificación binaria |
| QNN | Red neuronal cuántica | Clasificación/Regresión |
| QKERNEL | Kernel cuántico | SVM cuántico |

## 💡 Uso Rápido

```python
import asyncio
import numpy as np
from omnixan.quantum_cloud_architecture.quantum_ml_module.module import (
    QuantumMLModule,
    QMLConfig,
    QMLModelType,
    FeatureMapType,
    AnsatzType
)

async def main():
    config = QMLConfig(
        num_qubits=4,
        feature_map=FeatureMapType.ZZ,
        ansatz=AnsatzType.REAL_AMPLITUDES,
        num_layers=2,
        max_epochs=50,
        learning_rate=0.1
    )
    
    module = QuantumMLModule(config)
    await module.initialize()
    
    # Crear datos de ejemplo
    X_train = np.random.rand(50, 4) * 2 * np.pi
    y_train = (np.sum(X_train > np.pi, axis=1) % 2).astype(int)
    
    # Crear y entrenar modelo
    module.create_model("my_vqc", QMLModelType.VQC)
    result = await module.train_model("my_vqc", X_train, y_train)
    
    print(f"Accuracy: {result['final_accuracy']:.4f}")
    
    # Predicciones
    X_test = np.random.rand(10, 4) * 2 * np.pi
    predictions = await module.predict("my_vqc", X_test)
    
    await module.shutdown()

asyncio.run(main())
```

## 🔧 Configuración

```python
class QMLConfig:
    model_type: QMLModelType = VQC
    num_qubits: int = 4
    feature_map: FeatureMapType = ZZ
    ansatz: AnsatzType = REAL_AMPLITUDES
    num_layers: int = 2
    optimizer: OptimizerType = ADAM
    learning_rate: float = 0.01
    max_epochs: int = 100
    batch_size: int = 32
    shots: int = 1024
```

## 📦 Dependencias

- `qiskit>=1.0.0`
- `qiskit-aer>=0.13.0`
- `numpy>=1.26.0`
- `pydantic>=2.5.0`

---
**Status:** ✅ Implementado | **Última actualización:** 2025-01-XX
