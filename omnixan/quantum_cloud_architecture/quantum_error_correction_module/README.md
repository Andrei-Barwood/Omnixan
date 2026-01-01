# 🛡️ Quantum Error Correction Module

## 📖 Descripción

Módulo de corrección de errores cuánticos para OMNIXAN que implementa múltiples códigos de corrección de errores (Bit-flip, Phase-flip, Shor, Steane) con detección de síndrome, corrección de errores y estimación de fidelidad.

## 🎯 Características

- ✨ Múltiples códigos de corrección de errores
- 🔍 Detección de síndrome automatizada
- 🛠️ Corrección de errores bit-flip y phase-flip
- 📊 Métricas de rendimiento integradas
- 🧪 Simulación de ruido para pruebas

## 🏗️ Códigos Soportados

| Código | Qubits Físicos | Distancia | Errores Corregibles |
|--------|---------------|-----------|---------------------|
| Bit-Flip 3 | 3 | 3 | 1 bit-flip (X) |
| Phase-Flip 3 | 3 | 3 | 1 phase-flip (Z) |
| Shor 9 | 9 | 3 | 1 arbitrario |
| Steane 7 | 7 | 3 | 1 arbitrario |
| Repetition | n | n | (n-1)/2 bit-flip |

## 💡 Uso Rápido

```python
import asyncio
from omnixan.quantum_cloud_architecture.quantum_error_correction_module.module import (
    QuantumErrorCorrectionModule,
    ErrorCorrectionConfig,
    ErrorCorrectionCode
)

async def main():
    config = ErrorCorrectionConfig(
        default_code=ErrorCorrectionCode.BIT_FLIP_3,
        error_probability=0.1,
        shots=1024
    )
    
    module = QuantumErrorCorrectionModule(config)
    await module.initialize()
    
    try:
        # Ciclo completo de corrección
        result = await module.full_correction_cycle(
            code=ErrorCorrectionCode.SHOR_9,
            logical_state="0",
            error_probability=0.2
        )
        
        print(f"Status: {result['status']}")
        print(f"Error detectado: {result['error_detected']}")
        
    finally:
        await module.shutdown()

asyncio.run(main())
```

## 🔧 Configuración

```python
class ErrorCorrectionConfig:
    default_code: ErrorCorrectionCode = BIT_FLIP_3
    error_probability: float = 0.01
    shots: int = 1024
    enable_syndrome_history: bool = True
    max_correction_rounds: int = 3
    fidelity_threshold: float = 0.99
```

## 📦 Dependencias

- `qiskit>=1.0.0`
- `qiskit-aer>=0.13.0`
- `numpy>=1.26.0`
- `pydantic>=2.5.0`

---
**Status:** ✅ Implementado | **Última actualización:** 2025-01-XX
