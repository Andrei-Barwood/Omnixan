from __future__ import annotations

import asyncio
import importlib.util

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("qiskit") is None
    or importlib.util.find_spec("qiskit_aer") is None,
    reason="Quantum smoke tests require qiskit and qiskit-aer",
)


def test_quantum_algorithm_module_smoke() -> None:
    from omnixan.quantum_cloud_architecture.quantum_algorithm_module.module import (
        AlgorithmStatus,
        QuantumAlgorithmModule,
    )

    async def scenario() -> None:
        module = QuantumAlgorithmModule()
        module.initialize()
        try:
            result = await module.execute_algorithm(
                name="Grover",
                params={"num_qubits": 2, "target_state": "11", "iterations": 1},
                shots=32,
            )
            assert result.status == AlgorithmStatus.COMPLETED
            assert result.result is not None
            assert result.result["top_result"] is not None
            assert result.raw_output is not None
            assert sum(result.raw_output.values()) == 32
        finally:
            module.shutdown()

    asyncio.run(scenario())


def test_quantum_circuit_optimizer_module_smoke() -> None:
    from qiskit import QuantumCircuit

    from omnixan.quantum_cloud_architecture.quantum_circuit_optimizer_module.module import (
        QuantumCircuitOptimizerModule,
    )

    async def scenario() -> None:
        module = QuantumCircuitOptimizerModule()
        await module.initialize()
        try:
            circuit = QuantumCircuit(2)
            circuit.h(0)
            circuit.h(0)
            circuit.cx(0, 1)
            circuit.cx(0, 1)

            result = await module.optimize_circuit(circuit)
            assert result.optimized_metrics.gate_count <= result.original_metrics.gate_count
            assert "Gate Cancellation" in result.passes_applied
        finally:
            await module.shutdown()

    asyncio.run(scenario())


def test_quantum_error_correction_module_smoke() -> None:
    from omnixan.quantum_cloud_architecture.quantum_error_correction_module.module import (
        ErrorCorrectionCode,
        ErrorCorrectionConfig,
        QuantumErrorCorrectionModule,
    )

    async def scenario() -> None:
        module = QuantumErrorCorrectionModule(
            ErrorCorrectionConfig(
                default_code=ErrorCorrectionCode.BIT_FLIP_3,
                shots=64,
            )
        )
        await module.initialize()
        try:
            circuit = await module.encode_logical_qubit(
                ErrorCorrectionCode.BIT_FLIP_3,
                logical_state="1",
            )
            result = await module.detect_error(ErrorCorrectionCode.BIT_FLIP_3, circuit)
            assert len(result.syndrome) == 2
            assert result.error_detected is False
        finally:
            await module.shutdown()

    asyncio.run(scenario())


def test_quantum_simulator_module_smoke() -> None:
    from qiskit import QuantumCircuit

    from omnixan.quantum_cloud_architecture.quantum_simulator_module.module import (
        ExecutionStatus,
        QuantumSimulatorModule,
        SimulationMethod,
        SimulatorBackend,
        SimulatorConfig,
    )

    async def scenario() -> None:
        module = QuantumSimulatorModule(
            SimulatorConfig(
                backend=SimulatorBackend.QISKIT,
                method=SimulationMethod.STATEVECTOR,
                shots=32,
            )
        )
        await module.initialize()
        try:
            circuit = QuantumCircuit(1, 1)
            circuit.h(0)
            circuit.measure(0, 0)

            result = await module.simulate_circuit(circuit)
            statevector = await module.get_statevector(circuit)

            assert result.status == ExecutionStatus.COMPLETED
            assert result.counts is not None
            assert sum(result.counts.values()) == 32
            assert np.isclose(np.linalg.norm(statevector.data), 1.0)
        finally:
            await module.shutdown()

    asyncio.run(scenario())


def test_quantum_ml_module_smoke() -> None:
    from omnixan.quantum_cloud_architecture.quantum_ml_module.module import (
        QMLConfig,
        QuantumMLModule,
    )

    async def scenario() -> None:
        module = QuantumMLModule(
            QMLConfig(
                num_qubits=2,
                num_layers=1,
                max_epochs=1,
                batch_size=2,
                shots=16,
            )
        )
        await module.initialize()
        try:
            X = np.array([[0.0, np.pi], [np.pi, 0.0]])
            y = np.array([0, 1])

            module.create_model("smoke_vqc")
            train_result = await module.train_model("smoke_vqc", X, y)
            predictions = await module.predict("smoke_vqc", X)

            assert train_result["epochs_completed"] == 1
            assert predictions.shape == (2,)
            assert set(predictions.tolist()) <= {0, 1}
        finally:
            await module.shutdown()

    asyncio.run(scenario())
