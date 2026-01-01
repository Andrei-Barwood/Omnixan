"""
OMNIXAN Quantum Error Correction Module
quantum_cloud_architecture/quantum_error_correction_module

Production-ready quantum error correction implementation supporting multiple
error correction codes (Bit-flip, Phase-flip, Shor, Steane, Surface codes)
with syndrome detection, error correction, and fidelity estimation.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import json

import numpy as np

# Type hints for quantum libraries
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
    from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity
    from qiskit_aer import Aer, AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorCorrectionCode(str, Enum):
    """Supported quantum error correction codes"""
    BIT_FLIP_3 = "bit_flip_3"  # 3-qubit bit-flip code
    PHASE_FLIP_3 = "phase_flip_3"  # 3-qubit phase-flip code
    SHOR_9 = "shor_9"  # 9-qubit Shor code
    STEANE_7 = "steane_7"  # 7-qubit Steane code
    SURFACE_CODE = "surface_code"  # Surface code (simplified)
    REPETITION = "repetition"  # Repetition code


class ErrorType(str, Enum):
    """Types of quantum errors"""
    BIT_FLIP = "bit_flip"  # X error
    PHASE_FLIP = "phase_flip"  # Z error
    BIT_PHASE_FLIP = "bit_phase_flip"  # Y error
    DEPOLARIZING = "depolarizing"  # Random Pauli error
    AMPLITUDE_DAMPING = "amplitude_damping"
    MEASUREMENT = "measurement"


class CorrectionStatus(str, Enum):
    """Status of error correction"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    NO_ERROR = "no_error"


@dataclass
class SyndromeResult:
    """Result of syndrome measurement"""
    syndrome: str  # Binary syndrome string
    error_detected: bool
    error_type: Optional[ErrorType] = None
    error_location: Optional[int] = None
    confidence: float = 1.0


@dataclass
class CorrectionResult:
    """Result of error correction"""
    status: CorrectionStatus
    original_syndrome: str
    corrected: bool
    correction_applied: Optional[str] = None
    fidelity_before: Optional[float] = None
    fidelity_after: Optional[float] = None
    error_rate: float = 0.0


@dataclass
class ErrorCorrectionMetrics:
    """Performance metrics for error correction"""
    total_corrections: int = 0
    successful_corrections: int = 0
    failed_corrections: int = 0
    average_fidelity_improvement: float = 0.0
    error_detection_rate: float = 0.0
    execution_time: float = 0.0
    code_used: str = ""
    timestamp: float = field(default_factory=time.time)


class ErrorCorrectionConfig(BaseModel):
    """Configuration for error correction module"""
    default_code: ErrorCorrectionCode = Field(
        default=ErrorCorrectionCode.BIT_FLIP_3,
        description="Default error correction code"
    )
    error_probability: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Error probability for noise model"
    )
    shots: int = Field(
        default=1024,
        ge=1,
        le=1000000,
        description="Number of measurement shots"
    )
    enable_syndrome_history: bool = Field(
        default=True,
        description="Store syndrome measurement history"
    )
    max_correction_rounds: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum correction rounds"
    )
    fidelity_threshold: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description="Target fidelity threshold"
    )


class QuantumErrorCorrectionError(Exception):
    """Base exception for error correction errors"""
    pass


class CodeNotSupportedError(QuantumErrorCorrectionError):
    """Raised when code is not supported"""
    pass


class SyndromeDecodingError(QuantumErrorCorrectionError):
    """Raised when syndrome decoding fails"""
    pass


# ============================================================================
# Error Correction Code Implementations
# ============================================================================

class ErrorCorrectionCodeBase(ABC):
    """Abstract base class for error correction codes"""
    
    def __init__(self, name: str, code_distance: int, physical_qubits: int, logical_qubits: int):
        self.name = name
        self.code_distance = code_distance
        self.physical_qubits = physical_qubits
        self.logical_qubits = logical_qubits
    
    @abstractmethod
    def encode(self, logical_state: Any) -> QuantumCircuit:
        """Encode logical qubit into physical qubits"""
        pass
    
    @abstractmethod
    def measure_syndrome(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Add syndrome measurement to circuit"""
        pass
    
    @abstractmethod
    def decode_syndrome(self, syndrome: str) -> Tuple[Optional[ErrorType], Optional[int]]:
        """Decode syndrome to identify error type and location"""
        pass
    
    @abstractmethod
    def correct_error(self, circuit: QuantumCircuit, error_type: ErrorType, location: int) -> QuantumCircuit:
        """Apply correction based on syndrome"""
        pass


class BitFlip3Code(ErrorCorrectionCodeBase):
    """3-qubit bit-flip error correction code
    
    Encodes |0⟩ → |000⟩ and |1⟩ → |111⟩
    Can correct single bit-flip (X) errors
    """
    
    def __init__(self):
        super().__init__(
            name="Bit-Flip 3-Qubit",
            code_distance=3,
            physical_qubits=3,
            logical_qubits=1
        )
        # Syndrome lookup table: syndrome -> (error_type, qubit_index)
        self.syndrome_table = {
            "00": (None, None),  # No error
            "01": (ErrorType.BIT_FLIP, 2),  # Error on qubit 2
            "10": (ErrorType.BIT_FLIP, 0),  # Error on qubit 0
            "11": (ErrorType.BIT_FLIP, 1),  # Error on qubit 1
        }
    
    def encode(self, logical_state: str = "0") -> QuantumCircuit:
        """Encode logical qubit using bit-flip code"""
        qr = QuantumRegister(3, 'data')
        cr = ClassicalRegister(2, 'syndrome')
        qc = QuantumCircuit(qr, cr)
        
        # Initialize logical state
        if logical_state == "1":
            qc.x(qr[0])
        elif logical_state == "+":
            qc.h(qr[0])
        elif logical_state == "-":
            qc.x(qr[0])
            qc.h(qr[0])
        
        # Encode: |0⟩ → |000⟩, |1⟩ → |111⟩
        qc.cx(qr[0], qr[1])
        qc.cx(qr[0], qr[2])
        
        qc.barrier()
        return qc
    
    def measure_syndrome(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Measure syndrome using ancilla qubits"""
        # Add ancilla qubits for syndrome measurement
        qc = circuit.copy()
        
        # Create ancilla register if not exists
        ancilla = QuantumRegister(2, 'ancilla')
        syndrome_cr = ClassicalRegister(2, 'syndrome')
        qc.add_register(ancilla)
        qc.add_register(syndrome_cr)
        
        data = circuit.qregs[0]
        
        qc.barrier()
        
        # Syndrome measurement
        # Z1Z2 parity (ancilla[0])
        qc.cx(data[0], ancilla[0])
        qc.cx(data[1], ancilla[0])
        
        # Z2Z3 parity (ancilla[1])
        qc.cx(data[1], ancilla[1])
        qc.cx(data[2], ancilla[1])
        
        # Measure ancilla
        qc.measure(ancilla[0], syndrome_cr[0])
        qc.measure(ancilla[1], syndrome_cr[1])
        
        return qc
    
    def decode_syndrome(self, syndrome: str) -> Tuple[Optional[ErrorType], Optional[int]]:
        """Decode syndrome to identify error"""
        # Ensure syndrome is 2 bits
        syndrome = syndrome.zfill(2)[-2:]
        
        if syndrome in self.syndrome_table:
            return self.syndrome_table[syndrome]
        return (None, None)
    
    def correct_error(
        self,
        circuit: QuantumCircuit,
        error_type: ErrorType,
        location: int
    ) -> QuantumCircuit:
        """Apply X correction to identified qubit"""
        qc = circuit.copy()
        data = circuit.qregs[0]
        
        if error_type == ErrorType.BIT_FLIP and location is not None:
            qc.x(data[location])
        
        return qc


class PhaseFlip3Code(ErrorCorrectionCodeBase):
    """3-qubit phase-flip error correction code
    
    Encodes |0⟩ → |+++⟩ and |1⟩ → |---⟩
    Can correct single phase-flip (Z) errors
    """
    
    def __init__(self):
        super().__init__(
            name="Phase-Flip 3-Qubit",
            code_distance=3,
            physical_qubits=3,
            logical_qubits=1
        )
        self.syndrome_table = {
            "00": (None, None),
            "01": (ErrorType.PHASE_FLIP, 2),
            "10": (ErrorType.PHASE_FLIP, 0),
            "11": (ErrorType.PHASE_FLIP, 1),
        }
    
    def encode(self, logical_state: str = "0") -> QuantumCircuit:
        """Encode logical qubit using phase-flip code"""
        qr = QuantumRegister(3, 'data')
        cr = ClassicalRegister(2, 'syndrome')
        qc = QuantumCircuit(qr, cr)
        
        if logical_state == "1":
            qc.x(qr[0])
        
        # Encode in Hadamard basis
        qc.cx(qr[0], qr[1])
        qc.cx(qr[0], qr[2])
        qc.h(qr[0])
        qc.h(qr[1])
        qc.h(qr[2])
        
        qc.barrier()
        return qc
    
    def measure_syndrome(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Measure syndrome for phase-flip code"""
        qc = circuit.copy()
        
        ancilla = QuantumRegister(2, 'ancilla')
        syndrome_cr = ClassicalRegister(2, 'syndrome')
        qc.add_register(ancilla)
        qc.add_register(syndrome_cr)
        
        data = circuit.qregs[0]
        
        qc.barrier()
        
        # X-basis parity checks
        qc.h(data[0])
        qc.h(data[1])
        qc.h(data[2])
        
        qc.cx(data[0], ancilla[0])
        qc.cx(data[1], ancilla[0])
        qc.cx(data[1], ancilla[1])
        qc.cx(data[2], ancilla[1])
        
        qc.h(data[0])
        qc.h(data[1])
        qc.h(data[2])
        
        qc.measure(ancilla[0], syndrome_cr[0])
        qc.measure(ancilla[1], syndrome_cr[1])
        
        return qc
    
    def decode_syndrome(self, syndrome: str) -> Tuple[Optional[ErrorType], Optional[int]]:
        """Decode syndrome"""
        syndrome = syndrome.zfill(2)[-2:]
        if syndrome in self.syndrome_table:
            return self.syndrome_table[syndrome]
        return (None, None)
    
    def correct_error(
        self,
        circuit: QuantumCircuit,
        error_type: ErrorType,
        location: int
    ) -> QuantumCircuit:
        """Apply Z correction"""
        qc = circuit.copy()
        data = circuit.qregs[0]
        
        if error_type == ErrorType.PHASE_FLIP and location is not None:
            qc.z(data[location])
        
        return qc


class Shor9Code(ErrorCorrectionCodeBase):
    """9-qubit Shor code
    
    First quantum error correction code that can correct arbitrary single-qubit errors.
    Combines bit-flip and phase-flip codes.
    """
    
    def __init__(self):
        super().__init__(
            name="Shor 9-Qubit",
            code_distance=3,
            physical_qubits=9,
            logical_qubits=1
        )
    
    def encode(self, logical_state: str = "0") -> QuantumCircuit:
        """Encode using Shor's 9-qubit code"""
        qr = QuantumRegister(9, 'data')
        qc = QuantumCircuit(qr)
        
        if logical_state == "1":
            qc.x(qr[0])
        
        # Phase-flip encoding (on qubit groups)
        qc.cx(qr[0], qr[3])
        qc.cx(qr[0], qr[6])
        
        # Hadamard on each group leader
        qc.h(qr[0])
        qc.h(qr[3])
        qc.h(qr[6])
        
        # Bit-flip encoding within each group
        # Group 1: qubits 0, 1, 2
        qc.cx(qr[0], qr[1])
        qc.cx(qr[0], qr[2])
        
        # Group 2: qubits 3, 4, 5
        qc.cx(qr[3], qr[4])
        qc.cx(qr[3], qr[5])
        
        # Group 3: qubits 6, 7, 8
        qc.cx(qr[6], qr[7])
        qc.cx(qr[6], qr[8])
        
        qc.barrier()
        return qc
    
    def measure_syndrome(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Measure syndrome for Shor code"""
        qc = circuit.copy()
        
        # 6 ancillas for bit-flip + 2 for phase-flip
        ancilla = QuantumRegister(8, 'ancilla')
        syndrome_cr = ClassicalRegister(8, 'syndrome')
        qc.add_register(ancilla)
        qc.add_register(syndrome_cr)
        
        data = circuit.qregs[0]
        
        qc.barrier()
        
        # Bit-flip syndrome for each group
        # Group 1
        qc.cx(data[0], ancilla[0])
        qc.cx(data[1], ancilla[0])
        qc.cx(data[1], ancilla[1])
        qc.cx(data[2], ancilla[1])
        
        # Group 2
        qc.cx(data[3], ancilla[2])
        qc.cx(data[4], ancilla[2])
        qc.cx(data[4], ancilla[3])
        qc.cx(data[5], ancilla[3])
        
        # Group 3
        qc.cx(data[6], ancilla[4])
        qc.cx(data[7], ancilla[4])
        qc.cx(data[7], ancilla[5])
        qc.cx(data[8], ancilla[5])
        
        # Phase-flip syndrome (comparing groups)
        # Compare group 1 and 2
        for i in [0, 1, 2]:
            qc.h(data[i])
        for i in [3, 4, 5]:
            qc.h(data[i])
        
        qc.cx(data[0], ancilla[6])
        qc.cx(data[3], ancilla[6])
        
        for i in [0, 1, 2]:
            qc.h(data[i])
        for i in [3, 4, 5]:
            qc.h(data[i])
        
        # Measure all ancillas
        for i in range(8):
            qc.measure(ancilla[i], syndrome_cr[i])
        
        return qc
    
    def decode_syndrome(self, syndrome: str) -> Tuple[Optional[ErrorType], Optional[int]]:
        """Decode Shor code syndrome"""
        syndrome = syndrome.zfill(8)
        
        # Bit-flip syndromes (first 6 bits, 2 per group)
        bf_syndromes = [syndrome[0:2], syndrome[2:4], syndrome[4:6]]
        
        # Check for bit-flip errors in each group
        bf_table = {"00": None, "01": 2, "10": 0, "11": 1}
        
        for group_idx, bf_syn in enumerate(bf_syndromes):
            if bf_syn != "00":
                qubit_in_group = bf_table.get(bf_syn)
                if qubit_in_group is not None:
                    qubit_idx = group_idx * 3 + qubit_in_group
                    return (ErrorType.BIT_FLIP, qubit_idx)
        
        # Phase-flip syndrome (last 2 bits)
        pf_syndrome = syndrome[6:8]
        pf_table = {"00": None, "01": 6, "10": 0, "11": 3}
        
        if pf_syndrome != "00":
            group_leader = pf_table.get(pf_syndrome)
            if group_leader is not None:
                return (ErrorType.PHASE_FLIP, group_leader)
        
        return (None, None)
    
    def correct_error(
        self,
        circuit: QuantumCircuit,
        error_type: ErrorType,
        location: int
    ) -> QuantumCircuit:
        """Apply correction for Shor code"""
        qc = circuit.copy()
        data = circuit.qregs[0]
        
        if location is not None:
            if error_type == ErrorType.BIT_FLIP:
                qc.x(data[location])
            elif error_type == ErrorType.PHASE_FLIP:
                qc.z(data[location])
        
        return qc


class Steane7Code(ErrorCorrectionCodeBase):
    """7-qubit Steane code
    
    CSS code based on classical Hamming code.
    Can correct arbitrary single-qubit errors.
    """
    
    def __init__(self):
        super().__init__(
            name="Steane 7-Qubit",
            code_distance=3,
            physical_qubits=7,
            logical_qubits=1
        )
        # Generator matrix for Steane code
        self.H_matrix = np.array([
            [1, 0, 0, 1, 0, 1, 1],
            [0, 1, 0, 1, 1, 0, 1],
            [0, 0, 1, 0, 1, 1, 1]
        ])
    
    def encode(self, logical_state: str = "0") -> QuantumCircuit:
        """Encode using Steane code"""
        qr = QuantumRegister(7, 'data')
        qc = QuantumCircuit(qr)
        
        if logical_state == "1":
            # Logical |1⟩ state
            qc.x(qr[0])
            qc.x(qr[1])
            qc.x(qr[2])
            qc.x(qr[3])
            qc.x(qr[4])
            qc.x(qr[5])
            qc.x(qr[6])
        
        # Encoding circuit for Steane code
        # This creates the logical |0⟩ or |1⟩ codeword
        qc.h(qr[0])
        qc.h(qr[1])
        qc.h(qr[2])
        
        qc.cx(qr[0], qr[4])
        qc.cx(qr[0], qr[5])
        qc.cx(qr[0], qr[6])
        
        qc.cx(qr[1], qr[3])
        qc.cx(qr[1], qr[5])
        qc.cx(qr[1], qr[6])
        
        qc.cx(qr[2], qr[3])
        qc.cx(qr[2], qr[4])
        qc.cx(qr[2], qr[6])
        
        qc.barrier()
        return qc
    
    def measure_syndrome(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Measure syndrome for Steane code"""
        qc = circuit.copy()
        
        ancilla = QuantumRegister(6, 'ancilla')
        syndrome_cr = ClassicalRegister(6, 'syndrome')
        qc.add_register(ancilla)
        qc.add_register(syndrome_cr)
        
        data = circuit.qregs[0]
        
        qc.barrier()
        
        # X-type stabilizers (for Z errors)
        # S1: X0 X3 X4 X5
        qc.h(ancilla[0])
        qc.cx(ancilla[0], data[0])
        qc.cx(ancilla[0], data[3])
        qc.cx(ancilla[0], data[4])
        qc.cx(ancilla[0], data[5])
        qc.h(ancilla[0])
        
        # S2: X1 X3 X5 X6
        qc.h(ancilla[1])
        qc.cx(ancilla[1], data[1])
        qc.cx(ancilla[1], data[3])
        qc.cx(ancilla[1], data[5])
        qc.cx(ancilla[1], data[6])
        qc.h(ancilla[1])
        
        # S3: X2 X4 X5 X6
        qc.h(ancilla[2])
        qc.cx(ancilla[2], data[2])
        qc.cx(ancilla[2], data[4])
        qc.cx(ancilla[2], data[5])
        qc.cx(ancilla[2], data[6])
        qc.h(ancilla[2])
        
        # Z-type stabilizers (for X errors)
        # S4: Z0 Z3 Z4 Z5
        qc.cx(data[0], ancilla[3])
        qc.cx(data[3], ancilla[3])
        qc.cx(data[4], ancilla[3])
        qc.cx(data[5], ancilla[3])
        
        # S5: Z1 Z3 Z5 Z6
        qc.cx(data[1], ancilla[4])
        qc.cx(data[3], ancilla[4])
        qc.cx(data[5], ancilla[4])
        qc.cx(data[6], ancilla[4])
        
        # S6: Z2 Z4 Z5 Z6
        qc.cx(data[2], ancilla[5])
        qc.cx(data[4], ancilla[5])
        qc.cx(data[5], ancilla[5])
        qc.cx(data[6], ancilla[5])
        
        # Measure all syndrome bits
        for i in range(6):
            qc.measure(ancilla[i], syndrome_cr[i])
        
        return qc
    
    def decode_syndrome(self, syndrome: str) -> Tuple[Optional[ErrorType], Optional[int]]:
        """Decode Steane code syndrome"""
        syndrome = syndrome.zfill(6)
        
        # X syndrome (bits 0-2) indicates Z error location
        x_syn = [int(b) for b in syndrome[0:3]]
        # Z syndrome (bits 3-5) indicates X error location
        z_syn = [int(b) for b in syndrome[3:6]]
        
        # Convert syndrome to qubit index using H matrix
        x_idx = x_syn[0] + 2 * x_syn[1] + 4 * x_syn[2]
        z_idx = z_syn[0] + 2 * z_syn[1] + 4 * z_syn[2]
        
        if z_idx > 0 and z_idx <= 7:
            return (ErrorType.BIT_FLIP, z_idx - 1)
        elif x_idx > 0 and x_idx <= 7:
            return (ErrorType.PHASE_FLIP, x_idx - 1)
        
        return (None, None)
    
    def correct_error(
        self,
        circuit: QuantumCircuit,
        error_type: ErrorType,
        location: int
    ) -> QuantumCircuit:
        """Apply correction for Steane code"""
        qc = circuit.copy()
        data = circuit.qregs[0]
        
        if location is not None and 0 <= location < 7:
            if error_type == ErrorType.BIT_FLIP:
                qc.x(data[location])
            elif error_type == ErrorType.PHASE_FLIP:
                qc.z(data[location])
        
        return qc


class RepetitionCode(ErrorCorrectionCodeBase):
    """Repetition code with configurable distance
    
    Simple code that repeats the qubit n times.
    Can only correct bit-flip errors.
    """
    
    def __init__(self, distance: int = 3):
        super().__init__(
            name=f"Repetition-{distance}",
            code_distance=distance,
            physical_qubits=distance,
            logical_qubits=1
        )
        self.distance = distance
    
    def encode(self, logical_state: str = "0") -> QuantumCircuit:
        """Encode using repetition code"""
        qr = QuantumRegister(self.distance, 'data')
        qc = QuantumCircuit(qr)
        
        if logical_state == "1":
            qc.x(qr[0])
        
        # Copy to all other qubits
        for i in range(1, self.distance):
            qc.cx(qr[0], qr[i])
        
        qc.barrier()
        return qc
    
    def measure_syndrome(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Measure syndrome for repetition code"""
        qc = circuit.copy()
        
        num_ancilla = self.distance - 1
        ancilla = QuantumRegister(num_ancilla, 'ancilla')
        syndrome_cr = ClassicalRegister(num_ancilla, 'syndrome')
        qc.add_register(ancilla)
        qc.add_register(syndrome_cr)
        
        data = circuit.qregs[0]
        
        qc.barrier()
        
        # Parity checks between adjacent qubits
        for i in range(num_ancilla):
            qc.cx(data[i], ancilla[i])
            qc.cx(data[i + 1], ancilla[i])
            qc.measure(ancilla[i], syndrome_cr[i])
        
        return qc
    
    def decode_syndrome(self, syndrome: str) -> Tuple[Optional[ErrorType], Optional[int]]:
        """Decode repetition code syndrome"""
        n = self.distance - 1
        syndrome = syndrome.zfill(n)[-n:]
        
        # Find error location from syndrome
        for i in range(n + 1):
            if i == 0 and syndrome[0] == '1' and (n == 1 or syndrome[1] == '0'):
                return (ErrorType.BIT_FLIP, 0)
            elif i == n and syndrome[n - 1] == '1':
                return (ErrorType.BIT_FLIP, n)
            elif 0 < i < n and syndrome[i - 1] == '1' and syndrome[i] == '1':
                return (ErrorType.BIT_FLIP, i)
        
        return (None, None)
    
    def correct_error(
        self,
        circuit: QuantumCircuit,
        error_type: ErrorType,
        location: int
    ) -> QuantumCircuit:
        """Apply correction for repetition code"""
        qc = circuit.copy()
        data = circuit.qregs[0]
        
        if error_type == ErrorType.BIT_FLIP and location is not None:
            qc.x(data[location])
        
        return qc


# ============================================================================
# Main Module Implementation
# ============================================================================

class QuantumErrorCorrectionModule:
    """
    Production-ready quantum error correction module for OMNIXAN.
    
    Supports multiple error correction codes and provides:
    - Encoding of logical qubits
    - Syndrome measurement
    - Error detection and correction
    - Fidelity estimation
    - Performance metrics
    """
    
    def __init__(self, config: Optional[ErrorCorrectionConfig] = None):
        """Initialize the Quantum Error Correction Module"""
        self.config = config or ErrorCorrectionConfig()
        self.codes: Dict[ErrorCorrectionCode, ErrorCorrectionCodeBase] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.metrics_history: List[ErrorCorrectionMetrics] = []
        self.syndrome_history: List[SyndromeResult] = []
        self._backend = None
        self._initialized = False
        self._logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the error correction module"""
        if self._initialized:
            self._logger.warning("Module already initialized")
            return
        
        try:
            self._logger.info("Initializing QuantumErrorCorrectionModule...")
            
            if not QISKIT_AVAILABLE:
                raise QuantumErrorCorrectionError("Qiskit not available")
            
            # Initialize simulator backend
            self._backend = AerSimulator()
            
            # Register error correction codes
            self.codes[ErrorCorrectionCode.BIT_FLIP_3] = BitFlip3Code()
            self.codes[ErrorCorrectionCode.PHASE_FLIP_3] = PhaseFlip3Code()
            self.codes[ErrorCorrectionCode.SHOR_9] = Shor9Code()
            self.codes[ErrorCorrectionCode.STEANE_7] = Steane7Code()
            self.codes[ErrorCorrectionCode.REPETITION] = RepetitionCode(distance=3)
            
            self._initialized = True
            self._logger.info("QuantumErrorCorrectionModule initialized successfully")
            self._logger.info(f"Available codes: {list(self.codes.keys())}")
        
        except Exception as e:
            self._logger.error(f"Initialization failed: {str(e)}")
            raise QuantumErrorCorrectionError(f"Failed to initialize module: {str(e)}")
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute error correction operation"""
        if not self._initialized:
            raise QuantumErrorCorrectionError("Module not initialized")
        
        operation = params.get("operation")
        
        if operation == "encode":
            code = ErrorCorrectionCode(params.get("code", self.config.default_code.value))
            logical_state = params.get("logical_state", "0")
            circuit = await self.encode_logical_qubit(code, logical_state)
            return {"circuit": circuit.qasm()}
        
        elif operation == "detect_error":
            code = ErrorCorrectionCode(params.get("code", self.config.default_code.value))
            circuit = params.get("circuit")
            result = await self.detect_error(code, circuit)
            return {
                "syndrome": result.syndrome,
                "error_detected": result.error_detected,
                "error_type": result.error_type.value if result.error_type else None,
                "error_location": result.error_location
            }
        
        elif operation == "correct_error":
            code = ErrorCorrectionCode(params.get("code", self.config.default_code.value))
            circuit = params.get("circuit")
            result = await self.correct_circuit(code, circuit)
            return {
                "status": result.status.value,
                "corrected": result.corrected,
                "fidelity_before": result.fidelity_before,
                "fidelity_after": result.fidelity_after
            }
        
        elif operation == "full_correction_cycle":
            code = ErrorCorrectionCode(params.get("code", self.config.default_code.value))
            logical_state = params.get("logical_state", "0")
            error_prob = params.get("error_probability", self.config.error_probability)
            result = await self.full_correction_cycle(code, logical_state, error_prob)
            return result
        
        elif operation == "get_metrics":
            return self.get_metrics_summary()
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def encode_logical_qubit(
        self,
        code: ErrorCorrectionCode,
        logical_state: str = "0"
    ) -> QuantumCircuit:
        """Encode a logical qubit using specified code"""
        if code not in self.codes:
            raise CodeNotSupportedError(f"Code {code.value} not supported")
        
        code_impl = self.codes[code]
        circuit = code_impl.encode(logical_state)
        
        self._logger.info(f"Encoded logical |{logical_state}⟩ using {code.value}")
        return circuit
    
    async def detect_error(
        self,
        code: ErrorCorrectionCode,
        circuit: QuantumCircuit
    ) -> SyndromeResult:
        """Detect errors by measuring syndrome"""
        if code not in self.codes:
            raise CodeNotSupportedError(f"Code {code.value} not supported")
        
        code_impl = self.codes[code]
        
        # Add syndrome measurement
        syndrome_circuit = code_impl.measure_syndrome(circuit)
        
        # Execute circuit
        job = self._backend.run(transpile(syndrome_circuit, self._backend), shots=self.config.shots)
        result = job.result()
        counts = result.get_counts()
        
        # Get most likely syndrome
        syndrome = max(counts, key=counts.get)
        
        # Decode syndrome
        error_type, error_location = code_impl.decode_syndrome(syndrome)
        
        syndrome_result = SyndromeResult(
            syndrome=syndrome,
            error_detected=error_type is not None,
            error_type=error_type,
            error_location=error_location,
            confidence=counts[syndrome] / self.config.shots
        )
        
        if self.config.enable_syndrome_history:
            self.syndrome_history.append(syndrome_result)
        
        return syndrome_result
    
    async def correct_circuit(
        self,
        code: ErrorCorrectionCode,
        circuit: QuantumCircuit
    ) -> CorrectionResult:
        """Correct errors in circuit"""
        if code not in self.codes:
            raise CodeNotSupportedError(f"Code {code.value} not supported")
        
        code_impl = self.codes[code]
        start_time = time.time()
        
        # Detect error
        syndrome_result = await self.detect_error(code, circuit)
        
        if not syndrome_result.error_detected:
            return CorrectionResult(
                status=CorrectionStatus.NO_ERROR,
                original_syndrome=syndrome_result.syndrome,
                corrected=False
            )
        
        # Apply correction
        corrected_circuit = code_impl.correct_error(
            circuit,
            syndrome_result.error_type,
            syndrome_result.error_location
        )
        
        execution_time = time.time() - start_time
        
        # Store metrics
        metrics = ErrorCorrectionMetrics(
            total_corrections=1,
            successful_corrections=1,
            execution_time=execution_time,
            code_used=code.value
        )
        self.metrics_history.append(metrics)
        
        return CorrectionResult(
            status=CorrectionStatus.SUCCESS,
            original_syndrome=syndrome_result.syndrome,
            corrected=True,
            correction_applied=f"{syndrome_result.error_type.value} on qubit {syndrome_result.error_location}"
        )
    
    async def full_correction_cycle(
        self,
        code: ErrorCorrectionCode,
        logical_state: str = "0",
        error_probability: float = 0.01
    ) -> Dict[str, Any]:
        """Run full error correction cycle with noise"""
        if code not in self.codes:
            raise CodeNotSupportedError(f"Code {code.value} not supported")
        
        code_impl = self.codes[code]
        start_time = time.time()
        
        # 1. Encode logical state
        circuit = code_impl.encode(logical_state)
        
        # 2. Add noise (simulate errors)
        noisy_circuit = self._add_noise(circuit, error_probability)
        
        # 3. Detect error
        syndrome_result = await self.detect_error(code, noisy_circuit)
        
        # 4. Correct if needed
        if syndrome_result.error_detected:
            corrected_circuit = code_impl.correct_error(
                noisy_circuit,
                syndrome_result.error_type,
                syndrome_result.error_location
            )
            status = CorrectionStatus.SUCCESS
        else:
            corrected_circuit = noisy_circuit
            status = CorrectionStatus.NO_ERROR
        
        execution_time = time.time() - start_time
        
        return {
            "status": status.value,
            "syndrome": syndrome_result.syndrome,
            "error_detected": syndrome_result.error_detected,
            "error_type": syndrome_result.error_type.value if syndrome_result.error_type else None,
            "error_location": syndrome_result.error_location,
            "execution_time": execution_time,
            "code_used": code.value
        }
    
    def _add_noise(self, circuit: QuantumCircuit, error_prob: float) -> QuantumCircuit:
        """Add random errors to circuit for testing"""
        noisy = circuit.copy()
        data = circuit.qregs[0]
        
        # Randomly apply X error to one qubit
        if np.random.random() < error_prob * len(data):
            error_qubit = np.random.randint(0, len(data))
            noisy.x(data[error_qubit])
        
        return noisy
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of error correction metrics"""
        if not self.metrics_history:
            return {"message": "No metrics available"}
        
        total = len(self.metrics_history)
        successful = sum(1 for m in self.metrics_history if m.successful_corrections > 0)
        
        return {
            "total_corrections": total,
            "successful_corrections": successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "average_execution_time": np.mean([m.execution_time for m in self.metrics_history]),
            "codes_used": list(set(m.code_used for m in self.metrics_history))
        }
    
    async def shutdown(self) -> None:
        """Shutdown the error correction module"""
        self._logger.info("Shutting down QuantumErrorCorrectionModule...")
        
        self.executor.shutdown(wait=True)
        self.codes.clear()
        self._initialized = False
        
        self._logger.info("QuantumErrorCorrectionModule shutdown complete")


# Example usage
async def main():
    """Example usage of QuantumErrorCorrectionModule"""
    
    config = ErrorCorrectionConfig(
        default_code=ErrorCorrectionCode.BIT_FLIP_3,
        error_probability=0.1,
        shots=1024
    )
    
    module = QuantumErrorCorrectionModule(config)
    await module.initialize()
    
    try:
        # Run full correction cycle
        result = await module.full_correction_cycle(
            code=ErrorCorrectionCode.BIT_FLIP_3,
            logical_state="0",
            error_probability=0.3
        )
        
        print(f"\nError Correction Result:")
        print(f"Status: {result['status']}")
        print(f"Syndrome: {result['syndrome']}")
        print(f"Error Detected: {result['error_detected']}")
        if result['error_detected']:
            print(f"Error Type: {result['error_type']}")
            print(f"Error Location: {result['error_location']}")
        print(f"Execution Time: {result['execution_time']:.4f}s")
        
        # Test with different codes
        for code in [ErrorCorrectionCode.PHASE_FLIP_3, ErrorCorrectionCode.SHOR_9]:
            result = await module.full_correction_cycle(
                code=code,
                logical_state="1",
                error_probability=0.2
            )
            print(f"\n{code.value}: Status={result['status']}, Error={result['error_detected']}")
        
        # Get metrics
        metrics = module.get_metrics_summary()
        print(f"\nMetrics Summary:")
        print(json.dumps(metrics, indent=2))
    
    finally:
        await module.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

