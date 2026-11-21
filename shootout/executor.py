"""The 'executor': the one function Mitiq needs from us.

Mitiq is hardware-agnostic. It hands us circuits (possibly folded to amplify
noise) and expects back raw measurement bitstrings. This class runs them on a
noisy Aer simulator and counts how many circuits / shots were used, so we can
report the *cost* of each mitigation strategy.

NOTE: no ``from __future__ import annotations`` here on purpose. Mitiq reads
the return-type annotation (-> MeasurementResult) of the executor function to
know how to interpret what it gets back.
"""
import numpy as np
from mitiq import MeasurementResult
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from .benchmarks import NATIVE_GATES


class NoisyExecutor:
    def __init__(self, noise_model, shots=8192, seed=0):
        self.sim = AerSimulator(noise_model=noise_model)
        self.shots = shots
        self.rng = np.random.default_rng(seed)
        self.executions = 0  # number of circuits run

    @property
    def total_shots(self):
        return self.executions * self.shots

    def run(self, circuit):
        """Run one circuit, return an (shots x n_qubits) 0/1 numpy array."""
        qc = circuit.copy()
        if qc.num_clbits == 0:
            qc.measure_all()
        # optimization_level=0: keep folded gates (G G^dag G) exactly as given
        qc = transpile(qc, basis_gates=NATIVE_GATES + ["measure"], optimization_level=0)
        seed = int(self.rng.integers(2**31 - 1))
        result = self.sim.run(qc, shots=self.shots, memory=True, seed_simulator=seed).result()
        self.executions += 1
        # Qiskit prints qubit 0 as the RIGHTMOST character; Mitiq wants
        # qubit 0 first, so reverse every bitstring.
        return np.array([[int(b) for b in reversed(s)] for s in result.get_memory()])

    @property
    def fn(self):
        """The annotated function that is handed to Mitiq."""

        def execute(circuit: QuantumCircuit) -> MeasurementResult:
            return MeasurementResult(self.run(circuit).tolist())

        return execute
