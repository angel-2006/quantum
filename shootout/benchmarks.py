"""The four benchmark circuits and the observable measured on each.

Every benchmark is a noiseless circuit (no measurements) plus an observable
O = sum_k c_k * Z_{q_k}  (or a Z-product for GHZ). We compare how close each
mitigation strategy gets to the *exact* value <O> computed from the ideal
statevector.

Circuits are stored already translated to the native gate set
{rz, sx, x, cx}. This is what real IBM hardware uses, and it means every gate
that Mitiq later "folds" corresponds to a gate that carries noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mitiq import Observable, PauliString
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector

NATIVE_GATES = ["rz", "sx", "x", "cx"]


def to_native(circuit: QuantumCircuit) -> QuantumCircuit:
    """Translate to native gates WITHOUT optimising (level 0).

    Level 0 matters: higher levels would cancel the U U^dagger pairs that ZNE
    inserts and silently destroy the noise scaling.
    """
    return transpile(circuit, basis_gates=NATIVE_GATES, optimization_level=0)


@dataclass
class Benchmark:
    name: str
    description: str
    circuit: QuantumCircuit
    # list of (coefficient, {qubit_index: "Z"}) -- we only use Z-type terms
    terms: list

    @property
    def n_qubits(self) -> int:
        return self.circuit.num_qubits

    @property
    def measured_qubits(self) -> list:
        """Qubits the observable acts on (the only ones Mitiq measures)."""
        return sorted({q for _, ops in self.terms for q in ops})

    def mitiq_observable(self) -> Observable:
        paulis = []
        for coeff, ops in self.terms:
            qubits = tuple(sorted(ops))
            spec = "".join(ops[q] for q in qubits)
            paulis.append(PauliString(spec, coeff=coeff, support=qubits))
        return Observable(*paulis)

    def ideal_value(self) -> float:
        """Exact <O> from the noiseless statevector (independent of Mitiq)."""
        op = SparsePauliOp.from_sparse_list(
            [("".join(o[q] for q in sorted(o)), sorted(o), c) for c, o in self.terms],
            num_qubits=self.n_qubits,
        )
        return float(Statevector(self.circuit).expectation_value(op).real)

    def stats(self) -> dict:
        ops = self.circuit.count_ops()
        return {
            "benchmark": self.name,
            "qubits": self.n_qubits,
            "depth": self.circuit.depth(),
            "cx_gates": ops.get("cx", 0),
            "total_gates": sum(ops.values()),
            "ideal_value": round(self.ideal_value(), 4),
        }


def _signed_z_terms(bits, qubits=None):
    """Normalised sum (1/n) * sum_i (-1)^{b_i} Z_i  -> equals 1 when the
    register is exactly in basis state |bits>."""
    qubits = list(range(len(bits))) if qubits is None else qubits
    n = len(qubits)
    return [((-1) ** b / n, {q: "Z"}) for b, q in zip(bits, qubits)]


# --------------------------------------------------------------------------
def ghz(n: int = 4) -> Benchmark:
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return Benchmark(
        name=f"GHZ-{n}",
        description=f"{n}-qubit GHZ state, observable = parity Z^{n} (ideal +1)",
        circuit=to_native(qc),
        terms=[(1.0, {q: "Z" for q in range(n)})],
    )


def bernstein_vazirani(secret=(1, 0, 1, 1)) -> Benchmark:
    n = len(secret)
    qc = QuantumCircuit(n + 1)  # last qubit is the ancilla
    qc.x(n)
    qc.h(range(n + 1))
    for i, bit in enumerate(secret):
        if bit:
            qc.cx(i, n)
    qc.h(range(n))
    return Benchmark(
        name=f"BV-{''.join(map(str, secret))}",
        description="Bernstein-Vazirani, output register = secret string (ideal +1)",
        circuit=to_native(qc),
        terms=_signed_z_terms(secret),
    )


def _qft(qc: QuantumCircuit, qubits, inverse: bool = False):
    """Textbook QFT built from H and controlled-phase gates."""
    n = len(qubits)
    ops = []
    for j in range(n):
        ops.append(("h", qubits[j], None, None))
        for k in range(j + 1, n):
            ops.append(("cp", qubits[k], qubits[j], np.pi / 2 ** (k - j)))
    if inverse:
        ops = ops[::-1]
    for name, a, b, ang in ops:
        if name == "h":
            qc.h(a)
        else:
            qc.cp(-ang if inverse else ang, a, b)


def qft_roundtrip(bits=(0, 1, 1, 0)) -> Benchmark:
    n = len(bits)
    qc = QuantumCircuit(n)
    for i, b in enumerate(bits):
        if b:
            qc.x(i)
    _qft(qc, list(range(n)))
    _qft(qc, list(range(n)), inverse=True)
    return Benchmark(
        name=f"QFT-{n}",
        description="QFT followed by inverse QFT on a basis state (ideal +1); deepest circuit",
        circuit=to_native(qc),
        terms=_signed_z_terms(bits),
    )


def grover3(marked=(1, 0, 1), iterations: int = 2) -> Benchmark:
    n = 3
    qc = QuantumCircuit(n)
    qc.h(range(n))

    def ccz():
        qc.h(2)
        qc.ccx(0, 1, 2)
        qc.h(2)

    for _ in range(iterations):
        # oracle: phase-flip the marked state
        for q, b in enumerate(marked):
            if not b:
                qc.x(q)
        ccz()
        for q, b in enumerate(marked):
            if not b:
                qc.x(q)
        # diffusion operator
        qc.h(range(n))
        qc.x(range(n))
        ccz()
        qc.x(range(n))
        qc.h(range(n))
    return Benchmark(
        name="Grover-3",
        description="3-qubit Grover, marked state 101, 2 iterations (ideal < 1: algorithm itself is ~94.5% accurate)",
        circuit=to_native(qc),
        terms=_signed_z_terms(marked),
    )


def all_benchmarks() -> list[Benchmark]:
    return [ghz(4), bernstein_vazirani(), qft_roundtrip(), grover3()]
