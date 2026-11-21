"""A simple, controllable noise model (so no IBM account / internet needed).

Two independent error sources, mirroring what real devices have:

* gate noise  - depolarizing error after every 1-qubit gate (sx, x) and every
                2-qubit gate (cx). Zero-noise extrapolation (ZNE) targets this.
* readout noise - each qubit's measurement can flip 0->1 or 1->0, with a
                different probability per qubit. Readout-error mitigation
                (REM) targets this.

Default numbers are in the range of current IBM superconducting devices.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error


@dataclass(frozen=True)
class NoiseConfig:
    p1: float = 0.001   # depolarizing prob. per 1-qubit gate (sx, x)
    p2: float = 0.010   # depolarizing prob. per cx
    p01: float = 0.020  # base P(read 1 | prepared 0)
    p10: float = 0.040  # base P(read 0 | prepared 1)

    def scaled(self, gate: float = 1.0, readout: float = 1.0) -> "NoiseConfig":
        """Return a copy with gate / readout error rates multiplied."""
        return replace(
            self,
            p1=self.p1 * gate,
            p2=self.p2 * gate,
            p01=self.p01 * readout,
            p10=self.p10 * readout,
        )

    def readout_probs(self, n_qubits: int):
        """Per-qubit (p01, p10); qubits differ by up to +-30 % like real chips."""
        out = []
        for q in range(n_qubits):
            f = 0.7 + 0.6 * (q / max(n_qubits - 1, 1))
            out.append((min(self.p01 * f, 0.5), min(self.p10 * f, 0.5)))
        return out


def build_noise_model(cfg: NoiseConfig, n_qubits: int) -> NoiseModel:
    nm = NoiseModel(basis_gates=["rz", "sx", "x", "cx"])
    if cfg.p1 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(cfg.p1, 1), ["sx", "x"])
    if cfg.p2 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(cfg.p2, 2), ["cx"])
    for q, (p01, p10) in enumerate(cfg.readout_probs(n_qubits)):
        if p01 > 0 or p10 > 0:
            # rows = prepared state, columns = measured state
            nm.add_readout_error(ReadoutError([[1 - p01, p01], [p10, 1 - p10]]), [q])
    return nm
