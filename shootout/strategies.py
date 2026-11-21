"""The four strategies being compared.

    none     : raw noisy result
    rem      : readout-error mitigation (Mitiq REM)
    zne      : zero-noise extrapolation (Mitiq ZNE, global circuit folding)
    rem+zne  : both - REM cleans every noisy run, ZNE extrapolates the cleaned values
"""
from __future__ import annotations

import numpy as np
from mitiq import rem, zne
from mitiq.rem import generate_tensored_inverse_confusion_matrix
from mitiq.zne.inference import ExpFactory, LinearFactory, RichardsonFactory
from mitiq.zne.scaling import fold_global
from qiskit import QuantumCircuit

STRATEGIES = ["none", "rem", "zne", "rem+zne"]
LABELS = {
    "none": "No mitigation",
    "rem": "Readout (REM)",
    "zne": "ZNE",
    "rem+zne": "REM + ZNE",
}


def make_factory(kind: str = "richardson", scales=(1, 3, 5)):
    """ZNE extrapolation model: fit E(noise scale) and read off the value at 0."""
    kind = kind.lower()
    if kind == "richardson":
        return RichardsonFactory(list(scales))
    if kind == "linear":
        return LinearFactory(list(scales))
    if kind in ("exp", "exponential"):
        return ExpFactory(list(scales), asymptote=0.0)
    raise ValueError(f"unknown ZNE fit '{kind}'")


def calibrate_confusion_matrices(executor, n_qubits: int) -> list:
    """Measure the readout errors of each qubit, like on a real device.

    Prepare |00..0> and |11..1>, measure many times, and count how often each
    qubit is read wrongly. Returns one 2x2 confusion matrix per qubit
    (columns = prepared state, rows = measured state, Mitiq convention).
    """
    zeros = QuantumCircuit(n_qubits)
    ones = QuantumCircuit(n_qubits)
    ones.x(range(n_qubits))
    r0 = executor.run(zeros)  # shape (shots, n_qubits) of 0/1
    r1 = executor.run(ones)
    matrices = []
    for q in range(n_qubits):
        p01 = r0[:, q].mean()          # prepared 0, read 1
        p10 = 1.0 - r1[:, q].mean()    # prepared 1, read 0
        matrices.append(np.array([[1 - p01, p10], [p01, 1 - p10]]))
    return matrices


def inverse_confusion_matrix_for(matrices: list, qubits) -> np.ndarray:
    """Invert each measured qubit's 2x2 matrix and combine with a Kronecker
    product (assumes qubit readout errors are independent). Only the qubits the
    observable actually measures are included, in ascending order."""
    chosen = [matrices[q] for q in sorted(qubits)]
    return generate_tensored_inverse_confusion_matrix(len(chosen), chosen)


def estimate(strategy: str, circuit, observable, executor, inv_cm, factory) -> float:
    """Return the estimate of <observable> using the chosen strategy."""
    run = executor.fn  # annotated function Mitiq can inspect
    if strategy == "none":
        value = observable.expectation(circuit, run)
    elif strategy == "rem":
        value = rem.execute_with_rem(
            circuit, run, observable, inverse_confusion_matrix=inv_cm
        )
    elif strategy == "zne":
        value = zne.execute_with_zne(
            circuit, run, observable, factory=factory, scale_noise=fold_global
        )
    elif strategy == "rem+zne":
        # wrap the executor so EVERY run (at every noise scale) is readout-corrected,
        # then let ZNE extrapolate those cleaned values.
        rem_executor = rem.mitigate_executor(run, inverse_confusion_matrix=inv_cm)
        value = zne.execute_with_zne(
            circuit, rem_executor, observable, factory=factory, scale_noise=fold_global
        )
    else:
        raise ValueError(strategy)
    return float(np.real(value))
