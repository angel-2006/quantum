"""Experiment driver: sanity checks, main shoot-out, noise sweeps, ZNE picture."""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from mitiq import zne
from mitiq.zne.scaling import fold_global

from .benchmarks import Benchmark, all_benchmarks
from .executor import NoisyExecutor
from .noise import NoiseConfig, build_noise_model
from .strategies import (
    STRATEGIES,
    calibrate_confusion_matrices,
    estimate,
    inverse_confusion_matrix_for,
    make_factory,
)

CALIBRATION_CIRCUITS = 2  # |0..0> and |1..1>


# --------------------------------------------------------------------------
# One trial = one benchmark, one random seed, all four strategies
# --------------------------------------------------------------------------
def _run_task(task: dict) -> list[dict]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    bench = all_benchmarks()[task["bench_idx"]]
    cfg: NoiseConfig = task["cfg"]
    shots, trial, fit = task["shots"], task["trial"], task["fit"]
    ideal = bench.ideal_value()
    obs = bench.mitiq_observable()
    nm = build_noise_model(cfg, bench.n_qubits)

    def seed_for(k):
        return int(np.random.SeedSequence([task["seed"], task["bench_idx"], trial, k, task["cond_id"]]).generate_state(1)[0])

    np.random.seed(seed_for(77))  # Mitiq's REM resamples with the global NumPy RNG -> make runs reproducible

    # readout calibration (done once per trial, shared by the REM strategies)
    cal = NoisyExecutor(nm, shots, seed_for(99))
    cms = calibrate_confusion_matrices(cal, bench.n_qubits)
    inv_cm = inverse_confusion_matrix_for(cms, bench.measured_qubits)

    rows = []
    for k, strat in enumerate(STRATEGIES):
        ex = NoisyExecutor(nm, shots, seed_for(k))
        value = estimate(strat, bench.circuit, obs, ex, inv_cm, make_factory(fit))
        used_cal = strat.startswith("rem")
        execs = ex.executions + (CALIBRATION_CIRCUITS if used_cal else 0)
        rows.append(
            dict(
                experiment=task["experiment"],
                factor=task["factor"],
                benchmark=bench.name,
                trial=trial,
                strategy=strat,
                estimate=value,
                ideal=ideal,
                abs_error=abs(value - ideal),
                circuit_executions=execs,
                total_shots=execs * shots,
            )
        )
    return rows


def _run_tasks(tasks: list[dict], workers: int, label: str) -> pd.DataFrame:
    t0 = time.time()
    rows: list[dict] = []
    if workers <= 1:
        for i, t in enumerate(tasks, 1):
            rows += _run_task(t)
            if i % 10 == 0 or i == len(tasks):
                print(f"  [{label}] {i}/{len(tasks)} tasks  ({time.time() - t0:.0f}s)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, r in enumerate(pool.map(_run_task, tasks, chunksize=2), 1):
                rows += r
                if i % 20 == 0 or i == len(tasks):
                    print(f"  [{label}] {i}/{len(tasks)} tasks  ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def _make_tasks(experiment, factor, cfg, trials, shots, fit, seed, cond_id):
    n_bench = len(all_benchmarks())
    return [
        dict(experiment=experiment, factor=factor, cfg=cfg, bench_idx=b, trial=t,
             shots=shots, fit=fit, seed=seed, cond_id=cond_id)
        for b in range(n_bench) for t in range(trials)
    ]


# --------------------------------------------------------------------------
# Public experiments
# --------------------------------------------------------------------------
def run_main(cfg, trials, shots, fit, seed, workers) -> pd.DataFrame:
    tasks = _make_tasks("main", 1.0, cfg, trials, shots, fit, seed, cond_id=0)
    return _run_tasks(tasks, workers, "main")


def run_sweeps(cfg, trials, shots, fit, seed, workers, gate_factors, readout_factors) -> pd.DataFrame:
    frames = []
    cid = 1
    for f in gate_factors:
        tasks = _make_tasks("sweep_gate", f, cfg.scaled(gate=f), trials, shots, fit, seed, cid)
        frames.append(_run_tasks(tasks, workers, f"gate x{f}"))
        cid += 1
    for f in readout_factors:
        tasks = _make_tasks("sweep_readout", f, cfg.scaled(readout=f), trials, shots, fit, seed, cid)
        frames.append(_run_tasks(tasks, workers, f"readout x{f}"))
        cid += 1
    return pd.concat(frames, ignore_index=True)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per benchmark and strategy: error, bias, spread and cost."""
    g = df.groupby(["benchmark", "strategy"], sort=False)
    out = g.agg(
        ideal=("ideal", "first"),
        mean_estimate=("estimate", "mean"),
        std_estimate=("estimate", "std"),
        mean_abs_error=("abs_error", "mean"),
        circuit_executions=("circuit_executions", "first"),
        total_shots=("total_shots", "first"),
    ).reset_index()
    out["bias"] = out["mean_estimate"] - out["ideal"]
    none_err = out[out.strategy == "none"].set_index("benchmark")["mean_abs_error"]
    out["error_reduction_x"] = out.apply(lambda r: none_err[r.benchmark] / r.mean_abs_error, axis=1)
    return out


def overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Average over the four benchmarks (each benchmark weighted equally)."""
    per = df.groupby(["strategy", "benchmark"], sort=False)["abs_error"].mean().reset_index()
    out = per.groupby("strategy", sort=False)["abs_error"].mean().rename("mean_abs_error").reset_index()
    cost = df.groupby("strategy", sort=False)[["circuit_executions", "total_shots"]].mean().reset_index()
    out = out.merge(cost, on="strategy")
    none = out.loc[out.strategy == "none", "mean_abs_error"].iloc[0]
    out["error_reduction_x"] = none / out["mean_abs_error"]
    out["cost_vs_none_x"] = out["total_shots"] / out.loc[out.strategy == "none", "total_shots"].iloc[0]
    return out


def zne_illustration(bench: Benchmark, cfg: NoiseConfig, seed: int = 123, shots: int = 20000):
    """Collect E(lambda) at noise scales 1,3,5 with and without REM, for plotting."""
    obs = bench.mitiq_observable()
    nm = build_noise_model(cfg, bench.n_qubits)
    cal = NoisyExecutor(nm, shots, seed)
    inv = inverse_confusion_matrix_for(calibrate_confusion_matrices(cal, bench.n_qubits), bench.measured_qubits)
    out = {}
    for label, use_rem in (("no REM", False), ("with REM", True)):
        from mitiq import rem
        ex = NoisyExecutor(nm, shots, seed + 1)
        run = ex.fn
        if use_rem:
            run = rem.mitigate_executor(run, inverse_confusion_matrix=inv)
        fac = make_factory("richardson", (1, 3, 5))
        limit = zne.execute_with_zne(bench.circuit, run, obs, factory=fac, scale_noise=fold_global)
        out[label] = (list(fac.get_scale_factors()), [float(np.real(v)) for v in fac.get_expectation_values()], float(np.real(limit)))
    return out


# --------------------------------------------------------------------------
# Sanity checks - run before the experiments so wrong results can't slip by
# --------------------------------------------------------------------------
def sanity_checks(verbose: bool = True) -> bool:
    ok = True

    def report(name, passed, detail=""):
        nonlocal ok
        ok &= bool(passed)
        if verbose:
            print(f"  [{'PASS' if passed else 'FAIL'}] {name} {detail}")

    shots = 20000
    for i, b in enumerate(all_benchmarks()):
        obs, ideal = b.mitiq_observable(), b.ideal_value()
        v = float(np.real(obs.expectation(b.circuit, NoisyExecutor(None, shots, 1).fn)))
        report(f"noiseless pipeline reproduces exact value: {b.name}",
               abs(v - ideal) < 0.03, f"(sampled {v:.3f} vs exact {ideal:.3f})")

    b = all_benchmarks()[1]  # BV-1011: 4 measured qubits, one ancilla
    obs, ideal = b.mitiq_observable(), b.ideal_value()
    cfg = NoiseConfig().scaled(gate=0.0)  # readout noise only
    nm = build_noise_model(cfg, b.n_qubits)
    cms = calibrate_confusion_matrices(NoisyExecutor(nm, shots, 2), b.n_qubits)
    inv = inverse_confusion_matrix_for(cms, b.measured_qubits)
    raw = estimate("none", b.circuit, obs, NoisyExecutor(nm, shots, 3), inv, None)
    fixed = estimate("rem", b.circuit, obs, NoisyExecutor(nm, shots, 3), inv, None)
    report("REM removes pure readout noise (bit-order is correct)",
           abs(fixed - ideal) < 0.03 < abs(raw - ideal), f"(raw {raw:.3f} -> REM {fixed:.3f}, ideal {ideal:.3f})")

    b = all_benchmarks()[2]  # QFT-4: deepest, gate noise only
    obs, ideal = b.mitiq_observable(), b.ideal_value()
    cfg = NoiseConfig().scaled(readout=0.0)
    nm = build_noise_model(cfg, b.n_qubits)
    raw = estimate("none", b.circuit, obs, NoisyExecutor(nm, shots, 4), None, None)
    mit = estimate("zne", b.circuit, obs, NoisyExecutor(nm, shots, 4), None, make_factory())
    report("ZNE reduces pure gate noise",
           abs(mit - ideal) < abs(raw - ideal), f"(raw {raw:.3f} -> ZNE {mit:.3f}, ideal {ideal:.3f})")
    return ok
