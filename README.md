# Error Mitigation Shoot-out (Quantum Computing project #13)

Compares four ways of dealing with noise on a (simulated) quantum computer,
using the **Mitiq** error-mitigation library and **Qiskit Aer**:

| # | Strategy | What it fixes |
|---|----------|---------------|
| 1 | No mitigation | nothing (baseline) |
| 2 | Readout-error mitigation (REM) | wrong 0/1 readings at measurement time |
| 3 | Zero-noise extrapolation (ZNE) | noise from gates (sx, x, cx) |
| 4 | REM + ZNE combined | both |

on four benchmark circuits (GHZ-4, Bernstein-Vazirani, QFT-4 round trip, Grover-3).
No IBM account, internet, or real hardware is needed.

Base paper: *Mitiq: A software package for error mitigation on noisy quantum computers* (LaRose et al., Quantum 2022).

---------------------------------------------------------------------

## 1. How to run

**Requirement: Python 3.11 or 3.12** (Mitiq does not support 3.13 yet). Check with `python --version`.

```bash
# 1. go into the project folder
cd qc_project

# 2. create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell:  .venv\Scripts\Activate.ps1
                                   # Windows cmd:         .venv\Scripts\activate.bat

# 3. install dependencies
pip install -r requirements.txt

# 4. quick check that everything works (~20-60 seconds)
python run_all.py --quick

# 5. the real run (about 4-8 minutes depending on your CPU)
python run_all.py
```

If you have several CPU cores, the run uses half of them automatically.
Use `--workers 1` if you see any multiprocessing error (common on some Windows/Jupyter setups).

Other options:

```bash
python run_all.py --trials 30            # more repeats = smoother statistics
python run_all.py --shots 16384          # more shots per circuit
python run_all.py --zne-fit linear       # linear / richardson / exp extrapolation
python run_all.py --skip-sweeps          # only the main experiment (about 1 minute)
python run_all.py --out my_results       # different output folder
```

### What you get (in `results/`)

| File | Content |
|------|---------|
| `fig1_error_by_circuit.png` | **Main result**: error of each strategy per circuit |
| `fig2_estimate_spread.png` | Box plots: shows bias (off-target) vs variance (jitter) |
| `fig3_zne_extrapolation.png` | How ZNE works, visually, with and without REM |
| `fig4_noise_sweeps.png` | Which strategy wins as gate noise / readout noise grows |
| `main_summary.csv`, `main_overall.csv` | Tables for your report |
| `main_raw.csv`, `sweep_raw.csv` | Every single trial (for your own analysis) |

The folder shipped with this project already contains the output of one full run.

---------------------------------------------------------------------

## 2. Project layout

```
run_all.py               entry point (command-line options, prints tables, saves files)
shootout/
  benchmarks.py          the 4 circuits + their observables + exact ideal values
  noise.py               noise model: gate noise + readout noise (adjustable)
  executor.py            runs a circuit on noisy Aer, returns bitstrings, counts cost
  strategies.py          the 4 strategies, plus readout calibration
  experiments.py         trials, sweeps, summaries, sanity checks
  plots.py               figures
```

---------------------------------------------------------------------

## 3. What the code actually does (in detail)

### 3.1 The big picture

A real quantum computer returns wrong answers because of noise. Error
*mitigation* does not fix the hardware. It runs extra circuits and uses
classical post-processing to get a better estimate of the true answer. The price
is more circuit runs (and usually more statistical jitter).

For each circuit we compute one number, an **expectation value** of an observable
(for example "the average of ±Z over the output qubits", which equals 1 when the
circuit outputs exactly the right bitstring). Since the circuits are small we
know the exact answer from a noiseless statevector. Then, for each strategy:

> **error = | estimate from noisy machine − exact answer |**

We repeat everything 20 times (different random seeds) and report the mean and
the spread. That is the whole experiment.

### 3.2 The benchmark circuits (`benchmarks.py`)

| Circuit | Qubits | CX gates | Ideal value | Why chosen |
|---------|-------|----------|-------------|------------|
| GHZ-4 | 4 | 3 | 1.0 | shallow, entangling; observable is the 4-qubit parity ZZZZ |
| BV-1011 | 5 | 3 | 1.0 | Bernstein-Vazirani; output is a known secret string; few CX, many single-qubit gates |
| QFT-4 | 4 | 24 | 1.0 | QFT followed by inverse QFT, returns the starting state; deep (depth 52) |
| Grover-3 | 3 | 24 | 0.9375 | 3-qubit Grover search (2 iterations); deepest (depth 99); even ideally not perfectly 1 |

Each circuit is translated to the gate set `{rz, sx, x, cx}` that real IBM chips
use. This matters because noise is attached to individual native gates.

### 3.3 The noise model (`noise.py`)

Two independent noise sources, with defaults in the range of today's IBM devices:

* **Gate noise**: after every `sx`/`x` gate a depolarizing error of 0.1 %; after every `cx` gate 1 %.
* **Readout noise**: every qubit has its own chance of misreading: about 2 % for 0→1 and 4 % for 1→0 (varying ±30 % by qubit).

Both can be scaled (`NoiseConfig.scaled(gate=..., readout=...)`), which is how the sweeps work.

### 3.4 The executor (`executor.py`)

Mitiq doesn't care what machine it talks to. It only needs a function
`circuit -> measurement bitstrings`. Ours transpiles the circuit with
`optimization_level=0`, runs it on a noisy Aer simulator and returns the raw
bitstrings. It also counts how many circuits it ran, so we can report the **cost**
of each strategy.

Two details that are easy to get wrong:

1. **`optimization_level=0`**: ZNE inserts `U U† U` which a normal optimiser would cancel straight back, silently destroying the noise scaling.
2. **Bit order**: Qiskit prints qubit 0 as the *rightmost* character; Mitiq expects it first. We reverse every bitstring. The sanity checks verify this is right.

### 3.5 Strategy 1: no mitigation

Run the circuit once, compute the expectation value from the bitstrings.

### 3.6 Strategy 2: readout-error mitigation (REM)

Readout errors are a **classical** problem: some bits get flipped when read out.

1. **Calibrate**: prepare |0000…⟩ and |1111…⟩, measure many times, and count how often each qubit is wrong. This gives a 2×2 *confusion matrix* per qubit, `[[P(0|0), P(0|1)], [P(1|0), P(1|1)]]`.
2. **Invert** each matrix and combine them (Kronecker product) into one big inverse matrix.
3. **Apply it** to the measured outcome probabilities of the real experiment. This undoes the average effect of bit flips (`mitiq.rem`).

Cost: 2 extra calibration circuits (3 circuits total). It cannot fix gate errors.

### 3.7 Strategy 3: zero-noise extrapolation (ZNE)

We cannot turn noise off, but we can turn it **up**, and then extrapolate backwards.

1. Take the circuit U. Build a longer circuit that does the *same thing* but has more gates: **U U† U**, which has 3× the gates and so about 3× the noise. Similarly 5× with U U† U U† U.
2. Run the circuit at noise scale 1, 3 and 5 and measure the expectation value each time. It decays as noise grows (see `fig3_zne_extrapolation.png`).
3. Fit a curve through those points (Richardson extrapolation = a quadratic through 3 points) and read off its value at noise scale **0**.

Cost: 3 circuits. Trade-off: extrapolating amplifies the statistical jitter (visible as wider boxes in `fig2`).

**Important limitation:** folding only multiplies *gate* noise. Readout errors happen once at the end, so they are *not* amplified and ZNE cannot see them. That is the whole reason for strategy 4.

### 3.8 Strategy 4: REM + ZNE

Wrap the executor with REM so that **every** run (at every noise scale) is
readout-corrected first, then let ZNE extrapolate. REM removes the part ZNE can't
see; ZNE removes the part REM can't see. Cost: 3 + 2 = 5 circuits.

### 3.9 Sanity checks (run automatically)

Before the experiment, `run_all.py` verifies:

* the noiseless pipeline reproduces the exact value on all circuits (catches bit-order or circuit-conversion bugs);
* REM fully removes noise when **only** readout noise is present;
* ZNE improves the result when **only** gate noise is present.

If any check fails the program stops, so you can never get silently wrong numbers.

### 3.10 Statistics

Each (circuit, strategy) is repeated **20 times** with different random seeds
(run-to-run seeds are fixed, so results are reproducible). We report the mean
absolute error and standard deviation. A single run would be meaningless since
shot noise alone changes results by about 1 %.

---------------------------------------------------------------------

## 4. Results from the included run

Average absolute error over the 4 circuits (20 trials, 8192 shots):

| Strategy | Mean abs. error | Improvement vs none | Circuits run |
|----------|-----------------|---------------------|--------------|
| No mitigation | 0.188 | 1× | 1 |
| Readout (REM) | 0.094 | 2.0× | 3 |
| ZNE | 0.106 | 1.8× | 3 |
| **REM + ZNE** | **0.013** | **14.5×** | 5 |

Per circuit (mean abs. error):

| Circuit | None | REM | ZNE | REM+ZNE |
|---------|------|-----|-----|---------|
| GHZ-4 | 0.239 | 0.029 | 0.216 | **0.015** |
| BV-1011 | 0.083 | 0.018 | 0.063 | **0.008** |
| QFT-4 | 0.186 | 0.133 | 0.066 | **0.012** |
| Grover-3 | 0.246 | 0.195 | 0.080 | **0.018** |

### What to conclude

1. **Combining beats either alone**, on every circuit. The two techniques fix different noise sources, so their benefits stack (error ↓ ~14×) for only 5× the circuit runs.
2. **Which single method wins depends on the circuit.** Shallow circuits (GHZ, BV) are dominated by readout error, so REM wins and ZNE barely helps. Deep circuits (QFT, Grover; 24 CX gates) are dominated by gate error, so ZNE beats REM.
3. **The noise sweeps (`fig4`) confirm this**: with low gate noise (0.25×) REM (0.025) beats ZNE (0.100), but at 3× gate noise ZNE (0.167) beats REM (0.239). With readout noise switched off, ZNE alone is nearly perfect (0.009).
4. **Mitigation has a cost: variance.** ZNE and REM+ZNE have wider spread than the raw result (see `fig2`). Extrapolation amplifies shot noise. The bias goes away, the jitter grows.
5. **Mitigation does not scale forever.** At 3× gate noise even REM+ZNE error rises to 0.079. With too much noise the extrapolation gets unreliable.

---------------------------------------------------------------------

## 5. Things to mention in your report / viva

* **Why can't ZNE fix readout error?** Folding gates doesn't multiply readout noise.
* **What assumption does REM make?** Qubit readout errors are independent (tensor-product structure); it ignores crosstalk.
* **What assumption does ZNE make?** Noise scales (roughly) with gate count, and the expectation value varies smoothly with noise so a low-order polynomial fit is valid.
* **Why does the variance increase?** Richardson extrapolation multiplies noisy points by coefficients > 1 (for scales 1,3,5 the weights are 15/8, −5/4, 3/8).
* **Limitations of this study:** simulated noise (depolarizing + readout flips) is simpler than a real device (no crosstalk, no T1/T2 decay, no coherent errors); only four small circuits; only one observable per circuit.
* **Calibration caveat:** the |1…1⟩ calibration circuit uses X gates, which carry a tiny gate error (0.1 %) that leaks into the estimated readout error.

### Ideas to extend it (for extra marks)

* Try `--zne-fit linear` and `--zne-fit exp` and compare to Richardson (bias vs variance).
* Use other scale factors, e.g. `(1, 2, 3)` via `make_factory("richardson", (1, 2, 3))` in `strategies.py`.
* Use a realistic device noise model (`qiskit_ibm_runtime.fake_provider` + `NoiseModel.from_backend`) instead of the synthetic one.
* Add probabilistic error cancellation or dynamical decoupling from Mitiq (`mitiq.pec`, `mitiq.ddd`).
* Run the same experiment once on a real IBM device and compare with the simulation.

---------------------------------------------------------------------

## 6. Troubleshooting

| Problem | Fix |
|---------|-----|
| `No matching distribution found for mitiq` / install errors | You are on Python 3.13+. Use Python 3.11 or 3.12. |
| `ModuleNotFoundError: No module named 'ply'` | `pip install ply` (it is in `requirements.txt`). |
| Multiprocessing errors / program hangs | `python run_all.py --workers 1` |
| Takes too long | `python run_all.py --quick`, or `--skip-sweeps`, or fewer `--trials`. |
| A sanity check FAILS | Package versions differ from the tested ones. Reinstall in a fresh virtual environment from `requirements.txt`. |
| Numbers differ a little from the README | Normal: different Qiskit/Aer versions change the random stream. The ranking REM+ZNE ≪ others should stay. |

Tested with: Python 3.12, qiskit 2.6.2, qiskit-aer 0.18.4, mitiq 1.1.0.
