#!/usr/bin/env python
"""Error-mitigation shoot-out  -  run everything with:   python run_all.py

Compares  (1) no mitigation  (2) readout mitigation  (3) zero-noise extrapolation
(4) both combined  on four benchmark circuits, using Mitiq + Qiskit Aer.

Useful options:
    python run_all.py --quick            # ~1 minute smoke run
    python run_all.py --trials 30        # tighter statistics
    python run_all.py --zne-fit linear   # try a different ZNE extrapolation
    python run_all.py --skip-sweeps      # only the main experiment
"""
import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread per process: faster with many workers

import pandas as pd  # noqa: E402

from shootout import experiments as ex  # noqa: E402
from shootout import plots  # noqa: E402
from shootout.benchmarks import all_benchmarks  # noqa: E402
from shootout.noise import NoiseConfig  # noqa: E402
from shootout.strategies import LABELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=20, help="repeats per circuit (main experiment)")
    ap.add_argument("--sweep-trials", type=int, default=8, help="repeats per circuit in noise sweeps")
    ap.add_argument("--shots", type=int, default=8192, help="shots per circuit execution")
    ap.add_argument("--zne-fit", default="richardson", choices=["richardson", "linear", "exp"])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="parallel processes (use 1 if you hit multiprocessing problems)")
    ap.add_argument("--out", default="results", help="output folder")
    ap.add_argument("--quick", action="store_true", help="tiny run to check everything works")
    ap.add_argument("--skip-sweeps", action="store_true")
    ap.add_argument("--skip-checks", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.trials, args.sweep_trials, args.shots = 4, 2, 2048
    os.makedirs(args.out, exist_ok=True)
    cfg = NoiseConfig()
    t0 = time.time()

    print("=" * 70)
    print("Benchmarks")
    print(pd.DataFrame([b.stats() for b in all_benchmarks()]).to_string(index=False))
    print(f"\nNoise model: {cfg}")

    if not args.skip_checks:
        print("\nSanity checks")
        if not ex.sanity_checks():
            sys.exit("A sanity check failed - results would not be trustworthy. See README troubleshooting.")

    print(f"\nMain experiment: {args.trials} trials x 4 circuits x 4 strategies, {args.shots} shots")
    main_df = ex.run_main(cfg, args.trials, args.shots, args.zne_fit, args.seed, args.workers)
    main_df.to_csv(f"{args.out}/main_raw.csv", index=False)
    per_bench = ex.summarise(main_df)
    per_bench.to_csv(f"{args.out}/main_summary.csv", index=False)
    overall = ex.overall_summary(main_df)
    overall.to_csv(f"{args.out}/main_overall.csv", index=False)

    pd.set_option("display.width", 200)
    show = per_bench.copy()
    show["strategy"] = show["strategy"].map(LABELS)
    print("\nPer-circuit results")
    print(show[["benchmark", "strategy", "ideal", "mean_estimate", "std_estimate", "mean_abs_error",
                "error_reduction_x", "circuit_executions"]].round(4).to_string(index=False))
    show = overall.copy()
    show["strategy"] = show["strategy"].map(LABELS)
    print("\nOverall (average of the 4 circuits)")
    print(show.round(4).to_string(index=False))

    plots.fig_main_errors(main_df, f"{args.out}/fig1_error_by_circuit.png")
    plots.fig_distributions(main_df, f"{args.out}/fig2_estimate_spread.png")

    bench = all_benchmarks()[2]  # QFT-4 for the extrapolation picture
    illu = ex.zne_illustration(bench, cfg, seed=args.seed)
    plots.fig_zne(illu, bench.ideal_value(), bench.name, f"{args.out}/fig3_zne_extrapolation.png")

    if not args.skip_sweeps:
        gate_f, ro_f = ([0.5, 1, 2], [0, 1, 2]) if args.quick else ([0.25, 0.5, 1, 2, 3], [0, 0.5, 1, 2, 3])
        print(f"\nNoise sweeps ({args.sweep_trials} trials per point)")
        sweep_df = ex.run_sweeps(cfg, args.sweep_trials, args.shots, args.zne_fit, args.seed,
                                 args.workers, gate_f, ro_f)
        sweep_df.to_csv(f"{args.out}/sweep_raw.csv", index=False)
        plots.fig_sweeps(sweep_df, f"{args.out}/fig4_noise_sweeps.png")

    print(f"\nDone in {time.time() - t0:.0f}s. Everything is in the '{args.out}/' folder.")


if __name__ == "__main__":
    main()
