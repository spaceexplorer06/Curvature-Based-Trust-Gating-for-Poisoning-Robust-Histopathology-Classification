#!/usr/bin/env python3
"""
Batch experiment runner for Oxford Bioinformatics paper.
Streams output in real-time so errors are visible.
"""

import os
import sys
import subprocess
import json
import argparse
from datetime import datetime

EXPERIMENTS = [
    # Baselines (no defense) -- poison rate dose-response curve
    ("clean_baseline", 0.0, "none", "soft"),
    ("poison05_baseline", 0.05, "none", "soft"),
    ("poison10_baseline", 0.10, "none", "soft"),
    ("poison20_baseline", 0.20, "none", "soft"),

    # Ollivier-Ricci curvature defense (core contribution)
    ("poison05_ricci_soft", 0.05, "ricci", "soft"),
    ("poison10_ricci_soft", 0.10, "ricci", "soft"),
    ("poison20_ricci_soft", 0.20, "ricci", "soft"),
    # Ricci ablations
    ("poison10_ricci_hard", 0.10, "ricci", "hard"),
    ("poison10_ricci_curv", 0.10, "ricci", "curvature_only"),
    ("poison20_ricci_hard", 0.20, "ricci", "hard"),
    ("poison20_ricci_curv", 0.20, "ricci", "curvature_only"),

    # Small-loss trust baseline (non-geometric control -- the curvature
    # mechanism needs to clearly beat this to justify its complexity)
    ("poison05_smallloss", 0.05, "small_loss", "soft"),
    ("poison10_smallloss", 0.10, "small_loss", "soft"),
    ("poison20_smallloss", 0.20, "small_loss", "soft"),

    # Bloch-sphere defense -- kept as the rejected-alternative ablation,
    # not the headline result (see /areas/breakhis-histopathology.md for
    # why the quantum framing was dropped)
    ("poison10_bloch_soft", 0.10, "bloch", "soft"),
    ("poison20_bloch_soft", 0.20, "bloch", "soft"),
]


def run_experiment(dataset, data_root, run_name, poison_rate, defense, gate_mode, epochs, batch_size, lr, num_workers, freeze_backbone, seed, amp):
    cmd = [
        sys.executable, "training_defense.py",
        "--dataset", dataset,
        "--poison_rate", str(poison_rate),
        "--defense", defense,
        "--run_name", run_name,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--num_workers", str(num_workers),
        "--seed", str(seed),
    ]
    if dataset == "breakhis":
        cmd.extend(["--data_root", data_root])
    else:  # pcam
        cmd.extend(["--pcam_root", data_root])
    if freeze_backbone:
        cmd.append("--freeze_backbone")
    if amp:
        cmd.append("--amp")
    if defense in ("bloch", "ricci", "small_loss"):
        cmd.extend(["--gate_mode", gate_mode])

    print(f"\n{'='*60}")
    print(f"Running: {run_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

    if result.returncode != 0:
        print(f"\nERROR: {run_name} failed with code {result.returncode}\n")
        return False

    return True


def collect_results():
    """Reads every runs/<name>/test_report.json. Run folders named
    <base>_seed<N> (produced when --seeds has more than one value) are
    kept individually here and grouped by base name in print_summary."""
    results = {}
    runs_dir = "runs"
    if not os.path.exists(runs_dir):
        return results

    for run_name in os.listdir(runs_dir):
        report_path = os.path.join(runs_dir, run_name, "test_report.json")
        if os.path.exists(report_path):
            with open(report_path) as f:
                results[run_name] = json.load(f)
    return results


def _base_name(run_name):
    # "poison10_bloch_soft_seed1337" -> "poison10_bloch_soft"
    if "_seed" in run_name:
        return run_name.rsplit("_seed", 1)[0]
    return run_name


def _mean_std(values):
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, var ** 0.5


def print_summary(results):
    grouped = {}
    for run_name, data in results.items():
        grouped.setdefault(_base_name(run_name), []).append(data)

    print("\n" + "=" * 100)
    print("EXPERIMENT SUMMARY  (mean +/- std across seeds where n>1; single run shown as-is)")
    print("=" * 100)
    print(f"{'Experiment':<25} {'Poison':>7} {'Defense':>9} {'n':>3} "
          f"{'Acc':>16} {'BalAcc':>16} {'MacroF1':>16} {'AUC':>16}")
    print("-" * 100)

    for base_name, runs in sorted(grouped.items()):
        poison = f"{runs[0].get('poison_rate', 0) * 100:.0f}%"
        defense = runs[0].get('defense', 'none')
        n = len(runs)

        def fmt(key):
            vals = [r.get(key, 0) for r in runs]
            mean, std = _mean_std(vals)
            return f"{mean:.3f}+/-{std:.3f}" if n > 1 else f"{mean:.4f}"

        print(f"{base_name:<25} {poison:>7} {defense:>9} {n:>3} "
              f"{fmt('clean_test_accuracy'):>16} {fmt('balanced_accuracy'):>16} "
              f"{fmt('macro_f1'):>16} {fmt('auc'):>16}")

        # Flag likely class-collapse runs (defense "succeeding" by always
        # predicting one class) since raw accuracy alone hides this.
        for r in runs:
            b, m = r.get('benign_acc', -1), r.get('malignant_acc', -1)
            if b == 0.0 or m == 0.0:
                print(f"    ^ WARNING: {r.get('run_name')} collapsed to a single class "
                      f"(benign_acc={b:.3f}, malignant_acc={m:.3f}) -- accuracy for this run is not meaningful")

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="breakhis", choices=["breakhis", "pcam"])
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to BreaKHis_v1/ (--dataset breakhis) or the directory "
                             "containing the six PCam .h5 files (--dataset pcam)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--amp", action="store_true",
                        help="Mixed-precision training -- recommended for large datasets like PCam")
    parser.add_argument("--runs", type=str, default="all",
                        help="Comma-separated run names, or 'all'")
    parser.add_argument("--seeds", type=str, default="42",
                        help="Comma-separated seeds, e.g. '42,1337,2024'. Each "
                             "experiment is run once per seed so results in the "
                             "summary have a mean +/- std instead of being a "
                             "single (possibly non-representative) run.")
    args = parser.parse_args()

    selected = EXPERIMENTS
    if args.runs != "all":
        wanted = set(args.runs.split(","))
        selected = [e for e in EXPERIMENTS if e[0] in wanted]

    seeds = [int(s) for s in args.seeds.split(",")]
    multi_seed = len(seeds) > 1

    print(f"Starting {len(selected)} experiments x {len(seeds)} seed(s) at {datetime.now()}")

    # Runs from different datasets share the same EXPERIMENTS names (e.g.
    # "poison10_ricci_soft"), so pcam runs get a dataset prefix to avoid
    # silently overwriting/mixing with BreaKHis runs of the same base name
    # in runs/ and experiment_summary.json. BreaKHis keeps unprefixed names
    # for backward compatibility with existing results.
    name_prefix = "" if args.dataset == "breakhis" else f"{args.dataset}_"

    for run_name, poison_rate, defense, gate_mode in selected:
        for seed in seeds:
            base_name = f"{name_prefix}{run_name}"
            full_run_name = f"{base_name}_seed{seed}" if multi_seed else base_name
            success = run_experiment(
                args.dataset, args.data_root, full_run_name, poison_rate, defense, gate_mode,
                args.epochs, args.batch_size, args.lr, args.num_workers, args.freeze_backbone, seed, args.amp
            )
            if not success:
                print(f"WARNING: {full_run_name} failed!")

    results = collect_results()
    print_summary(results)

    with open("experiment_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSummary saved to experiment_summary.json")


if __name__ == "__main__":
    main()