#!/usr/bin/env python3
"""
Significance testing for defense comparisons.

Two tests, in order of what's available:

1. Paired seed-level test (works right now, from experiment_summary.json).
   Two defenses run at the same --seed use the exact same patient-wise
   test split (patientwise_split is keyed on --seed, not --defense), so
   the 5 seed-level metric values for defense A and defense B are
   naturally paired -- a paired t-test / Wilcoxon signed-rank test on
   those 5 pairs is valid. This is the same logic as a paired t-test
   across folds in cross-validation.

2. Per-seed DeLong AUC test (only for runs made *after* the test_probs.json
   change to training_defense.py -- older runs like your existing
   experiment_summary.json don't have per-sample predictions saved, so
   this will silently skip any run missing that file). This tests, within
   a single seed's shared test set, whether two defenses' AUCs differ
   significantly at the per-sample level -- a stronger test than #1, but
   only usable going forward.

Usage:
    python analyze_significance.py --summary experiment_summary.json \
        --compare poison10_ricci_soft:poison10_baseline \
        --compare poison10_ricci_soft:poison10_smallloss \
        --compare poison20_ricci_soft:poison20_baseline \
        --compare poison20_ricci_soft:poison20_smallloss

    # Also attempt per-seed DeLong tests using runs/<name>/test_probs.json
    # if present:
    python analyze_significance.py --summary experiment_summary.json \
        --compare poison10_ricci_soft:poison10_smallloss --runs_dir runs
"""

import argparse
import json
import os
import numpy as np
from scipy import stats


def base_name(run_name):
    if "_seed" in run_name:
        return run_name.rsplit("_seed", 1)[0]
    return run_name


def seed_of(run_name):
    if "_seed" in run_name:
        return run_name.rsplit("_seed", 1)[1]
    return None


def load_grouped(summary_path):
    """Returns {base_name: {seed: metrics_dict}}."""
    with open(summary_path) as f:
        results = json.load(f)
    grouped = {}
    for run_name, data in results.items():
        b = base_name(run_name)
        s = seed_of(run_name)
        if s is None:
            continue  # skip single (unseeded) runs -- can't pair them
        grouped.setdefault(b, {})[s] = data
    return grouped


def collapse_rate(runs_dir, base_name, threshold=0.05):
    """
    Fraction of seeds for a given experiment where the model collapsed to
    predicting a single class (either class's accuracy below `threshold`).
    Reads individual runs/<base_name>_seed<N>/test_report.json files rather
    than the aggregated summary, since collapse is a per-seed event that a
    mean+/-std can mask -- a run_experiments.py summary table can show a
    "reasonable" mean balanced accuracy while actually containing seeds
    that are pure single-class predictors, offset by seeds that did fine.
    """
    if not os.path.isdir(runs_dir):
        return None
    seed_dirs = [d for d in os.listdir(runs_dir) if d.startswith(base_name + "_seed")]
    if not seed_dirs:
        return None
    n_collapsed = 0
    collapsed_seeds = []
    for d in seed_dirs:
        report_path = os.path.join(runs_dir, d, "test_report.json")
        if not os.path.exists(report_path):
            continue
        with open(report_path) as f:
            r = json.load(f)
        b, m = r.get("benign_acc", -1), r.get("malignant_acc", -1)
        if b < threshold or m < threshold:
            n_collapsed += 1
            collapsed_seeds.append(d.rsplit("_seed", 1)[1])
    return {"n_total": len(seed_dirs), "n_collapsed": n_collapsed,
            "collapsed_seeds": collapsed_seeds,
            "collapse_rate": n_collapsed / len(seed_dirs) if seed_dirs else 0.0}


def paired_seed_test(grouped, base_a, base_b, metric):
    """Paired t-test + Wilcoxon signed-rank on matched-seed metric values."""
    if base_a not in grouped or base_b not in grouped:
        return None
    common_seeds = sorted(set(grouped[base_a]) & set(grouped[base_b]))
    if len(common_seeds) < 3:
        return {"error": f"only {len(common_seeds)} matched seeds (need >=3)"}

    a_vals = np.array([grouped[base_a][s][metric] for s in common_seeds])
    b_vals = np.array([grouped[base_b][s][metric] for s in common_seeds])
    diffs = a_vals - b_vals

    t_stat, t_p = stats.ttest_rel(a_vals, b_vals)
    try:
        w_stat, w_p = stats.wilcoxon(a_vals, b_vals)
    except ValueError:
        # all diffs are zero, or n too small for wilcoxon
        w_stat, w_p = float("nan"), float("nan")

    return {
        "n_seeds": len(common_seeds),
        "seeds": common_seeds,
        f"mean_{base_a}": a_vals.mean(),
        f"mean_{base_b}": b_vals.mean(),
        "mean_diff": diffs.mean(),
        "std_diff": diffs.std(ddof=1) if len(diffs) > 1 else 0.0,
        "paired_t_p": t_p,
        "wilcoxon_p": w_p,
    }


def delong_auc_variance(labels, probs):
    """
    Fast DeLong (Sun & Xu, 2014) computation of AUC and its covariance
    structure, restricted to the single-sample-set variance (used for the
    two-sample paired case below). Returns (auc, auc_var_placeholder) --
    the real work happens in delong_paired_test, this just gets the
    structural components.
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    pos = probs[labels == 1]
    neg = probs[labels == 0]
    n_pos, n_neg = len(pos), len(neg)

    # Midrank-based fast DeLong implementation
    def compute_midrank(x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1) + 1
            i = j
        T2 = np.empty(N)
        T2[J] = T
        return T2

    tx = compute_midrank(pos)
    ty = compute_midrank(neg)
    tz = compute_midrank(np.concatenate([pos, neg]))
    v01 = (tz[:n_pos] - tx) / n_neg
    v10 = 1.0 - (tz[n_pos:] - ty) / n_pos
    auc = v01.mean()
    return auc, v01, v10


def delong_paired_test(labels, probs_a, probs_b):
    """
    Paired DeLong test for whether AUC_a == AUC_b on the SAME set of
    labeled test samples (same patients, two different models' scores).
    Returns (auc_a, auc_b, z_stat, p_value).
    """
    auc_a, v01_a, v10_a = delong_auc_variance(labels, probs_a)
    auc_b, v01_b, v10_b = delong_auc_variance(labels, probs_b)

    n_pos = v01_a.shape[0]
    n_neg = v10_a.shape[0]

    v01 = np.stack([v01_a, v01_b])  # (2, n_pos)
    v10 = np.stack([v10_a, v10_b])  # (2, n_neg)

    s01 = np.cov(v01) / n_pos
    s10 = np.cov(v10) / n_neg
    s = s01 + s10  # (2,2) covariance of [auc_a, auc_b]

    diff = auc_a - auc_b
    var_diff = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    if var_diff <= 0:
        return auc_a, auc_b, float("nan"), 1.0
    z = diff / np.sqrt(var_diff)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc_a, auc_b, z, p


def try_delong_per_seed(runs_dir, base_a, base_b, grouped):
    """Attempts a per-seed DeLong test for every matched seed where both
    runs' test_probs.json exist. Silently skips seeds missing the file
    (i.e. any run made before the test_probs.json change)."""
    if base_a not in grouped or base_b not in grouped:
        return []
    common_seeds = sorted(set(grouped[base_a]) & set(grouped[base_b]))
    out = []
    for s in common_seeds:
        path_a = os.path.join(runs_dir, f"{base_a}_seed{s}", "test_probs.json")
        path_b = os.path.join(runs_dir, f"{base_b}_seed{s}", "test_probs.json")
        if not (os.path.exists(path_a) and os.path.exists(path_b)):
            continue
        with open(path_a) as f:
            da = json.load(f)
        with open(path_b) as f:
            db = json.load(f)
        if da["labels"] != db["labels"]:
            out.append({"seed": s, "error": "label mismatch -- test sets weren't actually shared, skipping"})
            continue
        auc_a, auc_b, z, p = delong_paired_test(da["labels"], da["malignant_probs"], db["malignant_probs"])
        out.append({"seed": s, "auc_a": auc_a, "auc_b": auc_b, "z": z, "p": p})
    return out


def variance_test(grouped, base_a, base_b, metric):
    """
    Levene's test for equal variance between two defenses' seed-level
    metric values. This is a DIFFERENT claim from the mean comparison in
    paired_seed_test: two defenses can have statistically indistinguishable
    means while one is meaningfully more consistent run-to-run -- which is
    itself a defensible thing to report ("comparable performance, lower
    variance") even when a straight performance-superiority claim doesn't
    survive testing.
    """
    if base_a not in grouped or base_b not in grouped:
        return None
    common_seeds = sorted(set(grouped[base_a]) & set(grouped[base_b]))
    if len(common_seeds) < 3:
        return {"error": f"only {len(common_seeds)} matched seeds (need >=3)"}
    a_vals = np.array([grouped[base_a][s][metric] for s in common_seeds])
    b_vals = np.array([grouped[base_b][s][metric] for s in common_seeds])
    stat, p = stats.levene(a_vals, b_vals)
    return {
        "n_seeds": len(common_seeds),
        "std_a": a_vals.std(ddof=1),
        "std_b": b_vals.std(ddof=1),
        "levene_stat": stat,
        "levene_p": p,
    }


def required_n_for_power(effect_size_d, alpha=0.05, power=0.80):
    """
    Normal-approximation sample size for a paired t-test given an observed
    effect size d = mean_diff / std_diff. This is a post-hoc estimate from
    a noisy small-sample std, so treat it as a rough planning number (how
    many more seeds would plausibly be needed), not a guarantee -- the
    true effect size could easily be smaller (or larger) than what 5 seeds
    happened to show.
    """
    if effect_size_d == 0:
        return float("inf")
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = ((z_alpha + z_beta) / abs(effect_size_d)) ** 2
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, help="Path to experiment_summary.json")
    parser.add_argument("--compare", action="append", required=True,
                        help="base_name_a:base_name_b, repeatable")
    parser.add_argument("--metrics", default="balanced_accuracy,macro_f1,auc")
    parser.add_argument("--runs_dir", default=None,
                        help="If given, also attempt per-seed DeLong AUC tests using "
                             "runs/<name>/test_probs.json where available")
    parser.add_argument("--variance_test", action="store_true",
                        help="Also run Levene's test comparing run-to-run variance "
                             "between the two defenses (a separate claim from the mean comparison)")
    parser.add_argument("--collapse_check", action="store_true",
                        help="Also report single-class-collapse rate per experiment "
                             "(requires --runs_dir; reads individual test_report.json files)")
    args = parser.parse_args()

    grouped = load_grouped(args.summary)
    metrics = args.metrics.split(",")

    # Collect every paired-t-test p-value across every comparison x metric
    # run in this invocation, so we can apply a multiple-comparisons
    # correction at the end. Running several comparisons and eyeballing
    # for any p<0.05 inflates the false-positive rate -- with e.g. 4
    # comparisons x 3 metrics = 12 tests at alpha=0.05, you'd expect
    # ~0.6 false positives by chance even if nothing were real.
    all_pvalues = []  # list of (label, p_value)

    for comparison in args.compare:
        a, b = comparison.split(":")
        print(f"\n{'='*80}\n{a}  vs  {b}\n{'='*80}")

        if args.collapse_check and args.runs_dir:
            for name in (a, b):
                cr = collapse_rate(args.runs_dir, name)
                if cr:
                    flag = f"  <-- {cr['n_collapsed']}/{cr['n_total']} seeds collapsed to single-class: {cr['collapsed_seeds']}" \
                           if cr["n_collapsed"] > 0 else ""
                    print(f"  [collapse check] {name}: {cr['collapse_rate']*100:.0f}% "
                          f"({cr['n_collapsed']}/{cr['n_total']}) collapsed to single-class prediction{flag}")

        for metric in metrics:
            result = paired_seed_test(grouped, a, b, metric)
            if result is None:
                print(f"  [{metric}] one or both experiments not found in summary")
                continue
            if "error" in result:
                print(f"  [{metric}] {result['error']}")
                continue
            print(f"  [{metric}] n={result['n_seeds']} seeds={result['seeds']}")
            print(f"    mean {a}: {result[f'mean_{a}']:.4f}   mean {b}: {result[f'mean_{b}']:.4f}"
                  f"   diff: {result['mean_diff']:+.4f} +/- {result['std_diff']:.4f}")
            print(f"    paired t-test p={result['paired_t_p']:.4f}   "
                  f"wilcoxon p={result['wilcoxon_p']:.4f}")
            all_pvalues.append((f"{a} vs {b} [{metric}]", result["paired_t_p"]))

            if args.variance_test:
                vresult = variance_test(grouped, a, b, metric)
                if vresult and "error" not in vresult:
                    print(f"    variance: std({a})={vresult['std_a']:.4f}  std({b})={vresult['std_b']:.4f}"
                          f"   Levene's p={vresult['levene_p']:.4f}"
                          f"{'  <-- variances differ' if vresult['levene_p'] < 0.05 else ''}")

        if args.runs_dir:
            delong_results = try_delong_per_seed(args.runs_dir, a, b, grouped)
            if delong_results:
                print(f"  [DeLong AUC, per-seed, paired on shared test set]")
                for r in delong_results:
                    if "error" in r:
                        print(f"    seed {r['seed']}: {r['error']}")
                    else:
                        print(f"    seed {r['seed']}: AUC {a}={r['auc_a']:.4f} vs {b}={r['auc_b']:.4f}  "
                              f"z={r['z']:.3f} p={r['p']:.4f}")
            else:
                print(f"  [DeLong AUC] no matching test_probs.json found for these runs "
                      f"(only available for runs made after the training_defense.py update)")

    if all_pvalues:
        labels, pvals = zip(*all_pvalues)
        pvals = np.array(pvals)
        # Holm-Bonferroni: sort ascending, compare p_(i) to alpha/(m-i+1),
        # stop at first failure -- less conservative than plain Bonferroni
        # but still controls the family-wise error rate, standard choice
        # for a handful of related hypothesis tests like this.
        order = np.argsort(pvals)
        m = len(pvals)
        holm_reject = np.zeros(m, dtype=bool)
        for rank, idx in enumerate(order):
            threshold = 0.05 / (m - rank)
            if pvals[idx] <= threshold:
                holm_reject[idx] = True
            else:
                break  # Holm's step-down procedure stops at first non-rejection

        print(f"\n{'='*80}\nHolm-Bonferroni correction across all {m} tests in this run "
              f"(family-wise alpha=0.05)\n{'='*80}")
        for label, p, sig in sorted(zip(labels, pvals, holm_reject), key=lambda x: x[1]):
            flag = "  <-- significant after correction" if sig else ""
            print(f"  p={p:.4f}  {label}{flag}")
        if not holm_reject.any():
            print("\n  No comparison survives correction for multiple testing. "
                  "Treat all of the above as not statistically distinguishable "
                  "at this sample size.")

        print(f"\n{'='*80}\nRough required-n for 80% power (post-hoc estimate, plan-ahead only)\n{'='*80}")
        for label, p in sorted(zip(labels, pvals), key=lambda x: x[1]):
            # recover the effect size from the underlying comparison
            comp_a, rest = label.split(" vs ")
            comp_b, metric = rest.rsplit(" [", 1)
            metric = metric.rstrip("]")
            result = paired_seed_test(grouped, comp_a, comp_b, metric)
            if result is None or result.get("std_diff", 0) == 0:
                continue
            d = result["mean_diff"] / result["std_diff"]
            n_needed = required_n_for_power(d)
            print(f"  {label}: observed d={d:+.2f}  ->  ~{n_needed:.0f} seeds needed for 80% power "
                  f"(have {result['n_seeds']})")


if __name__ == "__main__":
    main()