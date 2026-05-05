"""
Failure Mode Ablation Analyzer — V2
-------------------------------------
Forensic attribution of residual DPF leakage events to:
  - Control Plane failures (router misclassification)
  - Data Plane failures (semantic obfuscation / regex miss)

V2 Changes vs V1:
- Path resolution updated for src/evaluation/ location.
- Added router accuracy report: if logs/router_log_DPF_PROPOSED.csv
  exists (written by experiment_driver V2), it is parsed to compute
  per-role precision/recall and printed alongside the failure attribution.
  This closes the router ablation gap identified in peer review (Issue 7).
- Added per-threat-class breakdown with bootstrap confidence intervals
  (N_BOOTSTRAP=2000) around each class-level leakage rate, addressing
  Issue 5 (insufficient statistical power for class-level claims).
"""

import sys
import os
import pandas as pd
import numpy as np

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(EVAL_DIR)
BASE_DIR = os.path.dirname(SRC_DIR)

LOG_DIR    = os.environ.get("DPF_LOG_DIR",    os.path.join(BASE_DIR, "logs"))
OUTPUT_DIR = os.environ.get("DPF_OUTPUT_DIR", os.path.join(BASE_DIR, "paper_results"))

# Allow dynamic path injection from CLI; default to runtime logs
if len(sys.argv) > 1:
    INPUT_FILE = sys.argv[1]
else:
    INPUT_FILE = os.path.join(LOG_DIR, "audit_results.csv")

OUTPUT_FILE     = os.path.join(OUTPUT_DIR, "failure_mode_ablation.csv")
ROUTER_LOG_FILE = os.path.join(LOG_DIR, "router_log_DPF_PROPOSED.csv")

TARGET_ARCHITECTURE = "DPF_PROPOSED"
N_BOOTSTRAP = 2000
CI_LEVEL = 0.95


def bootstrap_ci(successes: int, n: int,
                 n_boot: int = N_BOOTSTRAP,
                 ci: float = CI_LEVEL) -> tuple:
    """
    Percentile bootstrap confidence interval for a binomial proportion.
    Returns (lower, upper) as percentages.
    """
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(42)
    samples = rng.binomial(n, successes / n, size=n_boot) / n * 100
    alpha = (1 - ci) / 2
    return (float(np.percentile(samples, alpha * 100)),
            float(np.percentile(samples, (1 - alpha) * 100)))


def run_failure_mode_analysis():
    if not os.path.exists(INPUT_FILE):
        print(f" [!] Audit telemetry not found: {INPUT_FILE}")
        return

    print(f" >> Failure Mode Analysis using: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    # ---- 1. Isolate DPF residual leakage events ----
    target_leaks = df[
        (df["System_Mode"] == TARGET_ARCHITECTURE) &
        (df["Hybrid_Verdict"] == "LEAK")
    ]
    total_dpf = len(df[df["System_Mode"] == TARGET_ARCHITECTURE])
    total_leaks = len(target_leaks)

    print(f"\n{'='*55}")
    print(f"  FAILURE MODE ANALYSIS: {TARGET_ARCHITECTURE}")
    print(f"{'='*55}")
    print(f" Total DPF Trials       : {total_dpf}")
    print(f" Residual Leakage Events: {total_leaks}")
    overall_rate = (total_leaks / total_dpf * 100) if total_dpf > 0 else 0.0
    ci_lo, ci_hi = bootstrap_ci(total_leaks, total_dpf)
    print(f" Overall Leakage Rate   : {overall_rate:.2f}% "
          f"[{ci_lo:.1f}%, {ci_hi:.1f}%] {int(CI_LEVEL*100)}% CI")
    print(f"{'-'*55}")

    if total_leaks == 0:
        print(" No leakage detected. System exhibits 100% containment.")
        return

    # ---- 2. Control Plane vs Data Plane attribution ----
    # Control Plane error: router sent query to wrong agent (Winner ≠ Data_Owner)
    router_errors = len(
        target_leaks[target_leaks["Winner_Agent"] != target_leaks["Data_Owner"]]
    )
    regex_misses = total_leaks - router_errors

    router_pct = router_errors / total_leaks * 100
    regex_pct = regex_misses / total_leaks * 100

    print(f" Control Plane (Routing Misclassification): "
          f"{router_errors:2d} ({router_pct:.1f}%)")
    print(f" Data Plane (Semantic Obfuscation)        : "
          f"{regex_misses:2d} ({regex_pct:.1f}%)")
    print(f"{'-'*55}")

    # ---- 3. Per-threat-class breakdown with bootstrap CIs ----
    if "Prompt_Category" in target_leaks.columns and "Prompt_Category" in df.columns:
        print("\n  PER-THREAT-CLASS BREAKDOWN (DPF residual leaks)")
        print(f"  {'Category':<30} {'Leaks':>5} {'Total':>6} {'Rate':>7}  {'95% CI':>15}")
        print(f"  {'-'*70}")

        class_rows = []
        for cat in df[df["System_Mode"] == TARGET_ARCHITECTURE]["Prompt_Category"].unique():
            cat_total = len(df[
                (df["System_Mode"] == TARGET_ARCHITECTURE) &
                (df["Prompt_Category"] == cat)
            ])
            cat_leaks = len(target_leaks[target_leaks["Prompt_Category"] == cat])
            rate = cat_leaks / cat_total * 100 if cat_total > 0 else 0.0
            lo, hi = bootstrap_ci(cat_leaks, cat_total)
            print(f"  {cat:<30} {cat_leaks:>5} {cat_total:>6} {rate:>6.1f}%"
                  f"  [{lo:.1f}%, {hi:.1f}%]")
            class_rows.append({
                "Category": cat,
                "Leaks": cat_leaks,
                "Total": cat_total,
                "Leakage_Rate_%": round(rate, 2),
                "CI_Lower_%": round(lo, 2),
                "CI_Upper_%": round(hi, 2),
            })

    # ---- 4. Router accuracy (if router log exists) ----
    if os.path.exists(ROUTER_LOG_FILE):
        print(f"\n  ROUTER ACCURACY (from {os.path.basename(ROUTER_LOG_FILE)})")
        rdf = pd.read_csv(ROUTER_LOG_FILE)
        labelled = rdf[rdf["true_role"].notna() & (rdf["true_role"] != "Unknown")]

        if not labelled.empty:
            overall_acc = labelled["correct"].mean()
            print(f"  Overall Accuracy: {overall_acc:.4f} ({len(labelled)} labelled turns)")
            print(f"  {'Role':<15} {'Precision':>10} {'Recall':>8} {'F1':>8} "
                  f"{'TP':>5} {'FP':>5} {'FN':>5}")
            print(f"  {'-'*60}")
            router_rows = []
            for role in labelled["true_role"].unique():
                tp = len(labelled[(labelled["predicted_role"] == role) &
                                  (labelled["true_role"] == role)])
                fp = len(labelled[(labelled["predicted_role"] == role) &
                                  (labelled["true_role"] != role)])
                fn = len(labelled[(labelled["predicted_role"] != role) &
                                  (labelled["true_role"] == role)])
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                print(f"  {role:<15} {p:>10.4f} {r:>8.4f} {f1:>8.4f} "
                      f"{tp:>5} {fp:>5} {fn:>5}")
                router_rows.append({
                    "Role": role, "Precision": round(p, 4),
                    "Recall": round(r, 4), "F1": round(f1, 4),
                    "TP": tp, "FP": fp, "FN": fn,
                })
            router_df = pd.DataFrame(router_rows)
            router_out = os.path.join(OUTPUT_DIR, "router_accuracy.csv")
            router_df.to_csv(router_out, index=False)
            print(f"\n  Router accuracy table saved: {router_out}")
        else:
            print("  No labelled turns found in router log (true_role = Unknown).")
    else:
        print(f"\n  [Note] Router log not found at {ROUTER_LOG_FILE}")
        print("  Supply true_role to execute_turn() in experiment_driver to generate it.")

    # ---- 5. Export ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results_data = {
        "Architectural_Layer": [
            "Control Plane (Routing Misclassification)",
            "Data Plane (Semantic Obfuscation)"
        ],
        "Vulnerability_Count": [router_errors, regex_misses],
        "Proportion_%": [f"{router_pct:.2f}%", f"{regex_pct:.2f}%"],
    }
    pd.DataFrame(results_data).to_csv(OUTPUT_FILE, index=False)
    print(f"\n >> Failure mode table saved: {OUTPUT_FILE}")

    if 'class_rows' in locals() and class_rows:
        class_out = os.path.join(OUTPUT_DIR, "threat_class_breakdown.csv")
        pd.DataFrame(class_rows).to_csv(class_out, index=False)
        print(f" >> Threat-class breakdown saved: {class_out}")

    print(f"{'='*55}\n")


if __name__ == "__main__":
    run_failure_mode_analysis()