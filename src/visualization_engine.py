"""
Analytical Visualization & Statistical Validation Engine — V2
--------------------------------------------------------------
Ingests telemetry/audit logs and produces publication-ready figures and tables.

V2 Changes:
- PALETTE and LABELS updated for 4 modes (STANDARD_POSTHOC_NLI added).
- generate_config_table() updated: POST-HOC NLI row added.
- generate_baseline_table() updated: LLM_Backend column shown when
  multiple backends are present in the data (cross-model sensitivity).
- generate_backend_sensitivity_table() NEW: summarizes leakage rates by
  (System_Mode, LLM_Backend) pair for the cross-model sensitivity analysis.
- plot_leakage_rate() and plot_category_breakdown() updated for 4 modes.
- generate_firewall_table() updated: reads Filter_Enforcement_Mode for
  cleaner DPF activity isolation.
- Columns added in V2 (LLM_Backend, Filter_Enforcement_Mode,
  Timing_Normalized) are read with .get() / fillna() so immutable
  paper_logs/ CSVs (which lack these columns) still work on Fast Path.
- BASE_DIR computed correctly for src/ location (unchanged).
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy.stats import mannwhitneyu, fisher_exact

try:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.dpi'] = 600
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if len(sys.argv) > 1:
    LOG_FILE = sys.argv[1]
else:
    LOG_FILE = os.path.join(BASE_DIR, "logs", "audit_results.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "paper_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# V2: 4-mode palette (colorblind-safe)
PALETTE = {
    "NAIVE_CONTROL":        "#D55E00",   # Vermillion
    "STANDARD_POSTHOC":     "#F0E442",   # Yellow
    "STANDARD_POSTHOC_NLI": "#56B4E9",   # Sky Blue (new fair baseline)
    "DPF_PROPOSED":         "#009E73",   # Bluish Green
}

LABELS = {
    "NAIVE_CONTROL":        "Naive Baseline",
    "STANDARD_POSTHOC":     "Post-Hoc (Regex)",
    "STANDARD_POSTHOC_NLI": "Post-Hoc (Regex+NLI)",
    "DPF_PROPOSED":         "DPF Architecture",
}

ALL_MODES = ["NAIVE_CONTROL", "STANDARD_POSTHOC", "STANDARD_POSTHOC_NLI", "DPF_PROPOSED"]


# ---------------------------------------------------------------------------
# Statistical Utilities
# ---------------------------------------------------------------------------

def calculate_cliffs_delta(u_stat, n1, n2):
    try:
        return (2 * u_stat) / (n1 * n2) - 1
    except ZeroDivisionError:
        return 0.0

def interpret_p_value(p):
    if p < 0.001: return "< 0.001"
    if p < 0.01:  return "< 0.01"
    if p < 0.05:  return "< 0.05"
    return "n.s."

def get_effect_size_label(d, metric="Cliffs"):
    ad = abs(d)
    if metric == "Cliffs":
        if ad < 0.147: return "Negligible"
        if ad < 0.33:  return "Small"
        if ad < 0.47:  return "Medium"
        return "Large"
    elif metric == "OddsRatio":
        if ad < 1.5: return "Negligible"
        if ad < 3.0: return "Small"
        if ad < 5.0: return "Medium"
        return "Large"
    return "Unknown"


# ---------------------------------------------------------------------------
# Table Generation
# ---------------------------------------------------------------------------

def generate_config_table(df):
    """
    Experimental configurations matrix — V2: includes POST-HOC NLI row.
    """
    n_counts = df["System_Mode"].value_counts()
    n_val = n_counts.get("DPF_PROPOSED", 0)

    data = {
        "Parameter": [
            "Semantic Routing", "Pre-Generation Firewall",
            "Post-Generation Filter (Regex)", "Post-Generation Filter (NLI)",
            "Private Memory Access", "Firewall Rule Set", "Prompt Count (N)"
        ],
        "NAIVE": ["Yes", "No", "No", "No", "No", "—", str(n_val)],
        "POST-HOC": ["Yes", "No", "Yes", "No", "Yes", "Regex (post)", str(n_val)],
        "POST-HOC+NLI": ["Yes", "No", "Yes", "Yes", "Yes", "Regex+NLI (post)", str(n_val)],
        "DPF (Proposed)": ["Yes", "Yes", "No", "No", "Yes", "Deterministic (pre)", str(n_val)],
    }

    out = pd.DataFrame(data)
    path = os.path.join(OUTPUT_DIR, "Table_Experimental_Configuration.csv")
    out.to_csv(path, index=False)
    print(f" >> [Table] Config Matrix: {path}")


def generate_baseline_table(df):
    """
    Core performance metrics. V2: shows LLM_Backend column when multiple
    backends are present (cross-model sensitivity analysis).
    """
    # Gracefully handle old CSVs lacking V2 columns
    if "LLM_Backend" not in df.columns:
        df["LLM_Backend"] = "llama3"
    if "Filter_Enforcement_Mode" not in df.columns:
        df["Filter_Enforcement_Mode"] = "UNKNOWN"

    present_modes = [m for m in ALL_MODES if m in df["System_Mode"].unique()]
    threat_df = df[~df["Prompt_Category"].isin(["General", "Benign_Utility_Test"])]
    naive_latency = df[df["System_Mode"] == "NAIVE_CONTROL"]["Total_Latency_ms"].mean()

    summary = []
    for mode in present_modes:
        subset = threat_df[threat_df["System_Mode"] == mode]
        leaks = len(subset[subset["Hybrid_Verdict"] == "LEAK"]) if "Hybrid_Verdict" in subset.columns else 0
        total = len(subset)
        leak_rate = leaks / total * 100 if total > 0 else 0

        swap_rate = 0.0
        if "Data_Owner" in subset.columns and "Winner_Agent" in subset.columns:
            swaps = len(subset[subset["Data_Owner"] != subset["Winner_Agent"]])
            swap_rate = swaps / total * 100 if total > 0 else 0

        full = df[df["System_Mode"] == mode]
        curr_lat = full["Total_Latency_ms"].mean()
        curr_std = full["Total_Latency_ms"].std()
        delta = curr_lat - naive_latency
        delta_str = "—" if mode == "NAIVE_CONTROL" else f"+{delta:.1f} ms"
        margin = full["Routing_Margin"].mean() if "Routing_Margin" in full.columns else 0.0

        backends = full["LLM_Backend"].unique().tolist()
        backend_str = ", ".join(backends) if backends else "llama3"

        summary.append({
            "Architecture": LABELS.get(mode, mode),
            "Cross-Context Leakage Events": f"{leaks}/{total}",
            "Leakage Rate (%)": f"{leak_rate:.1f}%",
            "Defensive Routing Rate (%)": f"{swap_rate:.1f}%",
            "Mean Latency (ms)": f"{curr_lat:.1f} ± {curr_std:.1f}",
            "Latency Overhead vs NAIVE": delta_str,
            "Routing Margin (Mean)": f"{margin:.4f}",
            "LLM Backend(s)": backend_str,
        })

    out = pd.DataFrame(summary)
    path = os.path.join(OUTPUT_DIR, "Table_Baseline_Comparison.csv")
    out.to_csv(path, index=False)
    print(f" >> [Table] Baseline Comparison: {path}")


def generate_backend_sensitivity_table(df):
    """
    V2 NEW: Cross-model leakage rate sensitivity table.
    Produces a (System_Mode × LLM_Backend) matrix of leakage rates.
    Only populated when multiple backends appear in the telemetry.
    """
    if "LLM_Backend" not in df.columns:
        print(" >> [Table] No LLM_Backend column — skipping sensitivity table.")
        return

    backends = df["LLM_Backend"].unique()
    if len(backends) < 2:
        print(f" >> [Table] Single backend ({backends[0]}) — sensitivity table skipped.")
        return

    threat_df = df[~df["Prompt_Category"].isin(["General", "Benign_Utility_Test"])]
    rows = []
    for mode in ALL_MODES:
        for backend in sorted(backends):
            subset = threat_df[
                (threat_df["System_Mode"] == mode) &
                (threat_df["LLM_Backend"] == backend)
            ]
            if subset.empty:
                continue
            if "Hybrid_Verdict" not in subset.columns:
                continue
            leaks = len(subset[subset["Hybrid_Verdict"] == "LEAK"])
            total = len(subset)
            rows.append({
                "Architecture": LABELS.get(mode, mode),
                "LLM_Backend": backend,
                "Leakage_Events": f"{leaks}/{total}",
                "Leakage_Rate_%": f"{leaks/total*100:.2f}%" if total > 0 else "N/A",
            })

    if rows:
        out = pd.DataFrame(rows)
        path = os.path.join(OUTPUT_DIR, "Table_Backend_Sensitivity.csv")
        out.to_csv(path, index=False)
        print(f" >> [Table] Backend Sensitivity: {path}")


def generate_category_table(df):
    if "Prompt_Category" not in df.columns:
        return
    threats = df[~df["Prompt_Category"].isin(["General", "Benign_Utility_Test"])]
    if "Hybrid_Verdict" not in threats.columns:
        return

    present_modes = [m for m in ALL_MODES if m in threats["System_Mode"].unique()]
    pivot = threats.pivot_table(
        index="Prompt_Category",
        columns="System_Mode",
        values="Hybrid_Verdict",
        aggfunc=lambda x: (len(x[x == "LEAK"]) / len(x) * 100)
    ).reindex(columns=present_modes)

    path = os.path.join(OUTPUT_DIR, "Table_Category_Analysis.csv")
    pivot.to_csv(path)
    print(f" >> [Table] Threat Vector Matrix: {path}")


def generate_stats_table(df):
    """Statistical significance tests. V2: includes POST-HOC-NLI comparisons."""
    if "Hybrid_Verdict" not in df.columns:
        return

    results = []

    def get_fisher_counts(mode):
        subset = df[(df["System_Mode"] == mode) & (~df["Prompt_Category"].isin(["General", "Benign_Utility_Test"]))]
        leaks = len(subset[subset["Hybrid_Verdict"] == "LEAK"])
        return leaks, len(subset) - leaks

    def get_vec(mode):
        return df[df["System_Mode"] == mode]["Total_Latency_ms"].dropna()

    comparisons = [
        ("NAIVE_CONTROL",        "STANDARD_POSTHOC",     "Leakage Rate", "fisher"),
        ("STANDARD_POSTHOC",     "STANDARD_POSTHOC_NLI", "Leakage Rate", "fisher"),
        ("STANDARD_POSTHOC_NLI", "DPF_PROPOSED",         "Leakage Rate", "fisher"),
        ("STANDARD_POSTHOC",     "DPF_PROPOSED",         "Leakage Rate", "fisher"),
        ("STANDARD_POSTHOC",     "DPF_PROPOSED",         "Latency",      "mwu"),
    ]

    for sys_a, sys_b, metric_name, test_type in comparisons:
        # Skip if either mode not in data
        if sys_a not in df["System_Mode"].values or sys_b not in df["System_Mode"].values:
            continue

        if test_type == "fisher":
            la, sa = get_fisher_counts(sys_a)
            lb, sb = get_fisher_counts(sys_b)
            if (la + sa) == 0 or (lb + sb) == 0:
                continue
            odds_ratio, p = fisher_exact([[la, sa], [lb, sb]], alternative='greater')
            effect_label = get_effect_size_label(odds_ratio, "OddsRatio")
            stat_val = f"OR={odds_ratio:.1f}"
            test_name = "Fisher's Exact"
        else:
            va, vb = get_vec(sys_a), get_vec(sys_b)
            if len(va) == 0 or len(vb) == 0:
                continue
            u_stat, p = mannwhitneyu(va, vb, alternative='two-sided')
            delta = calculate_cliffs_delta(u_stat, len(va), len(vb))
            effect_label = get_effect_size_label(delta, "Cliffs")
            stat_val = f"U={u_stat:.0f}"
            test_name = "Mann-Whitney U"

        label_a = LABELS.get(sys_a, sys_a)
        label_b = LABELS.get(sys_b, sys_b)
        results.append({
            "Comparison": f"{label_a} vs {label_b}",
            "Metric": metric_name,
            "Test": test_name,
            "Statistic": stat_val,
            "p-value": interpret_p_value(p),
            "Effect Size": effect_label,
        })

    if results:
        out = pd.DataFrame(results)
        path = os.path.join(OUTPUT_DIR, "Table_Statistical_Results.csv")
        out.to_csv(path, index=False)
        print(f" >> [Table] Statistical Results: {path}")


def generate_firewall_table(df):
    """
    DPF internal telemetry. V2: uses Filter_Enforcement_Mode == "PRE_GEN"
    to isolate DPF rows cleanly even when the CSV contains all 4 modes.
    """
    if "Filter_Enforcement_Mode" in df.columns:
        dpf = df[df["Filter_Enforcement_Mode"] == "PRE_GEN"]
    else:
        dpf = df[df["System_Mode"] == "DPF_PROPOSED"]

    if dpf.empty:
        dpf = df[df["System_Mode"] == "DPF_PROPOSED"]

    total = len(dpf)
    fw_triggers = len(dpf[dpf["Redaction_Count"] > 0]) if "Redaction_Count" in dpf.columns else 0
    routing_interventions = (
        len(dpf[dpf["Data_Owner"] != dpf["Winner_Agent"]])
        if "Data_Owner" in dpf.columns and "Winner_Agent" in dpf.columns else 0
    )
    total_redactions = dpf["Redaction_Count"].sum() if "Redaction_Count" in dpf.columns else 0
    mean_redactions = dpf["Redaction_Count"].mean() if "Redaction_Count" in dpf.columns else 0
    mean_fw_lat = dpf["Latency_Firewall_ms"].mean() if "Latency_Firewall_ms" in dpf.columns else 0

    timing_norm_count = 0
    if "Timing_Normalized" in dpf.columns:
        timing_norm_count = dpf["Timing_Normalized"].sum()

    data = [
        ("Total Prompts Processed",             total),
        ("Firewall Triggers (Regex/Static)",     fw_triggers),
        ("Routing Interventions (Context Swaps)", routing_interventions),
        ("Total Active Defenses",               fw_triggers + routing_interventions),
        ("Total Redactions Applied",            int(total_redactions)),
        ("Mean Redactions / Prompt",            f"{mean_redactions:.2f}"),
        ("Mean Firewall Latency (ms)",          f"{mean_fw_lat:.2f}"),
        ("Timing-Normalized Rejections [V2]",   int(timing_norm_count)),
    ]

    out = pd.DataFrame(data, columns=["Metric", "Value"])
    path = os.path.join(OUTPUT_DIR, "Table_Firewall_Activity.csv")
    out.to_csv(path, index=False)
    print(f" >> [Table] Firewall Telemetry: {path}")


# ---------------------------------------------------------------------------
# Figure Generation
# ---------------------------------------------------------------------------

def plot_latency_safety_tradeoff(df):
    present_modes = [m for m in ALL_MODES if m in df["System_Mode"].unique()]
    if "Hybrid_Verdict" not in df.columns:
        return

    def calc_leak_rate(grp):
        return (len(grp[grp["Hybrid_Verdict"] == "LEAK"]) / len(grp)) * 100 if len(grp) > 0 else 0

    stats = df.groupby("System_Mode").agg(
        mean_lat=("Total_Latency_ms", "mean"),
        std_lat=("Latency_Generation_ms", "std"),
    )
    stats["Leakage_Rate"] = df.groupby("System_Mode").apply(calc_leak_rate)
    stats = stats.reindex(present_modes).fillna(0)

    fig, ax1 = plt.subplots(figsize=(7.16, 4))
    x = np.arange(len(stats))
    bar_colors = [PALETTE.get(m, "#999999") for m in stats.index]

    ax1.bar(x, stats["mean_lat"], 0.5,
            yerr=stats["std_lat"], capsize=5,
            color=bar_colors, alpha=0.8, edgecolor='black', zorder=2)
    ax1.set_ylabel('Latency (ms)', fontweight='bold', color='#444444')
    ax1.set_xticks(x)
    ax1.set_xticklabels([LABELS.get(m, m) for m in stats.index], rotation=10)
    ax1.set_ylim(bottom=0, top=stats["mean_lat"].max() * 1.3)
    ax1.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

    ax2 = ax1.twinx()
    ax2.set_ylim(0, 110)
    ax2.plot(x, stats["Leakage_Rate"], marker='D', markersize=8,
             linewidth=2, linestyle='-', zorder=5)
    ax2.set_ylabel('Leakage Rate (%)', fontweight='bold')
    for i, v in enumerate(stats["Leakage_Rate"]):
        ax2.annotate(f"{v:.1f}%", (x[i], v + 4), ha='center', fontweight='bold')

    plt.title("Latency vs. Privacy Safety Trade-off", fontsize=11, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "Figure_Latency_Safety_Tradeoff.png")
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f" >> [Plot] Latency-Safety Tradeoff: {path}")


def plot_leakage_rate(df):
    if "Hybrid_Verdict" not in df.columns:
        return
    threats = df[~df["Prompt_Category"].isin(["General", "Benign_Utility_Test"])]
    present_modes = [m for m in ALL_MODES if m in threats["System_Mode"].unique()]
    if not present_modes:
        return

    rates = []
    for mode in present_modes:
        subset = threats[threats["System_Mode"] == mode]
        leaks = len(subset[subset["Hybrid_Verdict"] == "LEAK"])
        rates.append(leaks / len(subset) * 100 if len(subset) > 0 else 0)

    plt.figure(figsize=(4.5, 3.5))
    bars = plt.bar(
        [LABELS.get(m, m) for m in present_modes], rates,
        color=[PALETTE.get(m, "#999999") for m in present_modes],
        edgecolor='black', alpha=0.9, width=0.6
    )
    plt.title("Cross-Context Leakage Rate (Global)", fontsize=11, fontweight='bold')
    plt.ylabel("Leakage Rate (% of Threats)", fontsize=9)
    plt.ylim(0, 115)
    plt.xticks(rotation=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., h + 2,
                 f'{h:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "Figure_Leakage_Rate_Global.png")
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f" >> [Plot] Global Leakage Rate: {path}")


def plot_category_breakdown(df):
    if "Prompt_Category" not in df.columns or "Hybrid_Verdict" not in df.columns:
        return
    threats = df[~df["Prompt_Category"].isin(["General", "Benign_Utility_Test"])]
    present_modes = [m for m in ALL_MODES if m in threats["System_Mode"].unique()]
    categories = threats["Prompt_Category"].unique()
    if not len(categories):
        return

    x = np.arange(len(categories))
    width = 0.8 / len(present_modes)

    plt.figure(figsize=(7.16, 4))
    for i, mode in enumerate(present_modes):
        mode_data = threats[threats["System_Mode"] == mode]
        y_vals = []
        for cat in categories:
            sub = mode_data[mode_data["Prompt_Category"] == cat]
            rate = (len(sub[sub["Hybrid_Verdict"] == "LEAK"]) / len(sub) * 100
                    if len(sub) > 0 else 0)
            y_vals.append(rate)
        plt.bar(x + i * width, y_vals, width,
                label=LABELS.get(mode, mode),
                color=PALETTE.get(mode, "#999999"), edgecolor='black')

    plt.title("Leakage Rate by Attack Vector", fontsize=11, fontweight='bold')
    plt.ylabel("Leakage Rate (%)", fontsize=9)
    plt.xticks(x + width * (len(present_modes) - 1) / 2, categories, rotation=15)
    plt.legend(loc='best', fontsize=8)
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "Figure_Leakage_By_Category.png")
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f" >> [Plot] Category Breakdown: {path}")


def plot_backend_sensitivity(df):
    """
    V2 NEW: Bar chart comparing leakage rates across LLM backends for
    each architectural mode. Only rendered when ≥2 backends are present.
    """
    if "LLM_Backend" not in df.columns or "Hybrid_Verdict" not in df.columns:
        return
    backends = df["LLM_Backend"].unique()
    if len(backends) < 2:
        return

    threats = df[~df["Prompt_Category"].isin(["General", "Benign_Utility_Test"])]
    present_modes = [m for m in ALL_MODES if m in threats["System_Mode"].unique()]
    x = np.arange(len(present_modes))
    width = 0.35
    backend_colors = ["#0072B2", "#CC79A7"]  # blue, pink — colorblind safe

    plt.figure(figsize=(7.16, 4))
    for i, backend in enumerate(sorted(backends)):
        rates = []
        for mode in present_modes:
            sub = threats[(threats["System_Mode"] == mode) &
                          (threats["LLM_Backend"] == backend)]
            if len(sub) == 0:
                rates.append(0)
                continue
            rates.append(len(sub[sub["Hybrid_Verdict"] == "LEAK"]) / len(sub) * 100)
        plt.bar(x + i * width, rates, width,
                label=backend, color=backend_colors[i % len(backend_colors)],
                edgecolor='black', alpha=0.85)

    plt.title("Leakage Rate by Architecture & LLM Backend", fontsize=11, fontweight='bold')
    plt.ylabel("Leakage Rate (%)", fontsize=9)
    plt.xticks(x + width / 2, [LABELS.get(m, m) for m in present_modes], rotation=12)
    plt.legend(title="Backend", fontsize=8)
    plt.ylim(0, 115)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "Figure_Backend_Sensitivity.png")
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f" >> [Plot] Backend Sensitivity: {path}")


def plot_scalability_trends(df):
    present_modes = [m for m in ["STANDARD_POSTHOC", "DPF_PROPOSED"]
                     if m in df["System_Mode"].unique()]
    plt.figure(figsize=(7.16, 4))
    for mode in present_modes:
        subset = df[df["System_Mode"] == mode].sort_values("Response_Word_Count")
        if subset.empty or len(subset) < 2:
            continue
        x_data = subset["Response_Word_Count"]
        y_data = subset["Total_Latency_ms"]
        try:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            x_range = np.linspace(0, x_data.max() * 1.1, 100)
            plt.scatter(x_data, y_data, alpha=0.5, s=20,
                        color=PALETTE.get(mode), label=f"{LABELS.get(mode, mode)} (Raw)")
            plt.plot(x_range, p(x_range), linestyle="--", linewidth=2,
                     color=PALETTE.get(mode), label=f"{LABELS.get(mode, mode)} Trend")
        except Exception:
            pass

    plt.title("System Scalability (Latency vs Output Complexity)", fontsize=11, fontweight='bold')
    plt.xlabel("Response Complexity (Word Count)", fontsize=9)
    plt.ylabel("Processing Latency (ms)", fontsize=9)
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.legend(fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "Figure_Scalability_Trends.png")
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f" >> [Plot] Scalability Trends: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(LOG_FILE):
        print(f" [CRITICAL] Audit data not found: {LOG_FILE}")
        return

    print(f" >> [System] Data Synthesis Pipeline: {LOG_FILE}")
    try:
        df = pd.read_csv(LOG_FILE)

        # Ensure V2 columns exist (graceful fallback for old paper_logs CSVs)
        if "LLM_Backend" not in df.columns:
            df["LLM_Backend"] = "llama3"
        if "Filter_Enforcement_Mode" not in df.columns:
            df["Filter_Enforcement_Mode"] = "UNKNOWN"
        if "Timing_Normalized" not in df.columns:
            df["Timing_Normalized"] = 0

        generate_config_table(df)
        generate_baseline_table(df)
        generate_backend_sensitivity_table(df)
        generate_stats_table(df)
        generate_firewall_table(df)
        generate_category_table(df)

        plot_latency_safety_tradeoff(df)
        plot_leakage_rate(df)
        plot_category_breakdown(df)
        plot_backend_sensitivity(df)
        plot_scalability_trends(df)

        print(f" >> [Success] All artifacts saved to '{OUTPUT_DIR}'")

    except Exception as e:
        print(f" [CRITICAL] Visualization pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()