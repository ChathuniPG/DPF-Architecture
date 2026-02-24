"""
Analytical Visualization & Statistical Validation Engine
------------------------------------------------------
This module serves as the primary data synthesis pipeline. It ingests 
the raw telemetry and automated audit logs generated during system evaluation, 
and produces publication-ready statistical tables and high-resolution figures.

Data Provenance:
- Source: Dynamically routed via CLI (defaults to 'logs/audit_results.csv').
- Ground Truth: Derives performance metrics utilizing the 'Hybrid_Verdict' column.

Generated Artifacts:
1. Figures (.png):
   - Latency-Safety Trade-off (Dual Y-Axis)
   - Cross-Context Leakage Rate (Global Architecture Comparison)
   - Attack-Specific Leakage Analysis (Vector Category Breakdown)
   - Scalability Trends (Linear Regression Modeling)
2. Statistical Tables (.csv):
   - Experimental Configuration Matrix
   - Baseline Comparison Summary (Mean Latency & Leakage Rates)
   - Statistical Significance Analysis (Fisher's Exact & Mann-Whitney U)
   - Deterministic Firewall Activity Summary
   - Threat Category Analysis Matrix
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy.stats import mannwhitneyu, fisher_exact

# --- PUBLICATION GRAPHICS STANDARDS ---
# Enforces strict typesetting and DPI configurations suitable for 
# inclusion in high-impact scientific or engineering literature.
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
    pass # Fallback to system defaults if specific fonts are unavailable

# --- INFRASTRUCTURE CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Allow dynamic path injection from the CLI, default to standard runtime logs/
if len(sys.argv) > 1:
    LOG_FILE = sys.argv[1]
else:
    LOG_FILE = os.path.join(BASE_DIR, "logs", "audit_results.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "paper_results")
if not os.path.exists(OUTPUT_DIR):
    try: os.makedirs(OUTPUT_DIR)
    except: pass

# Accessibility-conscious color palette (Colorblind safe)
PALETTE = {
    "NAIVE_CONTROL": "#D55E00",      # Vermillion (High Risk Baseline)
    "STANDARD_POSTHOC": "#F0E442",   # Yellow (Caution / Intermediate)
    "DPF_PROPOSED": "#009E73"        # Bluish Green (Secured State)
}

LABELS = {
    "NAIVE_CONTROL": "Naive Baseline",
    "STANDARD_POSTHOC": "Post-Hoc Filter",
    "DPF_PROPOSED": "DPF Architecture"
}

# --- STATISTICAL UTILITIES ---

def calculate_cliffs_delta(u_stat, n1, n2):
    """Calculates Cliff's Delta (Non-parametric effect size) for latency analysis."""
    try:
        return (2 * u_stat) / (n1 * n2) - 1
    except ZeroDivisionError:
        return 0.0

def calculate_odds_ratio(a, b, c, d):
    """Calculates the Odds Ratio for categorical Fisher's Exact Tests."""
    try:
        return (a * d) / (b * c)
    except ZeroDivisionError:
        return 0.0

def interpret_p_value(p):
    """Standardizes the formatting of p-values for tabular display."""
    if p < 0.001: return "< 0.001"
    if p < 0.01: return "< 0.01"
    if p < 0.05: return "< 0.05"
    return "n.s."

def get_effect_size_label(d, metric="Cliffs"):
    """Maps continuous effect size values to qualitative standard interpretations."""
    ad = abs(d)
    if metric == "Cliffs":
        if ad < 0.147: return "Negligible"
        if ad < 0.33: return "Small"
        if ad < 0.47: return "Medium"
        return "Large"
    elif metric == "OddsRatio":
        if ad < 1.5: return "Negligible"
        if ad < 3.0: return "Small"
        if ad < 5.0: return "Medium"
        return "Large"
    return "Unknown"

# --- REPORT GENERATION (TABULAR) ---

def generate_config_table(df):
    """
    Serializes the Experimental Configurations matrix.
    """
    n_counts = df["System_Mode"].value_counts()
    n_val = n_counts.get("DPF_PROPOSED", 0) 
    
    data = {
        "Parameter": [
            "Semantic Routing", "Pre-Generation Firewall", "Post-Generation Filter", 
            "Private Memory Access", "Firewall Rule Set", "Prompt Count (N)"
        ],
        "NAIVE": ["Yes", "No", "No", "No", "—", f"{n_val}"],
        "POST-HOC": ["Yes", "No", "Yes", "Yes", "Regex (post)", f"{n_val}"],
        "DPF (Proposed)": ["Yes", "Yes", "No", "Yes", "Deterministic (pre)", f"{n_val}"]
    }
    
    out_df = pd.DataFrame(data)
    outfile = os.path.join(OUTPUT_DIR, "Table_Experimental_Configuration.csv")
    out_df.to_csv(outfile, index=False)
    print(f" >> [Table] Generated Configuration Matrix: {outfile}")

def generate_baseline_table(df):
    """
    Synthesizes the core architectural performance metrics (Latency & Leakage).
    """
    modes = ["NAIVE_CONTROL", "STANDARD_POSTHOC", "DPF_PROPOSED"]
    # Filter out benign traffic to calculate adversarial leakage accurately
    threat_df = df[df["Prompt_Category"] != "General"]

    summary = []
    naive_latency = df[df["System_Mode"] == "NAIVE_CONTROL"]["Total_Latency_ms"].mean()

    for mode in modes:
        subset = threat_df[threat_df["System_Mode"] == mode]
        
        # 1. Leakage Metrics (Ground Truth Verification)
        leaks = len(subset[subset["Hybrid_Verdict"] == "LEAK"])
        total = len(subset)
        leak_rate = (leaks / total * 100) if total > 0 else 0
        leak_str = f"{leaks}/{total}"

        # 2. Defensive Routing (Inter-Agent Handoff Verification)
        if "Data_Owner" in subset.columns and "Winner_Agent" in subset.columns:
            swaps = len(subset[subset["Data_Owner"] != subset["Winner_Agent"]])
            swap_rate = (swaps / total * 100) if total > 0 else 0
        else:
            swap_rate = 0.0

        # 3. Latency Profiling (Mean ± Std Dev)
        full_subset = df[df["System_Mode"] == mode]
        curr_latency = full_subset["Total_Latency_ms"].mean()
        curr_std = full_subset["Total_Latency_ms"].std() 
        
        latency_delta = curr_latency - naive_latency
        delta_str = "—" if mode == "NAIVE_CONTROL" else f"+{latency_delta:.1f} ms"
        
        # 4. Routing Decision Confidence
        margin = full_subset["Routing_Margin"].mean()
        
        summary.append({
            "Metric": LABELS[mode],
            "Cross-Context Leakage Events": leak_str,
            "Leakage Rate (%)": f"{leak_rate:.1f}%",
            "Defensive Routing Rate (%)": f"{swap_rate:.1f}%",
            "Mean Latency (ms)": f"{curr_latency:.1f} ± {curr_std:.1f}", 
            "Latency Overhead vs NAIVE": delta_str,
            "Routing Margin (Mean)": f"{margin:.4f}"
        })

    metrics = list(summary[0].keys())[1:] 
    final_data = {"Metric": metrics}
    for sys_data in summary:
        col_name = sys_data["Metric"]
        final_data[col_name] = [sys_data[m] for m in metrics]
        
    out_df = pd.DataFrame(final_data)
    outfile = os.path.join(OUTPUT_DIR, "Table_Baseline_Comparison.csv")
    out_df.to_csv(outfile, index=False)
    print(f" >> [Table] Generated Baseline Comparison: {outfile}")

def generate_category_table(df):
    """
    Pivot table mapping threat vectors to architectural resilience.
    """
    if "Prompt_Category" not in df.columns: return

    threats = df[df["Prompt_Category"] != "General"]
    
    pivot = threats.pivot_table(
        index="Prompt_Category", 
        columns="System_Mode", 
        values="Hybrid_Verdict",
        aggfunc=lambda x: (len(x[x=="LEAK"]) / len(x) * 100)
    ).reindex(columns=["NAIVE_CONTROL", "STANDARD_POSTHOC", "DPF_PROPOSED"])
    
    outfile = os.path.join(OUTPUT_DIR, "Table_Category_Analysis.csv")
    pivot.to_csv(outfile)
    print(f" >> [Table] Generated Threat Vector Matrix: {outfile}")

def generate_stats_table(df):
    """
    Executes rigorous statistical hypothesis testing on the performance data.
    - Fisher's Exact Test: Used for categorical/binary outcomes (Leak vs Safe).
    - Mann-Whitney U: Used for non-normally distributed continuous data (Latency).
    """
    results = []
    
    def get_fisher_counts(mode):
        subset = df[(df["System_Mode"] == mode) & (df["Prompt_Category"] != "General")]
        leaks = len(subset[subset["Hybrid_Verdict"] == "LEAK"])
        safe = len(subset) - leaks
        return leaks, safe

    def get_vec(mode, col):
        return df[df["System_Mode"] == mode][col].dropna()

    comparisons = [
        ("NAIVE_CONTROL", "STANDARD_POSTHOC", "Leakage Rate", "Outcomes", "fisher"),
        ("STANDARD_POSTHOC", "DPF_PROPOSED", "Leakage Rate", "Outcomes", "fisher"),
        ("STANDARD_POSTHOC", "DPF_PROPOSED", "Latency", "Performance", "mwu"),
    ]

    for sys_a, sys_b, metric_name, category, test_type in comparisons:
        
        if test_type == "fisher":
            leaks_a, safe_a = get_fisher_counts(sys_a)
            leaks_b, safe_b = get_fisher_counts(sys_b)
            
            # Formulate Contingency Table
            table = [[leaks_a, safe_a], [leaks_b, safe_b]]
            odds_ratio, p = fisher_exact(table, alternative='greater') 
            
            effect_label = get_effect_size_label(odds_ratio, "OddsRatio")
            stat_val = f"OR={odds_ratio:.1f}"
            test_name = "Fisher's Exact"

        else: 
            vec_a = get_vec(sys_a, "Total_Latency_ms")
            vec_b = get_vec(sys_b, "Total_Latency_ms")
            
            if len(vec_a) > 0 and len(vec_b) > 0:
                u_stat, p = mannwhitneyu(vec_a, vec_b, alternative='two-sided')
                delta = calculate_cliffs_delta(u_stat, len(vec_a), len(vec_b))
                effect_label = get_effect_size_label(delta, "Cliffs")
                stat_val = f"U={u_stat:.0f}"
                test_name = "Mann-Whitney U"
            else:
                continue

        name_a = sys_a.replace("_CONTROL", "").replace("_PROPOSED", "").replace("STANDARD_POSTHOC", "POST-HOC")
        name_b = sys_b.replace("_CONTROL", "").replace("_PROPOSED", "").replace("STANDARD_POSTHOC", "POST-HOC")
        
        results.append({
            "Comparison": f"{name_a} vs {name_b}",
            "Metric": metric_name,
            "Test": test_name,
            "Statistic": stat_val,
            "p-value": interpret_p_value(p),
            "Effect Size": effect_label
        })
            
    out_df = pd.DataFrame(results)
    outfile = os.path.join(OUTPUT_DIR, "Table_Statistical_Results.csv")
    out_df.to_csv(outfile, index=False)
    print(f" >> [Table] Generated Statistical Analysis: {outfile}")

def generate_firewall_table(df):
    """
    Summarizes the internal telemetry of the deterministic sanitization engine.
    """
    dpf = df[df["System_Mode"] == "DPF_PROPOSED"]
    
    total_prompts = len(dpf)
    fw_triggers = len(dpf[dpf["Redaction_Count"] > 0])
    
    if "Data_Owner" in dpf.columns and "Winner_Agent" in dpf.columns:
        routing_interventions = len(dpf[dpf["Data_Owner"] != dpf["Winner_Agent"]])
    else:
        routing_interventions = 0

    total_redactions = dpf["Redaction_Count"].sum()
    mean_redactions = dpf["Redaction_Count"].mean()
    mean_fw_latency = dpf["Latency_Firewall_ms"].mean()
    
    data = [
        ("Total Prompts Processed", total_prompts),
        ("Firewall Triggers (Regex/Static)", fw_triggers),
        ("Routing Interventions (Context Swaps)", routing_interventions),
        ("Total Active Defenses", fw_triggers + routing_interventions),
        ("Total Redactions Applied", int(total_redactions)),
        ("Mean Redactions / Prompt", f"{mean_redactions:.2f}"),
        ("Mean Firewall Latency (ms)", f"{mean_fw_latency:.2f}")
    ]
    
    out_df = pd.DataFrame(data, columns=["Metric", "Value"])
    outfile = os.path.join(OUTPUT_DIR, "Table_Firewall_Activity.csv")
    out_df.to_csv(outfile, index=False)
    print(f" >> [Table] Generated Firewall Telemetry: {outfile}")

# --- FIGURE GENERATION (PLOTS) ---

def plot_latency_safety_tradeoff(df):
    """ 
    Renders a dual-axis bar/line chart visualizing the inverse correlation 
    between computational overhead (latency) and system security (leakage rate).
    """
    def calc_leak_rate(x):
        return (len(x[x["Hybrid_Verdict"] == "LEAK"]) / len(x)) * 100

    stats = df.groupby("System_Mode").agg({
        "Total_Latency_ms": "mean",
        "Latency_Generation_ms": "std"
    })
    stats["Leakage_Rate"] = df.groupby("System_Mode").apply(calc_leak_rate)
    
    stats = stats.reindex(["NAIVE_CONTROL", "STANDARD_POSTHOC", "DPF_PROPOSED"]).fillna(0)

    fig, ax1 = plt.subplots(figsize=(7.16, 4))
    x = np.arange(len(stats))
    width = 0.5 
    bar_colors = [PALETTE[m] for m in stats.index]
    
    # Latency Bars
    bars = ax1.bar(x, stats["Total_Latency_ms"], width, 
                   yerr=stats["Latency_Generation_ms"], capsize=5,
                   color=bar_colors, alpha=0.8, edgecolor='black', zorder=2)
    
    ax1.set_ylabel('Latency (ms)', fontweight='bold', color='#444444')
    ax1.set_xticks(x)
    ax1.set_xticklabels([LABELS[m].replace("\n", " ") for m in stats.index])
    ax1.set_ylim(bottom=0, top=stats["Total_Latency_ms"].max() * 1.3)
    ax1.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

    # Leakage Rate Line overlay
    ax2 = ax1.twinx()
    ax2.set_ylim(0, 110)
    
    line = ax2.plot(x, stats["Leakage_Rate"], marker='D', markersize=8, 
                    label='Leakage Rate (%)', linewidth=2, linestyle='-', zorder=5)
    
    ax2.set_ylabel('Leakage Rate', fontweight='bold')
    ax2.tick_params(axis='y')
    
    for i, txt in enumerate(stats["Leakage_Rate"]):
        ax2.annotate(f"{txt:.1f}%", (x[i], stats["Leakage_Rate"].iloc[i] + 5), 
                     ha='center', fontweight='bold')

    plt.title("Latency vs. Privacy Safety Trade-off", fontsize=11, fontweight='bold')
    plt.tight_layout()
    
    outfile = os.path.join(OUTPUT_DIR, "Figure_Latency_Safety_Tradeoff.png")
    plt.savefig(outfile, dpi=600, bbox_inches='tight') 
    plt.close()
    print(f" >> [Plot] Generated Latency-Safety Vector: {outfile}")

def plot_leakage_rate(df):
    """ 
    Renders a standard bar chart depicting the global leakage 
    vulnerability of the tested architectures.
    """
    threats = df[df["Prompt_Category"] != "General"]
    if len(threats) == 0: return

    leakage_stats = []
    for mode in ["NAIVE_CONTROL", "STANDARD_POSTHOC", "DPF_PROPOSED"]:
        mode_data = threats[threats["System_Mode"] == mode]
        if len(mode_data) == 0:
             leakage_stats.append({"Mode": mode, "Leakage_Rate": 0})
             continue
             
        leaks = len(mode_data[mode_data["Hybrid_Verdict"] == "LEAK"])
        total = len(mode_data)
        leakage_stats.append({
            "Mode": mode,
            "Leakage_Rate": (leaks / total) * 100
        })
        
    stats_df = pd.DataFrame(leakage_stats).set_index("Mode")

    plt.figure(figsize=(3.5, 3.5))
    bars = plt.bar([LABELS[m].replace(" Baseline", "").replace(" Filter", "").replace(" Architecture", "") for m in stats_df.index], 
                   stats_df["Leakage_Rate"], 
                   color=[PALETTE[m] for m in stats_df.index], 
                   edgecolor='black', alpha=0.9, width=0.6)
    
    plt.title("Cross-Context Leakage Rate (Global)", fontsize=11, fontweight='bold')
    plt.ylabel("Leakage Rate (% of Threats)", fontsize=9)
    plt.ylim(0, 115)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                 f'{height:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, "Figure_Leakage_Rate_Global.png")
    plt.savefig(outfile, dpi=600, bbox_inches='tight')
    plt.close()
    print(f" >> [Plot] Generated Global Leakage Vector: {outfile}")

def plot_category_breakdown(df):
    """
    Renders a clustered bar chart decomposing architectural vulnerability 
    across specific adversarial attack vectors.
    """
    if "Prompt_Category" not in df.columns: return
    threats = df[df["Prompt_Category"] != "General"]
    
    stats = threats.groupby(["Prompt_Category", "System_Mode"]).apply(
        lambda x: (len(x[x["Hybrid_Verdict"] == "LEAK"]) / len(x)) * 100
    ).reset_index(name="Leakage_Rate")
    
    stats = stats[stats["System_Mode"].isin(["NAIVE_CONTROL", "STANDARD_POSTHOC", "DPF_PROPOSED"])]
    
    categories = stats["Prompt_Category"].unique()
    modes = ["NAIVE_CONTROL", "STANDARD_POSTHOC", "DPF_PROPOSED"]
    x = np.arange(len(categories))
    width = 0.25
    
    plt.figure(figsize=(7.16, 4))
    
    for i, mode in enumerate(modes):
        mode_data = stats[stats["System_Mode"] == mode]
        y_vals = []
        for cat in categories:
            val = mode_data[mode_data["Prompt_Category"] == cat]["Leakage_Rate"].values
            y_vals.append(val[0] if len(val) > 0 else 0)
            
        plt.bar(x + i*width, y_vals, width, label=LABELS[mode], 
                color=PALETTE[mode], edgecolor='black')

    plt.title("Leakage Rate by Attack Vector", fontsize=11, fontweight='bold')
    plt.ylabel("Leakage Rate (%)", fontsize=9)
    plt.xlabel("Attack Category", fontsize=9)
    plt.xticks(x + width, categories, rotation=15)
    plt.legend(loc='best', fontsize=8)
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, "Figure_Leakage_By_Category.png")
    plt.savefig(outfile, dpi=600, bbox_inches='tight')
    plt.close()
    print(f" >> [Plot] Generated Category Breakdown Vector: {outfile}")

def plot_scalability_trends(df):
    """ 
    Renders a scatter plot with linear regression overlays to demonstrate 
    how architectural latency scales with output complexity.
    """
    plt.figure(figsize=(7.16, 4))
    
    for mode in ["STANDARD_POSTHOC", "DPF_PROPOSED"]:
        subset = df[df["System_Mode"] == mode]
        if not subset.empty and len(subset) > 1:
            subset = subset.sort_values("Response_Word_Count")
            x_data = subset["Response_Word_Count"]
            y_data = subset["Total_Latency_ms"]
            try:
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                x_range = np.linspace(0, x_data.max() * 1.1, 100)
                plt.scatter(x_data, y_data, alpha=0.5, label=f"{LABELS[mode].split()[0]} (Raw)", 
                            color=PALETTE[mode], s=20)
                plt.plot(x_range, p(x_range), linestyle="--", linewidth=2, 
                         label=f"{LABELS[mode].split()[0]} Trend", color=PALETTE[mode])
            except Exception: pass

    plt.title("System Scalability (Latency vs Output Complexity)", fontsize=11, fontweight='bold')
    plt.xlabel("Response Complexity (Word Count)", fontsize=9)
    plt.ylabel("Processing Latency (ms)", fontsize=9)
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.legend(fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    outfile = os.path.join(OUTPUT_DIR, "Figure_Scalability_Trends.png")
    plt.savefig(outfile, dpi=600, bbox_inches='tight')
    plt.close()
    print(f" >> [Plot] Generated Scalability Vector: {outfile}")

def main():
    """
    Executes the master visualization and analytics pipeline.
    """
    if not os.path.exists(OUTPUT_DIR):
        try: os.makedirs(OUTPUT_DIR)
        except: pass
        
    if not os.path.exists(LOG_FILE):
        print(f" [CRITICAL ERROR] Target Audit Data not found: {LOG_FILE}")
        print(f" [REMEDIATION] Ensure the telemetry logs exist or verify the provided path.")
        return

    try:
        print(f" >> [System] Initializing Data Synthesis Pipeline using: {LOG_FILE}")
        df = pd.read_csv(LOG_FILE)
        
        # 1. Tabular Data Serialization
        generate_config_table(df)
        generate_baseline_table(df)
        generate_stats_table(df)
        generate_firewall_table(df)
        generate_category_table(df)
        
        # 2. High-Resolution Figure Generation
        plot_latency_safety_tradeoff(df)
        plot_leakage_rate(df)
        plot_category_breakdown(df)
        plot_scalability_trends(df)
        
        print(f" >> [Success] All analytical artifacts serialized to '{OUTPUT_DIR}'")
        
    except Exception as e:
        print(f" [CRITICAL ERROR] Visualization pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()