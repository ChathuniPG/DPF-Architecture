"""
Analytical Visualization Engine & Statistical Validation
------------------------------------------------------
This module processes audited telemetry logs to generate quantitative 
results.

DATA SOURCE:
- Reads 'audit_results.csv' (Output of evaluate_privacy_audit.py).
- Uses 'Hybrid_Verdict' (Ground Truth).

OUTPUTS:
1. Figures:
   - Latency-Safety Trade-off (Dual Axis)
   - Cross-Context Leakage Rate (Global)
   - Attack-Specific Leakage Analysis (Category Breakdown)
   - Scalability Trends
2. Tables (CSVs):
   - Experimental Configurations
   - Baseline Comparison Summary
   - Statistical Significance Results 
   - Firewall Activity Summary
   - Category Analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from scipy.stats import mannwhitneyu, fisher_exact

# --- PUBLICATION GRAPHICS STANDARDS ---
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

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CRITICAL UPDATE: Read the AUDIT results
LOG_FILE = os.path.join(BASE_DIR, "logs", "audit_results.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "paper_results")
if not os.path.exists(OUTPUT_DIR):
    try: os.makedirs(OUTPUT_DIR)
    except: pass

PALETTE = {
    "NAIVE_CONTROL": "#D55E00",      # Vermillion (High Risk)
    "STANDARD_POSTHOC": "#F0E442",   # Yellow (Caution)
    "DPF_PROPOSED": "#009E73"        # Bluish Green (Safe)
}

LABELS = {
    "NAIVE_CONTROL": "Naive Baseline",
    "STANDARD_POSTHOC": "Post-Hoc Filter",
    "DPF_PROPOSED": "DPF Architecture"
}

# --- STATISTICAL UTILITIES ---

def calculate_cliffs_delta(u_stat, n1, n2):
    try:
        return (2 * u_stat) / (n1 * n2) - 1
    except ZeroDivisionError:
        return 0.0

def calculate_odds_ratio(a, b, c, d):
    # Odds Ratio for Fisher's Test Effect Size
    try:
        return (a * d) / (b * c)
    except ZeroDivisionError:
        return 0.0

def interpret_p_value(p):
    if p < 0.001: return "< 0.001"
    if p < 0.01: return "< 0.01"
    if p < 0.05: return "< 0.05"
    return "n.s."

def get_effect_size_label(d, metric="Cliffs"):
    ad = abs(d)
    if metric == "Cliffs":
        if ad < 0.147: return "Negligible"
        if ad < 0.33: return "Small"
        if ad < 0.47: return "Medium"
        return "Large"
    elif metric == "OddsRatio":
        # Rule of thumb for OR
        if ad < 1.5: return "Negligible"
        if ad < 3.0: return "Small"
        if ad < 5.0: return "Medium"
        return "Large"
    return "Unknown"

# --- TABLE GENERATION ---

def generate_config_table(df):
    """
    Table: Experimental Configurations
    """
    n_counts = df["System_Mode"].value_counts()
    n_val = n_counts.get("DPF_PROPOSED", 0) 
    
    # We use PROMPT_COUNT (400) derived from data
    
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
    print(f" >> [Table] Generated Configuration Table: {outfile}")

def generate_baseline_table(df):
    """
    Table: Baseline Comparison Summary
    """
    modes = ["NAIVE_CONTROL", "STANDARD_POSTHOC", "DPF_PROPOSED"]
    threat_df = df[df["Prompt_Category"] != "General"]

    summary = []
    naive_latency = df[df["System_Mode"] == "NAIVE_CONTROL"]["Total_Latency_ms"].mean()

    for mode in modes:
        subset = threat_df[threat_df["System_Mode"] == mode]
        
        # 1. Leakage Metrics (Ground Truth)
        leaks = len(subset[subset["Hybrid_Verdict"] == "LEAK"])
        total = len(subset)
        leak_rate = (leaks / total * 100) if total > 0 else 0
        leak_str = f"{leaks}/{total}"

        # 2. Defensive Routing (Agent Swap)
        if "Data_Owner" in subset.columns and "Winner_Agent" in subset.columns:
            swaps = len(subset[subset["Data_Owner"] != subset["Winner_Agent"]])
            swap_rate = (swaps / total * 100) if total > 0 else 0
        else:
            swap_rate = 0.0

        # 3. Latency Metrics
        full_subset = df[df["System_Mode"] == mode]
        curr_latency = full_subset["Total_Latency_ms"].mean()
        curr_std = full_subset["Total_Latency_ms"].std()  # <-- NEW: Calculate Standard Deviation
        
        latency_delta = curr_latency - naive_latency
        delta_str = "—" if mode == "NAIVE_CONTROL" else f"+{latency_delta:.1f} ms"
        
        # 4. Routing Margin
        margin = full_subset["Routing_Margin"].mean()
        
        summary.append({
            "Metric": LABELS[mode],
            "Cross-Context Leakage Events": leak_str,
            "Leakage Rate (%)": f"{leak_rate:.1f}%",
            "Defensive Routing Rate (%)": f"{swap_rate:.1f}%",
            "Mean Latency (ms)": f"{curr_latency:.1f} ± {curr_std:.1f}", # <-- UPDATED: Add the ± format
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
    print(f" >> [Table] Generated Baseline Comparison Table: {outfile}")

def generate_category_table(df):
    """
    Table: Leakage by Attack Category
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
    print(f" >> [Table] Generated Category Analysis Table: {outfile}")

def generate_stats_table(df):
    """
    Table: Statistical Significance Results
    Uses Fisher's Exact Test for Leakage (Binary) 
    and Mann-Whitney U for Latency (Continuous).
    """
    results = []
    
    # Helper: Get counts for Fisher's Exact Test
    def get_fisher_counts(mode):
        subset = df[(df["System_Mode"] == mode) & (df["Prompt_Category"] != "General")]
        leaks = len(subset[subset["Hybrid_Verdict"] == "LEAK"])
        safe = len(subset) - leaks
        return leaks, safe

    # Helper: Get continuous vector for Mann-Whitney U
    def get_vec(mode, col):
        return df[df["System_Mode"] == mode][col].dropna()

    comparisons = [
        # 1. Leakage Rate: Naive vs Post-Hoc (Fisher's Exact)
        ("NAIVE_CONTROL", "STANDARD_POSTHOC", "Leakage Rate", "Outcomes", "fisher"),
        # 2. Leakage Rate: Post-Hoc vs DPF (Fisher's Exact)
        ("STANDARD_POSTHOC", "DPF_PROPOSED", "Leakage Rate", "Outcomes", "fisher"),
        # 3. Latency: Post-Hoc vs DPF (Mann-Whitney U)
        ("STANDARD_POSTHOC", "DPF_PROPOSED", "Latency", "Performance", "mwu"),
    ]

    for sys_a, sys_b, metric_name, category, test_type in comparisons:
        
        if test_type == "fisher":
            # Fisher's Exact Test for Binary Outcomes
            leaks_a, safe_a = get_fisher_counts(sys_a)
            leaks_b, safe_b = get_fisher_counts(sys_b)
            
            # Contingency Table: [[Leaks_A, Safe_A], [Leaks_B, Safe_B]]
            table = [[leaks_a, safe_a], [leaks_b, safe_b]]
            odds_ratio, p = fisher_exact(table, alternative='greater') # Testing if A > B (A leaks more)
            
            # Effect Size for Fisher is Odds Ratio
            effect_label = get_effect_size_label(odds_ratio, "OddsRatio")
            stat_val = f"OR={odds_ratio:.1f}"
            test_name = "Fisher's Exact"

        else: # test_type == "mwu"
            # Mann-Whitney U for Continuous Data
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
    print(f" >> [Table] Generated Stats Table: {outfile}")

def generate_firewall_table(df):
    """
    Table: Firewall Activity Summary (DPF Only)
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
    print(f" >> [Table] Generated Firewall Table: {outfile}")

# --- FIGURE GENERATION ---

def plot_latency_safety_tradeoff(df):
    """ 
    Figure: Latency vs Safety Tradeoff
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
    
    bars = ax1.bar(x, stats["Total_Latency_ms"], width, 
                   yerr=stats["Latency_Generation_ms"], capsize=5,
                   color=bar_colors, alpha=0.8, edgecolor='black', zorder=2)
    
    ax1.set_ylabel('Latency (ms)', fontweight='bold', color='#444444')
    ax1.set_xticks(x)
    ax1.set_xticklabels([LABELS[m].replace("\n", " ") for m in stats.index])
    ax1.set_ylim(bottom=0, top=stats["Total_Latency_ms"].max() * 1.3)
    ax1.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

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
    print(f" >> [Plot] Generated Latency-Safety Figure: {outfile}")

def plot_leakage_rate(df):
    """ 
    Figure: Cross-Context Leakage Rate (Global)
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
    print(f" >> [Plot] Generated Global Leakage Figure: {outfile}")

def plot_category_breakdown(df):
    """
    Figure: Leakage Rate by Attack Category
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
    print(f" >> [Plot] Generated Category Breakdown Figure: {outfile}")

def plot_scalability_trends(df):
    """ 
    Figure: Scalability Trends
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
    print(f" >> [Plot] Generated Scalability Figure: {outfile}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        try: os.makedirs(OUTPUT_DIR)
        except: pass
        
    if not os.path.exists(LOG_FILE):
        print(f" [ERROR] Audit Results file not found: {LOG_FILE}")
        print(f" [INFO] Please run 'src/evaluate_privacy_audit.py' first.")
        return

    try:
        df = pd.read_csv(LOG_FILE)
        
        # 1. Tables (Data Generation)
        generate_config_table(df)
        generate_baseline_table(df)
        generate_stats_table(df)
        generate_firewall_table(df)
        generate_category_table(df)
        
        # 2. Figures (High Res)
        plot_latency_safety_tradeoff(df)
        plot_leakage_rate(df)
        plot_category_breakdown(df)
        plot_scalability_trends(df)
        
    except Exception as e:
        print(f" [ERROR] Visualization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()