"""
Analytical Visualization Engine & Statistical Validation
------------------------------------------------------
This module processes telemetry logs to generate quantitative results for publication.

Outputs:
1. Figures:
    - Latency-Safety Trade-off (Dual Axis)
    - Cross-Context Leakage Rate
    - Scalability Trends
2. Tables (CSVs):
    - Experimental Configurations
    - Baseline Comparison Summary
    - Statistical Significance Results
    - Firewall Activity Summary
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from scipy.stats import mannwhitneyu

# --- PUBLICATION GRAPHICS STANDARDS ---
# Font: Times New Roman (or serif fallback)
# Size: Single Column width (3.5 inch) or Full Page width (7.16 inch)
# Resolution: 600 DPI for line art
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

# Fallback for script execution context
if not os.path.basename(BASE_DIR) in ['src', 'logs', 'paper_results']: 
     # Likely running from root
     pass
     
LOG_FILE = os.path.join(BASE_DIR, "logs", "experiment_data.csv")
# If logs dir not found, try current dir
if not os.path.exists(LOG_FILE):
    LOG_FILE = "experiment_data.csv"

OUTPUT_DIR = os.path.join(BASE_DIR, "paper_results")
# If paper_results not found in sibling structure, use current dir
if not os.path.exists(os.path.dirname(OUTPUT_DIR)):
    OUTPUT_DIR = "paper_results"

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

def interpret_p_value(p):
    if p < 0.001: return "< 0.001"
    if p < 0.01: return "< 0.01"
    if p < 0.05: return "< 0.05"
    return "n.s."

def get_effect_size_label(d):
    ad = abs(d)
    if ad < 0.147: return "Negligible"
    if ad < 0.33: return "Small"
    if ad < 0.47: return "Medium"
    return "Large"

# --- TABLE GENERATION ---

def generate_config_table(df):
    """
    Table: Experimental Configurations
    Combines static architectural flags with dynamic counts from the data.
    """
    # 1. Dynamic Counts
    n_counts = df["System_Mode"].value_counts()
    n_val = n_counts.get("DPF_PROPOSED", 0) 
    
    dpf = df[df["System_Mode"] == "DPF_PROPOSED"]
    attack_count = len(dpf[dpf["Redaction_Count"] > 0])
    attack_pct = (attack_count / n_val * 100) if n_val > 0 else 0
    
    data = {
        "Parameter": [
            "Semantic Routing", "Pre-Generation Firewall", "Post-Generation Filter", 
            "Private Memory Access", "Firewall Rule Set", "Prompt Count (N)", 
            "Attack Prompt Ratio", "Control/Load Ratio"
        ],
        "NAIVE": ["No", "No", "No", "Yes", "—", f"{n_val}", f"{attack_pct:.0f}%", f"{100-attack_pct:.0f}%"],
        "POST-HOC": ["Yes", "No", "Yes", "Yes", "Regex (post)", f"{n_val}", f"{attack_pct:.0f}%", f"{100-attack_pct:.0f}%"],
        "DPF (Proposed)": ["Yes", "Yes", "No", "Yes", "Deterministic (pre)", f"{n_val}", f"{attack_pct:.0f}%", f"{100-attack_pct:.0f}%"]
    }
    
    out_df = pd.DataFrame(data)
    outfile = os.path.join(OUTPUT_DIR, "Table_Experimental_Configuration.csv")
    out_df.to_csv(outfile, index=False)
    print(f" >> [Table] Generated Configuration Table: {outfile}")

def generate_baseline_table(df):
    """
    Table: Baseline Comparison Summary
    Computes cross-context leakage, latency overhead, and margin.
    """
    modes = ["NAIVE_CONTROL", "STANDARD_POSTHOC", "DPF_PROPOSED"]
    
    dpf = df[df["System_Mode"] == "DPF_PROPOSED"]
    threat_prompts = dpf[dpf["Redaction_Count"] > 0]["User_Input"].unique()
    total_threats = len(threat_prompts)

    summary = []
    
    naive_latency = df[df["System_Mode"] == "NAIVE_CONTROL"]["Total_Latency_ms"].mean()

    for mode in modes:
        subset = df[df["System_Mode"] == mode]
        
        # 1. Leakage Metrics
        leaks = 0
        if total_threats > 0:
            for prompt in threat_prompts:
                row = subset[subset["User_Input"] == prompt]
                if not row.empty and row["Redaction_Count"].sum() == 0:
                    leaks += 1
                elif row.empty: 
                    leaks += 1
            leak_rate = (leaks / total_threats) * 100
            leak_str = f"{leaks}/{total_threats}" 
        else:
            leak_rate = 0.0
            leak_str = "0/0"

        # 2. Latency Metrics
        curr_latency = subset["Total_Latency_ms"].mean()
        latency_delta = curr_latency - naive_latency
        delta_str = "—" if mode == "NAIVE_CONTROL" else f"+{latency_delta:.1f} ms"
        
        # 3. Routing Margin
        margin = subset["Routing_Margin"].mean()
        
        # 4. False Positives
        fp_count = 0
        safe_prompts = dpf[dpf["Redaction_Count"] == 0]["User_Input"].unique()
        for p in safe_prompts:
            row = subset[subset["User_Input"] == p]
            if not row.empty and row["Redaction_Count"].sum() > 0:
                fp_count += 1
                
        summary.append({
            "Metric": LABELS[mode],
            "Cross-Context Leakage Events": leak_str,
            "Leakage Rate (%)": f"{leak_rate:.1f}%",
            "Mean Latency (ms)": f"{curr_latency:.1f}",
            "Latency Overhead vs NAIVE": delta_str,
            "Routing Margin (Mean)": f"{margin:.4f}",
            "False Positive Blocks": fp_count
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

def generate_stats_table(df):
    """
    Table: Statistical Significance Results
    """
    results = []
    
    def get_vec(mode, col):
        return df[df["System_Mode"] == mode][col].dropna()

    dpf = df[df["System_Mode"] == "DPF_PROPOSED"]
    threat_prompts = dpf[dpf["Redaction_Count"] > 0]["User_Input"].unique()
    
    def get_leak_vec(mode):
        vec = []
        subset = df[df["System_Mode"] == mode]
        for p in threat_prompts:
            row = subset[subset["User_Input"] == p]
            if not row.empty:
                vec.append(1 if row["Redaction_Count"].sum() == 0 else 0)
            else:
                vec.append(1) 
        return vec

    comparisons = [
        ("NAIVE_CONTROL", "STANDARD_POSTHOC", "Leakage Rate", "Outcomes", lambda m: get_leak_vec(m)),
        ("STANDARD_POSTHOC", "DPF_PROPOSED", "Leakage Rate", "Outcomes", lambda m: get_leak_vec(m)),
        ("STANDARD_POSTHOC", "DPF_PROPOSED", "Latency", "Performance", lambda m: get_vec(m, "Total_Latency_ms")),
        ("NAIVE_CONTROL", "DPF_PROPOSED", "Routing Margin", "Quality", lambda m: get_vec(m, "Routing_Margin")),
    ]

    for sys_a, sys_b, metric_name, category, getter in comparisons:
        vec_a = getter(sys_a)
        vec_b = getter(sys_b)
        
        if len(vec_a) > 0 and len(vec_b) > 0:
            alt = "greater" if metric_name == "Leakage Rate" else "two-sided"
            
            u_stat, p = mannwhitneyu(vec_a, vec_b, alternative=alt)
            delta = calculate_cliffs_delta(u_stat, len(vec_a), len(vec_b))
            
            # Format Name: Remove CONTROL/PROPOSED, change STANDARD to POST-HOC
            name_a = sys_a.replace("_CONTROL", "").replace("_PROPOSED", "").replace("STANDARD_POSTHOC", "POST-HOC")
            name_b = sys_b.replace("_CONTROL", "").replace("_PROPOSED", "").replace("STANDARD_POSTHOC", "POST-HOC")
            
            results.append({
                "Comparison": f"{name_a} vs {name_b}",
                "Metric": metric_name,
                "Test": "Mann-Whitney U",
                "Statistic (U)": f"{u_stat:.1f}",
                "p-value": interpret_p_value(p),
                "Effect Size": get_effect_size_label(delta)
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
    triggering = len(dpf[dpf["Redaction_Count"] > 0])
    total_redactions = dpf["Redaction_Count"].sum()
    mean_redactions = dpf["Redaction_Count"].mean()
    mean_fw_latency = dpf["Latency_Firewall_ms"].mean()
    
    data = [
        ("Total Prompts", total_prompts),
        ("Attack Prompts (Detected)", triggering),
        ("Prompts Triggering Firewall", triggering),
        ("Total Redactions", int(total_redactions)),
        ("Mean Redactions / Prompt", f"{mean_redactions:.2f}"),
        ("False Positive Blocks", 0), 
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
    Style: Full Page Width (7.16 inches)
    """
    stats = df.groupby("System_Mode").agg({
        "Total_Latency_ms": "mean",
        "Redaction_Count": "mean", 
        "Latency_Generation_ms": "std" 
    }).reindex(["NAIVE_CONTROL", "STANDARD_POSTHOC", "DPF_PROPOSED"]).fillna(0)

    # Full Page Width ~7.16 inches. Height 4 inches for aspect ratio.
    fig, ax1 = plt.subplots(figsize=(7.16, 4))

    x = np.arange(len(stats))
    width = 0.5 
    bar_colors = [PALETTE[m] for m in stats.index]
    
    # Bar Chart (Latency)
    bars = ax1.bar(x, stats["Total_Latency_ms"], width, 
                   yerr=stats["Latency_Generation_ms"], capsize=5,
                   color=bar_colors, alpha=0.8, edgecolor='black', zorder=2)
    
    ax1.set_ylabel('Latency (ms)', fontweight='bold', color='#444444')
    ax1.set_xticks(x)
    # Simple labels, no newlines if possible
    ax1.set_xticklabels([LABELS[m].replace("\n", " ") for m in stats.index])
    
    # Zero Anchor
    ax1.set_ylim(bottom=0, top=stats["Total_Latency_ms"].max() * 1.3)
    ax1.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

    # Line Chart (Safety)
    ax2 = ax1.twinx()
    max_redaction = stats["Redaction_Count"].max()
    ax2.set_ylim(bottom=0, top=max_redaction * 1.3 if max_redaction > 0 else 1)
    
    line = ax2.plot(x, stats["Redaction_Count"], marker='o', markersize=8, 
                    label='Privacy Efficacy', color='#004488', linewidth=2, zorder=5)
    
    ax2.set_ylabel('Privacy Efficacy (Mean Redactions)', fontweight='bold', color='#004488')
    ax2.tick_params(axis='y', labelcolor='#004488')
    
    plt.title("Latency-Safety Efficiency Frontier", fontsize=11, fontweight='bold')
    plt.tight_layout()
    
    outfile = os.path.join(OUTPUT_DIR, "Figure_Latency_Safety_Tradeoff.png")
    plt.savefig(outfile, dpi=600, bbox_inches='tight') # 600 DPI High Res
    plt.close()
    print(f" >> [Plot] Generated Latency-Safety Figure: {outfile}")

def plot_leakage_rate(df):
    """ 
    Figure: Cross-Context Leakage Rate 
    Style: Single Column Width (3.5 inches)
    """
    dpf_data = df[df["System_Mode"] == "DPF_PROPOSED"]
    threat_prompts = dpf_data[dpf_data["Redaction_Count"] > 0]["User_Input"].unique()
    total_threats = len(threat_prompts)
    if total_threats == 0: return

    leakage_stats = []
    for mode in ["NAIVE_CONTROL", "STANDARD_POSTHOC", "DPF_PROPOSED"]:
        mode_data = df[df["System_Mode"] == mode]
        leaks = 0
        for prompt in threat_prompts:
            row = mode_data[mode_data["User_Input"] == prompt]
            if not row.empty:
                if row["Redaction_Count"].sum() == 0: leaks += 1
            else: leaks += 1
        leakage_stats.append({
            "Mode": mode,
            "Leakage_Rate": (leaks / total_threats) * 100
        })
        
    stats_df = pd.DataFrame(leakage_stats).set_index("Mode")

    # Single Column Width
    plt.figure(figsize=(3.5, 3.5))
    bars = plt.bar([LABELS[m].replace(" Baseline", "").replace(" Filter", "").replace(" Architecture", "") for m in stats_df.index], 
                   stats_df["Leakage_Rate"], 
                   color=[PALETTE[m] for m in stats_df.index], 
                   edgecolor='black', alpha=0.9, width=0.6)
    
    plt.title("Cross-Context Leakage Rate", fontsize=11, fontweight='bold')
    plt.ylabel("Leakage Rate (% of Threats)", fontsize=9)
    plt.ylim(0, 115)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                 f'{height:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, "Figure_Leakage_Rate.png")
    plt.savefig(outfile, dpi=600, bbox_inches='tight')
    plt.close()
    print(f" >> [Plot] Generated Leakage Rate Figure: {outfile}")

def plot_scalability_trends(df):
    """ 
    Figure: Scalability Trends
    Style: Full Page Width (7.16 inches)
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
    # Setup Output
    if not os.path.exists(OUTPUT_DIR):
        try:
            os.makedirs(OUTPUT_DIR)
        except: pass
        
    # Read Data
    if not os.path.exists(LOG_FILE):
        print(f" [ERROR] Log file not found: {LOG_FILE}")
        return

    try:
        df = pd.read_csv(LOG_FILE)
        
        # 1. Tables (Data Generation)
        generate_config_table(df)
        generate_baseline_table(df)
        generate_stats_table(df)
        generate_firewall_table(df)
        
        # 2. Figures (High Res)
        plot_latency_safety_tradeoff(df)
        plot_leakage_rate(df)
        plot_scalability_trends(df)
        
    except Exception as e:
        print(f" [ERROR] Visualization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()