"""
Automated Auditor Validation Metrics (Dual-Annotator Ground Truth)
------------------------------------------------------------------
Calculates statistical reliability metrics to validate the automated 
Hybrid Privacy Auditor (HPA) against TWO human annotators.

Calculates:
1. HPA vs Human 1
2. HPA vs Human 2
3. Human 1 vs Human 2 (Inter-Annotator Agreement)
"""

import sys
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import warnings

# Suppress undefined metric warnings if a mode has 0 variance in a sample
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- INFRASTRUCTURE CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if len(sys.argv) > 1:
    INPUT_FILE = sys.argv[1]
else:
    INPUT_FILE = os.path.join(BASE_DIR, "logs", "human_audit_set.csv")

RESULTS_DIR = os.path.join(BASE_DIR, "paper_results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_METRICS = os.path.join(RESULTS_DIR, "hpa_validation_metrics.csv")

def calculate_metrics_for_subset(y_true, y_pred, subset_name, comparison_name):
    """Helper function to calculate metrics for a specific dataset slice."""
    if len(y_true) == 0:
        return None
        
    matches = sum(1 for p, t in zip(y_pred, y_true) if p == t)
    agreement = (matches / len(y_true)) * 100
    
    try:
        kappa = cohen_kappa_score(y_true, y_pred)
        if pd.isna(kappa): kappa = 0.0
    except:
        kappa = 0.0

    # For Human 1 vs Human 2, TN/FP/FN/TP represents how H2 differs from H1 (H1 treated as ground truth)
    cm = confusion_matrix(y_true, y_pred, labels=['SAFE', 'LEAK'])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0 

    if kappa > 0.80: strength = "Excellent"
    elif kappa > 0.60: strength = "Substantial"
    elif kappa > 0.40: strength = "Moderate"
    elif kappa > 0.20: strength = "Fair"
    else: strength = "Poor"

    return {
        "Comparison": comparison_name,
        "Subset": subset_name,
        "N": len(y_true),
        "Agreement_%": round(agreement, 2),
        "Kappa": round(kappa, 3),
        "Strength": strength,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }

def run_validation():
    if not os.path.exists(INPUT_FILE):
        print(f" [!] CRITICAL ERROR: Ground truth file '{INPUT_FILE}' not found.")
        return

    print(f" >> [System] Initializing HPA Validation using: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    
    # Verify columns exist
    required_cols = ['Hybrid_Verdict', 'Human-Label_1', 'Human-Label_2']
    for col in required_cols:
        if col not in df.columns:
            print(f" [!] ERROR: Column '{col}' not found in {INPUT_FILE}.")
            print("     Ensure the CSV has columns: 'Human-Label_1' and 'Human-Label_2'.")
            return

    # Drop rows where either human hasn't annotated yet
    df = df.dropna(subset=['Human-Label_1', 'Human-Label_2'])
    
    if len(df) == 0:
        print(" [!] ERROR: No fully annotated rows found (both humans must complete).")
        return

    # Normalize data
    for col in required_cols:
        df[col] = df[col].astype(str).str.upper().str.strip()

    all_metrics = []
    
    # Define the 3 comparisons to run
    comparisons = [
        ("HPA_vs_Human_1", df['Human-Label_1'], df['Hybrid_Verdict']),
        ("HPA_vs_Human_2", df['Human-Label_2'], df['Hybrid_Verdict']),
        ("Human_1_vs_Human_2", df['Human-Label_1'], df['Human-Label_2']) # Human 1 acts as baseline here
    ]

    # 1. OVERALL METRICS
    for comp_name, y_true, y_pred in comparisons:
        overall = calculate_metrics_for_subset(
            y_true.tolist(), 
            y_pred.tolist(), 
            "OVERALL_SYSTEM",
            comp_name
        )
        all_metrics.append(overall)

    # 2. PER-MODE METRICS
    modes = df['System_Mode'].unique()
    for mode in sorted(modes):
        mode_df = df[df['System_Mode'] == mode]
        
        mode_comparisons = [
            ("HPA_vs_Human_1", mode_df['Human-Label_1'], mode_df['Hybrid_Verdict']),
            ("HPA_vs_Human_2", mode_df['Human-Label_2'], mode_df['Hybrid_Verdict']),
            ("Human_1_vs_Human_2", mode_df['Human-Label_1'], mode_df['Human-Label_2'])
        ]
        
        for comp_name, y_true, y_pred in mode_comparisons:
            mode_metrics = calculate_metrics_for_subset(
                y_true.tolist(), 
                y_pred.tolist(), 
                mode,
                comp_name
            )
            if mode_metrics:
                all_metrics.append(mode_metrics)

    # Export to CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUTPUT_METRICS, index=False)

    # Terminal Report (Focused on Overalls to avoid terminal flooding)
    print("\n" + "="*65)
    print("      AUTOMATED AUDITOR VALIDATION REPORT (DUAL-RATER)")
    print("="*65)
    
    overall_metrics = [m for m in all_metrics if m['Subset'] == "OVERALL_SYSTEM"]
    for m in overall_metrics:
        print(f" [{m['Comparison']}]")
        print(f"    Agreement: {m['Agreement_%']}% | Kappa: {m['Kappa']} ({m['Strength']})")
        if "Human_1_vs_Human_2" not in m['Comparison']:
            print(f"    TN: {m['TN']:<4} | FP: {m['FP']:<4} (HPA False Alarm)")
            print(f"    FN: {m['FN']:<4} | TP: {m['TP']:<4} (HPA Missed Leak)")
        else:
            print(f"    (High Kappa here proves the labeling task is objective)")
        print("-" * 65)

    print(f" >> Full mode-by-mode telemetry exported to: {OUTPUT_METRICS}")
    print("="*65)

if __name__ == "__main__":
    run_validation()