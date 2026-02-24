"""
Automated Auditor Validation Metrics (Ground Truth Verification)
----------------------------------------------------------------
This module calculates statistical reliability metrics to validate the 
efficacy of the automated Hybrid Privacy Auditor (HPA) against human-in-the-loop 
(HITL) ground truth evaluations.

Analytical Objectives:
1. Inter-Rater Reliability (IRR): Calculates raw agreement and Cohen's Kappa (\u03ba) 
   to measure the consensus between the algorithmic auditor and human annotators, 
   accounting for agreement occurring by chance.
2. Error Profiling (Confusion Matrix): Deconstructs the evaluation into True/False 
   Positives and Negatives to identify if the system biases towards over-flagging (FP) 
   or under-reporting (FN) data leaks.

Outputs:
- paper_results/hpa_validation_metrics.csv (Serialized statistical report)
"""

import sys
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, cohen_kappa_score

# --- INFRASTRUCTURE CONFIGURATION ---
# Robust absolute path resolution for reliable I/O operations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Allow dynamic path injection from the CLI (Fast Path).
# If no argument is provided, it defaults to looking in the standard logs directory.
if len(sys.argv) > 1:
    INPUT_FILE = sys.argv[1]
else:
    INPUT_FILE = os.path.join(BASE_DIR, "logs", "human_audit_set.csv")

# Explicitly route the final validation artifact to the publication results folder
RESULTS_DIR = os.path.join(BASE_DIR, "paper_results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_METRICS = os.path.join(RESULTS_DIR, "hpa_validation_metrics.csv")

def run_validation():
    """
    Executes the statistical validation pipeline.
    Compares the automated system's classification ('Hybrid_Verdict') 
    against the gold-standard manual annotations ('Human_Label').
    """
    # Pre-flight check for the required ground-truth dataset
    if not os.path.exists(INPUT_FILE):
        print(f" [!] CRITICAL ERROR: Ground truth file '{INPUT_FILE}' not found.")
        print("     Ensure manual labels have been committed before running validation.")
        return

    print(f" >> [System] Initializing HPA Validation using: {INPUT_FILE}")

    # --- 1. DATA INGESTION & PREPROCESSING ---
    df = pd.read_csv(INPUT_FILE)
    
    # Normalize data vectors to prevent case-sensitivity or whitespace mismatch errors
    y_hpa = df['Hybrid_Verdict'].astype(str).str.upper().str.strip().tolist()
    y_human = df['Human_Label'].astype(str).str.upper().str.strip().tolist()

    # --- 2. RELIABILITY METRICS (IRR) ---
    # Raw Agreement: The strict percentage of identical classifications.
    matches = sum(1 for hpa, hum in zip(y_hpa, y_human) if hpa == hum)
    agreement = (matches / len(df)) * 100
    
    # Cohen's Kappa: A robust metric factoring in the probability of random chance agreement.
    kappa = cohen_kappa_score(y_human, y_hpa)

    # --- 3. CONFUSION MATRIX CALCULATION ---
    # Expected Order: Top-Left=TN, Top-Right=FP, Bottom-Left=FN, Bottom-Right=TP
    # 'SAFE' = Negative (No Leak) | 'LEAK' = Positive (Privacy Violation)
    cm = confusion_matrix(y_human, y_hpa, labels=['SAFE', 'LEAK'])
    tn, fp, fn, tp = cm.ravel()

    # --- 4. QUALITATIVE STRENGTH INTERPRETATION ---
    # Applies standard interpretation thresholds (Landis & Koch)
    if kappa > 0.80: 
        strength = "Excellent"
    elif kappa > 0.60: 
        strength = "Substantial"
    elif kappa > 0.40:
        strength = "Moderate"
    else: 
        strength = "Fair/Poor"

    # --- 5. REPORT GENERATION & EXPORT ---
    metrics_data = {
        "Metric": [
            "Sample Size (N)", 
            "Agreement (%)", 
            "Cohen's Kappa (\u03ba)", 
            "Strength", 
            "True Negatives (TN) [Correctly Safe]", 
            "False Positives (FP) [False Alarm]", 
            "False Negatives (FN) [Missed Leak]", 
            "True Positives (TP) [Correctly Caught]"
        ],
        "Value": [
            len(df), 
            f"{agreement:.2f}", 
            f"{kappa:.4f}", 
            strength, 
            tn, fp, fn, tp
        ]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    # Persist the final statistical table to the paper_results directory
    metrics_df.to_csv(OUTPUT_METRICS, index=False)

    # --- 6. TERMINAL OBSERVABILITY REPORT ---
    print("\n" + "="*50)
    print("      AUTOMATED AUDITOR VALIDATION REPORT")
    print("="*50)
    print(f" Inter-Rater Agreement:    {agreement:.2f}%")
    print(f" Cohen’s Kappa (\u03ba):        {kappa:.3f} ({strength})")
    print("-" * 50)
    print(f" TN: {tn:<4} | FP: {fp:<4} (Human labeled SAFE)")
    print(f" FN: {fn:<4} | TP: {tp:<4} (Human labeled LEAK)")
    print("-" * 50)
    print(f" >> Telemetry exported to: {OUTPUT_METRICS}")
    print("="*50)

if __name__ == "__main__":
    run_validation()