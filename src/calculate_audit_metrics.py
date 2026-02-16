"""
HPA VALIDATION METRICS
----------------------
Calculates Inter-Rater Agreement and Confusion Matrix to validate 
the Hybrid Privacy Auditor against human judgment.

OUTPUT: logs/hpa_validation_metrics.csv
"""

import pandas as pd
import os
from sklearn.metrics import confusion_matrix, cohen_kappa_score

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "logs", "human_audit_set.csv")
OUTPUT_METRICS = os.path.join(BASE_DIR, "logs", "hpa_validation_metrics.csv")

def run_validation():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Please ensure you saved your labels.")
        return

    # Load data
    df = pd.read_csv(INPUT_FILE)
    
    # Ensure all strings are uppercase for matching and strip whitespace
    y_hpa = df['Hybrid_Verdict'].str.upper().str.strip().tolist()
    y_human = df['Human_Label'].str.upper().str.strip().tolist()

    # 1. Calculate Agreement & Kappa
    matches = sum(1 for hpa, hum in zip(y_hpa, y_human) if hpa == hum)
    agreement = (matches / len(df)) * 100
    kappa = cohen_kappa_score(y_human, y_hpa)

    # 2. Generate Confusion Matrix
    # Order: Top-Left=TN, Bottom-Right=TP
    cm = confusion_matrix(y_human, y_hpa, labels=['SAFE', 'LEAK'])
    tn, fp, fn, tp = cm.ravel()

    # 3. Determine Strength
    if kappa > 0.80: strength = "Excellent"
    elif kappa > 0.60: strength = "Substantial"
    else: strength = "Moderate"

    # --- 4. EXPORT TO CSV ---
    metrics_data = {
        "Metric": ["Sample Size (N)", "Agreement (%)", "Cohen's Kappa (κ)", "Strength", 
                   "True Negatives (TN)", "False Positives (FP)", 
                   "False Negatives (FN)", "True Positives (TP)"],
        "Value": [len(df), f"{agreement:.2f}", f"{kappa:.4f}", strength, tn, fp, fn, tp]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(OUTPUT_METRICS, index=False)

    # --- 5. PRINT SUMMARY ---
    print("\n" + "="*45)
    print("       HPA SCIENTIFIC VALIDATION REPORT")
    print("="*45)
    print(f"Inter-Rater Agreement:    {agreement:.2f}%")
    print(f"Cohen’s Kappa (κ):        {kappa:.3f} ({strength})")
    print("-" * 45)
    print(f"TN: {tn} | FP: {fp} (Human SAFE)")
    print(f"FN: {fn} | TP: {tp} (Human LEAK)")
    print("-" * 45)
    print(f" >> Metrics exported to: {OUTPUT_METRICS}")
    print("="*45)

if __name__ == "__main__":
    run_validation()