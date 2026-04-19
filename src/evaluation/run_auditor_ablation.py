"""
Failure Mode Ablation Analyzer (Vulnerability Attribution)
----------------------------------------------------------
This module conducts a forensic analysis on the specific leakage events 
detected within the proposed secured architecture. 

Analytical Objective:
To decompose the residual system vulnerabilities and attribute each failure 
to a specific architectural layer:
1. Control Plane Failures (Routing Misclassification): The semantic router 
   failed to identify the correct domain authority, bypassing the intended firewall.
2. Data Plane Failures (Semantic Obfuscation): The router acted correctly, but 
   the adversarial payload successfully evaded the deterministic regex boundaries.

Outputs:
- paper_results/failure_mode_ablation.csv (Structured tabular report)
"""

import sys
import os
import pandas as pd

# --- INFRASTRUCTURE CONFIGURATION ---
# Robust path resolution to ensure portability
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Allow dynamic path injection from the CLI, default to standard runtime logs/
if len(sys.argv) > 1:
    INPUT_FILE = sys.argv[1]
else:
    INPUT_FILE = os.path.join(BASE_DIR, "logs", "audit_results.csv")

# Define the output directory for analytical reports
OUTPUT_DIR = os.path.join(BASE_DIR, "paper_results")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "failure_mode_ablation.csv")

# The specific architectural mode under scrutiny
TARGET_ARCHITECTURE = "DPF_PROPOSED"

def run_failure_mode_analysis():
    """
    Executes the vulnerability attribution pipeline.
    Parses the audit logs, categorizes the leakage events, and serializes 
    the statistical breakdown to a CSV artifact.
    """
    # Pre-flight check for the required telemetry logs
    if not os.path.exists(INPUT_FILE):
        print(f" [!] CRITICAL ERROR: Audit telemetry not found at {INPUT_FILE}")
        return

    print(f" >> [System] Initializing Failure Mode Analysis using: {INPUT_FILE}")

    # 1. Data Ingestion
    df = pd.read_csv(INPUT_FILE)

    # 2. Vulnerability Isolation
    # Filter the dataset to isolate ONLY the successful data leaks that occurred 
    # while the proposed security architecture was active.
    target_leaks = df[(df["System_Mode"] == TARGET_ARCHITECTURE) & (df["Hybrid_Verdict"] == "LEAK")]
    total_leaks = len(target_leaks)

    print(f"\n=================================================")
    print(f"      FAILURE MODE ANALYSIS: {TARGET_ARCHITECTURE}")
    print(f"=================================================")
    print(f" Total Residual Vulnerabilities Detected: {total_leaks}")
    print(f"-------------------------------------------------")

    if total_leaks == 0:
        print(" >> No leaks detected. System exhibits 100% containment.")
        return

    # --- 3. ATTRIBUTION LOGIC ---
    
    # Category A: Control Plane Error (Routing Misclassification)
    # The system failed because the semantic router dispatched the query to an 
    # agent that did not own the targeted data, effectively circumventing the 
    # intended domain-specific firewall rules.
    router_errors = len(target_leaks[target_leaks["Winner_Agent"] != target_leaks["Data_Owner"]])

    # Category B: Data Plane Error (Semantic Obfuscation / Regex Miss)
    # The system correctly routed the query to the authoritative agent, but the 
    # generative model rephrased the sensitive data in a way that evaded the 
    # deterministic firewall's bounded constraints.
    regex_misses = total_leaks - router_errors

    # --- 4. STATISTICAL CALCULATION ---
    router_pct = (router_errors / total_leaks) * 100
    regex_pct = (regex_misses / total_leaks) * 100

    print(f" Control Plane (Routing Misclassification): {router_errors:02d} instances ({router_pct:.1f}%)")
    print(f" Data Plane (Semantic Obfuscation):       {regex_misses:02d} instances ({regex_pct:.1f}%)")
    print("=================================================\n")

    # --- 5. ARTIFACT GENERATION & EXPORT ---
    # Ensure the target directory exists before attempting I/O operations
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Construct a structured schema for the final analytical table
    results_data = {
        "Architectural_Layer": [
            "Control Plane (Routing Misclassification)", 
            "Data Plane (Semantic Obfuscation)"
        ],
        "Vulnerability_Count": [router_errors, regex_misses],
        "Proportion_Percentage": [f"{router_pct:.2f}%", f"{regex_pct:.2f}%"]
    }

    # Serialize to disk
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f" >> [Success] Failure Mode Ablation table exported to: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_failure_mode_analysis()