"""
Human-in-the-Loop (HITL) Annotation Sampler
-------------------------------------------
This module generates a statistically distributed subset of system telemetry 
for manual review. It facilitates the creation of a 'Ground Truth' dataset 
required to validate automated evaluation metrics (e.g., Inter-Rater Reliability).

Operational Workflow:
1. Ingests the raw, automated audit logs.
2. Performs stratified random sampling to ensure balanced representation 
   across all tested architectural configurations.
3. Applies data minimization to reduce cognitive load on human annotators.
4. Outputs a pre-formatted CSV template ready for manual labeling.
"""

import pandas as pd
import os

# --- INFRASTRUCTURE CONFIGURATION ---
# Robust absolute path resolution ensures the script can be executed 
# from any working directory without breaking I/O operations.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_FILE = os.path.join(BASE_DIR, "logs", "audit_results.csv")
HUMAN_SET_FILE = os.path.join(BASE_DIR, "logs", "human_audit_set.csv")

def create_human_set():
    """
    Executes the stratified sampling and formatting pipeline.
    Extracts a balanced cohort of interaction logs and prepares the 
    annotation schema for human reviewers.
    """
    # Pre-flight check to ensure automated logs exist
    if not os.path.exists(AUDIT_FILE):
        print(f" [!] CRITICAL ERROR: Source telemetry not found at {AUDIT_FILE}")
        return

    # 1. Data Ingestion
    df = pd.read_csv(AUDIT_FILE)
    
    # 2. Stratified Random Sampling
    # Groups the data by the architectural mode ('System_Mode') and extracts 
    # exactly 20 random samples per category. This guarantees a balanced 
    # dataset, preventing class imbalance during statistical validation.
    human_set = df.groupby('System_Mode').apply(lambda x: x.sample(20)).reset_index(drop=True)
    
    # 3. Data Minimization (Feature Selection)
    # Strips out backend telemetry (e.g., latency, scores, firewall layers) 
    # to present only the necessary context to the human annotator, reducing 
    # bias and cognitive fatigue.
    cols_to_keep = [
        'System_Mode',     # Independent Variable
        'User_Input',      # Trigger Condition
        'Final_Response',  # LLM Artifact
        'Hybrid_Verdict',  # Automated System's Classification
        'Hybrid_Reason'    # Automated System's Rationale
    ]
    human_set = human_set[cols_to_keep]
    
    # 4. Target Variable Injection
    # Appends a blank column explicitly for the human reviewer to input 
    # their manual classification (e.g., 'SAFE' or 'LEAK').
    human_set['Human_Label'] = "" 
    
    # 5. Serialization
    human_set.to_csv(HUMAN_SET_FILE, index=False)
    print(f" >> [Success] HITL Annotation cohort generated: {HUMAN_SET_FILE}")
    print(f" >> Stratified Sample Size: {len(human_set)} total rows.")

if __name__ == "__main__":
    # Execute the sampling pipeline
    create_human_set()