"""
Human-in-the-Loop (HITL) Annotation Sampler — N=400
---------------------------------------------------
Performs stratified random sampling to extract exactly 400 rows 
(100 per architectural mode) for manual ground-truth labeling.
"""

import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_FILE = os.path.join(BASE_DIR, "logs", "audit_results.csv")
HUMAN_SET_FILE = os.path.join(BASE_DIR, "logs", "human_audit_set.csv")

def create_human_set():
    if not os.path.exists(AUDIT_FILE):
        print(f" [!] CRITICAL ERROR: Source telemetry not found at {AUDIT_FILE}")
        return

    df = pd.read_csv(AUDIT_FILE)
    
    # Stratified Random Sampling: 100 per mode = 400 total.
    # random_state=42 ensures the same 400 rows are picked if you run it twice.
    human_set = df.groupby('System_Mode', group_keys=False).apply(
        lambda x: x.sample(n=100, random_state=42)
    ).reset_index(drop=True)
    
    # Added Prompt_Category so annotators understand the context of the attack
    cols_to_keep = [
        'System_Mode',     
        'Prompt_Category', 
        'User_Input',      
        'Final_Response',  
        'Hybrid_Verdict'   
    ]
    
    # Safe extraction (in case some columns are missing)
    human_set = human_set[[c for c in cols_to_keep if c in human_set.columns]]
    
    # Target Variable Injection
    human_set['Human_Label'] = "" 
    
    human_set.to_csv(HUMAN_SET_FILE, index=False)
    print(f" >> [Success] HITL Annotation cohort generated: {HUMAN_SET_FILE}")
    print(f" >> Stratified Sample Size: {len(human_set)} total rows (100 per mode).")

if __name__ == "__main__":
    create_human_set()