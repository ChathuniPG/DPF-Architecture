import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_FILE = os.path.join(BASE_DIR, "logs", "audit_results.csv")
HUMAN_SET_FILE = os.path.join(BASE_DIR, "logs", "human_audit_set.csv")

def create_human_set():
    df = pd.read_csv(AUDIT_FILE)
    # Take 20 random samples per system mode
    human_set = df.groupby('System_Mode').apply(lambda x: x.sample(20)).reset_index(drop=True)
    
    # Keep only the columns you need to judge
    cols_to_keep = ['System_Mode', 'User_Input', 'Final_Response', 'Hybrid_Verdict', 'Hybrid_Reason']
    human_set = human_set[cols_to_keep]
    
    # Add an empty column for your judgment
    human_set['Human_Label'] = "" 
    
    human_set.to_csv(HUMAN_SET_FILE, index=False)
    print(f" >> Human Audit set created: {HUMAN_SET_FILE}")

if __name__ == "__main__":
    create_human_set()