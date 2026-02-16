import pandas as pd

# Load your existing audit results
df = pd.read_csv("logs/audit_results.csv")

# Filter for the 27 DPF leaks
dpf_leaks = df[(df["System_Mode"] == "DPF_PROPOSED") & (df["Hybrid_Verdict"] == "LEAK")]

print(f"Total DPF Leaks: {len(dpf_leaks)}")

# Count Router Errors (Router sent query to the wrong agent)
router_errors = len(dpf_leaks[dpf_leaks["Winner_Agent"] != dpf_leaks["Data_Owner"]])

# Count Regex Misses (Router was correct, but Regex failed to catch the leak)
regex_misses = len(dpf_leaks) - router_errors

router_pct = (router_errors / len(dpf_leaks)) * 100
regex_pct = (regex_misses / len(dpf_leaks)) * 100

print(f"Router Misclassification: {router_errors} instances ({router_pct:.1f}%)")
print(f"Semantic Obfuscation (Regex Miss): {regex_misses} instances ({regex_pct:.1f}%)")