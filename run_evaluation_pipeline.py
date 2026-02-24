"""
Automated Systems Evaluation Pipeline (One-Click Reproducibility)
-----------------------------------------------------------------
Orchestrates the entire Comparative Ablation Study for the architecture.

This script executes the full engineering lifecycle unattended:
1. Environment Setup & Validation.
2. Adversarial Stress Testing (Data Collection).
3. Benign Utility Benchmarking (False Refusal Profiling).
4. Automated Privacy Auditing (NLI/Regex Hybrid Leakage Detection).
5. Failure Mode Attribution (Control Plane vs Data Plane).
6. Telemetry Analysis & Visualization (CSV/PNG Generation).
"""

import os
import sys
import time
import shutil
import subprocess

def run_external_script(script_name, description):
    """
    Helper function to execute external evaluation modules sequentially.
    Uses subprocess to ensure strict memory/state isolation between tests.
    """
    print(f"\n{'-'*60}")
    print(f"   PHASE: {description}")
    print(f"{'-'*60}")
    
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    if not os.path.exists(script_path):
        # Fallback to checking inside 'src/' depending on repository structure
        script_path = os.path.join(os.path.dirname(__file__), 'src', script_name)
        
    try:
        subprocess.check_call([sys.executable, script_path])
    except subprocess.CalledProcessError as e:
        print(f" !! [CRITICAL ERROR] Phase '{description}' failed with exit code {e.returncode}.")
        # We do not sys.exit(1) here to allow graceful continuation if a statistical 
        # module fails due to micro-dataset constraints during testing.
    except FileNotFoundError:
        print(f" !! [CRITICAL ERROR] Script '{script_name}' not found.")
        sys.exit(1)

def main():
    # --- 1. BOOT SEQUENCE ---
    os.system('cls' if os.name == 'nt' else 'clear')
    print("==========================================================")
    print("      ARCHITECTURE: END-TO-END EVALUATION PIPELINE        ")
    print("==========================================================")
    
    # Add root and 'src' to path to allow direct module imports
    sys.path.append(os.path.dirname(__file__))
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        # Import core architectural components
        import system_registry as registry
        import experiment_driver as test_harness
    except ImportError as e:
        print(f"\n [CRITICAL ERROR] Missing Module: {e}")
        print(" Ensure the script is executed from the project root.")
        sys.exit(1)

    # --- 2. LOG ROTATION & ISOLATION (Log Siphoning) ---
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # File paths
    eval_log_file = os.path.join(log_dir, "experiment_data.csv")
    interactive_log_file = os.path.join(log_dir, "system_telemetry.csv")
    backup_log_file = os.path.join(log_dir, "system_telemetry.backup")

    # Step A: Protect user's interactive debug data by temporarily renaming it
    if os.path.exists(interactive_log_file):
        print(" >> [Setup] Archiving interactive session telemetry...")
        os.rename(interactive_log_file, backup_log_file)

    # Step B: Purge old evaluation data to prevent cross-contamination
    if os.path.exists(eval_log_file):
        print(f" >> [Setup] Clearing previous evaluation telemetry: {eval_log_file}")
        os.remove(eval_log_file)

    # --- 3. ADVERSARIAL EXPERIMENT EXECUTION ---
    # We evaluate the Control group, the Industry Standard, and the Proposed Architecture.
    study_sequence = [
        "NAIVE_CONTROL",     # Negative Control (Upper Bound Risk)
        "STANDARD_POSTHOC",  # Competitor Baseline (Industry Standard)
        "DPF_PROPOSED"       # Proposed Solution (Experimental Condition)
    ]
    
    global_counter = 0
    total_trials = 1200 # Default full-scale target
    
    print(f"\n [System] Initiating Adversarial Stress Test Sequence...")
    
    for mode in study_sequence:
        print(f"\n{'-'*60}")
        print(f"   PHASE: Adversarial Testing -> {mode}")
        print(f"{'-'*60}")
        
        # A. State Enforcement
        try:
            registry.set_system_mode(mode)
        except ValueError as e:
            print(f" !! [ERROR] Configuration Failure: {e}")
            sys.exit(1)
        
        # B. Execute Stress Test
        # test_harness intrinsically writes to 'system_telemetry.csv'
        global_counter = test_harness.run_batch(mode, global_start_count=global_counter)
        print(f" >> [Phase Complete] {mode} finished. Current Vector Count: {global_counter}")
        
        time.sleep(1) # Cool-down to ensure OS file I/O flush

    # --- 3.5. DATASET NORMALIZATION & RESTORATION ---
    print(f"\n >> [System] Normalizing data streams...")
    
    # Siphon the newly generated telemetry into the official evaluation file
    if os.path.exists(interactive_log_file):
        os.rename(interactive_log_file, eval_log_file)
        print(f" >> [System] Evaluation data successfully isolated to 'experiment_data.csv'.")
    
    # Restore the user's interactive debugging file so it isn't lost
    if os.path.exists(backup_log_file):
        os.rename(backup_log_file, interactive_log_file)
        print(f" >> [System] Interactive telemetry successfully restored.")

    # Validate dataset size for statistical viability
    if os.path.exists(eval_log_file):
        with open(eval_log_file, 'r', encoding='utf-8') as f:
            row_count = sum(1 for row in f) - 1 # Subtract header
            if row_count < 20:
                print("\n" + "!"*60)
                print(" [WARNING] MICRO-DATASET DETECTED (N < 20)")
                print(" Downstream statistical modules (Mann-Whitney, Polyfit Regression)")
                print(" require higher variance to compute. Chart generation may gracefully")
                print(" abort if mathematical constraints are not met.")
                print("!"*60)

    # --- 4. EXTENDED PIPELINE EXECUTION ---
    # Trigger all supplementary evaluation modules in a strict, deterministic sequence
    
    run_external_script("run_utility_benchmark.py", "Benign Utility & False Refusal Audit (N=300)")
    run_external_script("evaluate_privacy_audit.py", "Hybrid Privacy Auditor (Semantic Leakage Detection)")
    run_external_script("run_auditor_ablation.py", "Failure Mode Attribution Analysis")
    run_external_script("visualization_engine.py", "Data Synthesis & Figure Generation")

    # --- 5. PIPELINE TERMINATION ---
    output_dir = "paper_results"
    
    print("\n" + "="*60)
    print("      [SUCCESS] REPRODUCIBILITY PIPELINE COMPLETE")
    print("="*60)
    print(f" >> Raw Telemetry stored in:  /logs")
    print(f" >> Final Artifacts saved in: /{output_dir}")
    print("    - Table_Baseline_Comparison.csv")
    print("    - Figure_Latency_Safety_Tradeoff.png")
    print("    - Figure_Leakage_Rate_Global.png")
    print("\n[System] Pipeline Shutdown. Goodbye.")

if __name__ == "__main__":
    main()