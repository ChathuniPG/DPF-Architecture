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
    Returns True if successful, False if the subprocess crashes.
    """
    print(f"\n{'-'*60}")
    print(f"   PHASE: {description}")
    print(f"{'-'*60}")
    
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    if not os.path.exists(script_path):
        script_path = os.path.join(os.path.dirname(__file__), 'src', script_name)
        
    try:
        subprocess.check_call([sys.executable, script_path])
        return True
    except subprocess.CalledProcessError as e:
        print(f" !! [CRITICAL ERROR] Phase '{description}' failed with exit code {e.returncode}.")
        # Return False to flag that the pipeline is degraded, but allow continuation
        return False
    except FileNotFoundError:
        print(f" !! [CRITICAL ERROR] Script '{script_name}' not found.")
        return False

def main():
    # --- 1. BOOT SEQUENCE ---
    os.system('cls' if os.name == 'nt' else 'clear')
    print("==========================================================")
    print("      ARCHITECTURE: END-TO-END EVALUATION PIPELINE        ")
    print("==========================================================")
    
    sys.path.append(os.path.dirname(__file__))
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
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

    eval_log_file = os.path.join(log_dir, "experiment_data.csv")
    interactive_log_file = os.path.join(log_dir, "system_telemetry.csv")
    backup_log_file = os.path.join(log_dir, "system_telemetry.backup")

    if os.path.exists(interactive_log_file):
        print(" >> [Setup] Archiving interactive session telemetry...")
        os.rename(interactive_log_file, backup_log_file)

    if os.path.exists(eval_log_file):
        print(f" >> [Setup] Clearing previous evaluation telemetry: {eval_log_file}")
        os.remove(eval_log_file)

    # --- 3. ADVERSARIAL EXPERIMENT EXECUTION ---
    study_sequence = ["NAIVE_CONTROL", "STANDARD_POSTHOC", "DPF_PROPOSED"]
    global_counter = 0
    pipeline_errors = False  # Track overall health
    
    print(f"\n [System] Initiating Adversarial Stress Test Sequence...")
    
    for mode in study_sequence:
        print(f"\n{'-'*60}")
        print(f"   PHASE: Adversarial Testing -> {mode}")
        print(f"{'-'*60}")
        
        try:
            registry.set_system_mode(mode)
        except ValueError as e:
            print(f" !! [ERROR] Configuration Failure: {e}")
            sys.exit(1)
        
        try:
            global_counter = test_harness.run_batch(mode, global_start_count=global_counter)
            print(f" >> [Phase Complete] {mode} finished. Current Vector Count: {global_counter}")
        except Exception as e:
            print(f" !! [ERROR] Stress test failed for {mode}: {e}")
            pipeline_errors = True
            
        time.sleep(1)

    # --- 3.5. DATASET NORMALIZATION & RESTORATION ---
    print(f"\n >> [System] Normalizing data streams...")
    
    if os.path.exists(interactive_log_file):
        os.rename(interactive_log_file, eval_log_file)
        
    if os.path.exists(backup_log_file):
        os.rename(backup_log_file, interactive_log_file)

    # Validate dataset size
    if os.path.exists(eval_log_file):
        with open(eval_log_file, 'r', encoding='utf-8') as f:
            row_count = sum(1 for row in f) - 1
            if row_count < 20:
                print("\n" + "!"*60)
                print(" [WARNING] MICRO-DATASET DETECTED (N < 20)")
                print(" Downstream statistical modules require higher variance to compute.")
                print("!"*60)

    # --- 4. EXTENDED PIPELINE EXECUTION ---
    # Track the success of each external module
    if not run_external_script("run_utility_benchmark.py", "Benign Utility & False Refusal Audit"):
        pipeline_errors = True
        
    if not run_external_script("evaluate_privacy_audit.py", "Hybrid Privacy Auditor"):
        pipeline_errors = True
        
    if not run_external_script("run_auditor_ablation.py", "Failure Mode Attribution"):
        pipeline_errors = True
        
    if not run_external_script("visualization_engine.py", "Data Synthesis & Figure Generation"):
        pipeline_errors = True

    # --- 5. PIPELINE TERMINATION ---
    output_dir = "paper_results"
    
    print("\n" + "="*60)
    # Dynamically adjust the final readout based on system health
    if pipeline_errors:
        print("      [FAILED] PIPELINE COMPLETED WITH CRITICAL ERRORS")
        print("="*60)
        print(" >> Please review the terminal output above for missing libraries or missing data files.")
    else:
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