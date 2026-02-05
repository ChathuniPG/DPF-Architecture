"""
run_evaluation_pipeline.py
--------------------------
Automated Systems Evaluation Pipeline.
Orchestrates the Comparative Ablation Study for the Deterministic Privacy Firewall.

This script executes the full engineering lifecycle:
1. Environment Setup & Validation.
2. Data Collection (Stress Testing 3 Architectural Modes).
3. Telemetry Analysis & Visualization (Latency/Safety Reporting).

Modes Executed:
1. NAIVE_CONTROL (Baseline Failure Rate)
2. STANDARD_POSTHOC (Latency Comparison)
3. DPF_PROPOSED (The Contribution)
"""

import os
import sys
import time
import shutil

def main():
    # --- 1. BOOT SEQUENCE ---
    os.system('cls' if os.name == 'nt' else 'clear')
    print("==========================================================")
    print("      DPF ARCHITECTURE: SYSTEMS EVALUATION PIPELINE       ")
    print("==========================================================")
    
    # Add 'src' to path to allow direct module imports
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        # Import core architectural components
        import system_registry as registry
        import experiment_driver as test_harness
        import visualization_engine as viz_engine
    except ImportError as e:
        print(f"\n [CRITICAL ERROR] Missing Module: {e}")
        print(" Ensure 'src/system_registry.py', 'src/experiment_driver.py',")
        print(" and 'src/visualization_engine.py' exist.")
        sys.exit(1)

    # --- 2. LOG ROTATION ---
    log_dir = "logs"
    log_file = os.path.join(log_dir, "experiment_data.csv")
    
    if os.path.exists(log_file):
        print(f" >> [Setup] Clearing previous telemetry: {log_file}")
        try:
            os.remove(log_file)
        except OSError as e:
            print(f" !! [WARNING] Could not delete log file: {e}")

    # --- 3. EXPERIMENT EXECUTION LOOP ---
    # We compare the Control group against the Industry Standard and the Proposed Architecture.
    # [CRITICAL]: These keys must match 'valid_modes' in src/system_registry.py
    study_sequence = [
        "NAIVE_CONTROL",     # Negative Control
        "STANDARD_POSTHOC",  # Competitor Baseline
        "DPF_PROPOSED"       # Proposed Solution
    ]
    
    print(f"\n [System] Starting Evaluation Sequence (N={len(study_sequence)} Modes)...")
    
    for mode in study_sequence:
        print(f"\n{'-'*60}")
        print(f"   PHASE: {mode}")
        print(f"{'-'*60}")
        
        # A. Set Architecture State
        try:
            registry.set_system_mode(mode)
        except ValueError as e:
            print(f" !! [ERROR] Configuration Failure: {e}")
            sys.exit(1)
        
        # B. Execute Stress Test (N=20 Vectors)
        # iterations=4 yields N=240 total data points, sufficient for Trend Analysis.
        test_harness.run_batch(mode, iterations=4)
        
        print(f" >> [Complete] Telemetry captured for: {mode}")
        
        # Cool-down to ensure file I/O flush
        time.sleep(1) 

    print("\n" + "="*60)
    print("      [SUCCESS] DATA COLLECTION COMPLETE")
    print("="*60)
    print(f" >> Raw Telemetry stored in: '{log_file}'")
    
    # --- 4. INTEGRATED VISUALIZATION ---
    print("\n[User Action Required]")
    user_choice = input(" >> Would you like to generate the Performance Charts now? (y/n): ").strip().lower()
    
    if user_choice in ['y', 'yes']:
        print(f"\n{'-'*60}")
        print("   EXECUTING VISUALIZATION ENGINE")
        print(f"{'-'*60}")
        
        try:
            # Ensure output directory exists
            output_dir = "paper_results_ieee"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Run Visualization
            viz_engine.main()
            
            print(f"\n [Done] Artifacts generated in: '/{output_dir}'")
            print("        - Latency Comparison Chart")
            print("        - Privacy Efficacy Chart")
            
        except Exception as e:
            print(f" !! [ERROR] Visualization Failed: {e}")
            print("    You can try running 'src/visualization_engine.py' manually.")
    
    else:
        print("\n >> Skipping visualization. You can run 'src/visualization_engine.py' later.")

    print("\n[System] Pipeline Shutdown. Goodbye.")

if __name__ == "__main__":
    main()