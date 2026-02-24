"""
System Entry Point & Environment Controller
-------------------------------------------
This script serves as the primary driver for the architecture repository.
It acts as a CLI wrapper that:
1. Validates the execution environment (dependency resolution & auto-install).
2. Provides a Fast-Path to reproduce exact published figures from immutable logs.
3. Provides a Slow-Path to completely re-run the 15+ hour evaluation pipeline.
4. Encapsulates subprocess execution to ensure strict memory isolation.
"""

import sys
import os
import subprocess

def check_dependencies():
    """
    Environment Verification Routine.
    Ensures all required engineering libraries are present before execution,
    preventing mid-experiment crashes. Automatically installs missing packages.
    """
    print(f" >> [System] Python Executable: {sys.executable}")
    print(" >> [System] Verifying environment dependencies...")
    try:
        import pandas               # Data manipulation for telemetry
        import matplotlib           # Visualization generation
        import torch                # Neural network backend (Crucial for Mac)
        import sentence_transformers # NLI Auditing
        import tqdm                 # Progress tracking
        import langchain            # Orchestration framework
        
        # Note: 'faiss' is often installed as 'faiss-cpu' but imported as 'faiss'
        try:
            import faiss
        except ImportError:
            pass # Faiss might be managed internally by LangChain, proceed with caution.
            
        print(" >> [System] All required dependencies are present.")
        
    except ImportError as e:
        print(f" >> [System] Missing library detected: {e}")
        print(" >> [System] Attempting auto-installation from requirements.txt...")
        try:
            if os.path.exists("requirements.txt"):
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
                print(" >> [System] Auto-installation complete.")
            else:
                print(" !! [Critical] requirements.txt not found. Please install dependencies manually.")
                sys.exit(1)
        except Exception as e:
            print(f" !! [Critical] Failed to install dependencies: {e}")
            sys.exit(1)

def run_module(script_name, args=None):
    """
    Safely executes a target Python script via subprocess.
    Allows passing dynamic arguments (like immutable file paths) to the child process.
    """
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    if not os.path.exists(script_path):
        script_path = os.path.join(os.path.dirname(__file__), 'src', script_name)
        
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
        
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n [System] Execution interrupted by user.")

def main():
    """
    Main Execution Loop (Reviewer CLI Menu).
    """
    # 1. Pre-Flight Check
    check_dependencies()
    
    # Define the paths to the immutable reference data
    paper_logs_path = os.path.join(os.path.dirname(__file__), "src", "data", "paper_logs", "audit_results.csv")
    utility_logs_path = os.path.join(os.path.dirname(__file__), "src", "data", "paper_logs", "ablation_utility_audit.csv")
    human_logs_path = os.path.join(os.path.dirname(__file__), "src", "data", "paper_logs", "human_audit_set.csv")
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("==========================================================")
        print("   MULTI-AGENT ARCHITECTURE: SYSTEM CONTROL INTERFACE     ")
        print("==========================================================")
        
        # Option 1: The "Instant" Reproducibility Path
        print("   [1] Generate Reference Figures & Tables (Fast Path)")
        print("       > Uses immutable telemetry logs from /src/data/paper_logs.")
        print("       > Instantly outputs identical CSVs and PNGs to /paper_results.")
        print("")
        
        # Option 2: The Full End-to-End Run
        print("   [2] Re-run Full Evaluation Pipeline (Slow Path)")
        print("       > WARNING: Takes ~15 hours (1200 Adversarial + 300 Benign trials).")
        print("       > NOTE: Due to LLM stochastic variance, exact metrics will ")
        print("         fluctuate slightly from the published manuscript if re-run.")
        print("       > Generates NEW telemetry in /logs and overwrites /paper_results.")
        print("")
        
        # Option 3: Human interaction
        print("   [3] Interactive Debug Console (Human-in-the-Loop)")
        print("       > Real-time chat interface to test agent routing and firewalls.")
        print("")
        
        # Option 4: Memory Inspection
        print("   [4] Inspect Vector Database State")
        print("       > Read-only audit of isolated memory partitions (FAISS).")
        print("")
        
        # Option 5: Methodological Validation
        print("   [5] Validate Auditor Accuracy (Cohen's Kappa)")
        print("       > Compares AI auditor results against human ground truth.")
        print("")
        
        print("   [6] Exit")
        print("==========================================================")
        
        print("\n   Type the number of your choice and press Enter.")
        choice = input("   >> ").strip()
        
        if choice == '1':
            print("\n   >> Generating metrics from Immutable Reference Logs...\n")
            if not os.path.exists(paper_logs_path) or not os.path.exists(utility_logs_path):
                print(f"   !! [ERROR] Immutable logs not found in 'src/data/paper_logs/'.")
                print("       Ensure both 'audit_results.csv' and 'ablation_utility_audit.csv' are securely placed.")
            else:
                # Pass the protected paths dynamically to the downstream analytical scripts
                run_module("run_utility_benchmark.py", [utility_logs_path])
                run_module("evaluate_privacy_audit.py", [paper_logs_path])
                run_module("run_auditor_ablation.py", [paper_logs_path])
                run_module("visualization_engine.py", [paper_logs_path])
            input("\n   [Press Enter to return to menu]")
            
        elif choice == '2':
            print("\n   !! WARNING: This operation takes 15+ hours and generates NEW stochastic data.")
            confirm = input("   >> Are you sure you want to proceed? (y/n): ").strip().lower()
            if confirm == 'y':
                print("\n   >> Initializing Master Evaluation Pipeline...\n")
                run_module("run_evaluation_pipeline.py")
            else:
                print("   >> Execution aborted.")
            input("\n   [Press Enter to return to menu]")
            
        elif choice == '3':
            print("\n   >> Booting Debug Console...\n")
            run_module("interactive_console.py")
            input("\n   [Press Enter to return to menu]")
            
        elif choice == '4':
            print("\n   >> Querying Storage Layer...\n")
            run_module("view_memory.py")
            input("\n   [Press Enter to return to menu]")
            
        elif choice == '5':
            print("\n   >> Executing Human-in-the-Loop Validation...\n")
            if not os.path.exists(human_logs_path):
                print(f"   !! [ERROR] Human Ground Truth not found at: {human_logs_path}")
                print("       Ensure 'human_audit_set.csv' is securely placed in 'src/data/paper_logs/'.")
            else:
                run_module("calculate_audit_metrics.py", [human_logs_path])
            input("\n   [Press Enter to return to menu]")

        elif choice == '6':
            print("\n   >> Exiting. Goodbye!")
            sys.exit(0)
            
        else:
            print(f"\n   !! Invalid input: '{choice}'")
            input("   !! Please type 1-6. Press Enter to try again...")

if __name__ == "__main__":
    main()