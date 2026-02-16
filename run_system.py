"""
run_system.py
-------------
System Entry Point & Environment Controller.

This script serves as the primary driver for the DPF Architecture.
It acts as a CLI wrapper that:
1. Validates the execution environment (dependencies).
2. Provides a menu-driven interface for switching between operation modes.
3. Encapsulates subprocess execution to ensure isolation.
"""

import sys
import os
import subprocess
import time

def check_dependencies():
    """
    Environment Verification Routine.
    Ensures all required engineering libraries are present before execution.
    """
    print(f" >> [System] Python Executable: {sys.executable}")
    print(" >> [System] Verifying environment dependencies...")
    try:
        # Import check for core architectural dependencies
        import pandas               # Data manipulation for telemetry
        import matplotlib           # Visualization generation
        import seaborn              # Statistical plotting
        import langchain            # Orchestration framework
        import langchain_community  # Community extensions
        
        # Note: 'faiss' is often installed as 'faiss-cpu' but imported as 'faiss'
        try:
            import faiss
        except ImportError:
            pass # Faiss might be managed internally by LangChain, proceed with caution.
        
        print(" >> [System] All dependencies are present.")
        
    except ImportError as e:
        # Fallback: Auto-Installation logic
        print(f" >> [System] Missing library detected: {e}")
        print(" >> [System] Attempting auto-installation from requirements.txt...")
        try:
            if os.path.exists("requirements.txt"):
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
                print(" >> [System] Installation complete.")
            else:
                print(" !! [Critical] requirements.txt not found. Please install dependencies manually.")
                sys.exit(1)
            
        except Exception as e:
            print(f" !! [Critical] Failed to install dependencies: {e}")
            print("    Please run 'pip install -r requirements.txt' manually.")
            sys.exit(1)

def main():
    """
    Main Execution Loop (CLI Menu).
    """
    
    # 1. Pre-Flight Check: Verify Environment Integrity
    check_dependencies()
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("==========================================================")
        print("   DPF ARCHITECTURE: ENGINEERING EVALUATION SYSTEM        ")
        print("==========================================================")
        
        # Option 1: Interactive Debugging
        print("   [1] Interactive Debug Console (User Mode)")
        print("       > Manual testing of routing, firewall, and latency.")
        print("")
        
        # Option 2: Automated Experiments
        print("   [2] Run Full Evaluation Pipeline (Research Mode)")
        print("       > Automates adversarial stress tests.")
        print("       > Dataset: N=1,200 (400 Vectors x 3 Architectures).")
        print("       > Generates Statistical Reports and Figures.")
        print("")
        
        # Option 3: Micro-Benchmark
        print("   [3] Run Micro-Benchmarks (Complexity Analysis)")
        print("       > Validates O(1) Firewall throughput and latency.")
        print("")
        
        print("   [4] Exit")
        print("==========================================================")
        
        print("\n   Type the number of your choice and press Enter.")
        choice = input("   >> ").strip()
        
        if choice == '1':
            print("\n   >> Booting Debug Console...\n")
            try:
                # Updated path to the new console script
                script_path = os.path.join("src", "interactive_console.py")
                subprocess.run([sys.executable, script_path])
            except KeyboardInterrupt:
                pass
            input("\n   [Press Enter to return to menu]")
            
        elif choice == '2':
            print("\n   >> Initializing Research Pipeline...\n")
            try:
                # Updated path to the new pipeline script
                subprocess.run([sys.executable, "run_evaluation_pipeline.py"])
            except KeyboardInterrupt:
                pass
            input("\n   [Press Enter to return to menu]")
            
        elif choice == '3':
            print("\n   >> Running Latency Micro-Benchmarks...\n")
            try:
                # Updated path to the new benchmark script
                script_path = os.path.join("tests", "benchmark_latency.py")
                if os.path.exists(script_path):
                    subprocess.run([sys.executable, script_path])
                else:
                    print(f" !! [ERROR] Benchmark script not found at: {script_path}")
            except KeyboardInterrupt:
                pass
            input("\n   [Press Enter to return to menu]")
            
        elif choice == '4':
            print("\n   >> Exiting. Goodbye!")
            sys.exit(0)
            
        else:
            print(f"\n   !! Invalid input: '{choice}'")
            input("   !! Please type '1', '2', '3', or '4'. Press Enter to try again...")

if __name__ == "__main__":
    main()