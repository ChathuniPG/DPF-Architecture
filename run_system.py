"""
System Entry Point & Environment Controller — V2
-------------------------------------------------
Primary CLI driver for the DPF architecture repository.

V2 Changes vs V1:
- run_module() now resolves scripts using a priority search list:
    1. project root
    2. src/evaluation/   ← new location for evaluation harnesses
    3. src/              ← legacy location (visualization_engine, etc.)
  This ensures all menu options work correctly after the git mv refactor.
- Menu text updated to reflect V2 new outputs (router_accuracy.csv,
  threat_class_breakdown.csv).
"""

import sys
import os
import subprocess


def check_dependencies():
    print(f" >> [System] Python: {sys.executable}")
    print(" >> [System] Verifying dependencies...")
    try:
        import pandas, matplotlib, torch, sentence_transformers, tqdm, langchain
        try:
            import faiss
        except ImportError:
            pass
        print(" >> [System] All dependencies present.")
    except ImportError as e:
        print(f" >> [System] Missing: {e}")
        print(" >> Attempting auto-install from requirements.txt...")
        try:
            if os.path.exists("requirements.txt"):
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"]
                )
                print(" >> Auto-install complete.")
            else:
                print(" !! requirements.txt not found.")
                sys.exit(1)
        except Exception as e:
            print(f" !! Install failed: {e}")
            sys.exit(1)


def run_module(script_name: str, args=None):
    """
    Execute a script as an isolated subprocess.
    V2: searches src/evaluation/ before src/ so moved harnesses resolve correctly.
    """
    base = os.path.dirname(os.path.abspath(__file__))

    candidates = [
        os.path.join(base, script_name),
        os.path.join(base, "src", "evaluation", script_name),
        os.path.join(base, "src", script_name),
    ]

    script_path = None
    for c in candidates:
        if os.path.exists(c):
            script_path = c
            break

    if script_path is None:
        print(f" !! [ERROR] Script not found: {script_name}")
        return

    cmd = [sys.executable, script_path] + (args or [])
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n [System] Interrupted.")


def main():
    check_dependencies()

    base = os.path.dirname(os.path.abspath(__file__))
    paper_logs = os.path.join(base, "src", "data", "paper_logs")
    audit_csv  = os.path.join(paper_logs, "audit_results.csv")
    utility_csv = os.path.join(paper_logs, "ablation_utility_audit.csv")
    human_csv  = os.path.join(paper_logs, "human_audit_set.csv")

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')

        print("=" * 60)
        print("   DPF ARCHITECTURE V2: SYSTEM CONTROL INTERFACE")
        print("=" * 60)
        print("   [1] Generate Reference Figures & Tables (Fast Path)")
        print("       > Immutable logs from src/data/paper_logs/")
        print("       > Instant CSV + PNG output to /paper_results/")
        print("")
        print("   [2] Re-run Full Evaluation Pipeline (Slow Path)")
        print("       > WARNING: ~17 hours (N=500 adversarial + benign).")
        print("       > Generates NEW telemetry — metrics will show")
        print("         stochastic variance vs published manuscript.")
        print("")
        print("   [3] Interactive Debug Console")
        print("       > Real-time chat to test routing and firewalls.")
        print("")
        print("   [4] Inspect Vector Database State")
        print("       > Read-only audit of FAISS memory partitions.")
        print("")
        print("   [5] Validate Auditor Accuracy (Cohen's Kappa)")
        print("       > HPA vs human annotation agreement.")
        print("")
        print("   [6] Exit")
        print("=" * 60)

        choice = input("\n   >> ").strip()

        if choice == '1':
            print("\n >> Fast Path: generating from immutable logs...\n")
            if not os.path.exists(audit_csv) or not os.path.exists(utility_csv):
                print(f" !! Logs not found in src/data/paper_logs/")
            else:
                run_module("run_utility_benchmark.py",  [utility_csv])
                run_module("evaluate_privacy_audit.py", [audit_csv])
                run_module("run_auditor_ablation.py",   [audit_csv])
                run_module("visualization_engine.py",   [audit_csv])
            input("\n [Press Enter to return to menu]")

        elif choice == '2':
            print("\n !! This takes ~17 hours and generates NEW stochastic data.")
            confirm = input(" >> Proceed? (y/n): ").strip().lower()
            if confirm == 'y':
                run_module("run_evaluation_pipeline.py")
            else:
                print(" >> Aborted.")
            input("\n [Press Enter to return to menu]")

        elif choice == '3':
            run_module("interactive_console.py")
            input("\n [Press Enter to return to menu]")

        elif choice == '4':
            run_module("view_memory.py")
            input("\n [Press Enter to return to menu]")

        elif choice == '5':
            if not os.path.exists(human_csv):
                print(f" !! Human ground truth not found: {human_csv}")
            else:
                run_module("calculate_audit_metrics.py", [human_csv])
            input("\n [Press Enter to return to menu]")

        elif choice == '6':
            print("\n >> Goodbye!")
            sys.exit(0)

        else:
            print(f"\n !! Invalid input: '{choice}'")
            input(" !! Enter 1–6. Press Enter to try again...")


if __name__ == "__main__":
    main()
