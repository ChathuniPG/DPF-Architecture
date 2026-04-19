"""
System Entry Point & Environment Controller — V3
-------------------------------------------------
Primary CLI driver for the DPF architecture repository.

V3 Changes:
- Menu updated to reflect V3 two-phase evaluation structure.
- GPU diagnostic displayed on launch.
- run_module() resolves scripts from src/evaluation/ first.
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


def check_gpu():
    """Quick GPU status for the startup banner."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            return f"GPU: {name} ({vram:.1f} GB) — CUDA available ✓"
        return "GPU: CUDA not available (CPU mode)"
    except Exception:
        return "GPU: Status unknown"


def run_module(script_name: str, args=None):
    """
    Execute a script as an isolated subprocess.
    Searches src/evaluation/ before src/ so moved harnesses resolve correctly.
    """
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, script_name),
        os.path.join(base, "src", "evaluation", script_name),
        os.path.join(base, "src", script_name),
    ]

    script_path = next((c for c in candidates if os.path.exists(c)), None)
    if not script_path:
        print(f" !! [ERROR] Script not found: {script_name}")
        return

    cmd = [sys.executable, script_path] + (args or [])
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n [System] Interrupted.")


def main():
    check_dependencies()
    gpu_status = check_gpu()

    base = os.path.dirname(os.path.abspath(__file__))
    paper_logs  = os.path.join(base, "src", "data", "paper_logs")
    audit_csv   = os.path.join(paper_logs, "audit_results.csv")
    utility_csv = os.path.join(paper_logs, "ablation_utility_audit.csv")
    human_csv   = os.path.join(paper_logs, "human_audit_set.csv")

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')

        print("=" * 60)
        print("   DPF ARCHITECTURE V3: SYSTEM CONTROL INTERFACE")
        print("=" * 60)
        print(f"   {gpu_status}")
        print("=" * 60)
        print()
        print("   [1] Generate Reference Figures & Tables (Fast Path)")
        print("       > Immutable logs from src/data/paper_logs/")
        print("       > Instant CSV + PNG output to /paper_results/")
        print()
        print("   [2] Re-run Full Evaluation Pipeline (Slow Path)")
        print("       > Phase 1: Llama-3-8B × 4 modes × N=500  (~17h)")
        print("       > Phase 2: Gemma-3-4B × 4 modes × N=500  (~17h)")
        print("       > WARNING: ~34 hours total.")
        print("       > Requires: ollama pull gemma3:4b")
        print()
        print("   [3] Interactive Debug Console")
        print("       > Real-time chat to test routing and firewalls.")
        print()
        print("   [4] Inspect Vector Database State")
        print("       > Read-only audit of FAISS memory partitions.")
        print()
        print("   [5] Validate Auditor Accuracy (Cohen's Kappa)")
        print("       > HPA vs human annotation agreement.")
        print()
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
            print("\n !! FULL PIPELINE — ~34 hours total.")
            print("    Phase 1: Llama-3-8B  (~17h)")
            print("    Phase 2: Gemma-3-4B  (~17h)")
            print()
            print("    Before proceeding, run: ollama pull gemma3:4b")
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
