"""
Automated Evaluation Pipeline (One-Click Reproducibility) — V2
---------------------------------------------------------------
Orchestrates the full comparative ablation study.

V2 Changes:
- STANDARD_POSTHOC_NLI added to adversarial study sequence.
- run_sensitivity_analysis() called after main adversarial sequence
  to produce cross-model backend data (Issue 1).
- Script resolution searches src/evaluation/ first.
"""

import os
import sys
import time
import subprocess


def run_external_script(script_name: str, description: str) -> bool:
    print(f"\n{'-'*60}")
    print(f"   PHASE: {description}")
    print(f"{'-'*60}")

    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, "src", "evaluation", script_name),
        os.path.join(base, "src", script_name),
        os.path.join(base, script_name),
    ]

    script_path = next((c for c in candidates if os.path.exists(c)), None)
    if not script_path:
        print(f" !! Script not found: {script_name}")
        return False

    try:
        subprocess.check_call([sys.executable, script_path])
        return True
    except subprocess.CalledProcessError as e:
        print(f" !! Phase '{description}' failed (exit {e.returncode}).")
        return False


def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 60)
    print("      DPF ARCHITECTURE V2: EVALUATION PIPELINE")
    print("=" * 60)

    base = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(base, "src"))
    sys.path.insert(0, os.path.join(base, "src", "evaluation"))

    try:
        import system_registry as registry
    except ImportError as e:
        print(f"\n [CRITICAL] {e}")
        sys.exit(1)

    try:
        import experiment_driver as test_harness
    except ImportError as e:
        print(f"\n [CRITICAL] {e}")
        sys.exit(1)

    log_dir = os.path.join(base, "logs")
    os.makedirs(log_dir, exist_ok=True)

    eval_log = os.path.join(log_dir, "experiment_data.csv")
    interactive_log = os.path.join(log_dir, "system_telemetry.csv")
    backup_log = os.path.join(log_dir, "system_telemetry.backup")

    if os.path.exists(interactive_log):
        os.rename(interactive_log, backup_log)
    if os.path.exists(eval_log):
        os.remove(eval_log)

    # V2: 4-mode adversarial sequence
    study_sequence = [
        "NAIVE_CONTROL",
        "STANDARD_POSTHOC",
        "STANDARD_POSTHOC_NLI",   # NEW: fair post-hoc baseline
        "DPF_PROPOSED",
    ]

    global_counter = 0
    pipeline_errors = False

    print(f"\n [System] Adversarial Stress Test Sequence ({len(study_sequence)} modes)...")

    for mode in study_sequence:
        print(f"\n{'-'*60}\n   Adversarial Testing → {mode}\n{'-'*60}")
        try:
            registry.set_system_mode(mode)
        except ValueError as e:
            print(f" !! {e}")
            sys.exit(1)
        try:
            global_counter = test_harness.run_batch(mode, global_start_count=global_counter)
            print(f" >> [Done] {mode} | Total: {global_counter}")
        except Exception as e:
            print(f" !! Batch failed ({mode}): {e}")
            pipeline_errors = True
        time.sleep(1)

    # V2 NEW: Cross-model sensitivity analysis (Issue 1, zero cost)
    print(f"\n{'-'*60}\n   Sensitivity Analysis → DPF × llama3:8b-instruct-fp16\n{'-'*60}")
    try:
        global_counter = test_harness.run_sensitivity_analysis(global_counter)
    except Exception as e:
        print(f" !! Sensitivity analysis failed: {e}")
        pipeline_errors = True

    # Log normalization
    if os.path.exists(interactive_log):
        os.rename(interactive_log, eval_log)
    if os.path.exists(backup_log):
        os.rename(backup_log, interactive_log)

    if os.path.exists(eval_log):
        with open(eval_log, 'r', encoding='utf-8') as f:
            row_count = sum(1 for _ in f) - 1
        if row_count < 20:
            print("\n" + "!"*60)
            print(" [WARNING] MICRO-DATASET (N < 20).")
            print("!"*60)

    # Extended pipeline phases
    phases = [
        ("run_utility_benchmark.py",  "Benign Utility & FRR Audit (4 modes)"),
        ("evaluate_privacy_audit.py", "Hybrid Privacy Auditor (NLI + Regex)"),
        ("run_auditor_ablation.py",   "Failure Mode Attribution + Router Accuracy"),
        ("visualization_engine.py",   "Data Synthesis & Figure Generation"),
    ]

    for script, desc in phases:
        if not run_external_script(script, desc):
            pipeline_errors = True

    print("\n" + "="*60)
    if pipeline_errors:
        print("      [DEGRADED] PIPELINE COMPLETED WITH ERRORS")
    else:
        print("      [SUCCESS] REPRODUCIBILITY PIPELINE COMPLETE")
        print("="*60)
        print(" >> Telemetry  : /logs/")
        print(" >> Artifacts  : /paper_results/")
        print("    Tables: Baseline_Comparison, Backend_Sensitivity,")
        print("            Statistical_Results, Firewall_Activity,")
        print("            Category_Analysis, router_accuracy,")
        print("            threat_class_breakdown, ablation_utility_summary")
        print("    Figures: Latency_Safety_Tradeoff, Leakage_Rate_Global,")
        print("             Leakage_By_Category, Backend_Sensitivity,")
        print("             Scalability_Trends")

    print("\n[System] Pipeline shutdown.")


if __name__ == "__main__":
    main()
