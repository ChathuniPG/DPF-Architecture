"""
Automated Evaluation Pipeline (One-Click Reproducibility) — V3
---------------------------------------------------------------
Orchestrates the full comparative ablation study.

V3 Changes:
- GPU diagnostic printed at startup.
- Gemma3:4b pre-flight check before sensitivity analysis begins.
- Sensitivity analysis label updated to reflect gemma3:4b and full N=500.
- run_sensitivity_analysis() now runs full 4-mode × 500-prompt protocol.

V2 Changes (retained):
- STANDARD_POSTHOC_NLI in the primary study sequence.
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
    print("      DPF ARCHITECTURE V3: EVALUATION PIPELINE")
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

    # V3: GPU diagnostic at startup
    print("\n [System] Checking GPU availability...")
    test_harness.check_gpu_inference()

    log_dir = os.path.join(base, "logs")
    os.makedirs(log_dir, exist_ok=True)

    eval_log       = os.path.join(log_dir, "experiment_data.csv")
    interactive_log = os.path.join(log_dir, "system_telemetry.csv")
    backup_log     = os.path.join(log_dir, "system_telemetry.backup")

    if os.path.exists(interactive_log):
        os.rename(interactive_log, backup_log)
    if os.path.exists(eval_log):
        os.remove(eval_log)

    # --- Phase 1: Primary adversarial study (Llama-3-8B, 4 modes × N=500) ---
    study_sequence = [
        "NAIVE_CONTROL",
        "STANDARD_POSTHOC",
        "STANDARD_POSTHOC_NLI",
        "DPF_PROPOSED",
    ]

    global_counter = 0
    pipeline_errors = False

    print(f"\n [System] PRIMARY STUDY: Llama-3-8B × 4 modes × N=500")
    print(f"          Estimated time: ~17 hours\n")

    for mode in study_sequence:
        print(f"\n{'-'*60}\n   Primary: {mode} × llama3\n{'-'*60}")
        try:
            registry.set_system_mode(mode)
        except ValueError as e:
            print(f" !! {e}")
            sys.exit(1)
        try:
            global_counter = test_harness.run_batch(
                mode, global_start_count=global_counter
            )
            print(f" >> [Done] {mode} | Total trials: {global_counter}")
        except Exception as e:
            print(f" !! Batch failed ({mode}): {e}")
            pipeline_errors = True
        time.sleep(1)

    # Log normalization after primary study
    if os.path.exists(interactive_log):
        os.rename(interactive_log, eval_log)
    if os.path.exists(backup_log):
        os.rename(backup_log, interactive_log)

    # --- Phase 2: Cross-model sensitivity (Gemma3:4b, 4 modes × N=500) ---
    print(f"\n{'='*60}")
    print(f" PHASE 2: CROSS-MODEL SENSITIVITY ANALYSIS")
    print(f" Backend: gemma3:4b | Modes: 4 | N: 500 per mode")
    print(f" Estimated time: ~17 hours")
    print(f" NOTE: Run 'ollama pull gemma3:4b' before this starts.")
    print(f"{'='*60}")

    try:
        global_counter = test_harness.run_sensitivity_analysis(global_counter)
    except Exception as e:
        print(f" !! Sensitivity analysis failed: {e}")
        pipeline_errors = True

    # Validate combined dataset
    if os.path.exists(eval_log):
        with open(eval_log, 'r', encoding='utf-8') as f:
            row_count = sum(1 for _ in f) - 1
        print(f"\n [System] Total telemetry rows: {row_count}")
        if row_count < 20:
            print("!" * 60)
            print(" [WARNING] MICRO-DATASET — statistical modules may fail.")
            print("!" * 60)

    # --- Phase 3: Analysis pipeline ---
    phases = [
        ("run_utility_benchmark.py",  "Benign Utility & FRR Audit (4 modes × 2 backends)"),
        ("evaluate_privacy_audit.py", "Hybrid Privacy Auditor (NLI + Regex)"),
        ("run_auditor_ablation.py",   "Failure Mode Attribution + Router Accuracy"),
        ("visualization_engine.py",   "Data Synthesis & Figure Generation"),
    ]

    for script, desc in phases:
        if not run_external_script(script, desc):
            pipeline_errors = True

    # --- Summary ---
    print("\n" + "=" * 60)
    if pipeline_errors:
        print("      [DEGRADED] PIPELINE COMPLETED WITH ERRORS")
        print("=" * 60)
        print(" Review terminal output for missing models or data.")
    else:
        print("      [SUCCESS] FULL REPRODUCIBILITY PIPELINE COMPLETE")
        print("=" * 60)
        print(f" >> Telemetry  : /logs/")
        print(f" >> Artifacts  : /paper_results/")
        print(f"    Primary study      : Llama-3-8B × 4 modes × N=500")
        print(f"    Sensitivity study  : Gemma-3-4B × 4 modes × N=500")
        print(f"    Tables: Baseline_Comparison, Backend_Sensitivity,")
        print(f"            Statistical_Results, Firewall_Activity,")
        print(f"            Category_Analysis, router_accuracy,")
        print(f"            threat_class_breakdown, ablation_utility_summary")
        print(f"    Figures: Latency_Safety_Tradeoff, Leakage_Rate_Global,")
        print(f"             Leakage_By_Category, Backend_Sensitivity,")
        print(f"             Scalability_Trends")

    print("\n[System] Pipeline shutdown.")


if __name__ == "__main__":
    main()
