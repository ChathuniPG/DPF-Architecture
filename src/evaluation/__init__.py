"""
Evaluation Package
------------------
Contains all evaluation harnesses for the DPF ablation study.
Separated from the core system to make the distinction between
what is under test and what is doing the testing explicit.

Modules:
    experiment_driver    — adversarial stress-test harness (N=400)
    run_utility_benchmark — benign FRR benchmark (N=100)
    evaluate_privacy_audit — Hybrid Privacy Auditor (HPA)
    run_auditor_ablation   — failure mode attribution
"""
