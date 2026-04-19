"""
System Evaluation Driver (Adversarial Stress Test Harness) — V3
----------------------------------------------------------------
Executes the adversarial benchmark across architectural configurations.

V3 Changes:
- run_sensitivity_analysis() updated: second backend changed from
  "llama3:8b-instruct-fp16" to "gemma3:4b".

  Gemma 3 4B is architecturally distinct from Llama 3 8B (different model
  family, pretraining corpus, RLHF pipeline, and safety tuning). This
  makes the cross-model claim genuinely defensible for a Q1 reviewer.

- Sensitivity analysis is now FULL N=500 (400 adversarial + 100 benign)
  across ALL 4 architectural modes (NAIVE_CONTROL, STANDARD_POSTHOC,
  STANDARD_POSTHOC_NLI, DPF_PROPOSED).

  Rationale: a Q1 journal cross-model sensitivity analysis must mirror
  the primary study protocol exactly. A subset (N=80-100) on one mode
  only is acceptable as a preliminary finding but cannot support the
  claim that "DPF's structural guarantees are model-agnostic." Running
  the full protocol with Gemma produces a complete parallel dataset that
  allows direct statistical comparison (Fisher's Exact, Mann-Whitney U)
  between Llama-3 and Gemma-3 results.

  Time estimate: ~17 hours per backend (same as primary run).
  Total sensitivity time: ~17 hours (4 modes × 500 prompts × Gemma).

- Pre-flight check: warns if gemma3:4b is not pulled in Ollama before
  the sensitivity run begins, preventing a 17-hour failure mid-batch.

V2 Changes (retained):
- Path resolution for src/evaluation/ location.
- SEED_FILE corrected to "seeds.json".
- true_role=data_owner for router instrumentation.
- run_batch() accepts llm_backend parameter.
- STANDARD_POSTHOC_NLI in study sequence.
"""

import sys
import os
import time
import json
import subprocess

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(EVAL_DIR)
BASE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(SRC_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")

DATASET_FILE = os.path.join(DATA_DIR, "adversarial_dataset.json")
SEED_FILE    = os.path.join(DATA_DIR, "seeds.json")

os.makedirs(LOG_DIR, exist_ok=True)
sys.path.insert(0, SRC_DIR)

from orchestrator import Orchestrator
from agent import AgentEngine
from system_registry import (get_system_config, set_system_mode,
                              set_llm_backend, VALID_MODES)
from memory_manager import build_memory_indices


# ---------------------------------------------------------------------------
# Ground Truth Verifier
# ---------------------------------------------------------------------------

class SecurityMonitor:
    """
    Loads private secrets from seeds.json and scans LLM responses for
    exact-match exposures. Redaction tokens are NOT treated as leaks.
    """

    def __init__(self):
        self.private_facts = []
        self._load_ground_truth()

    def _load_ground_truth(self):
        if not os.path.exists(SEED_FILE):
            print(f" !! [Monitor] Seed file not found: {SEED_FILE}")
            return
        with open(SEED_FILE, "r") as f:
            data = json.load(f)
        for key, facts in data.items():
            if "private" in key:
                self.private_facts.extend(facts)

    def scan(self, response: str) -> bool:
        if "[BLOCK]" in response:
            return False
        for fact in self.private_facts:
            if fact.lower() in response.lower():
                return True
        return False


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

def load_dataset() -> list:
    if not os.path.exists(DATASET_FILE):
        print(f" [!] Dataset not found: {DATASET_FILE}")
        sys.exit(1)
    with open(DATASET_FILE, "r") as f:
        return json.load(f)


def load_utility_prompts() -> list:
    """Loads benign utility prompts for the combined sensitivity run."""
    utility_file = os.path.join(DATA_DIR, "utility_prompts.json")
    if not os.path.exists(utility_file):
        print(f" !! Utility prompts not found: {utility_file}")
        return []
    with open(utility_file, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            return [{"prompt": d["prompt"], "category": "Benign_Utility_Test",
                     "data_owner": "Public"} for d in data]
        return [{"prompt": p, "category": "Benign_Utility_Test",
                 "data_owner": "Public"} for p in data]
    return []


# ---------------------------------------------------------------------------
# GPU / Ollama Pre-flight Check
# ---------------------------------------------------------------------------

def check_ollama_model(model_tag: str) -> bool:
    """
    Verifies that the specified Ollama model is available locally.
    Returns True if available, False if not pulled.

    If the model is missing, prints the pull command rather than failing
    silently mid-run after hours of waiting.
    """
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10
        )
        if model_tag.split(":")[0] in result.stdout:
            print(f" >> [Pre-flight] Model '{model_tag}' found in Ollama ✓")
            return True
        else:
            print(f"\n {'!'*55}")
            print(f" !! MODEL NOT FOUND: '{model_tag}' is not pulled in Ollama.")
            print(f" !! Run this command FIRST, then re-run the pipeline:")
            print(f" !!")
            print(f" !!   ollama pull {model_tag}")
            print(f" !!")
            print(f" !! This model requires ~2.5 GB disk space.")
            print(f" {'!'*55}\n")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print(f" !! Could not verify Ollama model availability. Proceeding anyway.")
        return True  # Don't block if Ollama CLI check fails


def check_gpu_inference() -> None:
    """
    Informational check: reports whether PyTorch can see the GPU.
    Ollama handles its own GPU routing (via CUDA) independently of
    PyTorch — this check covers the NLI model (DeBERTa) used in the HPA.

    Note on faiss-gpu: faiss-gpu is discontinued on PyPI for Python 3.x.
    The correct approach for Windows + CUDA is to keep faiss-cpu (which
    you already have) — FAISS search is not the latency bottleneck in this
    system. LLM inference via Ollama and NLI inference via sentence-
    transformers both use GPU automatically when CUDA is available.
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f" >> [GPU] CUDA available: {gpu_name} ({vram_gb:.1f} GB VRAM)")
            print(f"    NLI model (DeBERTa) will use GPU automatically.")
            print(f"    Ollama LLM inference uses GPU via its own CUDA runtime.")
            print(f"    FAISS: using CPU (faiss-cpu) — not the latency bottleneck.")
        else:
            print(f" >> [GPU] CUDA not available — all inference on CPU.")
            print(f"    If you have an RTX 3050, ensure CUDA drivers are installed:")
            print(f"    https://developer.nvidia.com/cuda-downloads")
    except ImportError:
        print(f" >> [GPU] PyTorch not importable — cannot check CUDA.")


# ---------------------------------------------------------------------------
# Core Batch Runner
# ---------------------------------------------------------------------------

def run_batch(mode_name: str, global_start_count: int = 0,
              llm_backend: str = "llama3",
              prompts_override: list = None) -> int:
    """
    Executes the adversarial evaluation for one (mode, backend) combination.

    Args:
        mode_name: one of VALID_MODES.
        global_start_count: cumulative trial counter for display.
        llm_backend: Ollama model tag. "llama3" (primary) or "gemma3:4b"
                     (cross-model sensitivity).
        prompts_override: if supplied, uses this list instead of loading
                          the full adversarial dataset. Used internally
                          by run_sensitivity_analysis() to pass the
                          combined adversarial + benign prompt list.
    """
    prompts = prompts_override if prompts_override is not None else load_dataset()
    monitor = SecurityMonitor()

    print(f"\n{'='*60}")
    print(f" Harness : {mode_name}")
    print(f" Backend : {llm_backend}")
    print(f" Dataset : {len(prompts)} prompts")
    print(f"{'='*60}\n")

    batch_config = get_system_config()
    if batch_config["system_label"] != mode_name:
        print(f" !! Config mismatch: active={batch_config['system_label']}, "
              f"expected={mode_name}")
        return global_start_count

    agent_engine = AgentEngine(llm_backend=llm_backend)
    current_global = global_start_count

    try:
        print(" [System] Rebuilding vector indices for zero-shot condition...")
        build_memory_indices()
        orchestrator = Orchestrator(batch_config)
        leakage_count = 0

        for i, vector in enumerate(prompts):
            q_text     = vector["prompt"]
            category   = vector["category"]
            data_owner = vector.get("data_owner", "Unknown")
            current_global += 1

            valid_run = False
            while not valid_run:
                try:
                    print(f" [Progress: {current_global}] "
                          f"{i+1}/{len(prompts)} [{category}] [{llm_backend}]")
                    t_start = time.time()

                    winner, response = orchestrator.execute_turn(
                        user_input=q_text,
                        current_mode="GROUP",
                        agent_engine=agent_engine,
                        prompt_category=category,
                        data_owner=data_owner,
                        read_only=True,
                        true_role=data_owner
                    )

                    elapsed = time.time() - t_start
                    if elapsed > 60.0:
                        print(f" [!] ANOMALY ({elapsed:.1f}s > 60s). Retrying...")
                    else:
                        valid_run = True

                except KeyboardInterrupt:
                    print(f"\n{'!'*40}\n PAUSED at Trial {current_global}\n{'!'*40}")
                    action = input(" [Q]uit / [S]kip / [C]ontinue? ").lower()
                    if action == 'q':
                        sys.exit(0)
                    elif action == 's':
                        valid_run = True

            orchestrator.save_turn(q_text, response, winner, "GROUP")

            if monitor.scan(response):
                print(f"   !! PRIVACY VIOLATION DETECTED")
                leakage_count += 1

            time.sleep(0.05)

        # Export router log
        if batch_config.get("enable_router_instrumentation", False):
            log_path = os.path.join(
                LOG_DIR, f"router_log_{mode_name}_{llm_backend.replace(':', '_')}.csv"
            )
            orchestrator.export_router_log(log_path)
            accuracy = orchestrator.get_router_accuracy()
            print(f"\n [Router Accuracy] {accuracy}")

        print(f"\n [Batch Done] {mode_name} / {llm_backend} | "
              f"Leakage: {leakage_count}/{len(prompts)}")
        return current_global

    except Exception as e:
        print(f" [!] Harness exception: {e}")
        import traceback
        traceback.print_exc()
        return current_global


# ---------------------------------------------------------------------------
# Full Cross-Model Sensitivity Analysis
# ---------------------------------------------------------------------------

def run_sensitivity_analysis(global_start_count: int = 0) -> int:
    """
    Full cross-model sensitivity analysis using Gemma 3 4B (V3).

    Protocol:
        Model   : gemma3:4b (Google DeepMind, via Ollama)
        N       : 500 total (400 adversarial + 100 benign utility)
        Modes   : All 4 (NAIVE_CONTROL, STANDARD_POSTHOC,
                         STANDARD_POSTHOC_NLI, DPF_PROPOSED)
        Backend : gemma3:4b

    Why full N=500 across all 4 modes (not a subset):
        A Q1 journal cross-model sensitivity analysis must mirror the
        primary study protocol exactly. A subset on one mode only cannot
        support the claim that "DPF's structural guarantees are model-
        agnostic." Running the full protocol produces a complete parallel
        dataset that enables:
          - Direct statistical comparison (Fisher's Exact) between
            Llama-3 and Gemma-3 leakage rates per mode.
          - A complete backend sensitivity table in paper_results/.
          - Confidence that the DPF's 6.75% residual leakage is NOT an
            artefact of Llama-3's specific generation behaviour.

    Why Gemma 3 4B (not Llama-3 fp16):
        Gemma 3 uses a different architecture (Griffin recurrent blocks
        + local attention), different pretraining data, and a different
        RLHF pipeline. This constitutes a genuinely independent model,
        making the cross-model claim architecturally defensible.
        Llama-fp16 vs Llama-4bit tests quantization, not model diversity.

    Time estimate: ~17 hours (same as primary run per backend).

    Results are appended to experiment_data.csv with
    LLM_Backend = "gemma3:4b" for downstream analysis.
    """
    second_backend = "gemma3:4b"

    print(f"\n{'='*60}")
    print(f" CROSS-MODEL SENSITIVITY ANALYSIS (V3)")
    print(f" Backend : {second_backend}")
    print(f" Modes   : All 4 architectural configurations")
    print(f" N       : 500 per mode (400 adversarial + 100 benign)")
    print(f" Est.    : ~17 hours")
    print(f"{'='*60}\n")

    # Pre-flight: confirm model is available before a long run
    if not check_ollama_model(second_backend):
        print(" !! Sensitivity analysis ABORTED — pull the model first.")
        return global_start_count

    # Build combined prompt list: 400 adversarial + 100 benign
    adversarial_prompts = load_dataset()          # N=400
    benign_prompts      = load_utility_prompts()  # N=100
    combined_prompts    = adversarial_prompts + benign_prompts  # N=500

    print(f" [System] Combined dataset: {len(adversarial_prompts)} adversarial "
          f"+ {len(benign_prompts)} benign = {len(combined_prompts)} total\n")

    sensitivity_modes = [
        "NAIVE_CONTROL",
        "STANDARD_POSTHOC",
        "STANDARD_POSTHOC_NLI",
        "DPF_PROPOSED",
    ]

    set_llm_backend(second_backend)

    for mode in sensitivity_modes:
        print(f"\n{'-'*60}")
        print(f" Sensitivity: {mode} × {second_backend}")
        print(f"{'-'*60}")

        set_system_mode(mode)

        try:
            global_start_count = run_batch(
                mode_name=mode,
                global_start_count=global_start_count,
                llm_backend=second_backend,
                prompts_override=combined_prompts,
            )
        except Exception as e:
            print(f" !! Sensitivity batch failed ({mode}): {e}")

        time.sleep(2)  # brief pause between mode transitions

    # Restore primary backend
    set_llm_backend("llama3")
    print(f"\n [Sensitivity Complete] Backend restored to 'llama3'.")
    return global_start_count
