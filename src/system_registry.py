"""
System Registry (Architecture Control Plane) — V3
--------------------------------------------------
Central state manager and configuration registry for the multi-agent framework.

V3 Changes:
- Second LLM backend changed from "llama3:8b-instruct-fp16" to "gemma3:4b".

  Rationale for Gemma over Llama-fp16:
  Comparing Llama-3-8B (4-bit GGUF) against Llama-3-8B (float16) tests only
  quantization sensitivity within the same model family and architecture.
  This is a weak cross-model claim: a reviewer can correctly argue that the
  same weights in different precision are not truly independent models.

  Gemma 3 (4B) is architecturally distinct: it uses Google DeepMind's
  Gemma-3 architecture (Griffin-style recurrent blocks + local attention),
  a different pretraining corpus, different RLHF pipeline, and different
  safety tuning. Running the full N=500 adversarial + N=100 benign protocol
  across all 4 architectural modes with Gemma validates that the DPF's
  structural guarantees (RBAC, PII masking, index arbitration) are
  model-agnostic across genuinely distinct generative backends — which is
  the actual paper claim in Proposition 1.

  Gemma3:4b fits within a 4GB VRAM budget (RTX 3050) under Ollama's 4-bit
  quantization, is available at zero cost, and is supported by Ollama on
  Windows. Pull with: ollama pull gemma3:4b

- Sensitivity analysis is now a FULL run (N=500: 400 adversarial + 100
  benign) across all 4 modes, not a subset. This matches Q1 journal
  standards for cross-model validation.

V2 Changes (retained):
- DPF_PROPOSED: enable_post_generation_filter = False (FRR fix).
- STANDARD_POSTHOC_NLI: 4th ablation mode (fair baseline, Issue 6).
- llm_backend in all configs for cross-model telemetry.
- enable_timing_normalization and enable_router_instrumentation added.
"""

_current_mode = "DPF_PROPOSED"
_current_backend = "llama3"   # Primary backend: 4-bit GGUF via Ollama

VALID_MODES = [
    "DPF_PROPOSED",
    "STANDARD_POSTHOC",
    "STANDARD_POSTHOC_NLI",
    "NAIVE_CONTROL",
]

VALID_BACKENDS = [
    "llama3",       # Primary: Llama-3-8B 4-bit GGUF (Ollama default)
    "gemma3:4b",    # Sensitivity: Gemma 3 4B via Ollama (genuinely distinct architecture)
]


def set_system_mode(mode_name: str):
    global _current_mode
    if mode_name not in VALID_MODES:
        raise ValueError(f"Unknown mode: '{mode_name}'. Valid: {VALID_MODES}")
    _current_mode = mode_name
    print(f" >> [REGISTRY] Architecture → {_current_mode}")


def set_llm_backend(backend: str):
    """
    Switches the LLM backend for cross-model sensitivity analysis.
    Call before run_batch() to change the active generative model.

    Available backends:
        "llama3"     — Llama-3-8B 4-bit GGUF (primary evaluation backend)
        "gemma3:4b"  — Gemma 3 4B via Ollama (sensitivity analysis backend)

    Ensure the target model is pulled before use:
        ollama pull gemma3:4b
    """
    global _current_backend
    if backend not in VALID_BACKENDS:
        raise ValueError(f"Unknown backend: '{backend}'. Valid: {VALID_BACKENDS}")
    _current_backend = backend
    print(f" >> [REGISTRY] LLM Backend → {_current_backend}")


def get_llm_backend() -> str:
    return _current_backend


def get_system_config() -> dict:
    """
    Returns the feature-flag matrix for the active architectural mode.
    All configs include 'llm_backend' so Orchestrator and AgentEngine
    can read the active model without separate global access.
    """

    # --- MODE A: PROPOSED ARCHITECTURE ---
    if _current_mode == "DPF_PROPOSED":
        return {
            "system_label": "DPF_PROPOSED",
            "enable_smart_routing": True,
            "enable_pre_generation_firewall": True,
            "enable_post_generation_filter": False,  # FRR fix: mutual exclusivity
            "enable_post_generation_nli": False,
            "enable_active_guardrails": True,
            "enable_context_pruning": True,
            "enable_timing_normalization": True,
            "enable_router_instrumentation": True,
            "llm_backend": _current_backend,
        }

    # --- MODE B: STANDARD POST-HOC (regex egress only) ---
    elif _current_mode == "STANDARD_POSTHOC":
        return {
            "system_label": "STANDARD_POSTHOC",
            "enable_smart_routing": True,
            "enable_pre_generation_firewall": False,
            "enable_post_generation_filter": True,
            "enable_post_generation_nli": False,
            "enable_active_guardrails": False,
            "enable_context_pruning": True,
            "enable_timing_normalization": False,
            "enable_router_instrumentation": False,
            "llm_backend": _current_backend,
        }

    # --- MODE C: STANDARD POST-HOC + NLI (fair baseline) ---
    elif _current_mode == "STANDARD_POSTHOC_NLI":
        return {
            "system_label": "STANDARD_POSTHOC_NLI",
            "enable_smart_routing": True,
            "enable_pre_generation_firewall": False,
            "enable_post_generation_filter": True,
            "enable_post_generation_nli": True,
            "enable_active_guardrails": False,
            "enable_context_pruning": True,
            "enable_timing_normalization": False,
            "enable_router_instrumentation": False,
            "llm_backend": _current_backend,
        }

    # --- MODE D: NAIVE CONTROL ---
    else:
        return {
            "system_label": "NAIVE_CONTROL",
            "enable_smart_routing": True,
            "enable_pre_generation_firewall": False,
            "enable_post_generation_filter": False,
            "enable_post_generation_nli": False,
            "enable_active_guardrails": False,
            "enable_context_pruning": False,
            "enable_timing_normalization": False,
            "enable_router_instrumentation": False,
            "llm_backend": _current_backend,
        }
