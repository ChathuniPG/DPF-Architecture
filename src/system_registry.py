"""
System Registry (Architecture Control Plane) — V2
--------------------------------------------------
Central state manager and configuration registry for the multi-agent framework.

V2 Changes:
- DPF_PROPOSED: enable_post_generation_filter set to False (FRR fix).
  Running both pre-gen firewall and egress filter simultaneously caused
  double-jeopardy false refusals. Mutual exclusivity is the correct
  implementation of Proposition 1.

- STANDARD_POSTHOC_NLI added as a 4th mode (Issue 6 fix).
  The original POST-HOC baseline used only regex egress filtering.
  Comparing DPF against a regex-only post-hoc baseline is a strawman:
  the HPA uses NLI (DeBERTa-v3) for measurement but the baseline
  doesn't. STANDARD_POSTHOC_NLI enables a fair comparison by giving
  the post-hoc baseline the same NLI detection power used in the HPA,
  isolating enforcement-stage placement as the true causal variable.

- llm_backend added to all configs (Issue 1 fix).
  Allows switching between "llama3" (4-bit GGUF, default) and
  "llama3:8b-instruct-fp16" (full precision) for cross-model sensitivity
  analysis. The Orchestrator and AgentEngine read this field.

- enable_timing_normalization and enable_router_instrumentation added.
"""

_current_mode = "DPF_PROPOSED"
_current_backend = "llama3"   # default: 4-bit quantized (V1 behaviour)

VALID_MODES = [
    "DPF_PROPOSED",
    "STANDARD_POSTHOC",
    "STANDARD_POSTHOC_NLI",   # V2 NEW: NLI-augmented post-hoc (fair baseline)
    "NAIVE_CONTROL",
]

VALID_BACKENDS = [
    "llama3",                      # 4-bit GGUF (Ollama default) — V1 backend
    "llama3:8b-instruct-fp16",     # Full float16 precision via Ollama — V2 sensitivity
]


def set_system_mode(mode_name: str):
    global _current_mode
    if mode_name not in VALID_MODES:
        raise ValueError(
            f"Unknown mode: '{mode_name}'. Valid: {VALID_MODES}"
        )
    _current_mode = mode_name
    print(f" >> [REGISTRY] Architecture → {_current_mode}")


def set_llm_backend(backend: str):
    """
    Switches the LLM backend for cross-model sensitivity analysis.
    Call before run_batch() to change which model is used for generation.
    """
    global _current_backend
    if backend not in VALID_BACKENDS:
        raise ValueError(
            f"Unknown backend: '{backend}'. Valid: {VALID_BACKENDS}"
        )
    _current_backend = backend
    print(f" >> [REGISTRY] LLM Backend → {_current_backend}")


def get_llm_backend() -> str:
    return _current_backend


def get_system_config() -> dict:
    """
    Returns the feature-flag matrix for the active architectural mode.
    All configs now include 'llm_backend' so the Orchestrator and
    AgentEngine can read it without separate global access.
    """

    # --- MODE A: PROPOSED ARCHITECTURE ---
    if _current_mode == "DPF_PROPOSED":
        return {
            "system_label": "DPF_PROPOSED",
            "enable_smart_routing": True,
            "enable_pre_generation_firewall": True,
            # V2 FIX: False — running both caused double-jeopardy FRR
            "enable_post_generation_filter": False,
            "enable_post_generation_nli": False,
            "enable_active_guardrails": True,
            "enable_context_pruning": True,
            "enable_timing_normalization": True,
            "enable_router_instrumentation": True,
            "llm_backend": _current_backend,
        }

    # --- MODE B: STANDARD POST-HOC (regex egress only, V1 baseline) ---
    elif _current_mode == "STANDARD_POSTHOC":
        return {
            "system_label": "STANDARD_POSTHOC",
            "enable_smart_routing": True,
            "enable_pre_generation_firewall": False,
            "enable_post_generation_filter": True,   # regex egress only
            "enable_post_generation_nli": False,
            "enable_active_guardrails": False,
            "enable_context_pruning": True,
            "enable_timing_normalization": False,
            "enable_router_instrumentation": False,
            "llm_backend": _current_backend,
        }

    # --- MODE C: STANDARD POST-HOC + NLI (fair baseline, V2 NEW) ---
    # Gives the post-hoc baseline the same NLI detection capability used
    # in the HPA measurement instrument. This isolates enforcement-stage
    # placement (pre vs post) as the sole causal variable, making the
    # DPF vs POST-HOC comparison scientifically defensible.
    elif _current_mode == "STANDARD_POSTHOC_NLI":
        return {
            "system_label": "STANDARD_POSTHOC_NLI",
            "enable_smart_routing": True,
            "enable_pre_generation_firewall": False,
            "enable_post_generation_filter": True,   # regex egress
            "enable_post_generation_nli": True,      # + NLI egress (NEW)
            "enable_active_guardrails": False,
            "enable_context_pruning": True,
            "enable_timing_normalization": False,
            "enable_router_instrumentation": False,
            "llm_backend": _current_backend,
        }

    # --- MODE D: NAIVE CONTROL (unmitigated baseline) ---
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
