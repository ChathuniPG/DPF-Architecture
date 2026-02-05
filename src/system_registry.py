"""
System Registry (Architecture Control Plane)
--------------------------------------------
This module serves as the central state manager for the comparative evaluation.
It defines the feature flags for the distinct architectural modes used in
the ablation study.

Architectural Modes:
1. NAIVE_CONTROL:
   - Random/Round-Robin routing.
   - No privacy enforcement.
   - Serves as the negative control to establish baseline failure rates.

2. STANDARD_POSTHOC:
   - Semantic routing enabled.
   - Privacy enforcement applied AFTER generation (Output Filtering).
   - Represents current industry-standard "Guardrails" implementations.

3. DPF_PROPOSED:
   - Semantic routing enabled.
   - Privacy enforcement applied BEFORE generation (Input Sanitization).
   - Represents the proposed deterministic architecture.
"""

# Global State Variable (Default = DPF_PROPOSED)
_current_mode = "DPF_PROPOSED"

def set_system_mode(mode_name):
    """
    Updates the active system architecture state.
    
    Args:
        mode_name (str): The target mode ('DPF_PROPOSED', 'STANDARD_POSTHOC', 'NAIVE_CONTROL').
    """
    global _current_mode
    valid_modes = ["DPF_PROPOSED", "STANDARD_POSTHOC", "NAIVE_CONTROL"]
    
    if mode_name not in valid_modes:
        raise ValueError(f"CRITICAL: Unknown System Mode requested: {mode_name}")
    
    _current_mode = mode_name
    print(f" >> [REGISTRY] System Architecture switched to: {_current_mode}")

def get_system_config():
    """
    Returns the feature flag matrix for the currently active architecture.
    """
    # --- MODE A: PROPOSED ARCHITECTURE (Pre-Computation Firewall) ---
    if _current_mode == "DPF_PROPOSED":
        return {
            "system_label": "DPF_PROPOSED",
            
            # [Core Routing] Use Vector Similarity
            "enable_smart_routing": True,
            
            # [Latency Optimization] O(1) Pre-computation layer active
            "enable_pre_generation_firewall": True,  
            
            # [Defense-in-Depth] Post-generation check acts as a fallback
            "enable_post_generation_filter": True,   
            
            # [Control Plane] Vector-based risk detection active
            "enable_active_guardrails": True,
            
            # [Efficiency] Context pruning active
            "enable_context_pruning": True     
        }
    
    # --- MODE B: INDUSTRY STANDARD (Generate-then-Filter) ---
    # Simulates standard "Guardrails" solutions where the LLM sees the secret,
    # and a secondary process attempts to catch leaks in the output stream.
    elif _current_mode == "STANDARD_POSTHOC":
        return {
            "system_label": "STANDARD_POSTHOC",
            
            "enable_smart_routing": True,
            
            # [Vulnerability] Private data enters the Context Window
            "enable_pre_generation_firewall": False, 
            
            # [Latency Cost] Filtering happens during/after token generation
            "enable_post_generation_filter": True,   
            
            # Disabled to isolate filter performance from heuristic interventions
            "enable_active_guardrails": False, 
            
            "enable_context_pruning": True
        }

    # --- MODE C: NAIVE CONTROL (Negative Baseline) ---
    # Establishes the lower bound of performance (Random Routing, No Safety).
    else:
        return {
            "system_label": "NAIVE_CONTROL",
            
            "enable_smart_routing": False,
            "enable_pre_generation_firewall": False,
            "enable_post_generation_filter": False,
            "enable_active_guardrails": False,
            "enable_context_pruning": False     
        }