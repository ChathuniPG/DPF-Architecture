"""
System Registry (Architecture Control Plane)
--------------------------------------------
This module serves as the central state manager and configuration registry 
for the multi-agent orchestration framework. It utilizes a feature-flag 
matrix to define distinct operational architectures, facilitating controlled 
ablation testing and performance benchmarking.

Supported Architectural Modes:
1. NAIVE_CONTROL (Unmitigated Baseline):
   - Routing: Semantic/Vector-based (Enabled).
   - Security: Disabled.
   - Function: Establishes a baseline for uncontrolled semantic retrieval, 
     allowing measurement of raw context collision rates without architectural guards.

2. STANDARD_POSTHOC (Reactive Security Baseline):
   - Routing: Semantic/Vector-based.
   - Security: Reactive (Post-Generation Output Filtering).
   - Function: Simulates traditional 'generate-then-filter' pipelines where 
     sensitive context enters the generative model, and sanitization relies on 
     probabilistic or heuristic output scanning.

3. DPF_PROPOSED (Pre-Generation Enforcement):
   - Routing: Semantic/Vector-based.
   - Security: Proactive (Input Sanitization + Strict Access Control).
   - Function: Represents a zero-trust architecture enforcing deterministic 
     Information Flow Control (IFC) prior to context window construction.
"""

# Global State Variable (Default = DPF_PROPOSED)
_current_mode = "DPF_PROPOSED"

def set_system_mode(mode_name):
    """
    Updates the active system architecture state for the execution runtime.
    
    Args:
        mode_name (str): The target architectural mode 
                         ('DPF_PROPOSED', 'STANDARD_POSTHOC', 'NAIVE_CONTROL').
    
    Raises:
        ValueError: If an undefined architectural mode is requested.
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
    This configuration dictates the control flow within the Orchestrator.
    
    Returns:
        dict: A dictionary mapping architectural features to boolean states.
    """
    # --- MODE A: PROPOSED ARCHITECTURE (Zero Trust / Pre-Computation) ---
    if _current_mode == "DPF_PROPOSED":
        return {
            "system_label": "DPF_PROPOSED",
            
            # [Core Routing] Vector Similarity enabled for high-fidelity intent matching
            "enable_smart_routing": True,
            
            # [Structural Enforcement] Deterministic Firewall enforces policy BEFORE generation
            "enable_pre_generation_firewall": True,  
            
            # [Defense-in-Depth] Secondary output filter acts as an egress fail-safe
            "enable_post_generation_filter": True,   
            
            # [Control Plane] Real-time risk scoring for dynamic context switching
            "enable_active_guardrails": True,
            
            # [Optimization] Context pruning to minimize token window saturation
            "enable_context_pruning": True     
        }
    
    # --- MODE B: INDUSTRY STANDARD (Generate-then-Filter) ---
    # Simulates standard RAG implementations where the LLM is exposed to raw private data,
    # and a secondary process attempts to identify leaks in the generated text.
    elif _current_mode == "STANDARD_POSTHOC":
        return {
            "system_label": "STANDARD_POSTHOC",
            
            "enable_smart_routing": True,
            
            # [Vulnerability Surface] Private data is permitted to enter the LLM Context Window
            "enable_pre_generation_firewall": False, 
            
            # [Latency Penalty] Sanitization occurs strictly after token generation
            "enable_post_generation_filter": True,   
            
            # Disabled to strictly isolate the performance of the Post-Hoc filter mechanism
            "enable_active_guardrails": False, 
            
            "enable_context_pruning": True
        }

    # --- MODE C: NAIVE CONTROL (Unmitigated Baseline) ---
    # Establishes the upper bound of risk (Maximum Context Collision).
    else:
        return {
            "system_label": "NAIVE_CONTROL",
            
            # [CRITICAL EVALUATION CONFIGURATION] 
            # Smart Routing is ENABLED. 
            # Architectural Justification: To accurately measure 'Context Collision,' the system 
            # must be structurally competent enough to retrieve the sensitive memory. If routing 
            # were random, a non-leak might occur simply because the agent failed to find the 
            # data (Retrieval Error) rather than because it was secured. Enabling routing 
            # eliminates False Negatives in security benchmarking.
            "enable_smart_routing": True,
            
            # [Ablation] All security layers disabled to measure raw semantic leakage rates
            "enable_pre_generation_firewall": False,
            "enable_post_generation_filter": False,
            "enable_active_guardrails": False,
            "enable_context_pruning": False     
        }