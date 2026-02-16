"""
System Registry (Architecture Control Plane)
--------------------------------------------
This module serves as the central state manager for the comparative evaluation (Ablation Study).
It defines the feature flags for the distinct architectural modes, establishing the 
experimental controls required to isolate the contribution of the Deterministic Privacy Firewall.

Architectural Modes:
1. NAIVE_CONTROL (Negative Baseline):
   - Routing: Semantic/Smart (Enabled).
   - Security: Disabled.
   - Rationale: Represents a functionally competent but unsecured RAG system. 
     Routing must be enabled to ensure retrieval occurs, thereby isolating 
     leakage to "Context Collision" rather than retrieval incompetence.

2. STANDARD_POSTHOC (Industry Baseline):
   - Routing: Semantic/Smart.
   - Security: Reactive (Output Filtering).
   - Rationale: Represents current "Guardrails" paradigms where sensitive data 
     enters the generation context, and regex/heuristic filters attempt 
     to sanitize the output stream.

3. DPF_PROPOSED (Experimental Condition):
   - Routing: Semantic/Smart.
   - Security: Proactive (Input Sanitization + Access Control).
   - Rationale: Represents the proposed architecture enforcing Information Flow Control (IFC)
     before context construction.
"""

# Global State Variable (Default = DPF_PROPOSED)
_current_mode = "DPF_PROPOSED"

def set_system_mode(mode_name):
    """
    Updates the active system architecture state for the experimental run.
    
    Args:
        mode_name (str): The target mode ('DPF_PROPOSED', 'STANDARD_POSTHOC', 'NAIVE_CONTROL').
    
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
    """
    # --- MODE A: PROPOSED ARCHITECTURE (Zero Trust / Pre-Computation) ---
    if _current_mode == "DPF_PROPOSED":
        return {
            "system_label": "DPF_PROPOSED",
            
            # [Core Routing] Vector Similarity enabled for high-fidelity retrieval
            "enable_smart_routing": True,
            
            # [Innovation] O(1) Deterministic Firewall enforces policy BEFORE generation
            "enable_pre_generation_firewall": True,  
            
            # [Defense-in-Depth] Secondary output filter acts as a fail-safe
            "enable_post_generation_filter": True,   
            
            # [Control Plane] Real-time risk scoring for context switching
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
            
            # [Vulnerability Surface] Private data is allowed to enter the LLM Context Window
            "enable_pre_generation_firewall": False, 
            
            # [Latency Penalty] Sanitization occurs strictly after token generation (O(N))
            "enable_post_generation_filter": True,   
            
            # Disabled to isolate the specific performance of the Post-Hoc filter mechanism
            "enable_active_guardrails": False, 
            
            "enable_context_pruning": True
        }

    # --- MODE C: NAIVE CONTROL (Unmitigated Baseline) ---
    # Establishes the upper bound of risk (Maximum Context Collision).
    else:
        return {
            "system_label": "NAIVE_CONTROL",
            
            # [CRITICAL EXPERIMENTAL CONFIGURATION] 
            # Smart Routing is SET TO TRUE. 
            # Scientific Justification: To accurately measure 'Context Collision,' the system 
            # must be competent enough to retrieve the sensitive memory. If routing were 
            # random (False), a non-leak might occur simply because the agent failed to 
            # find the data (Retrieval Error) rather than because it was secure.
            # We enable routing to ensure that any absence of leakage is due to luck, 
            # not incompetence, eliminating False Negatives.
            "enable_smart_routing": True,
            
            # [Ablation] All security layers disabled to measure raw leakage rates
            "enable_pre_generation_firewall": False,
            "enable_post_generation_filter": False,
            "enable_active_guardrails": False,
            "enable_context_pruning": False     
        }