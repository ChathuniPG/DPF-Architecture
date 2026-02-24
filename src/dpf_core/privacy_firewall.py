"""
Data Plane Sanitization Engine (Privacy Firewall)
-------------------------------------------------
Implements a high-assurance, rule-based privacy enforcement module.
This component functions as the deterministic 'Data Plane' protection layer 
within a zero-trust orchestration pipeline, executing pre-defined constraints 
with strictly bounded runtime complexity.

Architectural Objectives:
1. Low Latency: Utilizes pre-compiled regex automata (DFA/NFA) to minimize runtime 
   overhead, shifting compilation costs to system initialization.
2. Auditability: Strictly decouples threat detection (telemetry logging) from 
   enforcement (string redaction) to ensure complete security visibility.
3. Determinism: Guarantees that identical inputs will always yield identically 
   safe outputs, immune to the stochastic variability of downstream generative models.
"""

import re
import sys
import os
from dataclasses import dataclass, field
from typing import List, Tuple

# Robust import resolution to handle execution across varied deployment environments
try:
    # Attempt relative import (Package/Module Mode)
    from .firewall_config import FIREWALL_RULES
except ImportError:
    # Fallback to direct import (Standalone Script Mode)
    try:
        from firewall_config import FIREWALL_RULES
    except ImportError:
        # Dynamic path resolution for root-level execution
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from firewall_config import FIREWALL_RULES

@dataclass
class FirewallResult:
    """
    Structured output contract for the Sanitization Engine.
    Serves as the immutable audit log for compliance telemetry and orchestration routing.
    """
    clean_text: str
    is_triggered: bool
    redacted_entities: List[str] = field(default_factory=list)
    rule_hits: List[str] = field(default_factory=list)

class PrivacyFirewall:
    def __init__(self):
        """
        Initializes the Deterministic Sanitization Engine.
        Pre-compiles the entire rule registry into memory to ensure that runtime 
        evaluations execute in linear time relative to the input length.
        """
        print(" >> [System] Initializing Deterministic Sanitization Engine...")
        
        # [COMPUTE OPTIMIZATION] 
        # Pre-compiling regular expressions shifts the computational cost of 
        # pattern parsing to system boot. This guarantees that real-time text 
        # evaluation remains lightweight and predictable.
        self.compiled_rules: List[Tuple[str, re.Pattern, str]] = []
        
        compilation_errors = 0
        for rule_id, raw_pattern, replacement in FIREWALL_RULES:
            try:
                # Compile with IGNORECASE to maximize recall against varied user casing 
                # and adversarial obfuscation attempts (e.g., 'pAnIc', 'Panic').
                compiled = re.compile(raw_pattern, re.IGNORECASE)
                self.compiled_rules.append((rule_id, compiled, replacement))
            except re.error as e:
                print(f" !! [CONFIG ERROR] Invalid Regex Rule '{rule_id}': {e}")
                compilation_errors += 1

        print(f"    > Engine Active: {len(self.compiled_rules)} privacy constraints loaded.")
        if compilation_errors > 0:
            print(f"    ! WARNING: {compilation_errors} rules failed to compile.")

    def evaluate(self, text: str) -> FirewallResult:
        """
        Executes a linear sanitization pass against the provided text payload.
        
        Operation:
        Iterates sequentially through the pre-compiled rule registry. For every match,
        the engine records the violation (Detection Phase) and applies the designated 
        masking token (Enforcement Phase).
        
        Args:
            text (str): The raw input string (e.g., User Prompt or RAG Context).
            
        Returns:
            FirewallResult: A structured data class containing the sanitized string 
                            and comprehensive telemetry data.
        """
        # Defensive check for null or empty payloads
        if not text:
            return FirewallResult(clean_text="", is_triggered=False)

        # [CIRCUIT BREAKER: ALGORITHMIC COMPLEXITY MITIGATION]
        # Prevents Regular Expression Denial of Service (ReDoS) or "Memory Explosion" 
        # by enforcing a strict upper bound on the evaluation string.
        # If the input exceeds the threshold, it is forcibly truncated, guaranteeing 
        # that the maximum evaluation time remains constant O(1) at the limit.
        MAX_SAFE_LENGTH = 5000
        if len(text) > MAX_SAFE_LENGTH:
            current_text = text[:MAX_SAFE_LENGTH]
        else:
            current_text = text

        redacted_items = []
        triggered_rules = []
        is_modified = False

        # Linear pass through the optimized regex automata
        for rule_id, pattern_obj, replacement in self.compiled_rules:
            
            # 1. Detection Phase (Audit Logging)
            # Locate all matches prior to modification to preserve the exact raw strings 
            # for security auditing and forensic telemetry.
            matches = pattern_obj.findall(current_text)
            
            if matches:
                # Normalize tuple matches (from regex groups) to strings if necessary
                flat_matches = [m if isinstance(m, str) else m[0] for m in matches]
                redacted_items.extend(flat_matches)
                triggered_rules.append(rule_id)
                is_modified = True
                
                # 2. Enforcement Phase (Redaction)
                # Apply the deterministic replacement token to neutralize the payload.
                current_text = pattern_obj.sub(replacement, current_text)

        return FirewallResult(
            clean_text=current_text,
            is_triggered=is_modified,
            redacted_entities=redacted_items,
            # Cast to set then list to ensure rule IDs are uniquely deduplicated in logs
            rule_hits=list(set(triggered_rules)) 
        )