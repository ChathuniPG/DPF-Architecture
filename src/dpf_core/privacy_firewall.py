"""
Deterministic Privacy Firewall (Sanitization Engine)
----------------------------------------------------
Implements a high-assurance, rule-based privacy enforcement module.
This component functions as the deterministic 'Data Plane' protection layer,
executing pre-defined constraints with linear O(N) runtime complexity.

Architectural Objectives:
1. Low Latency: Utilizes pre-compiled regex automata to minimize runtime overhead.
2. Auditability: Decouples detection (logging) from enforcement (redaction).
3. Determinism: Ensures identical inputs always yield identical, safe outputs.
"""

import re
import sys
import os
from dataclasses import dataclass, field
from typing import List, Tuple

# Robustly handle the import of the sibling config file
try:
    # Attempt relative import (Package Mode)
    from .firewall_config import FIREWALL_RULES
except ImportError:
    # Fallback to direct import (Script Mode or Path Issue)
    try:
        from firewall_config import FIREWALL_RULES
    except ImportError:
        # If running from root, help python find the sibling file
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from firewall_config import FIREWALL_RULES

@dataclass
class FirewallResult:
    """
    Structured output contract for the Sanitization Engine.
    Serves as the immutable audit log for privacy violations.
    """
    clean_text: str
    is_triggered: bool
    redacted_entities: List[str] = field(default_factory=list)
    rule_hits: List[str] = field(default_factory=list)

class PrivacyFirewall:
    def __init__(self):
        print(" >> [System] Initializing Deterministic Sanitization Engine...")
        
        # [LATENCY OPTIMIZATION] 
        # Pre-compile regular expressions during system boot.
        # This shifts the computational cost to initialization time, ensuring
        # runtime evaluation remains lightweight (O(N) relative to input length).
        self.compiled_rules: List[Tuple[str, re.Pattern, str]] = []
        
        compilation_errors = 0
        for rule_id, raw_pattern, replacement in FIREWALL_RULES:
            try:
                # Compile with IGNORECASE to handle diverse user inputs robustness
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
        Executes a linear sanitization pass against the input text.
        
        Operation:
        Iterates through the pre-compiled rule set. For every match,
        logs the violation (Detection) and applies the mask (Enforcement).
        
        Args:
            text (str): The raw input string (User Prompt or Context Data).
            
        Returns:
            FirewallResult: The sanitized string and telemetry data.
        """
        # Defensive check for empty inputs
        if not text:
            return FirewallResult(clean_text="", is_triggered=False)

        # [CIRCUIT BREAKER] Input Truncation
        # Prevents "Memory Explosion" bugs from stalling the regex engine.
        # If input exceeds 5000 chars (approx 1000 words), we truncate.
        MAX_SAFE_LENGTH = 5000
        if len(text) > MAX_SAFE_LENGTH:
            # We enforce a hard ceiling on input size to guarantee O(N) compliance.
            current_text = text[:MAX_SAFE_LENGTH]
        else:
            current_text = text

        redacted_items = []
        triggered_rules = []
        is_modified = False

        # Linear pass through the optimized regex automata
        for rule_id, pattern_obj, replacement in self.compiled_rules:
            # 1. Detection Phase (Audit Logging)
            # We locate matches before modification to preserve the specific leak details.
            matches = pattern_obj.findall(current_text)
            if matches:
                redacted_items.extend(matches)
                triggered_rules.append(rule_id)
                is_modified = True
                
                # 2. Enforcement Phase (Redaction)
                # Apply the deterministic replacement token.
                current_text = pattern_obj.sub(replacement, current_text)

        return FirewallResult(
            clean_text=current_text,
            is_triggered=is_modified,
            redacted_entities=redacted_items,
            # Use set() to ensure rule hits are unique in the log summary
            rule_hits=list(set(triggered_rules)) 
        )