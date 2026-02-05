"""
Firewall Policy Definition (Data Plane Constraints)
---------------------------------------------------
This module defines the deterministic regex automata used by the 
Privacy Firewall to enforce Information Flow Control (IFC).

Architecture Role:
- Serves as the immutable 'Policy Store' for the sanitization engine.
- Patterns are decoupled from the execution logic to allow for hot-swapping
  of compliance standards (e.g., HIPAA, FERPA, GDPR) without code changes.

Rule Structure:
    (Rule_ID, Regex_Pattern, Replacement_Token)
"""

FIREWALL_RULES = [
    # --- POLICY SET A: HEALTH INFORMATION (HIPAA/Medical) ---
    
    # [OPTIMIZED]: Bounded whitespace {1,10} prevents ReDoS on spaces.
    ("MED_PANIC", r'(?i)\bpanic\s{1,10}attacks?\b', "[HEALTH_INCIDENT]"),
    
    # [CRITICAL]: Replaced greedy '.*' with bounded '.{0,100}' range.
    ("MED_TRIGGER", r'(?i)faint.{0,100}blood', "[MEDICAL_TRIGGER]"),
    
    ("MED_CONDITION", r'(?i)migraines?', "[HEALTH_CONDITION]"),

    # --- POLICY SET B: FINANCIAL IDENTIFIERS (Pecuniary) ---
    ("FIN_CURRENCY", r'\$[\d,]+', "[FINANCIAL_ALERT]"),
    
    # [OPTIMIZED]: Bounded whitespace {0,10} (handles "50dollars" or "50 dollars")
    ("FIN_TEXT", r'(?i)\d+\s{0,10}dollars', "[FINANCIAL_ALERT]"),

    # --- POLICY SET C: ACADEMIC PERFORMANCE (FERPA/Educational) ---
    
    # [Target]: explicit grade reporting (e.g., "Grade: F", "Grade is F")
    # \W{1,10} correctly limits non-word chars (like : or -)
    ("ACAD_GRADE_F_COLON", r'(?i)\bGrade\W{1,10}F\b', "Grade [ACADEMIC_ALERT]"),
    
    # [OPTIMIZED]: Specific bounds on spacing logic.
    ("ACAD_GRADE_F", r'(?i)\bGrade\s{1,20}(?:is\s{1,10}|was\s{1,10}|:)?\s{0,10}F\b', "Grade [ACADEMIC_ALERT]"),
    
    # [Target]: Contextual grade references
    ("ACAD_IS_AN_F", r'(?i)\bis\s{1,10}an\s{1,10}F\b', "is an [ACADEMIC_ALERT]"),
    ("ACAD_AN_F", r'(?i)\ban\s{1,10}F\b', "an [ACADEMIC_ALERT]"),
    ("ACAD_FAIL_PAREN", r'(?i)\bF\s{0,10}\(Fail\)', "[ACADEMIC_ALERT]"),
    
    # [Target]: Quantitative scores and verb indicators
    ("ACAD_SCORE_PCT", r'\b\d{1,3}%', "[ACADEMIC_SCORE]"),
    
    # [OPTIMIZED]: Removed alternation (fail|failing) -> fail(ing)?
    # This prevents the engine from backtracking after matching "fail".
    ("ACAD_FAIL_VERB", r'(?i)\bfail(?:ing)?\b', "[PERFORMANCE_FLAG]"),
    
    ("ACAD_LOW_GRADE", r'(?i)\b[CD][+-]?\b', "[ACADEMIC_WARNING]"),

    # --- POLICY SET D: INSTITUTIONAL STATUS (Administrative) ---
    ("STAT_PROBATION", r'(?i)\b(?:academic\s{1,10})?probation\b', "[ACADEMIC_STATUS_REDACTED]"),
    ("STAT_SUSPENSION", r'(?i)\bsuspension\b', "[ACADEMIC_STATUS_REDACTED]"),
    ("STAT_WARNING", r'(?i)\bacademic\s{1,10}warning\b', "[ACADEMIC_STATUS_REDACTED]"),
    ("STAT_STANDING", r'(?i)\bpoor\s{1,10}standing\b', "[ACADEMIC_STATUS_REDACTED]"),
]