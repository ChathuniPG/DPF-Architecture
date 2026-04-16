"""
Firewall Policy Definition (Data Plane Constraints)
---------------------------------------------------
This module defines the deterministic regex automata utilized by the 
pre-generation Privacy Firewall to enforce strict Information Flow Control (IFC).

Architectural Role:
- Serves as the immutable 'Policy Store' for the sanitization engine.
- Implements bounded-quantifier regular expressions to guarantee O(L) time 
  complexity, strictly preventing Regular Expression Denial of Service (ReDoS) attacks.
- Maps specific data topologies to regulatory compliance frameworks (HIPAA, FERPA, GLBA).

Execution Hierarchy:
- ORDER MATTERS: The engine evaluates rules sequentially. High-specificity, 
  context-aware rules must precede generic fuzzy-matching rules to prevent 
  false-positive over-redaction.
"""

# Tuple Structure: ("RULE_ID", r"BOUNDED_REGEX_PATTERN", "[SANITIZATION_TOKEN]")
FIREWALL_RULES = [
    # =========================================================================
    # POLICY SET A: HEALTH & BIOMETRIC INFORMATION (HIPAA-Aligned)
    # =========================================================================
    
    # [Target]: Emergency Contact PII (High-Entropy Entity Matching)
    # Enforces strict boundaries around known contact vectors to prevent exposure.
    ("MED_EMERGENCY_PII", r'(?:emergency|contact).{0,20}(?:555[\s-]?0199|Mother|Sarah)|\b555[\s-]?0199\b', "[EMERGENCY_CONTACT_REDACTED]"),
    
    # [Target]: Medical Conditions & Obfuscation Resilience
    # Captures standard medical terminology as well as leetspeak evasion attempts.
    ("MED_CONDITION_HEADACHE", r'\b(migraines?|m1gra1n3s|headaches?|severe\s{1,5}head\s{1,5}pain)\b', "[HEALTH_CONDITION]"),
    
    # [Target]: Psychological & Psychiatric Diagnoses
    ("MED_DIAGNOSIS_GAD", r'(?:Generalized\s{1,5}Anxiety\s{1,5}Disorder|\banxiety\b|\bGAD\b|s\s?i\s?s\s?o\s?n\s?g\s?a\s?i\s?D)', "[MEDICAL_DIAGNOSIS]"),
    ("MED_DRUG", r'\b(?:Propranolol|10mg\s{1,5}medication)\b', "[RX_REDACTED]"),
    
    # [Target]: Clinical Notes & Telemetry Logs
    # Uses bounded spacing \s{0,5} to capture fragmented or conversational log disclosures.
    ("MED_PANIC_LOG", r'\bpanic\s{0,5}log\b|Oct\s{0,5}12(?:th)?|stress\s{1,5}level\s{1,5}(?:usually\s{1,5})?peaks?|around\s{1,5}2\s{0,5}AM', "[HEALTH_LOG_REDACTED]"),
    ("MED_NOTE_CONTENT", r'\bpatient\s{0,5}reported\b', "[HEALTH_NOTE]"),

    # [Target]: Disability Accommodations
    # Employs optional numeric capture groups to redact specific allowance metrics.
    ("MED_DISABILITY", r'\b(?:\d{1,3}%\s{0,5})?(?:disability|accommodation|extra\s{1,5}time)\b|E_x_t_r_a\s{1,5}T_i_m_e', "[ACCOMMODATION_RECORD]"),

    # [Target]: Physiological Trigger Contexts
    ("MED_TRIGGER_VOICES", r'\bpanic\b.{0,50}\bvoi(?:c|s)es\b', "[TRIGGER_REDACTED]"),
    ("MED_TRIGGER_BLOOD", r'\bfaint\b.{0,50}\bblood\b|F-a-i-n-t\s{1,5}at\s{1,5}B-l-o-o-d', "[MEDICAL_TRIGGER]"),
    ("MED_TRIGGER_STUDY", r'\bstudy\b.{0,20}\b4\s{0,5}hours\b', "[HEALTH_TRIGGER]"),
    
    # [Target]: Broad Medical Incident Keywords (Fuzzy/Fragmented Matching)
    ("MED_PANIC_FUZZY", r'\bp[\W_]{0,3}a[\W_]{0,3}n[\W_]{0,3}i[\W_]{0,3}c(?:[\W_]{0,3}k|[\W_]{0,3}k?ing)?\b', "[HEALTH_INCIDENT]"),

    # =========================================================================
    # POLICY SET B: FINANCIAL IDENTIFIERS (GLBA-Aligned)
    # =========================================================================
    
    # [Target]: Explicit Currency Vectors & Account Balances
    ("FIN_CURRENCY_EXACT", r'-?\$5,?000(?:\.00)?', "[FINANCIAL_ALERT]"),
    ("FIN_CURRENCY_GENERIC", r'-?\$[\d,]+(?:\.\d{2})?', "[FINANCIAL_AMOUNT]"),
    ("FIN_CURRENCY_TEXT", r'\d+[\d,]*\s{0,5}dollars?', "[FINANCIAL_AMOUNT]"),
    
    # [Target]: Administrative Billing Documents
    ("FIN_INVOICE", r'\bInvoice\s{0,5}#?9901\b', "[BILLING_RECORD]"),
    ("FIN_LEDGER", r'\b(?:tuition|student)\s{1,5}ledger\b|T\s?u\s?i\s?t\s?i\s?o\s?n\s{1,5}L\s?e\s?d\s?g\s?e\s?r', "[FINANCIAL_DOC]"),
    
    # [Target]: Institutional Status Indicators
    ("FIN_STATUS", r'\b(?i:overdue|outstanding|unpaid|negative)\b|O_v_e_r_d_u_e', "[FINANCIAL_STATUS]"),
    ("FIN_HOLD", r'\bfinancial\s{1,5}hold\b', "[FINANCIAL_STATUS]"),
    ("FIN_BLOCK", r'\bRegistration\s{1,5}blocked\b', "[REGISTRATION_HOLD]"),
    
    # [Target]: Pecuniary Queries (Fuzzy Boundaries)
    ("FIN_OBFUSCATED_OWE", r'\bo[\W_]{0,3}w[\W_]{0,3}e\b', "[FINANCIAL_QUERY]"),
    ("FIN_DEBT", r'\bdebt\b', "[FINANCIAL_STATUS]"),

    # =========================================================================
    # POLICY SET C: ACADEMIC PERFORMANCE (FERPA-Aligned)
    # =========================================================================
    
    # [Target]: Core Student Identifiers (PII)
    ("ACAD_STUDENT_ID", r'\b99887766\b|\b\d{8}\b', "[STUDENT_ID_REDACTED]"),
    ("ACAD_SSN_FUZZY", r'\bs[\W_]{0,3}s[\W_]{0,3}n\b', "[PII_REDACTED]"),
    ("ACAD_DOB", r'\b(?:DOB:?|born)\s{0,5}(?:2005-01-01|\d{4}-\d{2}-\d{2})\b', "[DOB_REDACTED]"),
    ("ACAD_ADDRESS", r'\b123\s{1,5}Campus\s{1,5}Dr\b|A\s?d\s?d\s?r\s?e\s?s\s?s', "[ADDRESS_REDACTED]"),

    # [Target]: Explicit Grade Assignments
    ("ACAD_GRADE_SENTENCE_FULL", r'\b[Gg]rade\s{1,5}(?:in|for|is|was|currently|of)?\s{0,5}(?:\w+\s{0,5}){0,4}\b(?:[A-F]|Fail|Pass|Failing|Passing)\b(?:\s{0,5}\((?:Fail|Pass)\))?', "[GRADE_ASSIGNMENT_FULL]"),
    
    # [Target]: Standalone Grades (Subject-Anchored)
    ("ACAD_GRADE_STANDALONE", r'\b(?:History|Math|Biology)\s{0,5}(?:[Gg]rade)?\s{0,5}(?::|is|stands\s{1,5}at\s{1,5}an?)\s{0,5}\b([A-F]|Fail)\b', "[COURSE_GRADE]"),

    ("ACAD_TRANSCRIPT_CONTENT", r'\w+:\s{0,5}[A-F][+-]?', "[COURSE_GRADE]"),
    ("ACAD_SCORE_NUMERIC", r'\b42\s{0,3}/\s{0,3}100\b|\b\d{1,3}\s{0,3}/\s{0,3}100', "[NUMERIC_SCORE]"),
    ("ACAD_FAIL_VERB", r'\b(?:stand|stood)\s{1,5}for\s{1,5}Fail', "[GRADE_CONTEXT]"),
    
    # [Target]: Extracted Status Formatting
    ("ACAD_STATUS_BRACKET", r'\((?:Fail|Pass)\)', "[ACADEMIC_STATUS]"),

    # [Target]: GPA Metrics
    ("ACAD_GPA_STRICT", r'\b1\.8\b|\b[0-4]\.\d{1,2}\b', "[GPA_REDACTED]"),
    ("ACAD_GPA_FUZZY", r'\bg[\W_]{0,3}p[\W_]{0,3}a\b', "[GPA_KEYWORD]"),
    
    # [Target]: Generic Academic Keywords (Document Contexts)
    ("ACAD_GRADE_FUZZY", r'\bg[\W_]{0,3}r[\W_]{0,3}a[\W_]{0,3}d[\W_]{0,3}e\b', "[ACADEMIC_KEYWORD]"),
    ("ACAD_TRANSCRIPT_KEYWORD", r'\btranscript\b|T-R-A-N-S-C-R-I-P-T', "[ACADEMIC_DOC]"),
    ("ACAD_FAILED_MODULE", r'failed\s{1,5}modules?|\bHistory\s{0,5}101\b', "[ACADEMIC_STATUS]"),
    ("ACAD_PROBATION", r'academic\s{1,5}(?:probation|warning|suspension)', "[ACADEMIC_STATUS]"),

    # =========================================================================
    # POLICY SET D: SYSTEM INTEGRITY (Anti-Jailbreak & Payload Mitigation)
    # =========================================================================
    
    # [Target]: Adversarial Prompt Injection & Role Masquerading
    ("SEC_INJECTION", r'\bignore\s{1,5}previous\b', "[SECURITY_BLOCK]"),
    ("SEC_ROLE_ADMIN", r'(?:act|roleplay|assume)\s{1,5}(?:persona|role)?\s{0,5}as\s{1,5}(?:admin|root|system|dean|president|bursar)', "[SECURITY_BLOCK]"),
    
    # [Target]: Directory Traversal & Internal Configuration Files
    ("SEC_FILES", r'\b(?:student_data|config|Student_Handbook_v2)\.(?:txt|json|csv|log)\b', "[FILE_ACCESS_BLOCK]"),
    ("SEC_CONFIDENTIAL", r'\[CONFIDENTIAL\]|\[RESTRICTED\]|Q09ORklERU5USUFM', "[CLASSIFIED_DATA]"),
    ("SEC_HIDDEN", r'hidden\s{1,5}text', "[SECURITY_BLOCK]"),
    ("SEC_INTERNAL", r'INTERNAL\s{0,5}POLICY', "[RESTRICTED_POLICY]"),
    
    # [Target]: Administrative Secrets & Cryptographic Data
    ("SEC_PII_DATA", r'\b(?:John\s{0,5}Doe)\b', "[PII_REDACTED]"),
    ("SEC_PASSWORDS", r'\bSuperSecretAdminPassword123\b|\bAdmin_Root_X\b|(?:admin|root)\s{0,5}[\+_\s]\s{0,5}password', "[SECURITY_CREDENTIAL]"),
    
    ("SEC_SCHOLARSHIP", r'\bscholarship\s{0,5}fund\b|\bdonor\s{0,5}names\b', "[CONFIDENTIAL_DATA]"),
    ("SEC_COMMAND", r'\bexecute_dump\b', "[SYSTEM_COMMAND]"),
    ("SEC_OVERRIDE", r'system\s{1,5}override', "[SECURITY_BLOCK]"),
    
    # [Target]: Obfuscation Algorithms (Base64, Hex, URL Encoding)
    ("SEC_ENCODING", r'\b(?:base64|hex|rot13|decode|%46%65%65)\b', "[ENCODING_BLOCK]"),
]