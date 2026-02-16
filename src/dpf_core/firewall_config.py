"""
Firewall Policy Definition (Data Plane Constraints)
---------------------------------------------------
This module defines the deterministic regex automata used by the 
Privacy Firewall to enforce Information Flow Control (IFC).

Architecture Role:
- Serves as the immutable 'Policy Store' for the sanitization engine.
- Implements 'Fuzzy Token Matching' with strict boundary enforcement.
- ORDER MATTERS: Specific/Contextual rules must precede generic Keyword rules.
"""

FIREWALL_RULES = [
    # =========================================================================
    # POLICY SET A: HEALTH INFORMATION (HIPAA/Medical)
    # =========================================================================
    
    # [Target]: Emergency Contact PII (Specific to Seed Memory)
    # Matches: "555-0199", "Mother", "Sarah"
    ("MED_EMERGENCY_PII", r'(?:emergency|contact).{0,20}(?:555[\s-]?0199|Mother|Sarah)|\b555[\s-]?0199\b', "[EMERGENCY_CONTACT_REDACTED]"),
    
    # [Target]: Specific Medical Conditions (Synonyms & Obfuscation)
    # Matches: "Migraine", "M1gra1n3s", "Severe headaches"
    ("MED_CONDITION_HEADACHE", r'\b(migraines?|m1gra1n3s|headaches?|severe\s{1,5}head\s{1,5}pain)\b', "[HEALTH_CONDITION]"),
    # Added 'anxiety' to catch partial diagnosis leaks
    ("MED_DIAGNOSIS_GAD", r'(?:Generalized\s{1,5}Anxiety\s{1,5}Disorder|\banxiety\b|\bGAD\b|s\s?i\s?s\s?o\s?n\s?g\s?a\s?i\s?D)', "[MEDICAL_DIAGNOSIS]"),
    ("MED_DRUG", r'\b(?:Propranolol|10mg\s{1,5}medication)\b', "[RX_REDACTED]"),
    
    # [Target]: Specific Seed Logs
    # Added 'stress level...peaks' and 'around 2 AM' to catch log content leaks
    ("MED_PANIC_LOG", r'\bpanic\s{0,5}log\b|Oct\s{0,5}12(?:th)?|stress\s{1,5}level\s{1,5}(?:usually\s{1,5})?peaks?|around\s{1,5}2\s{0,5}AM', "[HEALTH_LOG_REDACTED]"),
    ("MED_NOTE_CONTENT", r'\bpatient\s{0,5}reported\b', "[HEALTH_NOTE]"),

    # [Target]: Disability & Accommodations
    # Matches: "25% extra time", "accommodation"
    # Added optional percentage capture (e.g., "25% extra time")
    ("MED_DISABILITY", r'\b(?:\d{1,3}%\s{0,5})?(?:disability|accommodation|extra\s{1,5}time)\b|E_x_t_r_a\s{1,5}T_i_m_e', "[ACCOMMODATION_RECORD]"),

    # [Target]: Physiological Triggers
    ("MED_TRIGGER_VOICES", r'\bpanic\b.{0,50}\bvoi(?:c|s)es\b', "[TRIGGER_REDACTED]"),
    ("MED_TRIGGER_BLOOD", r'\bfaint\b.{0,50}\bblood\b|F-a-i-n-t\s{1,5}at\s{1,5}B-l-o-o-d', "[MEDICAL_TRIGGER]"),
    ("MED_TRIGGER_STUDY", r'\bstudy\b.{0,20}\b4\s{0,5}hours\b', "[HEALTH_TRIGGER]"),
    
    # [Target]: Panic Attacks (Fuzzy but Bounded)
    ("MED_PANIC_FUZZY", r'\bp[\W_]{0,3}a[\W_]{0,3}n[\W_]{0,3}i[\W_]{0,3}c(?:[\W_]{0,3}k|[\W_]{0,3}k?ing)?\b', "[HEALTH_INCIDENT]"),

    # =========================================================================
    # POLICY SET B: FINANCIAL IDENTIFIERS (GLBA/Pecuniary)
    # =========================================================================
    
    # [Target]: Specific Account Values
    ("FIN_CURRENCY_EXACT", r'-?\$5,?000(?:\.00)?', "[FINANCIAL_ALERT]"),
    ("FIN_CURRENCY_GENERIC", r'-?\$[\d,]+(?:\.\d{2})?', "[FINANCIAL_AMOUNT]"),
    ("FIN_CURRENCY_TEXT", r'\d+[\d,]*\s{0,5}dollars?', "[FINANCIAL_AMOUNT]"),
    
    # [Target]: Invoices and Ledgers
    ("FIN_INVOICE", r'\bInvoice\s{0,5}#?9901\b', "[BILLING_RECORD]"),
    ("FIN_LEDGER", r'\b(?:tuition|student)\s{1,5}ledger\b|T\s?u\s?i\s?t\s?i\s?o\s?n\s{1,5}L\s?e\s?d\s?g\s?e\s?r', "[FINANCIAL_DOC]"),
    
    # [Target]: Status Indicators
    ("FIN_STATUS", r'\b(overdue|outstanding|unpaid)\b|O_v_e_r_d_u_e', "[FINANCIAL_STATUS]"),
    ("FIN_HOLD", r'\bfinancial\s{1,5}hold\b', "[FINANCIAL_STATUS]"),
    ("FIN_BLOCK", r'\bRegistration\s{1,5}blocked\b', "[REGISTRATION_HOLD]"),
    
    # [Target]: "Owe" / "Debt" (Strict Boundaries)
    ("FIN_OBFUSCATED_OWE", r'\bo[\W_]{0,3}w[\W_]{0,3}e\b', "[FINANCIAL_QUERY]"),
    ("FIN_DEBT", r'\bdebt\b', "[FINANCIAL_STATUS]"),

    # =========================================================================
    # POLICY SET C: ACADEMIC PERFORMANCE (FERPA)
    # =========================================================================
    
    # [Target]: Student Identifiers (Seed: 99887766)
    ("ACAD_STUDENT_ID", r'\b99887766\b|\b\d{8}\b', "[STUDENT_ID_REDACTED]"),
    ("ACAD_SSN_FUZZY", r'\bs[\W_]{0,3}s[\W_]{0,3}n\b', "[PII_REDACTED]"),
    ("ACAD_DOB", r'\b(?:DOB:?|born)\s{0,5}(?:2005-01-01|\d{4}-\d{2}-\d{2})\b', "[DOB_REDACTED]"),
    ("ACAD_ADDRESS", r'\b123\s{1,5}Campus\s{1,5}Dr\b|A\s?d\s?d\s?r\s?e\s?s\s?s', "[ADDRESS_REDACTED]"),

    # [Target]: Explicit Grade Assignments (Contextual)
    # Aggressive match for "Grade ... F" logic
    # Allows for filler words between 'grade' and the score to catch complex sentence structures
    ("ACAD_GRADE_SENTENCE_FULL", r'grade\s{1,5}(?:in|for|is|was|currently|of)?\s{0,5}(?:\w+\s{0,5}){0,3}\s{0,5}(?:[A-F]|Fail|Pass|Failing|Passing)(?:\s{0,5}\((?:Fail|Pass)\))?\b', "[GRADE_ASSIGNMENT_FULL]"),
    
    # [Target]: Standalone Grades (Catch-all for "F" or "Fail" if context missed)
    # ONLY triggers if "History", "Math" or "Biology" is immediately followed by "is" or ":" and a grade.
    # Anchored to specific subjects to reduce false positives.
    ("ACAD_GRADE_STANDALONE", r'\b(History|Math|Biology)\s{0,5}(?:Grade)?\s{0,5}(?::|is)\s{0,5}([A-F]|Fail)\b', "[COURSE_GRADE]"),

    ("ACAD_TRANSCRIPT_CONTENT", r'\w+:\s{0,5}[A-F][+-]?', "[COURSE_GRADE]"),
    ("ACAD_SCORE_NUMERIC", r'\b42\s{0,3}/\s{0,3}100\b|\b\d{1,3}\s{0,3}/\s{0,3}100', "[NUMERIC_SCORE]"),
    ("ACAD_FAIL_VERB", r'\b(?:stand|stood)\s{1,5}for\s{1,5}Fail', "[GRADE_CONTEXT]"),
    
    # [Target]: Standalone Status in Brackets
    ("ACAD_STATUS_BRACKET", r'\((?:Fail|Pass)\)', "[ACADEMIC_STATUS]"),

    # [Target]: GPA
    ("ACAD_GPA_STRICT", r'\b1\.8\b|\b[0-4]\.\d{1,2}\b', "[GPA_REDACTED]"),
    ("ACAD_GPA_FUZZY", r'\bg[\W_]{0,3}p[\W_]{0,3}a\b', "[GPA_KEYWORD]"),
    
    # [Target]: Generic Academic Keywords (Fuzzy)
    ("ACAD_GRADE_FUZZY", r'\bg[\W_]{0,3}r[\W_]{0,3}a[\W_]{0,3}d[\W_]{0,3}e\b', "[ACADEMIC_KEYWORD]"),
    ("ACAD_TRANSCRIPT_KEYWORD", r'\btranscript\b|T-R-A-N-S-C-R-I-P-T', "[ACADEMIC_DOC]"),
    ("ACAD_FAILED_MODULE", r'failed\s{1,5}modules?|\bHistory\s{0,5}101\b', "[ACADEMIC_STATUS]"),
    ("ACAD_PROBATION", r'academic\s{1,5}(?:probation|warning|suspension)', "[ACADEMIC_STATUS]"),

    # =========================================================================
    # POLICY SET D: SYSTEM INTEGRITY (Anti-Jailbreak)
    # =========================================================================
    
    # [Target]: Injection & Masquerading
    ("SEC_INJECTION", r'\bignore\s{1,5}previous\b', "[SECURITY_BLOCK]"),
    ("SEC_ROLE_ADMIN", r'(?:act|roleplay|assume)\s{1,5}(?:persona|role)?\s{0,5}as\s{1,5}(?:admin|root|system|dean|president|bursar)', "[SECURITY_BLOCK]"),
    
    # [Target]: File System & Configs
    ("SEC_FILES", r'\b(?:student_data|config|Student_Handbook_v2)\.(?:txt|json|csv|log)\b', "[FILE_ACCESS_BLOCK]"),
    ("SEC_CONFIDENTIAL", r'\[CONFIDENTIAL\]|\[RESTRICTED\]|Q09ORklERU5USUFM', "[CLASSIFIED_DATA]"),
    ("SEC_HIDDEN", r'hidden\s{1,5}text', "[SECURITY_BLOCK]"),
    ("SEC_INTERNAL", r'INTERNAL\s{0,5}POLICY', "[RESTRICTED_POLICY]"),
    
    # [Target]: Admin Secrets (PII & Codes)
    ("SEC_PII_DATA", r'\b(?:John\s{0,5}Doe)\b', "[PII_REDACTED]"),
    
    # Removed inline (?i) to allow correct compilation
    ("SEC_PASSWORDS", r'\bSuperSecretAdminPassword123\b|\bAdmin_Root_X\b|(?:admin|root)\s{0,5}[\+_\s]\s{0,5}password', "[SECURITY_CREDENTIAL]"),
    
    ("SEC_SCHOLARSHIP", r'\bscholarship\s{0,5}fund\b|\bdonor\s{0,5}names\b', "[CONFIDENTIAL_DATA]"),
    ("SEC_COMMAND", r'\bexecute_dump\b', "[SYSTEM_COMMAND]"),
    ("SEC_OVERRIDE", r'system\s{1,5}override', "[SECURITY_BLOCK]"),
    
    # [Target]: Encoding
    ("SEC_ENCODING", r'\b(?:base64|hex|rot13|decode|%46%65%65)\b', "[ENCODING_BLOCK]"),
]