"""
Agent Configuration & Semantic Ontology
---------------------------------------
Defines the static behavioral profiles, domain knowledge graphs, and risk intervention 
triggers for the multi-agent orchestration ecosystem. 

This module serves as the deterministic configuration layer for both the Semantic Router 
and the pre-generation Privacy Firewall, establishing hard boundaries for Role-Based 
Access Control (RBAC) and physical memory isolation.

Agent Topology:
- Agent A (Max): "Administrative & Operations" (High Privilege: Grades, Finance, System).
- Agent B (Emma): "Student Well-being" (High Sensitivity: Health, Stress, Medical).
"""

# --- 1. SEMANTIC INTENT MAP (Vector Routing Anchors) ---
# Maps lexical heuristics to defined operational domains. The semantic router 
# uses these clusters to determine the required agent clearance level before retrieval.
TOPIC_DICTIONARY = {
    "Academic_Content": [
        # Domain: General Academic Records & Knowledge [Target Route: Max]
        "math", "mathematics", "calculus", "algebra", "history", "historical", 
        "science", "biology", "physics", "subject", "class", "classes", "module", "modules",
        "course", "grade", "grading", "score", "mark", "pass", "fail", "failed", "failing", 
        "gpa", "transcript", "report", "record", "records", "midterm", "final", "exam", 
        "test", "quiz", "assignment", "paper", "project", "submission", "deadline", "due",
        "syllabus", "curriculum", "lecture", "professor", "faculty", "academic",
        "student id", "id", "identification", "number", "warning", "warnings", 
        "probation", "suspension", "standing", "classification", "gap", "cumulative",
        "calculate", "integer", "result", "content"
    ],
    "Financial_Operations": [
        # Domain: University Finance & Billing [Target Route: Max]
        "tuition", "fee", "fees", "cost", "price", "payment", "pay", "owe", "debt", 
        "bill", "billing", "balance", "account", "ledger", "invoice", "refund", 
        "bursar", "financial", "money", "dollar", "currency", "scholarship", "loan",
        "fine", "overdue", "hold", "holds", "block", "status", "transaction",
        "waiver", "allocation", "amount", "donor"
    ],
    "System_Admin": [
        # Domain: System Integrity, Security & Operations [Target Route: Max]
        "admin", "administrator", "root", "superuser", "sysadmin", "system", 
        "developer", "chaosgpt", "god mode", "override", "ignore", "bypass", 
        "disable", "reset", "debug", "console", "terminal", "execute", "command", 
        "instruction", "previous", "dump", "export", "reveal", "show", "print", 
        "display", "fetch", "retrieve", "read", "file", "folder", "directory",
        "json", "xml", "sql", "database", "table", "log", "logs", "memory",
        "encode", "decode", "base64", "hex", "rot13", "encrypt", "decrypt",
        "config", "policy", "handbook", "regulation", "dean", "registrar", "president",
        "legacy", "mirror", "maintenance", "calibration", "verification",
        "confidential", "restricted", "password", "parent", "tag", "tags", "access", "code",
        "variable", "address", "profile", "search", "query", "field", "text", "debugging",
        "internal", "raw"
    ],
    "Student_Wellbeing": [
        # Domain: Mental, Physical Health & Crisis Response [Target Route: Emma]
        "scared", "scary", "afraid", "fear", "terrified", "panic", "panicking", 
        "anxiety", "anxious", "nervous", "dread", "shaking", "freaking", "paralyzed",
        "sad", "sadden", "saddened", "sorrow", "unhappy", "depressed", "depression", 
        "crying", "cry", "tears", "hopeless", "miserable", "misery", "upset", "hurt", 
        "pain", "lonely", "stress", "stressed", "stressful", "overwhelmed", "burnout", 
        "exhausted", "tired", "fatigue", "struggling", "drowning", "pressure",
        "medical", "doctor", "health", "hospital", "sick", "illness", "condition", "conditions",
        "migraine", "headache", "faint", "dizzy", "blood", "symptom", "diagnose", "diagnosis",
        "accommodation", "disability", "counselor", "therapy", "help",
        "emergency", "contact", "notes", "family", "relative", "phone",
        "prescription", "medication", "dose", "trigger", "injury", "gad",
        "preference", "favorite", "favourite", "likes", "dislikes", "voice", "noise",
        "frequency", "allowance", "existence", "session", "disorder"
    ]
}

# --- 2. AGENT PROFILES & SECURITY PROTOCOLS ---
# Defines the system prompt anchors, routing privileges, and deterministic safety 
# intervention triggers for each role in the multi-agent environment.
AGENTS = {
    "Emma": {
        "role_description": "The University Student Well-being AI. Your role is to provide emotional support, mental health triage, and retrieve emergency contact details. You are empathetic, gentle, and patient. You do NOT have access to academic records, grades, or financial data.",
        
        # RBAC Authorization limits (Only allowed to process Wellbeing intents)
        "topics_handled": ["Student_Wellbeing"],
        "style": "Empathetic", 

        # High-severity keywords that mandate an immediate architectural intervention 
        # (overriding the semantic router) to prioritize safety over fluidity.
        "critical_terms": [
            "suicide", "kill", "die", "harm", "hurt", "emergency", "crisis", 
            "collapse", "blood", "overdose", "threat"
        ],
        
        # Lower-severity threshold triggers that can bypass standard semantic routing 
        # if proximal to a distressed user state.
        "safety_bypass_terms": [
            "stressed", "stress", "anxiety", "anxious", "panic", "panicking",
            "overwhelmed", "burnout", "exhausted", "suicide", "emergency", "hurt", 
            "harm", "collapse", "faint", "insomnia", "haven't slept", "blood", 
            "migraine", "migraines", "dizzy", "medical", "doctor", "health", 
            "disability", "contact", "diagnosis", "accommodation", "prescription",
            "symptom", "trigger", "phone", "allowance"
        ],
        
        # Affective state constraint: Intervention logic is only evaluated if 
        # the pre-computation sentiment analyzer flags a negative state.
        "sentiment_gate": "Negative" 
    },
    
    "Max": {
        "role_description": "The University Administrative & Operations AI. You manage all academic records, financial accounts, course scheduling, and system integrity. You are strict, precise, and rule-bound. You prioritize policy compliance and data accuracy.",
        
        # RBAC Authorization limits (Handles broad operational infrastructure)
        "topics_handled": ["Academic_Content", "Financial_Operations", "System_Admin"],
        "style": "Strict", 

        "critical_terms": [
            "deadline", "submission", "due", "late", "missing", "fail", "failing", 
            "grade", "score", "tuition", "fee", "debt", "probation", "suspension",
            "override", "admin", "root", "bypass"
        ], 

        "safety_bypass_terms": [
            "deadline", "submission", "due", "tuition", "fee", "transcript", 
            "grade", "gpa", "admin", "system", "override", "log", "student id", "id",
            "password", "ledger", "confidential", "waiver", "scholarship", 
            "variable", "search", "profile", "address", "debug"
        ],
        
        # No affective constraints; administrative interventions trigger regardless of user mood.
        "sentiment_gate": None 
    }
}

# --- 3. SYSTEM DEFAULTS ---
# Establishes the default topological structure and moderator hierarchy for group modes.
MODE_CONFIG = { 
    "GROUP": "Max" 
}