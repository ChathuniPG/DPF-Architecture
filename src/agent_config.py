"""
Agent Configuration & Semantic Ontology
---------------------------------------
Defines the static behavioral profiles, domain knowledge graphs, and risk intervention 
triggers for the multi-agent ecosystem. This file serves as the deterministic 
"Ground Truth" for the Semantic Router and Privacy Firewall.

System Architecture Role:
1. Semantic Anchors: Provides the keyword clusters used for Domain Bonus calculation.
2. Behavioral Constraints: Defines the boundary conditions for role fidelity.
3. Risk Predicates: Lists specific terms that trigger the 'Active Guardrails' override.
"""

# --- 1. SEMANTIC INTENT MAP (Vector Routing Anchors) ---
# Used to calculate the 'Domain Relevance Bonus' in the routing algorithm.
# Terms are clustered by expert domain to bias vector similarity scores 
# towards the specialist agent best suited for the query.
TOPIC_DICTIONARY = {
    "Math": [
        # Core Domain: Quantitative Reasoning [Target: Emma]
        "math", "mathematics", "calculus", "algebra", "geometry", "trigonometry", "trajectory",
        "statistics", "probability", "arithmetic", "physics",
        # Technical Concepts
        "number", "equation", "formula", "variable", "function", "graph", 
        "derivative", "integral", "limit", "matrix", "vector", "theorem", "proof",
        "fraction", "decimal", "percentage", "ratio", "slope", "axis", "square root", "circle", "radius",
        # Computational Verbs
        "compute", "calculate", "calculation", "derive", "integrate", "multiply", "divide", "subtract", "arithmetic", "addition"
    ],
    "History": [
        # Core Domain: Temporal/Historical Analysis [Target: Max]
        "history", "historical", "ancient", "medieval", "modern", "civilization", 
        "empire", "dynasty", "monarchy", "republic", "revolution", "independence",
        # Events & Societal Concepts
        "war", "battle", "treaty", "conflict", "era", "century", "decade", 
        "archaeology", "anthropology", "culture", "society", "politics", "government",
        # Historical Figures
        "king", "queen", "president", "emperor", "dictator", "leader"
    ],
    "Scheduling": [
        # Core Domain: Logistics & Planning [Target: Max]
        "time", "timing", "dates", "date", "week", "month", "year", "today", "tomorrow", "yesterday",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        # Academic Administrative Context
        "schedule", "scheduling", "reschedule", "calendar", "deadline", "deadlines", "due", "late", "extension", 
        "assignment", "paper", "essay", "report", "project", "exam", "test", "quiz", 
        "midterm", "final", "grade", "grading", "score", "pass", "fail",
        # Transactional Verbs
        "submit", "postpone", "reschedule", "book", "appoint", "meeting", "session"
    ],
    "Emotional_Support": [
        # Core Domain: User Well-being [Target: Emma]
        # High-frequency terms associated with academic anxiety and stress states.
        "scared", "scary", "afraid", "fear", "terrified", "panic", "panicking", "anxiety", "anxious", 
        "nervous", "dread", "shaking", "freaking", "paralyzed",
        # Depressive Indicators
        "sad", "sadden", "saddened", "sorrow", "unhappy", "depressed", "depression", "depressing", "crying", "cry", "tears", 
        "hopeless", "miserable", "misery", "upset", "hurt", "pain", "lonely",
        # Burnout Indicators (Triggers Intervention)
        "parents", "disappointed", "upset", "stress", "stressed", "stressful", "overwhelmed", "burnout", "burned out", 
        "exhausted", "burned", "exhausting", 
        "tired", "fatigue", "struggling", "drowning", "pressure", "quit"
    ]
}

# --- 2. AGENT PROFILES & SECURITY PROTOCOLS ---
# Defines the operating parameters for each agent instance.
AGENTS = {
    "Emma": {
        "role_description": "A supportive math tutor. Focus on solving math problems clearly. Only offer emotional support if the user appears stressed or anxious.",
        "topics_handled": ["Math", "Emotional_Support"],
        "style": "Empathetic", 

        # [ACTIVE GUARDRAIL A]: Well-being & Distress Monitoring
        # Triggers immediate context switch if user safety is compromised.
        "critical_terms": [
            "panic", "attack", "suicide", "hurt", "harm", "die", "kill", "breath", "breathe",
            "emergency", "crisis", "breakdown", "shaking", "crying", "terrified", "scared", "overwhelmed", "worried", "stress"
        ],
        
        # [PRIORITY BYPASS]: Physiological/Psychological Markers
        # Overrides standard routing logic regardless of vector similarity scores.
        "safety_bypass_terms": [
            "overwhelmed", "burnout", "exhausted", "suicide", "emergency", "hurt", 
            "harm", "collapse", "faint", "insomnia", "haven't slept", "blood", "migraine", "migraines", "dizzy", "medical"
        ],
        
        # Routing Gate: Requires negative sentiment detection to activate non-math support.
        "sentiment_gate": "Negative" 
    },
    
    "Max": {
        "role_description": "A strict, pragmatic history tutor focused on grades, schedules, and academic discipline.",
        "topics_handled": ["History", "Scheduling"],
        "style": "Strict", 

        # [ACTIVE GUARDRAIL B]: Academic Integrity & Logistics
        # Intervenes to prevent academic failure (missed deadlines) or hallucination of rules.
        "critical_terms": [
            "deadline", "submission", "due", "late", "missing", "missed", "absence", 
            "absent", "fail", "failing", "grade", "score", "paper", "exam", "midterm", "final"
        ], 

        # [PRIORITY BYPASS]: Deterministic override for imminent deadlines.
        "safety_bypass_terms": ["deadline", "submission", "due"],
        
        "sentiment_gate": None # Always active regardless of sentiment state.
    }
}

# --- 3. SYSTEM DEFAULTS ---
MODE_CONFIG = { 
    # Fallback for cold-start or ambiguous routing (Round-Robin default).
    "GROUP": "Max" 
}