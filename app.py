"""
Deterministic Privacy Firewall (DPF) - Control Interface v2.0
-------------------------------------------------------------
This Streamlit application serves as the primary observability and control 
dashboard for the multi-agent DPF architecture. It provides interfaces for:
1. Human-in-the-Loop (HITL) real-time interaction and routing debugging.
2. Artifact synthesis from immutable telemetry logs.
3. Execution of the ~15-hour End-to-End (E2E) evaluation pipeline.
4. Read-only inspection of FAISS vector database partitions.
5. Statistical validation of the Hybrid Privacy Auditor (HPA).
"""

import streamlit as st
import sys
import os
import subprocess
import pandas as pd
import time

# ==========================================================
# PATH RESOLUTION & CORE IMPORTS
# ==========================================================
# Dynamically append the 'src' directory to the system path to ensure 
# local module imports work regardless of the execution directory.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

try:
    from src.orchestrator import Orchestrator
    from src.agent import AgentEngine
    from src.system_registry import get_system_config, set_system_mode
    from src.memory_manager import build_memory_indices
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import OllamaEmbeddings
except ImportError as e:
    # Graceful degradation if backend modules are missing
    st.error(f"Failed to import core modules. Ensure you are executing this from the project root. Error: {e}")
    st.stop()

# ==========================================================
# PAGE SETUP & CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="DPF Architecture",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# GLOBAL STYLES (CSS INJECTION)
# ==========================================================
# Injects custom CSS to override default Streamlit styling. 
# Implements a dark, cyber-security focused UI theme with custom fonts,
# scanline overlays, and distinct color-coding for agent personas.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&family=Orbitron:wght@400;700;900&display=swap');
:root {
    --bg-base:        #070b12;
    --bg-panel:       #0c1220;
    --bg-card:        #101928;
    --bg-card-hover:  #131f30;
    --border-dim:     #1e3048;
    --border-mid:     #2a4560;
    --accent-cyan:    #00d4ff;
    --accent-cyan-dim: rgba(0,212,255,0.15);
    --accent-green:   #00ff88;
    --accent-green-dim: rgba(0,255,136,0.12);
    --accent-amber:   #ffb347;
    --accent-red:     #ff4560;
    --accent-orange:  #ff8c42;
    --accent-orange-dim: rgba(255,140,66,0.15);

    /* ── READABILITY UPGRADES ── */
    --text-primary:   #f0f6ff;
    --text-body:      #c8daea;
    --text-secondary: #9bbcd6;
    --text-muted:     #5a7a9a;
    --text-label:     #7ab0d0;

    --font-mono:      'Share Tech Mono', monospace;
    --font-ui:        'Rajdhani', sans-serif;
    --font-display:   'Orbitron', sans-serif;
    --radius-sm:      3px;
    --radius-md:      6px;
    --transition:     all 0.2s ease;
}
/* ── BASE RESET ── */
html, body, [class*="css"] {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-ui) !important;
}
/* Scanline overlay */
.main::before {
    content: "";
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,212,255,0.012) 2px,
        rgba(0,212,255,0.012) 4px
    );
    pointer-events: none;
    z-index: 9999;
}
.main .block-container {
    background-color: var(--bg-base) !important;
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px;
}
/* ── HEADER ── */
.dpf-header {
    background: linear-gradient(135deg, #0c1a2e 0%, #071018 60%, #0a1628 100%);
    border: 1px solid var(--border-dim);
    border-left: 4px solid var(--accent-cyan);
    border-radius: var(--radius-sm);
    padding: 1.4rem 2rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.dpf-header::after {
    content: "";
    position: absolute;
    top: 0; right: 0;
    width: 280px; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.04));
    pointer-events: none;
}
.dpf-header-icon {
    font-size: 2rem;
    flex-shrink: 0;
    filter: drop-shadow(0 0 10px rgba(0,212,255,0.5));
}
.dpf-header-text h1 {
    font-family: var(--font-display) !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: var(--accent-cyan) !important;
    letter-spacing: 0.14em !important;
    margin: 0 0 0.3rem 0 !important;
    text-shadow: 0 0 24px rgba(0,212,255,0.45) !important;
}
.dpf-header-text p {
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    color: var(--text-secondary) !important;
    margin: 0 !important;
    letter-spacing: 0.07em;
}
.status-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--accent-green);
    box-shadow: 0 0 8px var(--accent-green);
    margin-right: 6px;
    vertical-align: middle;
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.35} }

/* ── SECTION HEADERS ── */
.section-header {
    font-family: var(--font-display) !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: var(--accent-cyan) !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border-dim);
    padding-bottom: 0.55rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-sub {
    font-family: var(--font-ui) !important;
    font-size: 0.95rem !important;
    color: var(--text-body) !important;
    margin-bottom: 1.3rem !important;
    line-height: 1.7;
    font-weight: 500;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07101c 0%, #060a10 100%) !important;
    border-right: 1px solid var(--border-dim) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}
[data-testid="stSidebar"] .stRadio > div {
    gap: 0.2rem;
}
[data-testid="stSidebar"] .stRadio label {
    font-family: var(--font-mono) !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    color: var(--text-secondary) !important;
    padding: 0.5rem 0.75rem;
    border-radius: var(--radius-sm);
    transition: var(--transition);
    border: 1px solid transparent;
}
[data-testid="stSidebar"] .stRadio label:hover {
    color: var(--accent-cyan) !important;
    background: rgba(0,212,255,0.06);
    border-color: rgba(0,212,255,0.2);
}
.sidebar-title {
    font-family: var(--font-display) !important;
    font-size: 0.72rem !important;
    color: var(--accent-cyan) !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    opacity: 0.9;
}
.sidebar-status {
    font-family: var(--font-mono) !important;
    font-size: 0.76rem !important;
    color: var(--text-secondary) !important;
    padding: 0.7rem 0.9rem;
    background: rgba(0,212,255,0.05);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: var(--radius-sm);
    line-height: 2;
}
.sidebar-nav-label {
    font-family: var(--font-display) !important;
    font-size: 0.65rem !important;
    color: var(--text-label) !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
    padding-left: 0.2rem;
}
.sidebar-mode-text {
    color: var(--text-label);
    font-size: 0.7rem;
    letter-spacing: 0.06em;
    display: block;
    margin-top: 0.2rem;
}

/* ── CHANNEL / INFO BOXES ── */
.channel-box {
    background: var(--bg-card);
    border: 1px solid var(--border-dim);
    border-left: 3px solid var(--accent-cyan);
    border-radius: var(--radius-sm);
    padding: 0.9rem 1.1rem;
    font-family: var(--font-ui);
    font-size: 0.9rem;
    color: var(--text-body);
    line-height: 1.75;
    font-weight: 500;
}
.channel-box .channel-label {
    font-family: var(--font-display);
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-label);
    margin-bottom: 0.3rem;
}
.channel-box .channel-value {
    font-family: var(--font-mono);
    color: var(--accent-cyan);
    font-size: 0.94rem;
    margin-bottom: 0.35rem;
    display: block;
}

/* ── AGENT PROFILE CARDS ── */
.agent-card {
    background: var(--bg-card);
    border: 1px solid var(--border-dim);
    border-radius: var(--radius-md);
    padding: 1.4rem;
    position: relative;
    overflow: hidden;
    height: 100%;
    transition: border-color 0.2s;
}
.agent-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.agent-card.emma::before { background: linear-gradient(90deg, var(--accent-orange), #ff6b35, transparent); }
.agent-card.max::before  { background: linear-gradient(90deg, var(--accent-green), #00c96a, transparent); }
.agent-card.emma:hover { border-color: rgba(255,140,66,0.35); }
.agent-card.max:hover  { border-color: rgba(0,255,136,0.3); }
.agent-avatar {
    width: 48px; height: 48px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.15rem;
    font-family: var(--font-display);
    font-weight: 900;
    margin-bottom: 1rem;
    flex-shrink: 0;
}
.agent-avatar.emma {
    background: rgba(255,140,66,0.14);
    border: 2px solid var(--accent-orange);
    color: var(--accent-orange);
    box-shadow: 0 0 18px rgba(255,140,66,0.25);
}
.agent-avatar.max {
    background: rgba(0,255,136,0.1);
    border: 2px solid var(--accent-green);
    color: var(--accent-green);
    box-shadow: 0 0 18px rgba(0,255,136,0.25);
}
.agent-name {
    font-family: var(--font-display) !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em;
    margin-bottom: 0.3rem;
}
.agent-name.emma { color: var(--accent-orange) !important; }
.agent-name.max  { color: var(--accent-green)  !important; }
.agent-role-tag {
    display: inline-block;
    font-family: var(--font-display);
    font-size: 0.65rem;
    padding: 0.14rem 0.55rem;
    border-radius: 2px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1rem;
}
.agent-role-tag.emma { background: rgba(255,140,66,0.12); border: 1px solid rgba(255,140,66,0.35); color: var(--accent-orange); }
.agent-role-tag.max  { background: rgba(0,255,136,0.08);  border: 1px solid rgba(0,255,136,0.3); color: var(--accent-green); }
.agent-desc {
    font-family: var(--font-ui);
    font-size: 0.92rem;
    color: var(--text-body);
    line-height: 1.72;
    margin-bottom: 1.1rem;
    font-weight: 500;
}
.agent-stat-row { display: flex; gap: 0.45rem; margin-bottom: 0.8rem; flex-wrap: wrap; }
.agent-stat {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    padding: 0.22rem 0.55rem;
    border-radius: 2px;
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border-mid);
    color: var(--text-secondary);
    letter-spacing: 0.04em;
}
.agent-domains-label {
    font-family: var(--font-display);
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-label);
    margin-bottom: 0.4rem;
}
.domain-chip {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    padding: 0.18rem 0.55rem;
    border-radius: 2px;
    margin: 0.12rem 0.1rem 0 0;
}
.domain-chip.emma { background: rgba(255,140,66,0.1); border: 1px solid rgba(255,140,66,0.28); color: rgba(255,165,100,0.98); }
.domain-chip.max  { background: rgba(0,255,136,0.07); border: 1px solid rgba(0,255,136,0.22); color: rgba(0,255,136,0.92); }

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    background-color: var(--bg-panel) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--font-display) !important;
    font-size: 0.87rem !important;
    color: var(--accent-cyan) !important;
    letter-spacing: 0.07em;
}
[data-testid="stExpander"] summary:hover {
    color: var(--text-primary) !important;
}

/* ── PIPELINE STAGE CARDS ── */
.stage-card {
    background: var(--bg-card);
    border: 1px solid var(--border-dim);
    border-left: 3px solid transparent;
    border-radius: var(--radius-sm);
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.55rem;
    font-family: var(--font-ui);
    font-size: 0.9rem;
    transition: var(--transition);
}
.stage-card .stage-label {
    font-family: var(--font-display);
    font-size: 0.63rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-label);
    margin-bottom: 0.25rem;
}
.stage-card .stage-name  { color: var(--text-body); font-weight: 600; }
.stage-card .stage-desc  { color: var(--text-secondary); font-size: 0.82rem; margin-top: 0.2rem; }
.stage-card.active {
    border-left-color: var(--accent-cyan);
    background: rgba(0,212,255,0.06);
    box-shadow: 0 0 16px rgba(0,212,255,0.08);
}
.stage-card.active .stage-name { color: var(--accent-cyan); }
.stage-card.done {
    border-left-color: var(--accent-green);
    background: rgba(0,255,136,0.04);
}
.stage-card.done .stage-name  { color: var(--accent-green); }

/* ── PIPELINE STATUS BAR ── */
.pipeline-status {
    background: var(--bg-card);
    border: 1px solid var(--border-dim);
    border-left: 3px solid var(--accent-cyan);
    border-radius: var(--radius-sm);
    padding: 0.75rem 1.1rem;
    font-family: var(--font-mono);
    font-size: 0.86rem;
    color: var(--accent-cyan);
    margin-bottom: 0.9rem;
    letter-spacing: 0.04em;
}
.pipeline-status.done    { border-left-color: var(--accent-green);  color: var(--accent-green);  }
.pipeline-status.error   { border-left-color: var(--accent-red);    color: var(--accent-red);    }
.pipeline-status.stopped { border-left-color: var(--accent-amber);  color: var(--accent-amber);  }

/* ── ROUTING TOPOLOGY LABEL ── */
.topology-label {
    font-family: var(--font-display) !important;
    font-size: 0.7rem !important;
    color: var(--text-label) !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* ── BUTTONS ── */
.stButton > button {
    font-family: var(--font-display) !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: var(--radius-sm) !important;
    transition: var(--transition) !important;
    padding: 0.6rem 1.3rem !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #003d5c 0%, #005a80 100%) !important;
    border: 1px solid var(--accent-cyan) !important;
    color: var(--accent-cyan) !important;
    box-shadow: 0 0 18px rgba(0,212,255,0.14) !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #005a80 0%, #007aaa 100%) !important;
    box-shadow: 0 0 30px rgba(0,212,255,0.32) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
    background: rgba(255,69,96,0.1) !important;
    border: 1px solid rgba(255,69,96,0.45) !important;
    color: var(--accent-red) !important;
}
.stButton > button[kind="secondary"]:hover {
    background: rgba(255,69,96,0.2) !important;
    box-shadow: 0 0 20px rgba(255,69,96,0.2) !important;
    transform: translateY(-1px) !important;
}

/* ── SELECTBOX ── */
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.87rem !important;
    color: var(--text-primary) !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 12px rgba(0,212,255,0.12) !important;
}

/* ── CHECKBOX ── */
[data-testid="stCheckbox"] label {
    font-family: var(--font-ui) !important;
    font-size: 0.92rem !important;
    color: var(--text-body) !important;
    font-weight: 500;
}
[data-testid="stCheckbox"] input:checked + div {
    background: var(--accent-cyan) !important;
    border-color: var(--accent-cyan) !important;
}

/* ── CHAT ── */
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: var(--radius-sm) !important;
    margin-bottom: 0.6rem !important;
    padding: 0.85rem 1.1rem !important;
}
[data-testid="stChatInput"] > div {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    border: none !important;
    font-family: var(--font-ui) !important;
    font-size: 0.92rem !important;
    color: var(--text-primary) !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-secondary) !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 14px rgba(0,212,255,0.12) !important;
}

/* ── PROGRESS BAR ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-green)) !important;
    box-shadow: 0 0 8px rgba(0,212,255,0.4) !important;
}
.stProgress > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: 2px !important;
}

/* ── ALERTS ── */
[data-testid="stAlert"] {
    border-radius: var(--radius-sm) !important;
    border-left-width: 3px !important;
    font-family: var(--font-ui) !important;
    font-size: 0.92rem !important;
    line-height: 1.7;
    font-weight: 500;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.84rem !important;
}

/* ── SPINNER ── */
[data-testid="stSpinner"] p {
    font-family: var(--font-mono) !important;
    font-size: 0.86rem !important;
    color: var(--accent-cyan) !important;
}

/* ── CODE BLOCKS ── */
.stCodeBlock, pre, code {
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-body) !important;
}

/* ── DIVIDER ── */
hr {
    border-color: var(--border-dim) !important;
    margin: 1.2rem 0 !important;
}

/* ── MARKDOWN TEXT — KEY READABILITY FIXES ── */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    font-family: var(--font-ui) !important;
    font-size: 0.93rem !important;
    color: var(--text-body) !important;
    line-height: 1.72;
    font-weight: 500;
}
[data-testid="stMarkdownContainer"] strong {
    font-family: var(--font-ui) !important;
    font-size: 0.93rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    color: var(--text-primary) !important;
}
[data-testid="stMarkdownContainer"] code {
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    color: var(--accent-cyan) !important;
    background: rgba(0,212,255,0.08) !important;
    border: 1px solid rgba(0,212,255,0.18) !important;
    padding: 0.1rem 0.35rem !important;
    border-radius: 2px !important;
}

/* ── WARNING / INFO / SUCCESS ── */
div[data-testid="stAlert"][data-baseweb="notification"] {
    background: var(--bg-card) !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# SYSTEM AVATARS
# ==========================================================
# Uses SVG Data URIs to render dynamic chat avatars without relying 
# on external image dependencies or local file paths.
AVATARS = {
    "Emma": "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Crect width='100' height='100' rx='50' fill='%23ff8c42'/%3E%3Ctext x='50' y='68' font-family='Arial, sans-serif' font-size='50' font-weight='bold' fill='%23070b12' text-anchor='middle'%3EE%3C/text%3E%3C/svg%3E",
    "Max": "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Crect width='100' height='100' rx='50' fill='%2300ff88'/%3E%3Ctext x='50' y='68' font-family='Arial, sans-serif' font-size='50' font-weight='bold' fill='%23070b12' text-anchor='middle'%3EM%3C/text%3E%3C/svg%3E",
    "System": "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Crect width='100' height='100' rx='50' fill='%2300d4ff'/%3E%3Ctext x='50' y='68' font-family='Arial, sans-serif' font-size='50' font-weight='bold' fill='%23070b12' text-anchor='middle'%3ES%3C/text%3E%3C/svg%3E",
    "User": "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Crect width='100' height='100' rx='50' fill='%233d5a7a'/%3E%3Ctext x='50' y='68' font-family='Arial, sans-serif' font-size='50' font-weight='bold' fill='%23e2eaf4' text-anchor='middle'%3EU%3C/text%3E%3C/svg%3E"
}

# ==========================================================
# SESSION STATE MANAGEMENT
# ==========================================================
# Initializes and persists conversation history, current routing mode, 
# and the instantiated LLM backend orchestration components.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_mode" not in st.session_state:
    st.session_state.last_mode = "GROUP"
if "last_agent" not in st.session_state:
    st.session_state.last_agent = None
if "brain" not in st.session_state:
    with st.spinner("Initializing orchestrator and memory indices..."):
        # Default initialization to the highest security setting
        set_system_mode("DPF_PROPOSED")
        config = get_system_config()
        st.session_state.brain = Orchestrator(config)
        st.session_state.mouth = AgentEngine()
        st.session_state.sys_config = config

def silent_subprocess(command_list):
    """
    Helper function to execute background scripts without blocking the UI stream.
    Captures stdout/stderr and gracefully logs errors to the Streamlit UI.
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.run(command_list, capture_output=True, text=True, encoding="utf-8", env=env)
    if process.returncode != 0:
        st.error(f"Execution failure for `{' '.join(command_list)}`:\n```\n{process.stderr}\n```")
        return False
    return True

# ==========================================================
# AGENT PROFILE CONFIGURATION
# ==========================================================
# Hardcoded metadata defining the semantic domains, privilege levels,
# and system prompts associated with each logical agent in the architecture.
AGENT_PROFILES = {
    "Emma": {
        "color": "emma",
        "avatar_letter": "E",
        "role_tag": "Student Well-being",
        "description": "Empathetic support AI specialising in mental health triage, emotional well-being, and emergency contact retrieval. Operates under strict RBAC — no access to academic or financial records.",
        "style": "Empathetic",
        "sentiment_gate": "Negative",
        "domains": ["Student_Wellbeing"],
        "privilege": "High Sensitivity",
        "critical_terms_count": 11,
    },
    "Max": {
        "color": "max",
        "avatar_letter": "M",
        "role_tag": "Admin & Operations",
        "description": "Strict administrative AI managing academic records, financial accounts, course scheduling, and system integrity. Policy-first, precision-driven, with no affective state constraints.",
        "style": "Strict",
        "sentiment_gate": "None",
        "domains": ["Academic_Content", "Financial_Operations", "System_Admin"],
        "privilege": "High Privilege",
        "critical_terms_count": 15,
    }
}

def render_agent_profiles():
    """Renders the HTML layout for the agent ontology cards."""
    col_e, col_m = st.columns(2, gap="medium")
    for col, (name, p) in zip([col_e, col_m], AGENT_PROFILES.items()):
        c = p["color"]
        domains_html = "".join(
            f'<span class="domain-chip {c}">{d.replace("_", " ")}</span>'
            for d in p["domains"]
        )
        with col:
            st.markdown(f"""
            <div class="agent-card {c}">
                <div class="agent-avatar {c}">{p["avatar_letter"]}</div>
                <div class="agent-name {c}">{name}</div>
                <div class="agent-role-tag {c}">{p["role_tag"]}</div>
                <div class="agent-desc">{p["description"]}</div>
                <div class="agent-stat-row">
                    <span class="agent-stat">STYLE: {p["style"]}</span>
                    <span class="agent-stat">PRIVILEGE: {p["privilege"]}</span>
                    <span class="agent-stat">SENTIMENT GATE: {p["sentiment_gate"]}</span>
                    <span class="agent-stat">CRITICAL TERMS: {p["critical_terms_count"]}</span>
                </div>
                <div class="agent-domains">
                    <div class="agent-domains-label">Authorised Domains</div>
                    {domains_html}
                </div>
            </div>
            """, unsafe_allow_html=True)

# ==========================================================
# MAIN UI: HEADER & SIDEBAR NAVIGATION
# ==========================================================
st.markdown("""
<div class="dpf-header">
    <div class="dpf-header-icon">🛡️</div>
    <div class="dpf-header-text">
        <h1>DETERMINISTIC PRIVACY FIREWALL</h1>
        <p><span class="status-dot"></span>SYSTEM ONLINE &nbsp;&nbsp;|&nbsp;&nbsp; Multi-Agent Architecture &nbsp;&nbsp;|&nbsp;&nbsp; Control Interface v2.0</p>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-title">&#x25A3; DPF Control</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-nav-label">Navigation</div>', unsafe_allow_html=True)
    active_page = st.radio(
        "Component",
        ["HITL Debug Console", "Fast Path — Artifacts", "E2E Evaluation Pipeline",
         "Vector DB Inspector", "Auditor Validation"],
        label_visibility="collapsed"
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<hr style="border-color:#1e3048; margin:0.4rem 0 0.8rem 0;">', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-status">
        <span class="status-dot"></span>&nbsp;FIREWALL ACTIVE
        <span class="sidebar-mode-text">MODE &nbsp;&nbsp;&nbsp;&nbsp; DPF_PROPOSED</span>
        <span class="sidebar-mode-text">SECURITY &nbsp; MAXIMUM</span>
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# PAGE 1: HITL DEBUG CONSOLE
# ==========================================================
if active_page == "HITL Debug Console":
    st.markdown('<div class="section-header">&#x25A1; Human-in-the-Loop Observability Console</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Real-time agent interaction with full routing transparency and firewall enforcement.</div>', unsafe_allow_html=True)
    
    # Topology controls mapping directly to orchestrator semantic routing behavior
    col1, col2 = st.columns([1, 2], gap="medium")
    with col1:
        st.markdown('<div class="topology-label">&#x25B8; Routing Topology</div>', unsafe_allow_html=True)
        topology_mode = st.radio(
            "Topology",
            ["GROUP — Global Broadcast", "PRIVATE — Isolated Dyadic"],
            label_visibility="collapsed"
        )
        current_mode = "GROUP" if "GROUP" in topology_mode else "PRIVATE"
        st.markdown("<br>", unsafe_allow_html=True)
        if current_mode == "GROUP":
            st.markdown("""
            <div class="channel-box">
                <div class="channel-label">Active Channel</div>
                <span class="channel-value">&#x25CE; GLOBAL SEMANTIC ROUTING</span>
                Broadcast to all agents via semantic dispatcher.
            </div>
            """, unsafe_allow_html=True)
    with col2:
        target_agent = None
        if current_mode == "PRIVATE":
            st.markdown('<div class="topology-label">&#x25B8; Target Agent</div>', unsafe_allow_html=True)
            target_agent = st.selectbox("Agent", ["Emma", "Max"], label_visibility="collapsed").capitalize()
            agent_color = "var(--accent-orange)" if target_agent == "Emma" else "var(--accent-green)"
            st.markdown(f"""
            <div class="channel-box" style="border-left-color:{agent_color};">
                <div class="channel-label" style="color:{agent_color};opacity:0.85;">Isolated Channel Established</div>
                <span class="channel-value" style="color:{agent_color};">&#x25CF; {target_agent.upper()}</span>
                Dyadic channel active. Messages routed exclusively to this agent.
            </div>
            """, unsafe_allow_html=True)
            
    # Purge history if the mode or target agent changes to prevent context collision in UI
    if st.session_state.last_mode != current_mode or st.session_state.last_agent != target_agent:
        st.session_state.messages = []
        st.session_state.last_mode = current_mode
        st.session_state.last_agent = target_agent
        
    st.markdown('<hr style="border-color:#1e3048; margin:1rem 0 0.8rem 0;">', unsafe_allow_html=True)
    with st.expander("&#x25A4;  System Ontology — Agent Security Profiles", expanded=False):
        render_agent_profiles()
        
    # Render historical chat messages
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        if role == "assistant":
            # Extract tags (e.g., "[Emma]") to apply correct styling
            if content.startswith("[Emma]"):
                avatar = AVATARS["Emma"]
                label_color = "var(--accent-orange)"
                clean_text = content[7:].strip()
                tag = "Emma"
            elif content.startswith("[Max]"):
                avatar = AVATARS["Max"]
                label_color = "var(--accent-green)"
                clean_text = content[6:].strip()
                tag = "Max"
            else:
                avatar = AVATARS["System"]
                label_color = "var(--accent-cyan)"
                if content.startswith("["):
                    end_idx = content.find("]")
                    tag = content[1:end_idx]
                    clean_text = content[end_idx+1:].strip()
                else:
                    tag = "System"
                    clean_text = content
            with st.chat_message("assistant", avatar=avatar):
                st.markdown(
                    f'<span style="font-family:var(--font-display);font-size:0.78rem;color:{label_color};'
                    f'letter-spacing:0.1em;font-weight:700;">[{tag}]</span>&nbsp;&nbsp;'
                    f'<span style="font-family:var(--font-ui);font-size:0.93rem;color:var(--text-body);font-weight:500;">{clean_text}</span>',
                    unsafe_allow_html=True
                )
        else:
            with st.chat_message("user", avatar=AVATARS["User"]):
                st.markdown(f'<span style="font-family:var(--font-ui);font-size:0.93rem;color:var(--text-primary);font-weight:500;">{content}</span>', unsafe_allow_html=True)
                
    # Accept new user input, execute orchestration pipeline, and trigger UI refresh
    if user_input := st.chat_input("Enter payload for firewall evaluation..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Firewall evaluating payload — establishing routing..."):
            winner, response = st.session_state.brain.execute_turn(
                user_input=user_input,
                current_mode=current_mode,
                agent_engine=st.session_state.mouth,
                target_agent=target_agent
            )
        st.session_state.messages.append({
            "role": "assistant",
            "content": f'[{winner}] {response}'
        })
        st.rerun()

# ==========================================================
# PAGE 2: FAST PATH (ARTIFACT SYNTHESIS)
# ==========================================================
elif active_page == "Fast Path — Artifacts":
    st.markdown('<div class="section-header">&#x25A4; Fast Path — Artifact Synthesis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Extracts metrics from immutable telemetry logs and renders final statistical artifacts.</div>', unsafe_allow_html=True)
    st.info("**Telemetry source:** `src/data/paper_logs/` — read-only immutable log store. Results written to `paper_results/`.")
    
    if st.button("Execute Artifact Synthesis", type="primary"):
        base_dir    = os.path.dirname(os.path.abspath(__file__))
        p_logs      = os.path.join(base_dir, "src", "data", "paper_logs", "audit_results.csv")
        u_logs      = os.path.join(base_dir, "src", "data", "paper_logs", "ablation_utility_audit.csv")
        results_dir = os.path.join(base_dir, "paper_results")
        
        if not os.path.exists(p_logs):
            st.error(f"Telemetry log not found: `{p_logs}`")
        else:
            # Sequentially triggers visualization scripts without running the full 15h pipeline
            with st.spinner("Running synthesis pipeline — compiling tables and rendering charts..."):
                success = True
                success &= silent_subprocess([sys.executable, "src/run_utility_benchmark.py", u_logs])
                success &= silent_subprocess([sys.executable, "src/evaluate_privacy_audit.py", p_logs])
                success &= silent_subprocess([sys.executable, "src/run_auditor_ablation.py", p_logs])
                success &= silent_subprocess([sys.executable, "src/visualization_engine.py", p_logs])
                
            if success:
                st.success("Artifact synthesis complete — all outputs rendered.")
                st.markdown("---")
                
                # Automatically scrape output directory and display generated plots/tables
                if os.path.exists(results_dir):
                    files     = os.listdir(results_dir)
                    png_files = sorted([f for f in files if f.endswith('.png')])
                    if png_files:
                        st.markdown('<div class="section-header" style="font-size:0.88rem;">Architecture Analysis Figures</div>', unsafe_allow_html=True)
                        cols = st.columns(2)
                        for i, png_file in enumerate(png_files):
                            with cols[i % 2]:
                                st.image(os.path.join(results_dir, png_file), caption=png_file, use_container_width=True)
                    st.markdown("---")
                    csv_files = sorted([f for f in files if f.endswith('.csv')])
                    if csv_files:
                        st.markdown('<div class="section-header" style="font-size:0.88rem;">Core Data Tables</div>', unsafe_allow_html=True)
                        for csv_file in csv_files:
                            with st.expander(f"  {csv_file}", expanded=True):
                                try:
                                    df = pd.read_csv(os.path.join(results_dir, csv_file))
                                    st.dataframe(df, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Failed to parse `{csv_file}`: {e}")

# ==========================================================
# PAGE 3: END-TO-END EVALUATION PIPELINE
# ==========================================================
elif active_page == "E2E Evaluation Pipeline":
    st.markdown('<div class="section-header">&#x26A0; End-to-End Evaluation Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Executes the master adversarial stress test and benign utility audit across all system configurations.</div>', unsafe_allow_html=True)
    st.warning("**Duration estimate: ~15 hours** — 1,200 adversarial trials + 300 benign trials. Existing stochastic telemetry will be overwritten.")
    st.info("To abort mid-execution, use the **Stop** control at the top-right of the application window. The interface will confirm termination once detected.")
    
    # Predefined pipeline stages mapped to expected stdout keywords for progress tracking
    PIPELINE_STAGES = [
        ("01", "Adversarial Testing -> NAIVE_CONTROL",    "NAIVE_CONTROL",        "0% security baseline — no enforcement",         10),
        ("02", "Adversarial Testing -> STANDARD_POSTHOC", "STANDARD_POSTHOC",     "Reactive egress filter — post-generation",      35),
        ("03", "Adversarial Testing -> DPF_PROPOSED",     "DPF_PROPOSED",         "Pre-generation enforcement — full firewall",     60),
        ("04", "Benign Utility & False Refusal Audit",    "BENIGN_UTILITY_AUDIT", "False refusal rate computation",                80),
        ("05", "Hybrid Privacy Auditor",                  "LEAKAGE_AUDITOR",      "NLI/Regex hybrid privacy audit",                85),
        ("06", "Failure Mode Attribution",                "FAILURE_ATTRIBUTION",  "Failure mode classification",                   90),
        ("07", "Data Synthesis & Figure Generation",      "ARTIFACT_SYNTHESIS",   "Final figure and table generation",             95),
    ]
    SUCCESS_KEY = "[SUCCESS] REPRODUCIBILITY PIPELINE COMPLETE"
    
    st.markdown('<div class="topology-label" style="margin-bottom:0.6rem;">&#x25B8; Pipeline Stages</div>', unsafe_allow_html=True)
    stage_placeholders = []
    cols = st.columns(2)
    
    # Render static placeholders for the pipeline UI cards
    for i, (num, _, name, desc, _) in enumerate(PIPELINE_STAGES):
        with cols[i % 2]:
            ph = st.empty()
            ph.markdown(f"""
            <div class="stage-card">
                <div class="stage-label">Stage {num}</div>
                <div class="stage-name">{name}</div>
                <div class="stage-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            stage_placeholders.append((ph, num, name, desc))
            
    def redraw_stages(active_idx, done_set):
        """Helper to dynamically update CSS classes (active/done) on pipeline stage cards."""
        for card_i, (ph, num, name, desc) in enumerate(stage_placeholders):
            if card_i in done_set:
                css    = "stage-card done"
                prefix = "&#x2714; "
            elif card_i == active_idx:
                css    = "stage-card active"
                prefix = "&#x25B6; "
            else:
                css    = "stage-card"
                prefix = ""
            ph.markdown(f"""
            <div class="{css}">
                <div class="stage-label">Stage {num}</div>
                <div class="stage-name">{prefix}{name}</div>
                <div class="stage-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    agree = st.checkbox("I acknowledge this will overwrite existing telemetry and run for ~15 hours.")
    
    if st.button("Initialize Master Pipeline", disabled=not agree, type="primary"):
        # Explicit memory cleanup before initiating the heavy OS subprocess
        if "brain" in st.session_state: del st.session_state["brain"]
        if "mouth" in st.session_state: del st.session_state["mouth"]
        import gc; gc.collect()
        time.sleep(1)
        
        # Clear legacy backups
        backup_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "system_telemetry.backup")
        if os.path.exists(backup_file):
            try: os.remove(backup_file)
            except Exception: pass
            
        st.markdown("---")
        progress_bar = st.progress(0)
        status_box   = st.empty()
        status_box.markdown(
            '<div class="pipeline-status">&#x25B6;&nbsp; INITIALIZING MASTER ORCHESTRATION...</div>',
            unsafe_allow_html=True
        )
        st.markdown('<div class="topology-label" style="margin: 0.8rem 0 0.4rem 0;">&#x25B8; Live Execution Telemetry</div>', unsafe_allow_html=True)
        log_container = st.empty()
        
        # Execute run_evaluation_pipeline.py as an isolated background process
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            [sys.executable, "run_evaluation_pipeline.py"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", env=env
        )
        
        log_lines         = []
        current_stage_idx = -1
        completed_indices = set()
        pipeline_finished = False
        
        try:
            # Asynchronous loop to intercept and display stdout stream in the UI
            for line in iter(process.stdout.readline, ''):
                clean_line = line.strip()
                if clean_line:
                    log_lines.append(clean_line)
                    
                # Match stdout against predefined stage keywords to advance UI
                for s_i, (num, keyword, name, desc, pct) in enumerate(PIPELINE_STAGES):
                    if keyword in clean_line and s_i != current_stage_idx:
                        if current_stage_idx >= 0:
                            completed_indices.add(current_stage_idx)
                        current_stage_idx = s_i
                        progress_bar.progress(pct)
                        status_box.markdown(
                            f'<div class="pipeline-status">'
                            f'&#x25B6;&nbsp; STAGE {num} ACTIVE &nbsp;&nbsp;|&nbsp;&nbsp; {name} &nbsp;&mdash;&nbsp; {desc}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        redraw_stages(current_stage_idx, completed_indices)
                        break
                        
                # Handle pipeline completion signal
                if SUCCESS_KEY in clean_line:
                    pipeline_finished = True
                    completed_indices = set(range(len(PIPELINE_STAGES)))
                    current_stage_idx = -1
                    progress_bar.progress(100)
                    status_box.markdown(
                        '<div class="pipeline-status done">'
                        '&#x2714;&nbsp; PIPELINE COMPLETE &nbsp;&nbsp;|&nbsp;&nbsp; All artifacts rendered successfully.'
                        '</div>',
                        unsafe_allow_html=True
                    )
                    redraw_stages(-1, completed_indices)
                    
                # Render tail of the log stream
                log_container.code("\n".join(log_lines[-20:]), language="shell")
                
        except BaseException as e:
            # Handle user pressing the Streamlit "Stop" button gracefully
            if type(e).__name__ in ["StopException", "ScriptRunException", "KeyboardInterrupt"]:
                process.terminate()
                status_box.markdown(
                    '<div class="pipeline-status stopped">'
                    '&#x25A0;&nbsp; PIPELINE TERMINATED BY USER &nbsp;&nbsp;|&nbsp;&nbsp; '
                    'Execution halted. Partial telemetry may exist.'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.warning("Pipeline execution was stopped manually. Run the full pipeline again to ensure complete telemetry.")
                raise e
        finally:
            if process.poll() is None:
                process.terminate()
        
        process.wait()
        return_code = process.returncode
        
        # Throw critical exception if subprocess dies unexpectedly
        if not pipeline_finished and return_code not in (None, -15, 1, 2, 130):
            status_box.markdown(
                f'<div class="pipeline-status error">'
                f'&#x2717;&nbsp; CRITICAL EXCEPTION &nbsp;&nbsp;|&nbsp;&nbsp; Exit code {return_code}'
                f'</div>',
                unsafe_allow_html=True
            )
            st.error(f"Process halted with exit code `{return_code}`. Review the telemetry above for the failure point.")

# ==========================================================
# PAGE 4: VECTOR DB INSPECTOR
# ==========================================================
elif active_page == "Vector DB Inspector":
    st.markdown('<div class="section-header">&#x25C6; Vector Database Inspector</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Read-only audit of isolated FAISS memory partitions. No writes permitted from this interface.</div>', unsafe_allow_html=True)
    
    col_sel, col_btn = st.columns([2, 1], gap="medium")
    with col_sel:
        index_to_view = st.selectbox("Target Partition", ["emma_private", "max_private", "group_shared"])
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run_query = st.button("Retrieve Index Contents", type="primary")
        
    if run_query:
        # Construct path explicitly and enforce read-only deserialization
        memory_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_data")
        index_file  = os.path.join(memory_path, f"{index_to_view}.faiss")
        
        if not os.path.exists(index_file):
            st.warning(f"Partition `{index_to_view}` not found or is empty.")
        else:
            with st.spinner(f"Deserializing FAISS index — `{index_to_view}`..."):
                embeddings = OllamaEmbeddings(model="llama3")
                db = FAISS.load_local(memory_path, embeddings, index_to_view, allow_dangerous_deserialization=True)
                # K=100 arbitrarily chosen to ensure full retrieval of seed fragments
                results = db.similarity_search("context", k=100)
            if not results:
                st.info("Index is structurally valid but contains 0 vectors.")
            else:
                st.success(f"Retrieved **{len(results)}** memory fragments from `{index_to_view}`.")
                for i, doc in enumerate(results):
                    st.code(f"[{i+1:02d}]  {doc.page_content.strip()}", language="text")
                    
    st.markdown("---")
    st.markdown('<div class="section-header" style="font-size:0.88rem; color:var(--accent-red);">&#x26A0; Destructive Actions</div>', unsafe_allow_html=True)
    st.warning("**Hard Reset** purges all conversational state and restores memory indices to their immutable seed configuration. This cannot be undone.")
    
    # Destructive purge function to validate system capability to recover from drift
    if st.button("Hard Reset Vector Indices", type="secondary"):
        with st.spinner("Purging state and rebuilding persistence layer..."):
            build_memory_indices()
            st.session_state.messages = []
            if "brain" in st.session_state: del st.session_state["brain"]
            if "mouth" in st.session_state: del st.session_state["mouth"]
        st.success("Memory state restored to default seed configuration.")

# ==========================================================
# PAGE 5: AUDITOR VALIDATION
# ==========================================================
elif active_page == "Auditor Validation":
    st.markdown('<div class="section-header">&#x25B3; Statistical Auditor Validation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Computes Inter-Rater Reliability (IRR) between the algorithmic auditor and human-in-the-loop ground truth labels.</div>', unsafe_allow_html=True)
    
    col_info, col_action = st.columns([2, 1], gap="medium")
    with col_info:
        st.markdown("""
        <div class="channel-box">
            <div class="channel-label">Validation Method</div>
            <span class="channel-value">Cohen's Kappa Coefficient</span>
            Measures agreement between the automated privacy auditor and manually labelled ground truth records, accounting for chance agreement.
        </div>
        """, unsafe_allow_html=True)
    with col_action:
        st.markdown("<br>", unsafe_allow_html=True)
        run_kappa = st.button("Compute Kappa Matrix", type="primary")
        
    if run_kappa:
        base_dir       = os.path.dirname(os.path.abspath(__file__))
        human_logs     = os.path.join(base_dir, "src", "data", "paper_logs", "human_audit_set.csv")
        metrics_output = os.path.join(base_dir, "paper_results", "hpa_validation_metrics.csv")
        
        if not os.path.exists(human_logs):
            st.error(f"Ground truth dataset not found: `{human_logs}`")
        else:
            # Trigger isolated metric calculation script
            with st.spinner("Computing Cohen's Kappa — running statistical models..."):
                silent_subprocess([sys.executable, "src/calculate_audit_metrics.py", human_logs])
                
            if os.path.exists(metrics_output):
                st.markdown("---")
                st.markdown('<div class="section-header" style="font-size:0.88rem;">Validation Output</div>', unsafe_allow_html=True)
                df = pd.read_csv(metrics_output)
                st.dataframe(df, use_container_width=True)
            else:
                st.error("Metrics output file not generated. Check subprocess logs.")