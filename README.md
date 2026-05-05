# Deterministic Privacy Firewall (DPF) for Multi-Agent LLM Systems

**Evaluation & Reproducibility Repository — V2**

This repository contains the official implementation, evaluation pipeline, and reproducibility suite for the **Deterministic Privacy Firewall (DPF)** architecture — a pre-generation information flow control mechanism for mitigating cross-context data leakage in multi-agent LLM systems. The system is validated through rigorous adversarial benchmarking (N=1000 trials), benign utility auditing (N=200 trials), and dual human annotator inter-rater reliability assessment.

---

## What's New in V2

| Change | Details |
|---|---|
| **4th architectural mode** | `STANDARD_POSTHOC_NLI` — post-hoc baseline with DeBERTa NLI detection, matching the DPF's detection power for a fair comparison |
| **Cross-model evaluation** | Gemma 3 (4B, Google DeepMind) added as a second backend alongside Llama-3-8B-Instruct, validating model-agnostic structural guarantees |
| **Doubled trial count** | N=1000 total (800 adversarial + 200 benign), up from N=500 in V1 |
| **Bootstrap CIs** | Per-threat-class leakage rates reported with 95% bootstrap confidence intervals |
| **Router accuracy analysis** | Precision, recall, and F1 per agent role measured independently from end-to-end leakage outcome |
| **Timing normalisation** | `TimingNormalizer` implemented to mitigate binary timing side-channel leakage |
| **FRR bug fix** | V1 ran pre-generation firewall and post-generation filter simultaneously in DPF mode, causing double-jeopardy false refusals. Now mutually exclusive |
| **V2 Fast Path** | Dedicated Fast Path option synthesizes figures and tables from `src/data/paper_logs_v2/` into `paper_results_v2/` |
| **System Demo panel** | New UI panel with a structured five-phase guided walkthrough of the core DPF claims, firing live inference turns |

---

## 🛠️ 1. Prerequisites & Environment Setup

### A. Core Requirements

- **Python 3.13.5** — Required for LangChain and FAISS compatibility.
  [Download Python 3.13](https://www.python.org/downloads/)
- **Ollama** — Required for local, privacy-preserving LLM inference.
  [Download Ollama](https://ollama.com/download)
- **Primary LLM** — Llama-3-8B-Instruct (4-bit GGUF), the primary evaluation backend.
- **Secondary LLM** — Gemma 3 4B (Google DeepMind), the cross-model sensitivity backend.

### B. Setup Instructions

**Step 1: Install Ollama and pull the required models**

```bash
# Primary backend (Llama-3-8B-Instruct, 4-bit quantized)
ollama pull llama3

# Secondary backend (Gemma 3 4B — required for cross-model sensitivity analysis)
ollama pull gemma3:4b
```

**Step 2: Obtain the repository**

Access the repository via the permanent Zenodo DOI link provided in the manuscript's Data Availability Statement. Download `DPF-ARCHITECTURE.zip`, extract it, and navigate into the extracted root directory:

```bash
cd DPF-ARCHITECTURE
```

Ensure your terminal is inside the folder containing `run_system.py` and `app.py`.

**Step 3: Configure the Python environment**

```bash
# Create and activate a virtual environment
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install streamlit
```

---

## 🚀 2. System Execution Guide

Two interfaces are provided: a graphical web dashboard and a terminal CLI.

### ⚠️ Important: Always launch the web UI with `streamlit run`, not `python`

```bash
# Correct
streamlit run app.py

# Incorrect — will not open the browser interface
python app.py
```

### 🖥️ Option A: Graphical Web Interface (Streamlit)

```bash
streamlit run app.py
```

The dashboard provides six panels accessible from the left sidebar:

| Panel | Description |
|---|---|
| **HITL Debug Console** | Real-time agent interaction with configurable system mode and LLM backend. Supports both GROUP (global broadcast) and PRIVATE (dyadic) routing topologies. |
| **Fast Path — Artifacts** | Renders figures and tables from immutable telemetry. Supports two result sets: **V1** (original paper, N=500) and **V2** (extended multi-model, N=1000). |
| **E2E Evaluation Pipeline** | Executes the full ~15-hour adversarial + benign evaluation pipeline with live progress tracking. |
| **Vector DB Inspector** | Read-only audit of isolated FAISS memory partitions. Includes a hard reset option. |
| **Auditor Validation** | Computes Cohen's Kappa IRR between the Hybrid Privacy Auditor and human ground-truth labels. |
| **System Demo** | Structured five-phase guided walkthrough of the core DPF claims. Fires live inference turns — responses are real, not scripted. |

#### System Demo Panel

The System Demo panel is the primary interface for live architectural demonstrations. It progresses through five phases in sequence:

| Phase | Mode | What it shows |
|---|---|---|
| 1 | NAIVE_CONTROL | Academic/financial data (GPA, probation, tuition) exposed via group query — Max vault |
| 2 | NAIVE_CONTROL | Health/medical data (GAD diagnosis, prescription, triggers) exposed — Emma vault |
| 3 | DPF_PROPOSED | Same academic query intercepted pre-generation — [GPA_REDACTED], [ACADEMIC_STATUS] |
| 4 | DPF_PROPOSED | Same health query blocked via RBAC cross-vault isolation |
| 5 | DPF_PROPOSED | Benign syllabus query passes without redaction — FRR demonstration |

Each phase displays its system context, the suggested input prompt, the expected outcome, and a **Run Phase** button that fires a live inference turn. The mode switches automatically between phases. A full transcript is displayed on completion with a reset option.

#### Fast Path — V2 Result Set

To use the V2 Fast Path:

1. Create the folder `src/data/paper_logs_v2/`
2. Copy the following files from your `logs/` folder (generated after a complete V2 pipeline run) into it:
   - `experiment_data.csv`
   - `audit_results.csv`
   - `ablation_utility_audit.csv`
   - `router_log_DPF_PROPOSED.csv`
   - `human_audit_set.csv`
3. Select **V2** in the Fast Path panel and click **Execute V2 Artifact Synthesis**

Outputs are written to `paper_results_v2/` — separate from the V1 outputs in `paper_results/`.

### ⌨️ Option B: Terminal CLI

```bash
python run_system.py
```

Available options:

| Option | Description |
|---|---|
| **1 — Fast Path (V1)** | Generate V1 reference figures and tables from `src/data/paper_logs/` |
| **2 — Full Pipeline** | Re-run the complete N=1000 evaluation (~15 hours) |
| **3 — Interactive Console** | HITL terminal REPL with `/demo`, `/mode`, `/backend`, `/private` commands |
| **4 — Vector DB Inspection** | Read-only audit of FAISS memory partitions |
| **5 — Auditor Validation** | Compute Cohen's Kappa against human ground-truth labels |

The terminal console includes a `/demo` command that runs the same five-phase demonstration as the UI demo panel, accepting live keyboard input at each phase.

---

## 📁 3. Repository Architecture

```text
DPF-ARCHITECTURE/
│
├── app.py                         # Streamlit web dashboard (v2.1) — 6 panels
├── run_system.py                  # Primary terminal CLI
├── run_evaluation_pipeline.py     # Master orchestrator for the full evaluation pipeline
├── requirements.txt               # pip dependencies
├── README.md                      # This file
│
├── logs/                          # [Auto-generated] Runtime telemetry from full pipeline runs
├── memory_data/                   # [Auto-generated] Ephemeral FAISS vector indices
├── paper_results/                 # [Auto-generated] V1 synthesized tables and figures
├── paper_results_v2/              # [Auto-generated] V2 synthesized tables and figures
│
└── src/                           # Core system source code
    │
    ├── data/
    │   ├── paper_logs/            # IMMUTABLE — V1 ground-truth telemetry (Fast Path V1 source)
    │   │   ├── ablation_utility_audit.csv
    │   │   ├── audit_results.csv
    │   │   ├── experiment_data.csv
    │   │   └── human_audit_set.csv
    │   │
    │   ├── paper_logs_v2/         # V2 raw telemetry (Fast Path V2 source — copy from logs/)
    │   │   ├── experiment_data.csv
    │   │   ├── audit_results.csv
    │   │   ├── ablation_utility_audit.csv
    │   │   ├── router_log_DPF_PROPOSED.csv
    │   │   └── human_audit_set.csv
    │   │
    │   ├── adversarial_dataset.json   # N=400 adversarial prompt vectors (per backend)
    │   ├── seeds.json                 # Seed knowledge base and private PII for agent vaults
    │   └── utility_prompts.json       # N=100 benign queries for false refusal rate testing
    │
    ├── dpf_core/
    │   ├── firewall_config.py         # Regex rules and semantic policy definitions
    │   └── privacy_firewall.py        # Deterministic O(L) filtering engine
    │
    ├── evaluation/                    # Evaluation harnesses (separated from core system)
    │   ├── evaluate_privacy_audit.py  # Multi-layer NLI + Regex leakage detection
    │   ├── experiment_driver.py       # Adversarial trial execution harness
    │   ├── run_auditor_ablation.py    # Layer-wise auditor ablation + bootstrap CIs
    │   └── run_utility_benchmark.py   # Benign false-refusal rate evaluator
    │
    ├── agent_config.py                # Agent persona and RBAC domain definitions
    ├── agent.py                       # LangChain LLM bindings and retrieval logic
    ├── calculate_audit_metrics.py     # Cohen's Kappa IRR computation
    ├── generate_dataset.py            # Utility: build adversarial JSON prompt vectors
    ├── human_audit.py                 # Utility: stratified sampling for human annotation
    ├── interactive_console.py         # Terminal HITL console with /demo command (V5)
    ├── memory_manager.py              # FAISS index lifecycle (build, load, HNSW conversion)
    ├── metrics_logger.py              # Sub-millisecond telemetry capture with V2 columns
    ├── orchestrator.py                # Semantic router, timing normalizer, filter orchestration
    ├── system_registry.py             # Architectural mode and backend configuration registry
    ├── view_memory.py                 # Diagnostic: read isolated vector index contents
    └── visualization_engine.py        # Matplotlib charts and final CSV table rendering
```

---

## 🔒 4. Architectural Modes

Four architectural configurations are supported, switchable at runtime via the UI or `/mode` command:

| Mode | Enforcement Stage | Determinism | Description |
|---|---|---|---|
| `NAIVE_CONTROL` | None | None | Shared global index, no access control |
| `STANDARD_POSTHOC` | Post-generation | Partial (Regex) | Partitioned memory with reactive Regex egress filter |
| `STANDARD_POSTHOC_NLI` | Post-generation | Partial (Regex + NLI) | Partitioned memory with Regex + DeBERTa NLI egress filter |
| `DPF_PROPOSED` | Pre-generation | Rule-based, auditable | RBAC + dual-index partitioning + PII masking before LLM inference |

---

## 🖥️ 5. LLM Backends

| Backend | Description | Pull command |
|---|---|---|
| `llama3` | Llama-3-8B-Instruct · 4-bit GGUF · Primary evaluation backend | `ollama pull llama3` |
| `gemma3:4b` | Gemma 3 4B · Google DeepMind · Cross-model sensitivity backend | `ollama pull gemma3:4b` |

---

## 📊 6. Key V2 Results

| Architecture | Leakage Rate | Backends | N |
|---|---|---|---|
| Naive Baseline | 62.10% | llama3, gemma3:4b | 800 |
| Post-Hoc (Regex) | 17.10% | llama3, gemma3:4b | 800 |
| Post-Hoc (Regex+NLI) | 9.80% | llama3, gemma3:4b | 800 |
| **DPF Architecture** | **10.10%** | llama3, gemma3:4b | 800 |

Post-Hoc (Regex) vs DPF: OR=1.8, p<0.001, Small effect.
Post-Hoc (Regex+NLI) vs DPF: OR=1.0, n.s., Negligible effect.

Mean firewall evaluation latency: **0.36 ms** (<0.01% of total turn latency).
Timing-normalised rejections applied: **642** of 1000 turns.

---

## 🗂️ 7. Ephemeral Directories

`logs/`, `memory_data/`, `paper_results/`, and `paper_results_v2/` are excluded from version control via `.gitignore` and created dynamically at runtime. The immutable V1 reference data is in `src/data/paper_logs/`. V2 raw telemetry is placed manually in `src/data/paper_logs_v2/` after a completed pipeline run.

---

## 📋 8. Data Availability

The codebase, full prompt dataset (N=1000), synthetic PII injection schema, firewall policy configurations, HPA scoring scripts, and agent configuration profiles are publicly available at the permanent Zenodo DOI provided in the manuscript's Data Availability Statement.