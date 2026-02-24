# Deterministic Privacy Firewall (DPF) for Multi-Agent Systems

**Evaluation & Reproducibility Repository**

This repository contains the official implementation, evaluation pipeline, and automated reproducibility suite for the **Deterministic Privacy Firewall (DPF)** architecture. The system demonstrates a robust Information Flow Control (IFC) mechanism for mitigating cross-context data leakage in multi-agent LLM architectures, validated through rigorous adversarial and benign utility benchmarking.

---

## 🛠️ 1. Prerequisites & Environment Setup

To ensure strict zero-shot reproducibility and avoid dependency conflicts, please match the following environment specifications:

### A. Core Requirements

* **Python (v3.13.5):** Required for specific LangChain and FAISS Vector DB compatibility.
* 🔗 [Download Python 3.13 here](https://www.python.org/downloads/)
* **Ollama Engine:** Required for local, privacy-preserving LLM inference.
* 🔗 [Download Ollama here](https://ollama.com/download)
* **LLM Model:** We utilize the 4-bit quantized **Llama-3-8B-Instruct** model to balance computational efficiency with high-fidelity reasoning.

### B. Setup Instructions

**Step 1: Install Ollama & Pull the Model**
Ensure Ollama is installed and running in the background. Open a terminal and execute the following command (an active internet connection is required to pull the model weights):

```bash
ollama run llama3

```

*(Note: This automatically downloads the exact `Llama-3-8B-Instruct (4-bit)` model used in our manuscript's evaluation).*

**Step 2: Acquire the Repository**
Access the repository via the secure reviewer link provided in the manuscript.

1. Click the **Download ZIP** option on the web interface to download the repository archive.
2. Extract the `.zip` file to your local machine.
3. Open your terminal or command prompt and navigate into the extracted root directory:

```bash
cd DPF-ARCHITECTURE

```

*(Note: Depending on your extraction tool, the folder might be named slightly differently, such as `DPF-ARCHITECTURE-main`. Please ensure your terminal is inside the folder containing `run_system.py`).*

**Step 3: Configure the Python Environment**
We highly recommend isolating dependencies using a virtual environment. **Please ensure you have an active internet connection** so the package manager can download the required libraries.

```bash
# Create and activate a virtual environment
python -m venv venv

# For MacOS/Linux:
source venv/bin/activate  
# For Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

---

## 🚀 2. System Execution Guide (Reviewer CLI)

We have engineered a "Zero-Friction" entry point. Reviewers do not need to manually chain scripts together. **Everything is orchestrated through a single command:**

```bash
python run_system.py

```

Upon execution, you will be presented with a terminal UI containing the following options:

### ⚡ Option 1: Generate Reference Figures & Tables (Fast Path)

**Recommended for Reviewers.** This instantly generates all publication-ready `.csv` tables and `.png` charts into the `/paper_results` folder. It bypasses the 15-hour computational loop by utilizing the immutable reference telemetry (`src/data/paper_logs/`) captured during the manuscript's original stress tests.

### 🐢 Option 2: Re-run Full Evaluation Pipeline (Slow Path)

Executes the complete $N=1500$ ablation study (1200 Adversarial + 300 Benign trials).

* **Warning:** Requires ~15+ hours of continuous GPU/CPU compute.
* **Note:** Due to the inherent stochasticity of generative LLMs, exact statistical metrics will fluctuate slightly from the published manuscript if re-run. Output is safely routed to the `/logs` directory to prevent overwriting our immutable paper artifacts.

### 💬 Option 3: Interactive Debug Console (Human-in-the-Loop)

Opens a Read-Eval-Print Loop (REPL) allowing you to chat with the multi-agent system in real-time. You can manually test the semantic router, observe the deterministic firewall, and switch between `Group` (Broadcast) and `Private` (Dyadic) topologies using slash commands (e.g., `/help`, `/private Emma`).

### 🗄️ Option 4: Inspect Vector Database State

Provides a read-only audit of the isolated FAISS memory partitions to verify that strict memory segregation is being enforced between domain agents.

### 📊 Option 5: Validate Auditor Accuracy (Cohen's Kappa)

Runs the statistical validation suite comparing our automated AI Hybrid Privacy Auditor (HPA) against Human-in-the-Loop ground-truth labels, generating the Inter-Rater Reliability metrics found in the paper.

---

## 📁 3. Repository Architecture

Below is the structural breakdown of the repository and the specific engineering role of each component.

```text
DPF-ARCHITECTURE/
│
├── run_system.py                 # Primary Reviewer CLI & Environment Controller
├── run_evaluation_pipeline.py    # Master orchestrator for the 15-hour Slow Path test
├── requirements.txt              # Standardized pip dependencies
├── README.md                     # Documentation (You are here)
│
├── logs/                         # [Auto-Generated] Target dir for runtime telemetry 
├── memory_data/                  # [Auto-Generated] Ephemeral FAISS vector storage 
├── paper_results/                # [Auto-Generated] Target dir for final tables and charts
│
└── src/                          # Core System Source Code
    │
    ├── data/                     # Data Layer (Decoupled from logic)
    │   ├── paper_logs/           # IMMUTABLE Ground Truth (Used by Option 1/Fast Path)
    │   │   ├── ablation_utility_audit.csv
    │   │   ├── audit_results.csv 
    │   │   ├── experiment_data.csv 
    │   │   └── human_audit_set.csv
    │   ├── adversarial_dataset.json # N=400 malicious prompt vectors
    │   ├── seeds.json               # Initial knowledge base/secrets for Agent Vector DBs
    │   └── utility_prompts.json     # N=100 benign queries for False Refusal testing
    │
    ├── dpf_core/                 # The Proposed Solution Mechanics
    │   ├── firewall_config.py    # Regex boundaries and semantic rulesets
    │   └── privacy_firewall.py   # O(1) Deterministic filtering engine
    │
    ├── agent_config.py           # Persona and access control definitions
    ├── agent.py                  # LangChain LLM bindings and retrieval logic
    ├── calculate_audit_metrics.py# IRR & Cohen's Kappa statistical validation
    ├── evaluate_privacy_audit.py # Multi-layer NLI DeBERTa leakage detection
    ├── experiment_driver.py      # Adversarial execution harness
    ├── generate_dataset.py       # Utility script to build JSON testing vectors
    ├── human_audit.py            # Developer script for stratified human sampling
    ├── interactive_console.py    # The Real-time Chat UI (Option 3)
    ├── memory_manager.py         # FAISS vector DB instantiation and wiping
    ├── metrics_logger.py         # Sub-millisecond telemetry capture
    ├── orchestrator.py           # The Semantic Router (Control Plane)
    ├── run_auditor_ablation.py   # Failure Mode Attribution (Control vs. Data Plane)
    ├── run_utility_benchmark.py  # Benign False-Refusal Rate (FRR) evaluator
    ├── system_registry.py        # Ablation state controller (Naive vs PostHoc vs DPF)
    ├── view_memory.py            # Diagnostic tool to read isolated vector indices
    └── visualization_engine.py   # Renders matplotlib charts and final CSV tables

```

### 🔒 Note on Ephemeral Directories

To maintain a clean repository footprint, the folders `logs/`, `memory_data/`, and `paper_results/` are ignored via `.gitignore` (if cloning via Git). They are **dynamically created at runtime** by `run_system.py` to store temporary session states, active telemetry, and final graphical artifacts. Your exact, manuscript-matching reference data is safely protected inside `src/data/paper_logs/`.

---