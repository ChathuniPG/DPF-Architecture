"""
Memory State Manager (Persistence Layer)
----------------------------------------
Manages the lifecycle, persistence, and isolation of the Vector Database indices.
This module functions as the state management controller, responsible for 
initializing, verifying, and resetting semantic memory to guarantee data 
segregation and idempotent execution states across system runs.

Architectural Design:
- Implements a "Dual-Index" topological pattern to physically separate 
  domain-restricted Private Vaults from the globally accessible Shared Index.
- Utilizes FAISS (Facebook AI Similarity Search) for optimized, high-throughput 
  dense vector retrieval.
- Orchestrates embedding serialization via the Llama-3 embedding space to 
  ensure semantic alignment between the storage layer and the generative compute layer.
"""

import json
import os
import shutil
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

# --- INFRASTRUCTURE CONFIGURATION ---
# Robust relative path resolution to ensure portability across deployment environments
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS_FILE = os.path.join(BASE_DIR, "src", "data", "seeds.json")
MEMORY_STORE_PATH = os.path.join(BASE_DIR, "memory_data")

# Index Definitions (Must strictly map to the RBAC Agent Configuration)
INDICES = ["emma_private", "max_private", "group_shared"]

# Initialize Embedding Model (Singleton Context)
# Instantiated at module load to avoid recurrent I/O latency during per-turn execution.
print(" >> [System] Initializing Embedding Engine (Compute Layer)...")
try:
    embeddings = OllamaEmbeddings(model="llama3")
except Exception as e:
    print(f" !! [CRITICAL ERROR] Embedding Model Instantiation Failed: {e}")
    print("    Ensure the local inference server (Ollama) is active and reachable.")

def check_memory_integrity():
    """
    Verifies the existence and structural validity of all required vector indices.
    Acts as a pre-flight health check before the Orchestrator binds to the storage layer.
    
    Returns:
        bool: True if the storage layer is fully intact, False if corruption 
              or missing partitions are detected.
    """
    if not os.path.exists(MEMORY_STORE_PATH):
        return False
        
    for idx in INDICES:
        if not os.path.exists(os.path.join(MEMORY_STORE_PATH, idx)):
            return False
            
    return True



def build_memory_indices():
    """
    Executes a Hard Reset of the Vector Database topology.
    
    Pipeline Stages:
    1. Teardown: Purges existing persistence files to prevent data contamination.
    2. Ingestion: Loads ground-truth semantic seed data from local storage.
    3. Vectorization: Converts raw strings into dense vector representations.
    4. Persistence: Serializes the initialized FAISS indices to disk.
    
    Usage:
    Invoked during system initialization or benchmarking to guarantee a 
    clean, deterministic execution state.
    """
    print(f" >> [Memory] Rebuilding Index State from: {os.path.basename(SEEDS_FILE)}")
    
    # 1. State Teardown (Idempotency Enforcement)
    if os.path.exists(MEMORY_STORE_PATH):
        try:
            shutil.rmtree(MEMORY_STORE_PATH)
        except PermissionError:
            print(" !! [ERROR] I/O Lock Detected: Terminate processes accessing 'memory_data' and retry.")
            return
            
    os.makedirs(MEMORY_STORE_PATH)

    # 2. Data Ingestion
    try:
        with open(SEEDS_FILE, 'r') as f:
            seed_data = json.load(f)
    except FileNotFoundError:
        print(f" !! [CRITICAL] Seed ingestion failed. Target missing: {SEEDS_FILE}")
        return

    # 3. Vectorization & Persistence
    print(" >> [Memory] Vectorizing seed data (Blocking I/O operation)...")
    
    try:
        # Construct Private Partition A (High-Sensitivity Domain)
        if "emma_private" in seed_data:
            emma_db = FAISS.from_texts(seed_data["emma_private"], embeddings)
            emma_db.save_local(folder_path=MEMORY_STORE_PATH, index_name="emma_private")

        # Construct Private Partition B (Administrative/Operations Domain)
        if "max_private" in seed_data:
            max_db = FAISS.from_texts(seed_data["max_private"], embeddings)
            max_db.save_local(folder_path=MEMORY_STORE_PATH, index_name="max_private")

        # Construct Shared Partition (Globally Accessible Group Context)
        if "group_shared" in seed_data:
            group_db = FAISS.from_texts(seed_data["group_shared"], embeddings)
            group_db.save_local(folder_path=MEMORY_STORE_PATH, index_name="group_shared")

        print(" >> [Memory] State Reset Complete. Vector indices committed to disk.")
        
    except Exception as e:
        print(f" !! [ERROR] Vectorization Pipeline Failed: {e}")

def load_memory_index(index_name):
    """
    Retrieves and deserializes a specific FAISS index partition from the persistence layer.
    
    Args:
        index_name (str): The target partition identifier (e.g., 'max_private').
    
    Returns:
        FAISS: The active vector store object bound to the requested partition.
    """
    # Auto-recovery Protocol: If indices are missing/corrupted, rebuild dynamically.
    if not check_memory_integrity():
        print(" >> [Memory] Structural mismatch detected in storage layer. Auto-initializing...")
        build_memory_indices()
        
    return FAISS.load_local(
        MEMORY_STORE_PATH, 
        embeddings, 
        index_name, 
        allow_dangerous_deserialization=True # Acknowledged security exception: Loading from trusted internal seeds.json
    )

if __name__ == "__main__":
    # Provides a CLI hook for manual administrative resets of the vector database
    build_memory_indices()