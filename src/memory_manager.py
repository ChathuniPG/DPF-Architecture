"""
Memory State Manager (Persistence Layer)
----------------------------------------
Manages the lifecycle of the Vector Database indices.
Responsible for initializing, verifying, and resetting the semantic memory
state to ensure experimental isolation between trials.

Architecture:
- Implements the "Dual-Index" storage pattern (Private vs. Shared).
- Uses FAISS for high-performance dense vector retrieval.
- Handles embedding serialization via Ollama/Llama-3.
"""

import json
import os
import shutil
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

# --- CONFIGURATION ---
# Robust path resolution relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS_FILE = os.path.join(BASE_DIR, "src", "data", "seeds.json")
MEMORY_STORE_PATH = os.path.join(BASE_DIR, "memory_data")

# Index Definitions (Must match Agent Configuration)
INDICES = ["emma_private", "max_private", "group_shared"]

# Initialize Embedding Model (Singleton)
# We use Llama 3 embeddings to maintain semantic alignment with the generative model.
print(" >> [System] Initializing Embedding Engine (Llama 3)...")
try:
    embeddings = OllamaEmbeddings(model="llama3")
except Exception as e:
    print(f" !! [CRITICAL ERROR] Embedding Model Failed: {e}")
    print("    Ensure Ollama is running.")

def check_memory_integrity():
    """
    Verifies the existence and validity of all required vector indices.
    Returns:
        bool: True if the storage layer is intact, False otherwise.
    """
    if not os.path.exists(MEMORY_STORE_PATH):
        return False
        
    for idx in INDICES:
        if not os.path.exists(os.path.join(MEMORY_STORE_PATH, idx)):
            return False
            
    return True

def build_memory_indices():
    """
    Executes a Hard Reset of the Vector Database.
    1. Purges existing persistence files (Wipe).
    2. Ingests ground-truth data from 'seeds.json' (Load).
    3. Vectorizes and serializes new indices to disk (Persist).
    
    Usage:
    Called at the start of every experimental batch to guarantee Zero-Shot conditions.
    """
    print(f" >> [Memory] Rebuilding Index State from: {os.path.basename(SEEDS_FILE)}")
    
    # 1. State Teardown
    if os.path.exists(MEMORY_STORE_PATH):
        try:
            shutil.rmtree(MEMORY_STORE_PATH)
        except PermissionError:
            print(" !! [ERROR] File Lock: Close programs accessing 'memory_data' and retry.")
            return
            
    os.makedirs(MEMORY_STORE_PATH)

    # 2. Data Ingestion
    try:
        with open(SEEDS_FILE, 'r') as f:
            seed_data = json.load(f)
    except FileNotFoundError:
        print(f" !! [CRITICAL] Seed file missing: {SEEDS_FILE}")
        return

    # 3. Vectorization & Persistence
    print(" >> [Memory] Vectorizing seed data (This may take a moment)...")
    
    try:
        # Construct Private Index A (Emma/Wellness)
        if "emma_private" in seed_data:
            emma_db = FAISS.from_texts(seed_data["emma_private"], embeddings)
            emma_db.save_local(folder_path=MEMORY_STORE_PATH, index_name="emma_private")

        # Construct Private Index B (Max/Logistics)
        if "max_private" in seed_data:
            max_db = FAISS.from_texts(seed_data["max_private"], embeddings)
            max_db.save_local(folder_path=MEMORY_STORE_PATH, index_name="max_private")

        # Construct Shared Index (Group Context)
        if "group_shared" in seed_data:
            group_db = FAISS.from_texts(seed_data["group_shared"], embeddings)
            group_db.save_local(folder_path=MEMORY_STORE_PATH, index_name="group_shared")

        print(" >> [Memory] State Reset Complete. Indices ready.")
        
    except Exception as e:
        print(f" !! [ERROR] Vectorization Process Failed: {e}")

def load_memory_index(index_name):
    """
    Retrieves a deserialized FAISS index from the persistence layer.
    
    Args:
        index_name (str): The target index (e.g., 'max_private').
    
    Returns:
        FAISS: The usable vector store object.
    """
    # Auto-recovery: If indices are missing, build them on the fly.
    if not check_memory_integrity():
        print(" >> [Memory] Index mismatch detected. Auto-initializing...")
        build_memory_indices()
        
    return FAISS.load_local(
        MEMORY_STORE_PATH, 
        embeddings, 
        index_name, 
        allow_dangerous_deserialization=True # Safe here as we control the seeds.json source
    )

if __name__ == "__main__":
    # Allow manual execution to reset the DB
    build_memory_indices()