"""
Memory State Manager (Persistence Layer) — V2
----------------------------------------------
Manages the lifecycle, persistence, and isolation of FAISS vector indices.

V2 Changes:
- load_memory_index() now accepts index_type="flat" (default, V1 behaviour)
  or index_type="hnsw". HNSW conversion lives here because memory_manager
  owns all FAISS lifecycle operations. The Orchestrator simply calls
  load_memory_index(name, index_type) and receives the correct index type.
- _convert_to_hnsw() is a module-level helper (not buried in Orchestrator).
- build_memory_indices() is unchanged — initial build always uses FlatL2
  because HNSW in FAISS does not support incremental add after construction.
  Writes (save_turn) also use FlatL2 for the same reason.
"""

import json
import os
import shutil
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS_FILE = os.path.join(BASE_DIR, "src", "data", "seeds.json")
MEMORY_STORE_PATH = os.path.join(BASE_DIR, "memory_data")

INDICES = ["emma_private", "max_private", "group_shared"]

print(" >> [System] Initializing Embedding Engine (Compute Layer)...")
try:
    embeddings = OllamaEmbeddings(model="llama3")
except Exception as e:
    print(f" !! [CRITICAL ERROR] Embedding Model Failed: {e}")
    print("    Ensure Ollama is running.")


def check_memory_integrity() -> bool:
    if not os.path.exists(MEMORY_STORE_PATH):
        return False
    for idx in INDICES:
        if not os.path.exists(os.path.join(MEMORY_STORE_PATH, idx)):
            return False
    return True


def _convert_to_hnsw(faiss_db: FAISS) -> FAISS:
    """
    Converts a loaded FAISS FlatL2 index to HNSW in-place.

    HNSW provides O(log n) approximate nearest-neighbour retrieval,
    suitable for enterprise-scale deployments (≥10^6 vectors).

    Parameters chosen to balance recall and construction cost:
        M=32         — number of bidirectional links per node
        efConstruction=200 — search width during graph construction
        efSearch=64  — search width during query time

    Falls back to the original FlatL2 index if conversion fails
    (e.g., faiss not installed with HNSW support).

    Note: HNSW does not support add() after construction. All write
    paths (save_turn, build_memory_indices) continue to use FlatL2.
    """
    try:
        import faiss as faiss_lib

        old_index = faiss_db.index
        d = old_index.d
        n = old_index.ntotal

        if n == 0:
            return faiss_db

        vectors = np.zeros((n, d), dtype=np.float32)
        old_index.reconstruct_n(0, n, vectors)

        hnsw = faiss_lib.IndexHNSWFlat(d, 32)
        hnsw.hnsw.efConstruction = 200
        hnsw.hnsw.efSearch = 64
        hnsw.add(vectors)

        faiss_db.index = hnsw
        print(f"    [HNSW] Converted: {n} vectors, d={d}")
        return faiss_db

    except Exception as e:
        print(f"    [HNSW] Conversion failed ({e}), keeping FlatL2.")
        return faiss_db


def build_memory_indices():
    """
    Hard reset of the vector database topology.
    Always builds FlatL2 indices (HNSW does not support incremental add).
    """
    print(f" >> [Memory] Rebuilding from: {os.path.basename(SEEDS_FILE)}")

    if os.path.exists(MEMORY_STORE_PATH):
        try:
            shutil.rmtree(MEMORY_STORE_PATH)
        except PermissionError:
            print(" !! I/O Lock: terminate processes using 'memory_data' and retry.")
            return

    os.makedirs(MEMORY_STORE_PATH)

    try:
        with open(SEEDS_FILE, 'r') as f:
            seed_data = json.load(f)
    except FileNotFoundError:
        print(f" !! [CRITICAL] Seed file missing: {SEEDS_FILE}")
        return

    print(" >> [Memory] Vectorizing seed data...")
    try:
        if "emma_private" in seed_data:
            db = FAISS.from_texts(seed_data["emma_private"], embeddings)
            db.save_local(MEMORY_STORE_PATH, "emma_private")

        if "max_private" in seed_data:
            db = FAISS.from_texts(seed_data["max_private"], embeddings)
            db.save_local(MEMORY_STORE_PATH, "max_private")

        if "group_shared" in seed_data:
            db = FAISS.from_texts(seed_data["group_shared"], embeddings)
            db.save_local(MEMORY_STORE_PATH, "group_shared")

        print(" >> [Memory] Vector indices committed to disk.")
    except Exception as e:
        print(f" !! Vectorization failed: {e}")


def load_memory_index(index_name: str, index_type: str = "flat") -> FAISS:
    """
    Loads a FAISS index partition from disk.

    Args:
        index_name: partition identifier ("emma_private", "max_private",
                    "group_shared").
        index_type: "flat" returns IndexFlatL2 (exact search, default,
                    V1 behaviour).
                    "hnsw" converts the loaded index to HNSW before
                    returning it (O(log n) ANN retrieval).

    Returns:
        FAISS vector store bound to the requested partition.
    """
    if not check_memory_integrity():
        print(" >> [Memory] Integrity check failed. Auto-initializing...")
        build_memory_indices()

    db = FAISS.load_local(
        MEMORY_STORE_PATH,
        embeddings,
        index_name,
        allow_dangerous_deserialization=True
    )

    if index_type == "hnsw":
        db = _convert_to_hnsw(db)

    return db


if __name__ == "__main__":
    build_memory_indices()
