"""
Memory State Manager (Persistence Layer) — V4
----------------------------------------------
Manages lifecycle, persistence, and isolation of FAISS vector indices.

V4 Bug Fix — Triple Rebuild:
  load_memory_index() previously called check_memory_integrity() and
  build_memory_indices() inside itself. The orchestrator calls
  load_memory_index() three times (group_shared, emma_private, max_private).
  Each call triggered: check_integrity → fail → wipe → rebuild → next call
  checks again → wipe again → rebuild again. Three full rebuilds = ~5 minutes.

  Fix: ensure_memory_ready() does one integrity check and one optional build,
  then sets a module-level flag so subsequent calls within the same session
  skip the check entirely. load_memory_index() calls ensure_memory_ready()
  which short-circuits after the first successful build.

V4 Bug Fix — Ollama crash resilience:
  build_memory_indices() now checks Ollama is reachable before attempting
  embedding calls. If Ollama is down (e.g., killed by OOM), prints a clear
  actionable message instead of a 200-line traceback.

V2 Changes (retained):
  load_memory_index() accepts index_type="flat" or "hnsw".
  _convert_to_hnsw() is the module-level HNSW helper.
"""

import json
import os
import shutil
import numpy as np
import requests
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS_FILE = os.path.join(BASE_DIR, "src", "data", "seeds.json")
MEMORY_STORE_PATH = os.path.join(BASE_DIR, "memory_data")

INDICES = ["emma_private", "max_private", "group_shared"]

# Module-level singleton — instantiated once, reused everywhere.
# This eliminates ~2 seconds of HTTP handshake per embedding model init.
print(" >> [System] Initializing Embedding Engine (Compute Layer)...")
try:
    embeddings = OllamaEmbeddings(model="llama3")
    _embeddings_ready = True
except Exception as e:
    print(f" !! [CRITICAL ERROR] Embedding Model Failed: {e}")
    _embeddings_ready = False

# Session flag: set to True after the first successful ensure_memory_ready()
# so subsequent load_memory_index() calls skip the integrity check entirely.
_memory_confirmed_ready = False


def check_ollama_health() -> bool:
    """
    Lightweight check that Ollama is reachable at localhost:11434.
    Returns True if healthy, False if the server is down or unreachable.
    Used as a pre-flight guard before any embedding or build operation.
    """
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def check_memory_integrity() -> bool:
    if not os.path.exists(MEMORY_STORE_PATH):
        return False
    for idx in INDICES:
        if not os.path.exists(os.path.join(MEMORY_STORE_PATH, idx)):
            return False
    return True


def ensure_memory_ready() -> bool:
    """
    Single-check gate: verifies memory integrity once per session and
    rebuilds if needed. Sets _memory_confirmed_ready = True on success
    so all subsequent calls are instant (no disk check, no rebuild).

    Returns True if memory is ready, False if Ollama is unreachable.
    """
    global _memory_confirmed_ready
    if _memory_confirmed_ready:
        return True

    if not check_memory_integrity():
        print(" >> [Memory] Integrity check failed. Auto-initializing...")
        if not build_memory_indices():
            return False

    _memory_confirmed_ready = True
    return True


def invalidate_memory_cache():
    """
    Call after /reset to force a fresh integrity check on the next
    load_memory_index() call. Without this, the session flag would skip
    the check even after a full wipe.
    """
    global _memory_confirmed_ready
    _memory_confirmed_ready = False


def _convert_to_hnsw(faiss_db: FAISS) -> FAISS:
    """
    Converts a FlatL2 index to HNSW for O(log n) retrieval.
    Falls back to FlatL2 if conversion fails.
    Note: HNSW does not support add() after construction.
    """
    try:
        import faiss as faiss_lib
        old_index = faiss_db.index
        d, n = old_index.d, old_index.ntotal
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


def build_memory_indices() -> bool:
    """
    Hard reset of the vector database.
    Returns True on success, False if Ollama is unreachable.

    V4: checks Ollama health before attempting any embedding calls.
    A dead Ollama server (killed by OOM) previously caused a 200-line
    traceback. Now prints a clear actionable message and returns False.
    """
    print(f" >> [Memory] Rebuilding from: {os.path.basename(SEEDS_FILE)}")

    # Pre-flight: verify Ollama is alive before a slow embedding call fails
    if not check_ollama_health():
        print("\n" + "!" * 55)
        print(" !! OLLAMA SERVER IS NOT RUNNING OR IS UNRESPONSIVE.")
        print(" !!")
        print(" !! This usually means:")
        print(" !!   1. Ollama crashed (out of memory — close other apps)")
        print(" !!   2. Ollama was not started (run: ollama serve)")
        print(" !!")
        print(" !! ACTION: Restart Ollama, then run /reset again.")
        print("!" * 55 + "\n")
        return False

    if os.path.exists(MEMORY_STORE_PATH):
        try:
            shutil.rmtree(MEMORY_STORE_PATH)
        except PermissionError:
            print(" !! I/O Lock: terminate processes using 'memory_data' and retry.")
            return False

    os.makedirs(MEMORY_STORE_PATH)

    try:
        with open(SEEDS_FILE, 'r') as f:
            seed_data = json.load(f)
    except FileNotFoundError:
        print(f" !! [CRITICAL] Seed file missing: {SEEDS_FILE}")
        return False

    print(" >> [Memory] Vectorizing seed data...")
    try:
        for index_name in ["emma_private", "max_private", "group_shared"]:
            if index_name in seed_data:
                db = FAISS.from_texts(seed_data[index_name], embeddings)
                db.save_local(MEMORY_STORE_PATH, index_name)
        print(" >> [Memory] Vector indices committed to disk.")
        return True
    except Exception as e:
        print(f" !! Vectorization failed: {e}")
        print(" !! Is Ollama running? Try: ollama serve")
        return False


def load_memory_index(index_name: str, index_type: str = "flat") -> FAISS:
    """
    Loads a FAISS index partition from disk.

    V4: calls ensure_memory_ready() instead of check_memory_integrity()
    inline. ensure_memory_ready() is a no-op after the first successful
    call in a session, eliminating the triple-rebuild bug.

    Args:
        index_name: "emma_private", "max_private", or "group_shared".
        index_type: "flat" (exact, default) or "hnsw" (approximate).
    """
    if not ensure_memory_ready():
        raise RuntimeError(
            "Memory indices unavailable — Ollama may be down. "
            "Restart Ollama and run /reset."
        )

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
