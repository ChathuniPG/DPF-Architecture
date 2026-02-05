"""
view_memory.py
--------------
Database Inspection & Integrity Audit Tool.

This utility provides a read-only interface to inspect the contents of the 
local FAISS vector indices. It serves as an administrative tool to verify:
1. Data Segregation: ensuring private data resides ONLY in authorized indices.
2. Index Integrity: confirming that memory commits are persisting correctly.

Usage:
Executes a broad similarity search to retrieve and display the stored 
knowledge fragments from each segregated memory partition.
"""

import os
import sys

# 1. Setup Path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 2. Import Libraries
try:
    from langchain_community.vectorstores import FAISS
    # Try importing new Ollama class, fallback to old if needed
    from langchain_community.embeddings import OllamaEmbeddings
except ImportError:
    print(" [Error] Required libraries not found.")
    sys.exit(1)

# Configuration: Point to the main folder containing the .faiss files
MEMORY_FOLDER = os.path.join(project_root, "memory_data")

def view_database(index_filename):
    print(f"\n{'='*60}")
    print(f"   AUDIT TARGET: {index_filename}")
    print(f"{'='*60}")
    
    # Check if the specific .faiss file exists
    file_path = os.path.join(MEMORY_FOLDER, f"{index_filename}.faiss")
    
    if not os.path.exists(file_path):
        print(f" !! [Warning] File not found: {file_path}")
        print(f"    (The system has not saved this memory yet.)")
        return

    try:
        print("   [Status] Loading vector store...")
        embeddings = OllamaEmbeddings(model="llama3")
        
        # FIX: 
        # folder_path = MEMORY_FOLDER (Where the files are)
        # index_name = index_filename (The name of the file without extension)
        db = FAISS.load_local(
            folder_path=MEMORY_FOLDER, 
            embeddings=embeddings, 
            index_name=index_filename,
            allow_dangerous_deserialization=True
        )
        
        # Search for everything
        results = db.similarity_search("context", k=100)
        
        if len(results) == 0:
            print("   [Status] Index is empty (No vectors found).")
        else:
            print(f"   [Status] Found {len(results)} memories:\n")
            for i, doc in enumerate(results): 
                print(f"   [{i+1:02d}] {doc.page_content.strip()}")
                
    except Exception as e:
        print(f" !! [Error] Failed to load: {e}")

if __name__ == "__main__":
    # Note: These names must match your .faiss filenames exactly
    view_database("emma_private")
    view_database("max_private")
    view_database("group_shared")