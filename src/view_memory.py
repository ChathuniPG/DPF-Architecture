"""
Vector Store Inspection & Integrity Audit Utility
-------------------------------------------------
Provides a read-only administrative interface to inspect the raw contents of 
the local FAISS vector indices. 

Architectural Purpose:
1. State Verification: Confirms that conversational state and external data 
   are persisting correctly to the disk storage layer.
2. Data Isolation Audit: Allows system administrators to manually verify that 
   strict domain segregation is maintained (i.e., private data resides exclusively 
   in authorized user vaults and has not leaked into the global shared partition).

Operation:
Executes a broad, low-specificity similarity search to retrieve and render 
the stored knowledge fragments from each requested memory partition.
"""

import os
import sys

# --- 1. INFRASTRUCTURE & PATH RESOLUTION ---
# Ensure robust absolute path resolution so the utility can be executed 
# reliably from any working directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# --- 2. DEPENDENCY INGESTION ---
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import OllamaEmbeddings
except ImportError:
    print(" [CRITICAL] Required dependencies not found.")
    print(" Ensure 'langchain_community' and FAISS are installed in the active environment.")
    sys.exit(1)

# Centralized configuration pointing to the persistence layer directory
MEMORY_FOLDER = os.path.join(project_root, "memory_data")

def view_database(index_filename):
    """
    Loads a specific vector partition from disk and retrieves its stored contents.
    
    Args:
        index_filename (str): The specific FAISS partition identifier 
                              (e.g., 'emma_private' or 'group_shared').
    """
    print(f"\n{'='*60}")
    print(f"   AUDIT TARGET: {index_filename}")
    print(f"{'='*60}")
    
    # Pre-flight check to verify the physical file exists before attempting deserialization
    file_path = os.path.join(MEMORY_FOLDER, f"{index_filename}.faiss")
    
    if not os.path.exists(file_path):
        print(f" !! [Warning] Partition file not found: {file_path}")
        print(f"    (The system has not yet committed data to this vault.)")
        return

    try:
        print("   [Status] Initializing embedding model and loading vector store...")
        # Note: Must perfectly match the embedding dimensions used during ingestion
        embeddings = OllamaEmbeddings(model="llama3")
        
        # Deserializes the FAISS index from local storage.
        # Security Note: allow_dangerous_deserialization is required for FAISS local loading.
        # This is strictly acceptable here as the memory folder is a trusted local directory.
        db = FAISS.load_local(
            folder_path=MEMORY_FOLDER, 
            embeddings=embeddings, 
            index_name=index_filename,
            allow_dangerous_deserialization=True
        )
        
        # Execute a broad, generic query to fetch the maximum allowable chunks.
        # 'context' is used as a neutral anchor point to retrieve highly populated indices.
        results = db.similarity_search("context", k=100)
        
        if len(results) == 0:
            print("   [Status] Index is structurally sound but empty (0 vectors).")
        else:
            print(f"   [Status] Successfully retrieved {len(results)} memory fragments:\n")
            for i, doc in enumerate(results): 
                # Render the raw string payload associated with the stored vector
                print(f"   [{i+1:02d}] {doc.page_content.strip()}")
                
    except Exception as e:
        print(f" !! [CRITICAL ERROR] Failed to load or parse the index: {e}")

if __name__ == "__main__":
    # Execute the audit across the defined architectural partitions.
    # Note: These identifiers must map perfectly to the serialized FAISS filenames.
    view_database("emma_private")
    view_database("max_private")
    view_database("group_shared")