import torch
import psutil
import platform
import sys

def get_specs():
    # 1. CPU
    print(f"CPU: {platform.processor()}")
    
    # 2. RAM
    total_ram = round(psutil.virtual_memory().total / (1024**3))
    print(f"RAM: {total_ram} GB")

    # 3. GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = round(torch.cuda.get_device_properties(0).total_memory / (1024**3))
        print(f"GPU: {gpu_name} ({vram}GB VRAM)")
    else:
        print("GPU: None (CPU Only)")

    # 4. Software Versions
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    
    try:
        import faiss
        print(f"FAISS: {faiss.__version__}")
    except ImportError:
        print("FAISS: Not installed or version not found")
        
    try:
        import langchain
        print(f"LangChain: {langchain.__version__}")
    except ImportError:
        print("LangChain: Not installed")

get_specs()