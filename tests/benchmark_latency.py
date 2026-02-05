"""
Computational Overhead Benchmark (Micro-Latency)
------------------------------------------------
Quantifies the runtime performance cost of the Deterministic Sanitization Engine.
Executes a high-volume iteration loop to measure processing latency isolated
from network or generative model overhead.

Objective:
- Validate the linear O(N) complexity of the regex engine.
- Establish baseline throughput metrics (Requests/Second).
- Determine P99 latency bounds for real-time SLA compliance.
"""

import time
import sys
import os
import numpy as np

# --- ENVIRONMENT SETUP ---
# Adjust path to include the source directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import from dpf_core (Robust import handling)
try:
    from dpf_core.privacy_firewall import PrivacyFirewall
except ImportError:
    # Fallback if python doesn't resolve the package structure immediately
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'dpf_core'))
    from privacy_firewall import PrivacyFirewall

def run_benchmark():
    print("==================================================")
    print("   DPF ARCHITECTURE: ENGINE LATENCY BENCHMARK     ")
    print("==================================================")
    
    # Initialize Engine (One-time compile cost)
    print(" >> [Setup] Initializing Pre-compiled Regex Engine...")
    try:
        fw = PrivacyFirewall()
    except Exception as e:
        print(f" !! [CRITICAL] Could not load Firewall: {e}")
        return
    
    # 1. Define Test Vectors (Varying Complexity)
    test_suite = [
        (
            "Baseline (Clean/Short)", 
            "The quick brown fox jumps over the lazy dog."
        ),
        (
            "Adversarial (Dirty/Short)", 
            "I am having a panic attack because I owe $5000 and my grade is F."
        ),
        (
            "Stress Test (Clean/Long - 1k Tokens)", 
            "History of Art context " * 200 # Approx 1KB payload
        )
    ]
    
    ITERATIONS = 10000
    
    # 2. Execution Loop
    for label, text in test_suite:
        payload_size = len(text.encode('utf-8'))
        print(f"\n>> Benchmark: {label}")
        print(f"   Payload Size: {payload_size} bytes | Iterations: {ITERATIONS}")
        
        latencies = []
        
        # Warmup
        for _ in range(10): fw.evaluate(text)
        
        # Measurement
        t_start_batch = time.perf_counter()
        
        for _ in range(ITERATIONS):
            t0 = time.perf_counter()
            _ = fw.evaluate(text)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000) # Microseconds -> Milliseconds
            
        t_end_batch = time.perf_counter()
        
        # 3. Statistical Analysis
        avg_lat = np.mean(latencies)
        p95_lat = np.percentile(latencies, 95)
        p99_lat = np.percentile(latencies, 99)
        total_time = t_end_batch - t_start_batch
        throughput = ITERATIONS / total_time
        
        print(f"   -------------------------------------------")
        print(f"   Avg Latency:  {avg_lat:.4f} ms")
        print(f"   P99 Latency:  {p99_lat:.4f} ms")
        print(f"   Throughput:   {throughput:.2f} req/sec")
        print(f"   -------------------------------------------")

if __name__ == "__main__":
    try:
        run_benchmark()
    except KeyboardInterrupt:
        print("\n [!] Benchmark aborted by user.")