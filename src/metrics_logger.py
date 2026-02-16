"""
Metrics Logging System (System Telemetry)
-----------------------------------------
Primary observability module for the DPF Architecture evaluation.
Captures real-time performance data, routing stability metrics, and 
privacy enforcement events for downstream statistical analysis.

Data Schema:
1. Routing Telemetry: Quantifies the stability of the Vector Router (Margin Analysis).
2. System Latency: Breakdown of computational overhead (O(1) vs O(N)).
3. Safety Metrics: Automatic counts of firewall redaction events.
"""

import csv
import os
from datetime import datetime

class MetricsLogger:
    def __init__(self, filename="experiment_data.csv"):
        # Ensure log directory availability
        self.log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        self.filepath = os.path.join(self.log_dir, filename)
        
        # --- ENGINEERING DATA SCHEMA ---
        self.headers = [
            # 1. Experimental Conditions
            "Timestamp", 
            "System_Mode",          # Independent Variable (Naive vs PostHoc vs DPF)
            "Conversation_Mode",    # Context Setting
            "Prompt_Category",      # Attack Type (e.g. ROLE_MASQUERADING)
            "Data_Owner",           # Target Agent (Max/Emma)
            "User_Input",           # Input Vector
            
            # 2. Routing Stability Metrics
            "Winner_Agent",         # Selected Execution Path
            "Winning_Score",        # Primary Vector Similarity
            "Runner_Up_Score",      # Secondary Vector Similarity
            "Routing_Margin",       # Stability Indicator (Winner - RunnerUp)
            
            # 3. Control Plane Flags
            "Intervention_Active",  # Active Guardrail Triggered?
            "Privacy_Active",       # Data Plane Firewall Triggered?
            "Pruning_Active",       # Context Window Optimization?
            
            # 4. System Performance (Latency)
            "Latency_Routing_ms",   # Decision Overhead
            "Latency_Firewall_ms",  # Sanitization Overhead
            "Latency_Generation_ms",# LLM Inference Time
            "Total_Latency_ms",     # End-to-End Latency
            
            # 5. Output Metrics
            "Response_Word_Count",  # Throughput Proxy
            "Redaction_Count",      # Safety Efficacy Metric
            "Final_Response"        # Artifact
        ]
        
        # Initialize CSV with Engineering Header
        if not os.path.exists(self.filepath):
            try:
                with open(self.filepath, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.headers)
            except PermissionError:
                print(f" [CRITICAL] File Lock Error: Please close '{filename}'")

    def log_turn(self, system_mode, conv_mode, user_input, winner, scores, 
                 intervention, privacy, pruning, response, 
                 lat_routing=0, lat_firewall=0, lat_generation=0, redaction_count=0,
                 prompt_category="General", data_owner="Unknown"): 
        """
        Persists a single atomic transaction to the telemetry log.
        Calculates derived stability metrics (Margins) on the fly.
        """
        
        # --- 1. Calculate Routing Stability Margin ---
        # A high margin indicates a confident, stable architectural decision.
        # A low margin (<0.05) indicates semantic ambiguity.
        winning_score = 0.0
        runner_up_score = 0.0
        routing_margin = 0.0
        
        if scores:
            # Sort scores descending to isolate top candidates
            sorted_scores = sorted(scores.values(), reverse=True)
            winning_score = sorted_scores[0] if len(sorted_scores) > 0 else 0.0
            runner_up_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
            
            # Precision rounding to 4 decimal places for consistency
            winning_score = round(winning_score, 4)
            runner_up_score = round(runner_up_score, 4)
            routing_margin = round(winning_score - runner_up_score, 4)

        # --- 2. Calculate System Performance ---
        word_count = len(response.split())
        total_latency = round(lat_routing + lat_firewall + lat_generation, 2)

        # --- 3. Construct Data Row ---
        row = [
            # Conditions
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            system_mode,
            conv_mode,
            prompt_category, 
            data_owner, 
            user_input.replace("\n", " "), # Sanitize newline chars
            
            # Routing
            winner,
            f"{winning_score:.4f}",
            f"{runner_up_score:.4f}",
            f"{routing_margin:.4f}", # The "Stability" Metric
            
            # Flags (Boolean -> Integer for CSV compatibility)
            1 if intervention else 0,
            1 if privacy else 0,
            1 if pruning else 0,
            
            # Latency
            f"{lat_routing:.2f}",
            f"{lat_firewall:.2f}",
            f"{lat_generation:.2f}",
            f"{total_latency:.2f}",
            
            # Output
            word_count,
            redaction_count,
            response.replace("\n", " ")
        ]
        
        # --- 4. Write to Disk ---
        try:
            with open(self.filepath, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            # Console Feedback for Observability
            print(f"    [Log] Latency: {total_latency}ms | Margin: {routing_margin} | Safe: {privacy}")

        except PermissionError:
             print(f" [CRITICAL ERROR] Could not write to log. Is '{self.filepath}' open?")