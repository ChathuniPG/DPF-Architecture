"""
Metrics Logging System (System Telemetry)
-----------------------------------------
Primary observability and telemetry module for the multi-agent orchestration framework.
Captures real-time performance data, routing stability metrics, and privacy 
enforcement events to enable rigorous downstream analysis and system auditing.

Data Schema Design:
1. Routing Telemetry: Quantifies the confidence and stability of the Semantic Router 
   by tracking nearest-neighbor vector margins.
2. System Latency Profiling: Provides a granular breakdown of computational overhead 
   (Routing decision vs. Security enforcement vs. LLM generation).
3. Safety & Compliance Metrics: Automates the auditing of firewall redaction events 
   and active guardrail interventions.
"""

import csv
import os
from datetime import datetime

class MetricsLogger:
    def __init__(self, filename="system_telemetry.csv"):
        """
        Initializes the Observability Logger.
        Establishes the data schema and ensures safe I/O operations for the 
        persistent telemetry log.
        """
        # Ensure log directory availability using robust relative paths
        self.log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        self.filepath = os.path.join(self.log_dir, filename)
        
        # --- ENGINEERING DATA SCHEMA ---
        # Defines the strictly ordered columns for the CSV data store
        self.headers = [
            # 1. Context & State Conditions
            "Timestamp", 
            "System_Mode",          # Active architectural state (e.g., Baseline vs Secure)
            "Conversation_Mode",    # Topological context (Dyadic Private vs Group Shared)
            "Prompt_Category",      # Classification of the input intent (e.g., Threat Vector)
            "Data_Owner",           # Target Agent/Domain Authority (Max/Emma)
            "User_Input",           # Raw Input Vector (Sanitized for CSV)
            
            # 2. Routing Stability Metrics
            "Winner_Agent",         # Selected Execution Path / Target Node
            "Winning_Score",        # Primary Vector Similarity (Cosine/L2)
            "Runner_Up_Score",      # Secondary Vector Similarity
            "Routing_Margin",       # Semantic Stability Indicator (Winner - RunnerUp)
            
            # 3. Control Plane Event Flags
            "Intervention_Active",  # Boolean: Was an Active Guardrail triggered?
            "Privacy_Active",       # Boolean: Did the Data Plane Firewall redact tokens?
            "Pruning_Active",       # Boolean: Was context window truncation applied?
            
            # 4. System Performance Profiling (Latency)
            "Latency_Routing_ms",   # O(L) Decision & Orchestration Overhead
            "Latency_Firewall_ms",  # O(1) Bounded Sanitization Overhead
            "Latency_Generation_ms",# O(N) LLM Autoregressive Inference Time
            "Total_Latency_ms",     # End-to-End Turn Latency
            
            # 5. Output Artifacts & Efficacy
            "Response_Word_Count",  # System Throughput Proxy
            "Redaction_Count",      # Quantitative Safety Efficacy Metric
            "Final_Response"        # Sanitized Generation Artifact
        ]
        
        # Initialize the CSV store with the Engineering Header if it does not exist
        if not os.path.exists(self.filepath):
            try:
                with open(self.filepath, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.headers)
            except PermissionError:
                print(f" [CRITICAL] File Lock Error: Please release the lock on '{filename}'")

    def log_turn(self, system_mode, conv_mode, user_input, winner, scores, 
                 intervention, privacy, pruning, response, 
                 lat_routing=0, lat_firewall=0, lat_generation=0, redaction_count=0,
                 prompt_category="General", data_owner="Unknown"): 
        """
        Persists a single atomic conversational transaction to the telemetry log.
        Dynamically calculates derived stability metrics (e.g., Routing Margins) 
        and total latency overheads prior to disk I/O.
        """
        
        # --- 1. Calculate Semantic Routing Stability Margin ---
        # A high margin indicates a confident, stable architectural decision.
        # A low margin (e.g., < 0.05) indicates semantic ambiguity and potential context collision.
        winning_score = 0.0
        runner_up_score = 0.0
        routing_margin = 0.0
        
        if scores:
            # Sort scores descending to isolate the top two candidate nodes
            sorted_scores = sorted(scores.values(), reverse=True)
            winning_score = sorted_scores[0] if len(sorted_scores) > 0 else 0.0
            runner_up_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
            
            # Enforce precision rounding to 4 decimal places for scientific consistency
            winning_score = round(winning_score, 4)
            runner_up_score = round(runner_up_score, 4)
            routing_margin = round(winning_score - runner_up_score, 4)

        # --- 2. Calculate System Performance Aggregates ---
        word_count = len(response.split())
        total_latency = round(lat_routing + lat_firewall + lat_generation, 2)

        # --- 3. Construct Telemetry Row ---
        row = [
            # State Conditions
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            system_mode,
            conv_mode,
            prompt_category, 
            data_owner, 
            user_input.replace("\n", " "), # Sanitize newline chars to prevent CSV corruption
            
            # Routing Diagnostics
            winner,
            f"{winning_score:.4f}",
            f"{runner_up_score:.4f}",
            f"{routing_margin:.4f}", # The derived 'Stability' Metric
            
            # Boolean Flags (Cast to Integer for strict schema compatibility)
            1 if intervention else 0,
            1 if privacy else 0,
            1 if pruning else 0,
            
            # Latency Profiling
            f"{lat_routing:.2f}",
            f"{lat_firewall:.2f}",
            f"{lat_generation:.2f}",
            f"{total_latency:.2f}",
            
            # Output Artifacts
            word_count,
            redaction_count,
            response.replace("\n", " ") # Sanitize output string
        ]
        
        # --- 4. Disk I/O Commits ---
        try:
            with open(self.filepath, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            # Console Feedback for real-time observability
            print(f"    [Telemetry] Latency: {total_latency}ms | Margin: {routing_margin} | Privacy Triggered: {privacy}")

        except PermissionError:
             print(f" [CRITICAL ERROR] Could not commit telemetry to disk. Is '{self.filepath}' open in another process?")