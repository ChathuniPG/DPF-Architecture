"""
Metrics Logging System (System Telemetry) — V2
-----------------------------------------------
Captures real-time performance data, routing stability, and privacy
enforcement events for downstream analysis.

V2 Schema Changes:
- "LLM_Backend" column added: records which model generated the response.
  Required for cross-model sensitivity analysis (Issue 1 fix).
- "Filter_Enforcement_Mode" column added: records "PRE_GEN", "POST_HOC",
  "POST_HOC_NLI", or "NONE" — makes the causal enforcement-stage variable
  explicit in the telemetry, removing ambiguity when ablation_utility and
  audit CSVs are read by visualization_engine.
- "Timing_Normalized" column added: boolean flag recording whether this
  turn's response time was delayed by the TimingNormalizer (intervention
  path). Allows post-hoc measurement of how often the timing oracle
  mitigation fired.

All new columns are appended to the end of the schema so existing
paper_logs/ CSVs (which lack these columns) remain readable by
visualization_engine via pd.read_csv with graceful NaN fill.
"""

import csv
import os
from datetime import datetime


class MetricsLogger:

    def __init__(self, filename: str = "system_telemetry.csv"):
        self.log_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs"
        )
        os.makedirs(self.log_dir, exist_ok=True)
        self.filepath = os.path.join(self.log_dir, filename)

        self.headers = [
            # 1. Context & State Conditions
            "Timestamp",
            "System_Mode",           # Active architectural mode
            "Conversation_Mode",     # GROUP or PRIVATE
            "Prompt_Category",       # Threat class or Benign_Utility_Test
            "Data_Owner",            # Ground-truth target agent (Emma/Max)
            "User_Input",            # Raw input (newlines sanitized)

            # 2. Routing Stability
            "Winner_Agent",          # Agent selected by router
            "Winning_Score",         # Top cosine similarity score
            "Runner_Up_Score",       # Second-best score
            "Routing_Margin",        # Winner - RunnerUp (stability indicator)

            # 3. Control Plane Flags
            "Intervention_Active",   # 1 if safety guardrail fired
            "Privacy_Active",        # 1 if firewall redacted any token
            "Pruning_Active",        # 1 if context window was pruned

            # 4. Latency Decomposition
            "Latency_Routing_ms",    # Routing + orchestration overhead
            "Latency_Firewall_ms",   # Firewall evaluation (pre or post)
            "Latency_Generation_ms", # LLM autoregressive inference
            "Total_Latency_ms",      # End-to-end turn latency

            # 5. Output Artifacts
            "Response_Word_Count",   # Throughput proxy
            "Redaction_Count",       # Number of PII tokens redacted
            "Final_Response",        # Sanitized LLM output

            # 6. V2 NEW: Cross-Model & Enforcement Metadata
            "LLM_Backend",           # Model identifier (e.g., "llama3")
            "Filter_Enforcement_Mode",  # PRE_GEN | POST_HOC | POST_HOC_NLI | NONE
            "Timing_Normalized",     # 1 if TimingNormalizer delayed this turn
        ]

        if not os.path.exists(self.filepath):
            try:
                with open(self.filepath, mode='w', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow(self.headers)
            except PermissionError:
                print(f" [CRITICAL] File lock on '{filename}' — release and retry.")

    def log_turn(self, system_mode, conv_mode, user_input, winner, scores,
                 intervention, privacy, pruning, response,
                 lat_routing=0.0, lat_firewall=0.0, lat_generation=0.0,
                 redaction_count=0, prompt_category="General",
                 data_owner="Unknown",
                 llm_backend="llama3",
                 filter_enforcement_mode="NONE",
                 timing_normalized=False):
        """
        Persists one conversational turn to the telemetry log.

        New V2 args:
            llm_backend: model identifier string from system config.
            filter_enforcement_mode: one of "PRE_GEN", "POST_HOC",
                "POST_HOC_NLI", "NONE" — derived by the Orchestrator
                from its active config flags.
            timing_normalized: True if this turn was delayed by the
                TimingNormalizer (INTERVENTION path only).
        """
        winning_score = 0.0
        runner_up_score = 0.0
        routing_margin = 0.0

        if scores:
            sorted_scores = sorted(scores.values(), reverse=True)
            winning_score = round(sorted_scores[0], 4) if sorted_scores else 0.0
            runner_up_score = round(sorted_scores[1], 4) if len(sorted_scores) > 1 else 0.0
            routing_margin = round(winning_score - runner_up_score, 4)

        word_count = len(response.split())
        total_latency = round(lat_routing + lat_firewall + lat_generation, 2)

        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            system_mode,
            conv_mode,
            prompt_category,
            data_owner,
            user_input.replace("\n", " "),
            winner,
            f"{winning_score:.4f}",
            f"{runner_up_score:.4f}",
            f"{routing_margin:.4f}",
            1 if intervention else 0,
            1 if privacy else 0,
            1 if pruning else 0,
            f"{lat_routing:.2f}",
            f"{lat_firewall:.2f}",
            f"{lat_generation:.2f}",
            f"{total_latency:.2f}",
            word_count,
            redaction_count,
            response.replace("\n", " "),
            # V2 new columns
            llm_backend,
            filter_enforcement_mode,
            1 if timing_normalized else 0,
        ]

        try:
            with open(self.filepath, mode='a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(row)
            print(f"    [Telemetry] {total_latency}ms | Margin: {routing_margin} "
                  f"| Mode: {filter_enforcement_mode} | Backend: {llm_backend} "
                  f"| Privacy: {privacy}")
        except PermissionError:
            print(f" [CRITICAL] Cannot write telemetry — is '{self.filepath}' open?")
