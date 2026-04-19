"""
System Orchestrator (Control Plane) — V2
-----------------------------------------
Central coordination engine for the multi-agent architecture.

V2 Changes (full detail in inline comments):

1. FRR Fix — Mutually Exclusive Filter Enforcement
   Post-generation egress filter only runs when DPF is DISABLED.
   Running both simultaneously in V1 caused double-jeopardy false refusals.

2. STANDARD_POSTHOC_NLI support
   When enable_post_generation_nli is True, the egress filter also runs a
   DeBERTa NLI pass on the response — giving the POST-HOC baseline the same
   detection power as the HPA measurement instrument. This makes the
   enforcement-stage comparison scientifically fair (Issue 6).

3. Removed [[[INTERNAL_PRIVATE_KNOWLEDGE (DO NOT LEAK)]]] prompt wrapper
   Probabilistic and counterproductive with 4-bit quantized models.

4. HNSW support moved to memory_manager._convert_to_hnsw()
   Orchestrator calls load_memory_index(name, index_type) — the manager
   owns all FAISS lifecycle operations.

5. TimingNormalizer — eliminates binary timing side-channel (Section VI.B)

6. Router Instrumentation — per-turn RouterLog for accuracy analysis

7. metrics_logger.log_turn() now receives llm_backend,
   filter_enforcement_mode, and timing_normalized for V2 paper columns.

8. AgentEngine initialized with llm_backend from config for cross-model
   sensitivity analysis (Issue 1).
"""

import warnings
import re
import numpy as np
import time
import os
import sys
import random
import statistics
import csv

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from agent_config import AGENTS, MODE_CONFIG, TOPIC_DICTIONARY
from metrics_logger import MetricsLogger
from system_registry import get_system_config
from memory_manager import load_memory_index, build_memory_indices, MEMORY_STORE_PATH

try:
    from dpf_core.privacy_firewall import PrivacyFirewall
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dpf_core'))
    from privacy_firewall import PrivacyFirewall

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ---------------------------------------------------------------------------
# Timing Oracle Mitigation (Section VI.B)
# ---------------------------------------------------------------------------

class TimingNormalizer:
    """
    Eliminates the binary timing side-channel from fail-closed firewall design.

    Blocked queries (~1 ms) are delayed to match the passing-query latency
    distribution, making them statistically indistinguishable to an attacker
    probing response times to map the policy scope.
    """

    _WARM_START = [random.gauss(23346.1, 2636.6) for _ in range(50)]

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._samples: list = list(self._WARM_START)

    def record_passing_latency(self, latency_ms: float):
        self._samples.append(latency_ms)
        if len(self._samples) > 500:
            self._samples.pop(0)

    def normalize_rejection(self, firewall_eval_ms: float):
        if not self.enabled or not self._samples:
            return
        target_ms = random.choice(self._samples)
        delay_ms = max(0.0, target_ms - firewall_eval_ms)
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

    @property
    def stats(self) -> dict:
        if len(self._samples) < 2:
            return {}
        return {
            "n": len(self._samples),
            "mean_ms": round(statistics.mean(self._samples), 2),
            "stdev_ms": round(statistics.stdev(self._samples), 2),
        }


# ---------------------------------------------------------------------------
# Router Instrumentation
# ---------------------------------------------------------------------------

class RouterLog:
    __slots__ = ("query_prefix", "predicted_role", "true_role",
                 "margin", "reason", "correct")

    def __init__(self, query, predicted, true_role, margin, reason):
        self.query_prefix = query[:60]
        self.predicted_role = predicted
        self.true_role = true_role
        self.margin = margin
        self.reason = reason
        self.correct = (true_role is None) or (predicted == true_role)

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:

    def __init__(self, config=None, index_type: str = "flat"):
        """
        Args:
            config: system configuration dict. Loaded from system_registry
                    if not supplied.
            index_type: "flat" — IndexFlatL2 exact search (default, V1).
                        "hnsw" — HNSW approximate search via memory_manager.
        """
        print(" >> [System] Initializing Orchestrator Control Plane (V2)...")

        self.config = config if config else get_system_config()
        self.index_type = index_type

        llm_backend = self.config.get("llm_backend", "llama3")
        print(f" >> [Config] Mode     : {self.config['system_label']}")
        print(f" >> [Config] Index    : {self.index_type.upper()}")
        print(f" >> [Config] Backend  : {llm_backend}")

        self.firewall = PrivacyFirewall()
        self.logger = MetricsLogger()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.embeddings = OllamaEmbeddings(model="llama3")
        self.llm = Ollama(model=llm_backend)   # used for save_turn summaries only

        timing_on = self.config.get("enable_timing_normalization", False)
        self.timing_normalizer = TimingNormalizer(enabled=timing_on)
        self.router_log: list = []

        # NLI model for STANDARD_POSTHOC_NLI egress — loaded lazily
        self._nli_model = None

        # Session state
        self.last_winner = None
        self.last_response_memory = None
        self.round_robin_index = 0
        self.last_detected_topics = []
        self.last_sentiment = "Neutral"

        self._load_memory_indices()

    # -----------------------------------------------------------------------
    # Memory Loading — delegates to memory_manager
    # -----------------------------------------------------------------------

    def _load_memory_indices(self):
        """
        Loads dual-index memory via memory_manager.load_memory_index().
        HNSW conversion (when index_type="hnsw") happens inside the manager.
        """
        self.shared_db = None
        try:
            self.shared_db = load_memory_index("group_shared", self.index_type)
        except Exception:
            print(" !! [Warning] Shared Memory index not found.")

        self.private_memories = {}
        for agent in ["emma", "max"]:
            try:
                db = load_memory_index(f"{agent}_private", self.index_type)
                self.private_memories[agent.capitalize()] = db
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # Signal Analysis
    # -----------------------------------------------------------------------

    def analyze_signal(self, user_input: str):
        self.previous_sentiment = getattr(self, 'last_sentiment', 'Neutral')
        score = self.sentiment_analyzer.polarity_scores(user_input)['compound']
        sentiment = ("Negative" if score < -0.05 else
                     "Positive" if score > 0.05 else "Neutral")
        self.last_sentiment = sentiment

        detected_topics = []
        user_lower = user_input.lower()
        for topic, keywords in TOPIC_DICTIONARY.items():
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', user_lower):
                    detected_topics.append(topic)
                    break

        if (re.search(r'\d+\s*[\+\-\*\/]\s*\d+', user_lower)
                or "solve for" in user_lower):
            if "Math" not in detected_topics:
                detected_topics.append("Math")

        if not detected_topics:
            detected_topics.append("General")

        self.last_detected_topics = detected_topics
        print(f"    [Signal] Sentiment: {sentiment} | Topics: {detected_topics}")
        return sentiment, detected_topics

    # -----------------------------------------------------------------------
    # Semantic Routing
    # -----------------------------------------------------------------------

    def determine_speaker(self, user_input: str, current_mode: str,
                          target_agent: str = None, true_role: str = None):
        print(f"\n--- ROUTING (Mode: {current_mode}) ---")

        all_scores = {}
        user_vector = np.array([self.embeddings.embed_query(user_input)])
        margin = 0.0

        if current_mode == "PRIVATE":
            self._record_routing(user_input, target_agent, true_role, 0.0, "PRIVATE_FORCED")
            return target_agent, "PRIVATE_FORCED", {}

        user_lower = user_input.lower()
        for agent_name in AGENTS.keys():
            if agent_name.lower() in user_lower:
                self._record_routing(user_input, agent_name, true_role, 2.0, "DIRECT_MENTION")
                return agent_name, "DIRECT_MENTION", {agent_name: 2.0}

        if not self.config["enable_smart_routing"]:
            agents = list(AGENTS.keys())
            winner = agents[self.round_robin_index % len(agents)]
            self.round_robin_index += 1
            self._record_routing(user_input, winner, true_role, 0.0, "NAIVE_BASELINE")
            return winner, "NAIVE_BASELINE", {}

        for agent_name, profile in AGENTS.items():
            agent_vector = np.array([self.embeddings.embed_query(profile["role_description"])])
            similarity = cosine_similarity(user_vector, agent_vector)[0][0]
            role_bias = 0.2 if agent_name == MODE_CONFIG.get("GROUP") else 0.0
            domain_bonus = 1.5 if any(
                t in profile["topics_handled"] for t in self.last_detected_topics
            ) else 0.0
            all_scores[agent_name] = similarity + role_bias + domain_bonus

        if self.last_winner and self.last_winner in all_scores:
            all_scores[self.last_winner] += 0.5

        sorted_scores = sorted(all_scores.values(), reverse=True)
        margin = (sorted_scores[0] - sorted_scores[1]
                  if len(sorted_scores) > 1 else sorted_scores[0] if sorted_scores else 0.0)
        final_best = max(all_scores, key=all_scores.get)

        if self.config["enable_active_guardrails"]:
            intervention = self._scan_for_risk_triggers(user_input)
            if intervention:
                print(f"    !!! SAFETY INTERVENTION → {intervention} !!!")
                self._record_routing(user_input, intervention, true_role, margin, "INTERVENTION")
                return intervention, "INTERVENTION", all_scores

        self._record_routing(user_input, final_best, true_role, margin, "SEMANTIC_ROUTING")
        return final_best, "SEMANTIC_ROUTING", all_scores

    def _record_routing(self, query, predicted, true_role, margin, reason):
        if self.config.get("enable_router_instrumentation", False):
            self.router_log.append(RouterLog(query, predicted, true_role, margin, reason))

    def export_router_log(self, path: str):
        if not self.router_log:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=RouterLog.__slots__)
            w.writeheader()
            w.writerows([r.as_dict() for r in self.router_log])
        print(f"    [Router Log] {len(self.router_log)} entries → {path}")

    def get_router_accuracy(self) -> dict:
        labelled = [r for r in self.router_log if r.true_role is not None]
        if not labelled:
            return {"error": "No labelled entries."}
        roles = set(r.true_role for r in labelled)
        results = {}
        for role in roles:
            tp = sum(1 for r in labelled if r.predicted_role == role and r.true_role == role)
            fp = sum(1 for r in labelled if r.predicted_role == role and r.true_role != role)
            fn = sum(1 for r in labelled if r.predicted_role != role and r.true_role == role)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r_ = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r_ / (p + r_) if (p + r_) > 0 else 0.0
            results[role] = {"precision": round(p, 4), "recall": round(r_, 4),
                             "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}
        results["_overall_accuracy"] = round(
            sum(1 for r in labelled if r.correct) / len(labelled), 4
        )
        return results

    # -----------------------------------------------------------------------
    # Risk Scanner
    # -----------------------------------------------------------------------

    def _scan_for_risk_triggers(self, user_input: str):
        is_question = ("?" in user_input or
                       user_input.lower().startswith(("what", "when", "who", "how")))
        if len(user_input.split()) < 5 and self.last_sentiment != "Negative":
            return None

        lowest = float('inf')
        winner = None

        for agent_name, private_db in self.private_memories.items():
            if not private_db:
                continue
            try:
                for res, score in private_db.similarity_search_with_score(user_input, k=5):
                    risk_text = res.page_content.lower()
                    critical_list = AGENTS[agent_name].get("critical_terms", [])
                    bypass_list = AGENTS[agent_name].get("safety_bypass_terms", [])

                    hit_critical = any(t in risk_text for t in critical_list)
                    hit_bypass = any(t in risk_text for t in bypass_list)
                    is_valid_critical = hit_critical and score < 8000
                    is_valid_bypass = hit_bypass and score < 19000

                    if is_valid_critical or is_valid_bypass:
                        if agent_name == "Max" and "History" in self.last_detected_topics:
                            continue
                        if agent_name == "Max" and is_question:
                            continue

                        if is_valid_bypass:
                            score -= 8000
                        elif is_valid_critical:
                            if agent_name == "Emma" and self.last_sentiment != "Negative":
                                continue
                            score -= 4000

                        if score < lowest:
                            lowest = score
                            winner = agent_name
            except Exception as e:
                print(f"    (Error scanning {agent_name}: {e})")

        return winner

    # -----------------------------------------------------------------------
    # Context Construction Helpers
    # -----------------------------------------------------------------------

    def _resolve_filter_enforcement_mode(self) -> str:
        """
        Returns the filter enforcement mode string for telemetry logging.
        This makes the active enforcement stage explicit in the CSV,
        which the ablation study uses to causally attribute leakage rates.
        """
        pre_gen = self.config.get("enable_pre_generation_firewall", False)
        post_nli = self.config.get("enable_post_generation_nli", False)
        post_regex = self.config.get("enable_post_generation_filter", False)

        if pre_gen:
            return "PRE_GEN"
        if post_nli:
            return "POST_HOC_NLI"
        if post_regex:
            return "POST_HOC"
        return "NONE"

    def _apply_pre_generation_firewall(self, chunks: list) -> tuple:
        """
        DPF enforcement at retrieval ingress. Runs before LLM inference.
        Returns (sanitized_text, redactions, fw_latency_ms).
        """
        processed = []
        total_redactions = 0
        total_fw_ms = 0.0

        for chunk in chunks:
            t0 = time.perf_counter()
            fw_result = self.firewall.evaluate(chunk.page_content)
            total_fw_ms += (time.perf_counter() - t0) * 1000

            processed.append(f"- {fw_result.clean_text}")
            if fw_result.is_triggered:
                total_redactions += len(fw_result.redacted_entities)
                print(f"       -> [DPF] Redacted: {fw_result.redacted_entities}")

        return "\n".join(processed), total_redactions, total_fw_ms

    def _apply_egress_filter(self, response: str,
                             use_nli: bool = False) -> tuple:
        """
        Post-generation egress filter. Called ONLY when DPF is disabled.

        Args:
            use_nli: when True (STANDARD_POSTHOC_NLI mode), also runs a
                     DeBERTa NLI pass to catch paraphrased leakage.
                     This gives the POST-HOC baseline the same detection
                     power as the HPA, making the comparison fair.

        Returns (filtered_response, redactions, latency_ms).
        """
        t0 = time.perf_counter()
        fw_result = self.firewall.evaluate(response)
        lat_ms = (time.perf_counter() - t0) * 1000

        if fw_result.is_triggered:
            print(f"       -> [Egress-Regex] Caught: {fw_result.redacted_entities}")
            return fw_result.clean_text, len(fw_result.redacted_entities), lat_ms

        # NLI pass (STANDARD_POSTHOC_NLI only)
        if use_nli and response.strip():
            nli_redacted, nli_ms = self._apply_nli_egress(response)
            lat_ms += nli_ms
            if nli_redacted != response:
                print(f"       -> [Egress-NLI] Semantic leak caught.")
                return nli_redacted, 1, lat_ms

        return response, 0, lat_ms

    def _apply_nli_egress(self, response: str) -> tuple:
        """
        Runs DeBERTa-v3 NLI on the response to catch paraphrased leakage.
        Model is loaded lazily on first call to avoid startup overhead
        in configurations that don't use NLI egress.
        Returns (filtered_response, latency_ms).
        """
        from evaluation.evaluate_privacy_audit import SECRETS_HYPOTHESES
        t0 = time.perf_counter()

        if self._nli_model is None:
            from sentence_transformers import CrossEncoder
            self._nli_model = CrossEncoder(
                'cross-encoder/nli-deberta-v3-base'
            )

        pairs = [(response, h) for h in SECRETS_HYPOTHESES]
        scores = self._nli_model.predict(pairs)
        import numpy as np_
        for score_set in scores:
            if isinstance(score_set, (list, np_.ndarray)) and len(score_set) == 3:
                entailment = float(score_set[1])
            else:
                entailment = float(score_set)
            if entailment > 0.50:
                lat_ms = (time.perf_counter() - t0) * 1000
                return "[NLI_EGRESS_REDACTED: Semantic leakage detected]", lat_ms

        return response, (time.perf_counter() - t0) * 1000

    def _build_group_context(self, winner: str, user_input: str) -> tuple:
        """
        Assembles the GROUP-mode context window.

        V2: neutral context labels only (no probabilistic "DO NOT LEAK" wrapper).
        Returns (context_str, redactions, fw_latency_ms, privacy_active).
        """
        total_redactions = 0
        fw_latency_ms = 0.0
        privacy_active = False

        retrieved_history = ""
        if self.shared_db:
            results = self.shared_db.similarity_search(user_input, k=3)
            retrieved_history = "\n".join([r.page_content for r in results])

        if self.config["enable_context_pruning"] and len(user_input.split()) < 5:
            retrieved_history = ""

        silent_knowledge = ""
        private_db = self.private_memories.get(winner)

        if private_db:
            priv_results = private_db.similarity_search(user_input, k=5)
            if priv_results:
                if self.config["enable_pre_generation_firewall"]:
                    print("    [DPF] Pre-Generation Firewall: ACTIVE")
                    sanitized, redactions, fw_ms = self._apply_pre_generation_firewall(
                        priv_results
                    )
                    silent_knowledge = sanitized
                    total_redactions += redactions
                    fw_latency_ms += fw_ms
                    if redactions > 0:
                        privacy_active = True
                else:
                    print("    [DPF] Pre-Generation Firewall: DISABLED")
                    silent_knowledge = "\n".join(
                        [f"[PRIVATE_LEAK]: {r.page_content}" for r in priv_results]
                    )

        parts = []
        if silent_knowledge:
            parts.append(f"[AGENT CONTEXT]\n{silent_knowledge}")
        if self.last_response_memory:
            parts.append(f"[LAST TURN]\n{self.last_response_memory}")
        if retrieved_history:
            parts.append(f"[GROUP HISTORY]\n{retrieved_history}")

        return "\n\n".join(parts), total_redactions, fw_latency_ms, privacy_active

    # -----------------------------------------------------------------------
    # Primary Execution Pipeline
    # -----------------------------------------------------------------------

    def execute_turn(self, user_input: str, current_mode: str,
                     agent_engine, target_agent: str = None,
                     prompt_category: str = "General",
                     data_owner: str = "Unknown",
                     read_only: bool = False,
                     true_role: str = None):
        """
        Primary pipeline for a conversation turn.

        V2 Enforcement Matrix (mutually exclusive):
          DPF_PROPOSED      → pre-gen firewall ON;  egress OFF.
          STANDARD_POSTHOC  → pre-gen OFF; regex egress ON.
          STANDARD_POSTHOC_NLI → pre-gen OFF; regex + NLI egress ON.
          NAIVE_CONTROL     → all OFF.
        """
        if not user_input.strip():
            return "System", "Input was empty."

        lat_firewall = 0.0
        total_redactions = 0
        intervention_active = False
        privacy_active = False
        timing_normalized = False

        pre_gen_active = self.config.get("enable_pre_generation_firewall", False)
        post_hoc_active = self.config.get("enable_post_generation_filter", False)
        post_nli_active = self.config.get("enable_post_generation_nli", False)

        # Runtime mutual exclusivity guard
        if pre_gen_active and (post_hoc_active or post_nli_active):
            post_hoc_active = False
            post_nli_active = False
            print("    [V2] DPF active → egress filter suppressed.")

        filter_mode = self._resolve_filter_enforcement_mode()
        llm_backend = self.config.get("llm_backend", "llama3")

        # ---- Stage 1: Signal Analysis & Routing ----
        self.analyze_signal(user_input)

        t0 = time.perf_counter()
        winner, reason, scores = self.determine_speaker(
            user_input, current_mode, target_agent, true_role=true_role
        )
        lat_routing = (time.perf_counter() - t0) * 1000

        if not winner:
            winner = "Max"
        if reason == "INTERVENTION":
            intervention_active = True

        context_data = None

        # ---- Stage 2: Context Construction ----

        if reason == "INTERVENTION":
            private_db = self.private_memories.get(winner)
            results = private_db.similarity_search(user_input, k=3)
            all_triggers = (AGENTS[winner].get("critical_terms", [])
                            + AGENTS[winner].get("safety_bypass_terms", []))
            found_terms = []
            for res in results:
                content = res.page_content.lower()
                found_terms.extend([t for t in all_triggers if t in content])

            if found_terms:
                raw_str = ", ".join(list(set(found_terms))).upper()
                t0 = time.perf_counter()
                fw_result = self.firewall.evaluate(raw_str)
                lat_firewall += (time.perf_counter() - t0) * 1000
                context_data = {
                    "type": "RISK_METADATA",
                    "content": f"Critical Risk Factor: {fw_result.clean_text}"
                }

        elif current_mode == "PRIVATE":
            private_db = self.private_memories.get(winner)
            valid_memories = []
            if private_db:
                for res, score in private_db.similarity_search_with_score(user_input, k=10):
                    if score < 28000:
                        valid_memories.append(res.page_content)

            shared_context = ""
            if self.shared_db:
                shared_context = "\n".join(
                    [f"[GROUP-LOG]: {r.page_content}"
                     for r in self.shared_db.similarity_search(user_input, k=3)]
                )

            if valid_memories or shared_context:
                combined = "\n".join([f"- {m}" for m in valid_memories])
                secret_text = (f"RELEVANT PRIVATE MEMORIES:\n{combined}\n\n"
                               f"RECENT GROUP CONTEXT:\n{shared_context}")
            else:
                secret_text = "No specific prior memory found regarding this topic."

            context_data = {"type": "RAW_TEXT", "content": secret_text}

        elif current_mode == "GROUP":
            final_context, redactions, fw_ms, pv_active = self._build_group_context(
                winner, user_input
            )
            lat_firewall += fw_ms
            total_redactions += redactions
            if pv_active:
                privacy_active = True
            context_data = {"type": "SHARED_TEXT", "content": final_context}

        # ---- Stage 3: LLM Generation ----
        response, lat_generation = agent_engine.generate_response(
            winner, user_input, context_data, current_mode
        )

        # ---- Stage 4: Egress Filter (POST-HOC and POST-HOC-NLI baselines ONLY) ----
        if (post_hoc_active or post_nli_active) and current_mode != "PRIVATE":
            mode_label = "POST_HOC_NLI" if post_nli_active else "POST_HOC"
            print(f"    [Egress] {mode_label}: ACTIVE")
            response, eg_redactions, eg_ms = self._apply_egress_filter(
                response, use_nli=post_nli_active
            )
            lat_firewall += eg_ms
            if eg_redactions > 0:
                total_redactions += eg_redactions
                privacy_active = True

        # ---- Stage 5: Timing Normalization ----
        if reason == "INTERVENTION" and self.timing_normalizer.enabled:
            self.timing_normalizer.normalize_rejection(lat_firewall)
            timing_normalized = True
        else:
            self.timing_normalizer.record_passing_latency(
                lat_routing + lat_firewall + lat_generation
            )

        # ---- Stage 6: Telemetry ----
        self.logger.log_turn(
            system_mode=self.config["system_label"],
            conv_mode=current_mode,
            prompt_category=prompt_category,
            data_owner=data_owner,
            user_input=user_input,
            winner=winner,
            scores=scores,
            intervention=intervention_active,
            privacy=privacy_active,
            pruning=self.config["enable_context_pruning"],
            response=response,
            lat_routing=lat_routing,
            lat_firewall=lat_firewall,
            lat_generation=lat_generation,
            redaction_count=total_redactions,
            llm_backend=llm_backend,
            filter_enforcement_mode=filter_mode,
            timing_normalized=timing_normalized,
        )

        self.last_winner = winner
        self.last_response_memory = f"Agent ({winner}): {response}"

        if not read_only:
            self.save_turn(user_input, response, winner, current_mode)

        return winner, response

    # -----------------------------------------------------------------------
    # State Persistence
    # -----------------------------------------------------------------------

    def save_turn(self, user_input: str, agent_response: str,
                  winner: str, current_mode: str):
        """
        Writes conversational state to the appropriate partitioned index.
        Always uses FlatL2 (HNSW does not support incremental add in FAISS).
        """
        memory_text = f"User: {user_input}\nAgent ({winner}): {agent_response}"

        if current_mode == "PRIVATE":
            target_db = self.private_memories.get(winner)
            if target_db:
                if winner == "Max":
                    try:
                        summary = self.llm.invoke(
                            f"Summarize this interaction as a fact:\n{memory_text}"
                        )
                        memory_text = f"[Fact]: {summary.strip()}"
                    except Exception:
                        pass
                target_db.add_texts([memory_text])
                target_db.save_local(MEMORY_STORE_PATH, f"{winner.lower()}_private")
                print(f"    [Disk IO] Saved to {winner}'s PRIVATE vault.")

        elif current_mode == "GROUP" and self.shared_db:
            self.shared_db.add_texts([memory_text])
            self.shared_db.save_local(MEMORY_STORE_PATH, "group_shared")
            print("    [Disk IO] Saved to SHARED Group history.")
