"""
System Orchestrator (Control Plane)
-----------------------------------
Central coordination engine for the DPF Architecture.
Manages the lifecycle of a single conversation turn, coordinating:
1. Semantic Routing (Vector Classification).
2. Policy Enforcement (Active Guardrails).
3. Privacy Sanitization (Data Plane Firewall).
4. Context Retrieval & Generation.
"""

import warnings
import re 
import numpy as np
import time
import os
import sys

# External ML Libraries
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from sklearn.metrics.pairwise import cosine_similarity

# Internal Modules
# Internal Modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_config import AGENTS, MODE_CONFIG, TOPIC_DICTIONARY
from metrics_logger import MetricsLogger
from system_registry import get_system_config 

# Import from the nested core package
try:
    from dpf_core.privacy_firewall import PrivacyFirewall
except ImportError:
    # Fallback if path mapping varies
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dpf_core'))
    from privacy_firewall import PrivacyFirewall

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Orchestrator:
    def __init__(self, config=None):
        print(" >> [System] Initializing Orchestrator Control Plane...")
        
        # --- 1. CONFIGURATION LOADING ---
        if config:
            self.config = config
        else:
            self.config = get_system_config()
            
        print(f" >> [Config] Architecture Mode: {self.config['system_label']}")
        
        # --- 2. SUBSYSTEM INITIALIZATION ---
        self.firewall = PrivacyFirewall()
        self.logger = MetricsLogger()
        
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.embeddings = OllamaEmbeddings(model="llama3")
        self.llm = Ollama(model="llama3") 
        
        # State Tracking
        self.last_winner = None 
        self.last_response_memory = None 
        self.round_robin_index = 0  
        self.last_detected_topics = []
        self.last_sentiment = "Neutral"
        
        # --- 3. MEMORY STORE LOADING ---
        self._load_memory_indices()

    def _load_memory_indices(self):
        base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memory_data")
        try:
            self.shared_db = FAISS.load_local(base_path, self.embeddings, "group_shared", allow_dangerous_deserialization=True)
        except Exception:
            print(" !! [Warning] Shared Memory index not found (Fresh run?)")
            self.shared_db = None

        self.private_memories = {}
        for agent in ["emma", "max"]:
            try:
                db = FAISS.load_local(base_path, self.embeddings, f"{agent}_private", allow_dangerous_deserialization=True)
                self.private_memories[agent.capitalize()] = db
            except Exception:
                pass 

    def analyze_signal(self, user_input):
        # A. Sentiment Analysis
        self.previous_sentiment = getattr(self, 'last_sentiment', 'Neutral')
        score = self.sentiment_analyzer.polarity_scores(user_input)['compound']
        
        if score < -0.05: sentiment = "Negative"
        elif score > 0.05: sentiment = "Positive"
        else: sentiment = "Neutral"
        self.last_sentiment = sentiment

        # B. Topic Extraction
        detected_topics = [] 
        user_lower = user_input.lower()
        for topic, keywords in TOPIC_DICTIONARY.items():
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', user_lower): 
                    detected_topics.append(topic)
                    break 

        if re.search(r'\d+\s*[\+\-\*\/]\s*\d+', user_lower) or "solve for" in user_lower:
            if "Math" not in detected_topics: detected_topics.append("Math")

        if not detected_topics: detected_topics.append("General") 

        self.last_detected_topics = detected_topics
        print(f"    [Signal Analysis] Sentiment: {sentiment} | Topics: {detected_topics}")
        return sentiment, detected_topics

    def determine_speaker(self, user_input, current_mode, target_agent=None):
        print(f"\n--- ROUTING LOGIC (Context: {current_mode}) ---")
        
        all_scores = {}
        user_vector = np.array([self.embeddings.embed_query(user_input)])
        
        # --- PATH 1: PRIVATE OVERRIDE ---
        if current_mode == "PRIVATE":
            print(f"    [Mode Check] Private Chat with {target_agent}. Skipping competition.")
            return target_agent, "PRIVATE_FORCED", {}

        # --- PATH 2: DIRECT MENTION (+2.0 Bonus) ---
        user_lower = user_input.lower()
        for agent_name in AGENTS.keys():
            if agent_name.lower() in user_lower:
                print(f"    [Direct Mention] User explicitly called '{agent_name}'.")
                return agent_name, "DIRECT_MENTION", {agent_name: 2.0}

        # --- CONTROL CONDITION: BASELINE MODE ---
        if not self.config["enable_smart_routing"]:
            agents = list(AGENTS.keys())
            winner = agents[self.round_robin_index % len(agents)]
            self.round_robin_index += 1
            print(f"    [Baseline] Round-Robin Selection: {winner}")
            return winner, "NAIVE_BASELINE", {}

        # --- EXPERIMENTAL CONDITION: SEMANTIC ROUTING ---
        calculated_highest = -1
        
        for agent_name, profile in AGENTS.items():
            # A. Similarity (Vector Match)
            agent_vector = np.array([self.embeddings.embed_query(profile["role_description"])])
            similarity = cosine_similarity(user_vector, agent_vector)[0][0]
            
            # B. Bias (Group Context)
            role_bias = 0.2 if agent_name == MODE_CONFIG.get("GROUP") else 0.0
            
            # C. Domain Bonus
            domain_bonus = 0.0
            for topic in self.last_detected_topics:
                if topic in profile["topics_handled"]:
                    domain_bonus = 1.5
                    break
            
            final_score = similarity + role_bias + domain_bonus
            all_scores[agent_name] = final_score

        # Momentum Bonus
        if self.last_winner and self.last_winner in all_scores:
            all_scores[self.last_winner] += 0.5
            print(f"    [Bonus] {self.last_winner} has Conversational Momentum (+0.5)")

        final_best_agent = max(all_scores, key=all_scores.get)

        # --- PATH 3: SAFETY INTERVENTION (Passive Monitoring) ---
        if self.config["enable_active_guardrails"]:
            print("    [Background Check] Scanning ALL Private Memories for risks...")
            intervention_agent = self._scan_for_risk_triggers(user_input)
            
            if intervention_agent:
                print(f"    !!! SAFETY INTERVENTION WINNER: {intervention_agent} !!!")
                return intervention_agent, "INTERVENTION", all_scores

        return final_best_agent, "SEMANTIC_ROUTING", all_scores

    def _scan_for_risk_triggers(self, user_input):
        """
        Scans private memory stores using FAISS L2 Distance logic.
        (Lower Score = Higher Similarity)
        """
        lowest_risk_score = float('inf')
        intervention_agent = None

        # Heuristic optimization
        is_question = "?" in user_input or user_input.lower().startswith(("what", "when", "who", "how")) 
        word_count = len(user_input.split())
        should_skip = (word_count < 5) and (self.last_sentiment != "Negative")
        if should_skip: return None

        for agent_name, private_db in self.private_memories.items():
            if not private_db: continue
            try:
                # FAISS returns L2 distance (Lower is better/closer)
                results = private_db.similarity_search_with_score(user_input, k=5)
                
                for res, score in results:
                    risk_text = res.page_content.lower()
                    current_score = score # This is L2 Distance
                    
                    critical_list = AGENTS[agent_name].get("critical_terms", [])
                    bypass_list = AGENTS[agent_name].get("safety_bypass_terms", [])
                    
                    hit_critical = any(t in risk_text for t in critical_list)
                    hit_bypass = any(t in risk_text for t in bypass_list)
                    
                    # Exact Thresholds from original code
                    is_valid_critical = hit_critical and current_score < 8000
                    is_valid_bypass = hit_bypass and current_score < 19000

                    if is_valid_critical or is_valid_bypass:
                        # Exclusions
                        if agent_name == "Max" and "History" in self.last_detected_topics: continue
                        if agent_name == "Max" and is_question: continue
                        
                        if is_valid_bypass:
                            print(f"    --> {agent_name} triggered SAFETY BYPASS (Super Bonus): '{risk_text[:15]}...' (Score: {current_score:.2f})")
                            current_score -= 8000 # Artificial boost (lowering distance)
                        
                        elif is_valid_critical:
                            if agent_name == "Emma" and self.last_sentiment != "Negative": continue
                            print(f"    --> {agent_name} found CRITICAL memory (Standard Bonus): '{risk_text[:15]}...'")
                            current_score -= 4000
                        
                        if current_score < lowest_risk_score:
                            lowest_risk_score = current_score
                            intervention_agent = agent_name
                            
            except Exception as e:
                print(f"    (Error scanning {agent_name}: {e})")
                
        return intervention_agent

    def execute_turn(self, user_input, current_mode, agent_engine, target_agent=None):
        if not user_input.strip(): return "System", "Input was empty."

        # Metrics Accumulators
        lat_firewall = 0.0
        total_redactions = 0
        intervention_active = False
        privacy_active = False
        
        # 1. Signal Analysis & Routing
        self.analyze_signal(user_input)
        
        t0 = time.perf_counter()
        winner, reason, scores = self.determine_speaker(user_input, current_mode, target_agent)
        t1 = time.perf_counter()
        lat_routing = (t1 - t0) * 1000
        
        if not winner: winner = "Max"
        if reason == "INTERVENTION": intervention_active = True

        context_data = None
        
        # --- SCENARIO A: INTERVENTION (Risk Metadata) ---
        if reason == "INTERVENTION":
            # Sanitize the Trigger terms themselves
            private_db = self.private_memories.get(winner)
            results = private_db.similarity_search(user_input, k=3)
            all_triggers = AGENTS[winner].get("critical_terms", []) + AGENTS[winner].get("safety_bypass_terms", [])
            found_terms = []
            
            for res in results:
                content = res.page_content.lower()
                found = [t for t in all_triggers if t in content]
                found_terms.extend(found)
            
            if found_terms:
                raw_str = ", ".join(list(set(found_terms))).upper()
                t0 = time.perf_counter()
                fw_result = self.firewall.evaluate(raw_str)
                t1 = time.perf_counter()
                lat_firewall += (t1 - t0) * 1000
                
                print(f"    [Firewall] Dynamic Sanitization of Risk Trigger: {found_terms}")
                risk_topic = f"Critical Risk Factor: {fw_result.clean_text}"
                context_data = {"type": "RISK_METADATA", "content": risk_topic}

        # --- SCENARIO B: PRIVATE CHAT ---
        elif current_mode == "PRIVATE":
            private_db = self.private_memories.get(winner)
            valid_memories = []
            if private_db:
                results_with_score = private_db.similarity_search_with_score(user_input, k=10)
                if results_with_score:
                    print(f"    [Memory Search] Inspecting top {len(results_with_score)} matches:")
                    for res, score in results_with_score:
                        if score < 28000: # Threshold check
                            valid_memories.append(res.page_content)
                            print(f"       - Accepted: '{res.page_content[:30]}...' (Score: {score:.2f})")

            shared_context = ""
            if self.shared_db:
                results_shared = self.shared_db.similarity_search(user_input, k=3)
                shared_context = "\n".join([f"[GROUP-LOG]: {res.page_content}" for res in results_shared])

            if valid_memories or shared_context:
                combined_private = "\n".join([f"- {mem}" for mem in valid_memories])
                secret_text = f"RELEVANT PRIVATE MEMORIES:\n{combined_private}\n\nRECENT GROUP CONTEXT:\n{shared_context}"
            else:
                secret_text = "No specific prior memory found regarding this topic."

            context_data = { "type": "RAW_TEXT", "content": secret_text }

        # --- SCENARIO C: GROUP CHAT (Main Experiment) ---
        elif current_mode == "GROUP":
            retrieved_history = ""
            if self.shared_db:
                results = self.shared_db.similarity_search(user_input, k=3)
                retrieved_history = "\n".join([res.page_content for res in results])

            if self.config["enable_context_pruning"]:
                if len(user_input.split()) < 5:
                    print("    [Context Control] Pruning older history.")
                    retrieved_history = "" 

            silent_knowledge = ""
            if winner:
                private_db = self.private_memories.get(winner)
                if private_db:
                    priv_results = private_db.similarity_search(user_input, k=5)
                    if priv_results:
                        processed_list = []
                        
                        # [BRANCH A] Pre-Generation Firewall
                        if self.config["enable_pre_generation_firewall"]:
                            print("    [DPF] Pre-Generation Firewall: ACTIVE")
                            for r in priv_results:
                                t0 = time.perf_counter()
                                fw_result = self.firewall.evaluate(r.page_content)
                                t1 = time.perf_counter()
                                lat_firewall += (t1 - t0) * 1000
                                
                                processed_list.append(f"- {fw_result.clean_text}")
                                if fw_result.is_triggered:
                                    privacy_active = True
                                    total_redactions += len(fw_result.redacted_entities)
                                    print(f"       -> Redacted: {fw_result.redacted_entities}")
                            
                            silent_knowledge = "\n".join(processed_list)
                            print(f"    [Silent Awareness] Injected {len(priv_results)} memories (Sanitized).")
                        
                        # [BRANCH B] Baseline (Leak)
                        else:
                            print("    [DPF] Pre-Generation Firewall: DISABLED (Potential Leakage)")
                            for r in priv_results:
                                processed_list.append(f"[PRIVATE_LEAK]: {r.page_content}")
                            silent_knowledge = "\n".join(processed_list)

            final_context = ""
            if silent_knowledge:
                final_context += f"[[[INTERNAL_PRIVATE_KNOWLEDGE (DO NOT LEAK)]]]\n{silent_knowledge}\n\n"
            if self.last_response_memory:
                 final_context += f"[[[IMMEDIATE_LAST_TURN]]]\n{self.last_response_memory}\n\n"
            if retrieved_history:
                final_context += f"[[[OLDER_HISTORY]]]\n{retrieved_history}"

            context_data = {"type": "SHARED_TEXT", "content": final_context}

        # 4. Generate Response
        response, lat_generation = agent_engine.generate_response(winner, user_input, context_data, current_mode)

        # 5. Post-Generation Filter
        if self.config["enable_post_generation_filter"]:
            print("    [DPF] Egress Filter: ACTIVE")
            t0 = time.perf_counter()
            fw_result = self.firewall.evaluate(response)
            t1 = time.perf_counter()
            lat_firewall += (t1 - t0) * 1000
            
            if fw_result.is_triggered:
                print(f"       -> Caught Leak in Output: {fw_result.redacted_entities}")
                response = fw_result.clean_text
                privacy_active = True
                total_redactions += len(fw_result.redacted_entities)

        # 6. Telemetry
        self.logger.log_turn(
            system_mode=self.config["system_label"],
            conv_mode=current_mode,
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
            redaction_count=total_redactions
        )

        self.last_winner = winner 
        self.last_response_memory = f"Agent ({winner}): {response}"
        return winner, response

    def save_turn(self, user_input, agent_response, winner, current_mode):
        base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memory_data")
        memory_text = f"User: {user_input}\nAgent ({winner}): {agent_response}"
        
        if current_mode == "PRIVATE":
            target_db = self.private_memories.get(winner)
            if target_db:
                if winner == "Max": 
                    try:
                        summary = self.llm.invoke(f"Summarize this interaction as a fact:\n{memory_text}")
                        memory_text = f"[Fact]: {summary.strip()}"
                    except Exception: pass
                target_db.add_texts([memory_text])
                target_db.save_local(base_path, f"{winner.lower()}_private")
                print(f"    [Disk IO] Saved to {winner}'s PRIVATE vault.")

        elif current_mode == "GROUP" and self.shared_db:
            self.shared_db.add_texts([memory_text])
            self.shared_db.save_local(base_path, "group_shared")
            print("    [Disk IO] Saved to SHARED Group history.")