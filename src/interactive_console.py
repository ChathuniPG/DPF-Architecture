"""
Interactive System Console (HITL Debug Interface) — V3
-------------------------------------------------------
Provides a real-time, Human-in-the-Loop (HITL) command shell for interacting
with the multi-agent orchestration architecture.

V3 Changes:
- /mode command added: allows switching architectural mode during a live
  session without restarting. Useful for viva demonstrations — you can
  show the NAIVE baseline leaking data, then switch to DPF and show it
  blocked, all in the same terminal session.
  Usage: /mode DPF_PROPOSED | /mode NAIVE_CONTROL |
         /mode STANDARD_POSTHOC | /mode STANDARD_POSTHOC_NLI

- AgentEngine() now reads llm_backend from the active config rather than
  hardcoding "llama3". This ensures the console respects whatever backend
  is set in system_registry (e.g., if you set it to "gemma3:4b" before
  launching the console for manual testing).

- /backend command added: switch LLM backend during session.
  Usage: /backend llama3 | /backend gemma3:4b

- Dashboard header updated to show Post-Gen NLI flag (new in V2/V3).
"""

import sys
import os
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.orchestrator import Orchestrator
    from src.agent import AgentEngine
    from src.system_registry import (get_system_config, set_system_mode,
                                      set_llm_backend, VALID_MODES, VALID_BACKENDS)
    from src.memory_manager import check_memory_integrity, build_memory_indices
except ImportError as e:
    print(f" [CRITICAL] Module Import Error: {e}")
    print(" Ensure you are executing this script from the project root directory.")
    sys.exit(1)


def print_help_menu():
    """Renders available system commands to the console."""
    print("\n-------------------------------------------------")
    print(" SYSTEM COMMANDS:")
    print("   /help                  -> Show this command menu")
    print("   /quit                  -> Gracefully terminate session")
    print("   /reset                 -> Hard purge of Vector DB and context state")
    print("   /group                 -> Switch to Global Multi-Agent topology")
    print("   /private [Agent]       -> Open isolated channel (e.g., /private Emma)")
    print("   /mode [MODE]           -> Switch architectural mode (live, no restart)")
    print(f"                            Options: {', '.join(VALID_MODES)}")
    print("   /backend [MODEL]       -> Switch LLM backend (live)")
    print(f"                            Options: {', '.join(VALID_BACKENDS)}")
    print("   /status                -> Show current configuration flags")
    print("-------------------------------------------------\n")


def print_dashboard(config: dict):
    """Renders the security feature flag dashboard."""
    system_mode = config["system_label"]
    backend     = config.get("llm_backend", "llama3")

    print("\n=================================================")
    print(f"      ARCHITECTURE: HITL DEBUGGING CONSOLE V3")
    print("=================================================")
    print(f" [!] ACTIVE MODE    : {system_mode}")
    print(f" [-] LLM Backend    : {backend}")
    print(f" [-] Smart Routing  : {'[ACTIVE]' if config.get('enable_smart_routing') else '[DISABLED]'}")
    print(f" [-] Pre-Gen FW     : {'[ACTIVE]' if config.get('enable_pre_generation_firewall') else '[DISABLED]'}")
    print(f" [-] Post-Gen Regex : {'[ACTIVE]' if config.get('enable_post_generation_filter') else '[DISABLED]'}")
    print(f" [-] Post-Gen NLI   : {'[ACTIVE]' if config.get('enable_post_generation_nli') else '[DISABLED]'}")
    print(f" [-] Guardrails     : {'[ACTIVE]' if config.get('enable_active_guardrails') else '[DISABLED]'}")
    print("=================================================")


def run_console_session():
    """
    Initializes state managers and enters the primary REPL event loop.
    """

    # --- 1. INITIALIZATION ---
    set_system_mode("DPF_PROPOSED")

    if not check_memory_integrity():
        print(" >> [System] Missing vector indices. Initializing Memory Store...")
        build_memory_indices()

    config      = get_system_config()
    backend     = config.get("llm_backend", "llama3")

    os.system('cls' if os.name == 'nt' else 'clear')
    print_dashboard(config)
    print_help_menu()

    brain = Orchestrator(config)
    mouth = AgentEngine(llm_backend=backend)  # V3 fix: reads backend from config

    current_context = "GROUP"
    target_agent    = None

    # --- 2. REPL EVENT LOOP ---
    while True:
        status_label = f"[{current_context}]"
        if current_context == "PRIVATE":
            status_label += f"@{target_agent}"

        try:
            user_input = input(f"\nUser {status_label} (Type /help): ").strip()
        except KeyboardInterrupt:
            print("\n [System] Interrupt received. Terminating...")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        # ---- Command Parsing ----

        if cmd in ["/quit", "/exit", "quit", "exit"]:
            print(" >> [System] Session terminated. Context state preserved on disk.")
            break

        elif cmd == "/help":
            print_help_menu()

        elif cmd == "/status":
            # Re-read config in case mode or backend changed
            print_dashboard(get_system_config())

        elif cmd == "/reset":
            print("\n >> [System] Hard Reset — purging persistence layer...")
            memory_dir = os.path.join(os.path.dirname(__file__), '..', 'memory_data')
            if os.path.exists(memory_dir):
                shutil.rmtree(memory_dir)
            build_memory_indices()
            # Rebind orchestrator to fresh indices
            config = get_system_config()
            brain  = Orchestrator(config)
            print(" >> [System] Memory State: Clean. Routing paths reset.")

        elif cmd.startswith("/mode"):
            # Live mode switching — no restart required
            parts = user_input.split()
            if len(parts) < 2:
                print(f" !! Usage: /mode [MODE]")
                print(f"    Options: {', '.join(VALID_MODES)}")
            else:
                requested = parts[1].upper()
                try:
                    set_system_mode(requested)
                    config = get_system_config()
                    # Reinitialize orchestrator with new config
                    brain  = Orchestrator(config)
                    print_dashboard(config)
                    print(f"\n >> [Mode] Switched to {requested}. "
                          f"Orchestrator reinitialized.")
                    print(f"    Note: Vector DB NOT reset — memory state preserved.")
                    print(f"    Use /reset if you want a clean slate.")
                except ValueError as e:
                    print(f" !! {e}")

        elif cmd.startswith("/backend"):
            # Live backend switching
            parts = user_input.split()
            if len(parts) < 2:
                print(f" !! Usage: /backend [MODEL]")
                print(f"    Options: {', '.join(VALID_BACKENDS)}")
            else:
                requested_backend = parts[1]
                try:
                    set_llm_backend(requested_backend)
                    config = get_system_config()
                    mouth  = AgentEngine(llm_backend=requested_backend)
                    print(f" >> [Backend] Switched to '{requested_backend}'.")
                    print(f"    Ensure model is pulled: ollama pull {requested_backend}")
                except ValueError as e:
                    print(f" !! {e}")

        elif cmd.startswith("/private"):
            parts = cmd.split()
            if len(parts) >= 2:
                current_context = "PRIVATE"
                target_agent    = parts[1].capitalize()
                brain.last_winner = target_agent
                print(f" >> [Router] Isolated channel → {target_agent}")
            else:
                print(" !! Usage: /private [AgentName]  (e.g., /private Emma)")

        elif cmd == "/group":
            current_context = "GROUP"
            target_agent    = None
            print(f" >> [Router] Global Group Context restored.")

        # ---- Core Execution ----
        else:
            winner, response = brain.execute_turn(
                user_input=user_input,
                current_mode=current_context,
                agent_engine=mouth,
                target_agent=target_agent
            )
            print(f"\n{winner}: {response}")


if __name__ == "__main__":
    run_console_session()
