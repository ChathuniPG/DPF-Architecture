"""
Interactive System Console (Debug Interface)
--------------------------------------------
Provides a real-time, human-in-the-loop command shell for interacting with
the DPF Architecture. Used for qualitative verification and system debugging.

Capabilities:
- Real-time switching between Routing Contexts (Group vs. Private).
- Manual triggering of System Reset (Factory Settings).
- Live telemetry observation (Latency/Routing Decisions).
"""

import sys
import os

# Ensure robust path resolution for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.orchestrator import Orchestrator
    from src.agent import AgentEngine
    from src.system_registry import get_system_config, set_system_mode
    from src.memory_manager import check_memory_integrity, build_memory_indices
except ImportError as e:
    print(f" [CRITICAL] Module Import Error: {e}")
    print(" Ensure you are running this from the project root.")
    sys.exit(1)

def run_console_session():
    """
    Primary event loop for the interactive debugging session.
    """

    # --- 1. SYSTEM INITIALIZATION ---
    # Default to the Proposed Architecture for debugging
    set_system_mode("DPF_PROPOSED")
    
    # Auto-Initialization of Persistence Layer
    if not check_memory_integrity():
        print(" >> [System] Corrupt or missing indices detected. Initializing Memory Store...")
        build_memory_indices()

    # Load Architectural Configuration
    config = get_system_config()
    system_mode = config["system_label"]

    print(f" >> [System] Booting {system_mode} environment...")
    
    # Initialize Core Subsystems
    brain = Orchestrator(config) 
    mouth = AgentEngine()

    # --- 2. UI RENDER ---
    os.system('cls' if os.name == 'nt' else 'clear')

    print("\n=================================================")
    print(f"      DPF ARCHITECTURE: DEBUGGING CONSOLE")
    print("=================================================")
    print(f" [!] ARCHITECTURE: {system_mode}")
    print(f" [-] Smart Routing:       {'[ACTIVE]' if config.get('enable_smart_routing') else '[DISABLED]'}")
    
    # Feature Flags Display
    pre_fw = config.get('enable_pre_generation_firewall', False)
    post_fw = config.get('enable_post_generation_filter', False)
    guardrails = config.get('enable_active_guardrails', False)
    
    print(f" [-] Pre-Gen Firewall:    {'[ACTIVE]' if pre_fw else '[DISABLED]'}")
    print(f" [-] Post-Gen Filter:     {'[ACTIVE]' if post_fw else '[DISABLED]'}")
    print(f" [-] Active Guardrails:   {'[ACTIVE]' if guardrails else '[DISABLED]'}")
    print("=================================================\n")
    
    print(" COMMANDS:")
    print("  :quit            -> Exit Session (History Preserved)")
    print("  :reset           -> Factory Reset (Wipe Memory Indices)")
    print("  :mode group      -> Switch to Multi-Agent Context")
    print("  :mode private [Agent] -> Switch to Direct Channel (e.g., ':mode private Emma')")
    print("-------------------------------------------------")

    current_context = "GROUP"
    target_agent = None

    # --- 3. EVENT LOOP ---
    while True:
        # Dynamic Prompt Construction
        status_label = f"[{current_context}]"
        if current_context == "PRIVATE":
            status_label += f"@{target_agent}"
        
        try:
            user_input = input(f"\nUser {status_label}: ").strip()
        except KeyboardInterrupt:
            print("\n [System] Force Exit.")
            break 

        if not user_input:
            continue

        # --- COMMAND HANDLING ---
        # Normalize input for command parsing
        cmd = user_input.lower()
        
        if cmd in [":quit", ":exit", "quit", "exit"]:
            print(" >> [System] Session terminated. Context preserved.")
            break
            
        elif cmd == ":reset":
            print("\n >> [System] Executing Factory Reset (Wiping Vector DB)...")
            build_memory_indices() 
            
            # Re-instantiate Orchestrator to bind to new indices
            brain = Orchestrator(config)
            print(" >> [System] Memory State: Clean.")
            continue

        elif cmd.startswith(":mode private"):
            parts = user_input.split()
            if len(parts) > 2:
                current_context = "PRIVATE"
                target_agent = parts[2].capitalize() 
                # Reset routing memory for clean context switch
                brain.last_winner = target_agent
                print(f" >> [Router] Channel switched: Direct Link established with {target_agent}")
            else:
                print(" !! [Syntax Error] Usage: :mode private [AgentName]")
            continue

        elif cmd == ":mode group":
            current_context = "GROUP"
            target_agent = None
            print(f" >> [Router] Channel switched: Global Group Context")
            continue

        # --- CORE EXECUTION PIPELINE ---
        
        # 1. Pipeline Execution
        # Passes input through Routing -> Firewall -> Generation
        # read_only defaults to False, so this AUTOMATICALLY SAVES to memory.
        winner, response = brain.execute_turn(
            user_input=user_input, 
            current_mode=current_context, 
            agent_engine=mouth, 
            target_agent=target_agent
        )

        # 2. Output Rendering
        print(f"\n{winner}: {response}")

        # [REMOVED]: Manual brain.save_turn() call.
        # Reasoning: The Orchestrator now handles saving internally within execute_turn.
        # Leaving this here would cause duplicate memory entries.

if __name__ == "__main__":
    run_console_session()