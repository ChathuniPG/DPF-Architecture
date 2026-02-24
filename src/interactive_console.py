"""
Interactive System Console (HITL Debug Interface)
-------------------------------------------------
Provides a real-time, Human-in-the-Loop (HITL) command shell for interacting 
with the multi-agent orchestration architecture. 

Architectural Role:
- Serves as the primary observability layer for qualitative verification, 
  allowing engineers to inspect routing behavior, vector retrieval, and 
  firewall sanitization in real-time.
- Implements a Read-Eval-Print Loop (REPL) for dynamic state switching 
  between global broadcasting (Group) and isolated dyadic channels (Private).
"""

import sys
import os
import shutil

# Ensure robust path resolution for module imports to allow standalone execution
# regardless of the current working directory of the terminal.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.orchestrator import Orchestrator
    from src.agent import AgentEngine
    from src.system_registry import get_system_config, set_system_mode
    from src.memory_manager import check_memory_integrity, build_memory_indices
except ImportError as e:
    print(f" [CRITICAL] Module Import Error: {e}")
    print(" Ensure you are executing this script from the project root directory.")
    sys.exit(1)

def print_help_menu():
    """Renders the available system commands to the console."""
    print("\n-------------------------------------------------")
    print(" SYSTEM COMMANDS:")
    print("   /help            -> Show this command menu")
    print("   /quit            -> Gracefully terminate session")
    print("   /reset           -> Hard purge of Vector DB and context state")
    print("   /group           -> Switch routing topology to Global Multi-Agent")
    print("   /private [Agent] -> Open isolated channel (e.g., '/private Emma')")
    print("-------------------------------------------------\n")

def run_console_session():
    """
    Initializes the state managers and enters the primary REPL event loop 
    for the interactive debugging session.
    """

    # --- 1. SYSTEM INITIALIZATION & STATE VERIFICATION ---
    # Default to the most restrictive/secure architecture for standard debugging
    set_system_mode("DPF_PROPOSED")
    
    # Auto-Initialization of Persistence Layer
    # Ensures the vector database is structurally sound before binding the Orchestrator.
    if not check_memory_integrity():
        print(" >> [System] Corrupt or missing vector indices detected. Initializing Memory Store...")
        build_memory_indices()

    # Load Active Architectural Configuration
    config = get_system_config()
    system_mode = config["system_label"]

    print(f" >> [System] Booting {system_mode} environment...")
    
    # Initialize Core Subsystems
    # brain = Control Plane (Routing, Security, Memory)
    # mouth = Compute Plane (LLM Generation)
    brain = Orchestrator(config) 
    mouth = AgentEngine()

    # --- 2. UI RENDER & OBSERVABILITY DASHBOARD ---
    # Clear terminal buffer for a clean session start
    os.system('cls' if os.name == 'nt' else 'clear')

    print("\n=================================================")
    print(f"      ARCHITECTURE: HITL DEBUGGING CONSOLE")
    print("=================================================")
    print(f" [!] ACTIVE MODE: {system_mode}")
    print(f" [-] Smart Routing:       {'[ACTIVE]' if config.get('enable_smart_routing') else '[DISABLED]'}")
    
    # Display Security Feature Flags
    pre_fw = config.get('enable_pre_generation_firewall', False)
    post_fw = config.get('enable_post_generation_filter', False)
    guardrails = config.get('enable_active_guardrails', False)
    
    print(f" [-] Pre-Gen Firewall:    {'[ACTIVE]' if pre_fw else '[DISABLED]'}")
    print(f" [-] Post-Gen Filter:     {'[ACTIVE]' if post_fw else '[DISABLED]'}")
    print(f" [-] Active Guardrails:   {'[ACTIVE]' if guardrails else '[DISABLED]'}")
    print("=================================================")
    
    print_help_menu()

    # Initial Context State
    current_context = "GROUP"
    target_agent = None

    # --- 3. PRIMARY REPL EVENT LOOP ---
    while True:
        # Dynamic CLI Prompt Construction reflecting current topology
        status_label = f"[{current_context}]"
        if current_context == "PRIVATE":
            status_label += f"@{target_agent}"
        
        try:
            # Minimalist prompt that reminds the user how to pull up the commands
            user_input = input(f"\nUser {status_label} (Type /help): ").strip()
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully without corrupting state
            print("\n [System] Received interrupt signal. Terminating...")
            break 

        if not user_input:
            continue

        # --- COMMAND PARSING & STATE MANAGEMENT ---
        # Normalize input for strict command evaluation
        cmd = user_input.lower()
        
        if cmd in ["/quit", "/exit", "quit", "exit"]:
            print(" >> [System] Session terminated. Context state preserved on disk.")
            break
            
        elif cmd == "/help":
            print_help_menu()
            continue
            
        elif cmd == "/reset":
            print("\n >> [System] Executing Hard Reset (Purging Persistence Layer)...")
            
            # 1. Hard wipe the storage directory to prevent zombie vectors
            memory_dir = os.path.join(os.path.dirname(__file__), '..', 'memory_data')
            if os.path.exists(memory_dir):
                shutil.rmtree(memory_dir)
                
            # 2. Rebuild pristine indices from the seed state
            build_memory_indices() 
            
            # 3. Re-instantiate the Orchestrator to bind to the fresh memory indices
            brain = Orchestrator(config)
            print(" >> [System] Memory State: Clean. Routing paths reset.")
            continue

        elif cmd.startswith("/private"):
            parts = cmd.split()
            if len(parts) >= 2:
                # Transition to isolated dyadic communication
                current_context = "PRIVATE"
                target_agent = parts[1].capitalize() 
                
                # Reset orchestration momentum to prevent cross-channel context bleeding
                brain.last_winner = target_agent
                print(f" >> [Router] Topology isolated: Direct Link established with {target_agent}")
            else:
                print(" !! [Syntax Error] Usage: /private [AgentName] (e.g., /private Emma)")
            continue

        elif cmd == "/group":
            # Transition to global broadcasting and semantic competition
            current_context = "GROUP"
            target_agent = None
            print(f" >> [Router] Topology updated: Global Group Context restored")
            continue

        # --- CORE EXECUTION PIPELINE ---
        
        # 1. Orchestration
        # Dispatches the raw input through the Control Plane (Routing -> Security -> Memory)
        # and ultimately to the Compute Plane for generation. 
        # Note: 'read_only' defaults to False, so this state will automatically persist to the Vector DB.
        winner, response = brain.execute_turn(
            user_input=user_input, 
            current_mode=current_context, 
            agent_engine=mouth, 
            target_agent=target_agent
        )

        # 2. Render Agent Artifact
        print(f"\n{winner}: {response}")

if __name__ == "__main__":
    # Boot the interactive REPL
    run_console_session()