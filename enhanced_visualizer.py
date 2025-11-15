#!/usr/bin/env python3
"""
Enhanced visualization and tracing for ARC LangGraph Agent.
"""

import os
from dotenv import load_dotenv
load_dotenv()

def setup_langsmith_tracing():
    """Enable LangSmith tracing for detailed execution visualization."""
    print("🔍 Setting up LangSmith tracing...")
    
    # Check if LangSmith is configured
    langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
    
    if langchain_api_key:
        print("✅ LangSmith API key found")
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "arc-langgraph-agent"
        
        print("🎯 LangSmith tracing enabled!")
        print("View traces at: https://smith.langchain.com/")
        print(f"Project: {os.environ.get('LANGCHAIN_PROJECT')}")
        
        return True
    else:
        print("❌ LangSmith API key not found in .env")
        print("To enable tracing:")
        print("1. Sign up at https://smith.langchain.com/")
        print("2. Add LANGCHAIN_API_KEY=your-key to .env")
        print("3. Re-run this script")
        
        return False

def run_with_detailed_logging():
    """Run the agent with detailed step-by-step logging."""
    print("\n🚀 Running ARC agent with detailed logging...\n")
    
    try:
        from arc_langgraph_agent import ARCLangGraphAgent
        from run_arc_prompt import load_arc_tasks, get_task_by_index
        from run_langgraph_agent import initialize_llm_from_config
        from model_configs import DEFAULT_MODEL
        
        # Initialize LLM
        llm = initialize_llm_from_config(DEFAULT_MODEL)
        if not llm:
            print("❌ Could not initialize LLM")
            return
            
        # Load a simple task
        tasks = load_arc_tasks("data/arc-2024/arc-agi_training_challenges.json")
        task_id, task_data = get_task_by_index(tasks, 0)  # Use first task
        
        print(f"📝 Testing task: {task_id}")
        print(f"Training examples: {len(task_data['train'])}")
        
        # Create agent with verbose logging
        agent = ARCLangGraphAgent(llm, max_attempts=2)
        
        # Custom logging wrapper
        class LoggingLLM:
            def __init__(self, base_llm):
                self.base_llm = base_llm
                self.call_count = 0
                
            def invoke(self, prompt):
                self.call_count += 1
                print(f"\n🔥 LLM CALL #{self.call_count}")
                print("=" * 60)
                print("PROMPT:")
                print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
                print("=" * 60)
                
                response = self.base_llm.invoke(prompt)
                
                print("RESPONSE:")
                response_text = response.content if hasattr(response, 'content') else str(response)
                print(response_text[:300] + "..." if len(response_text) > 300 else response_text)
                print("=" * 60)
                
                return response
        
        # Wrap the LLM for logging
        agent.llm = LoggingLLM(agent.llm)
        
        # Run the agent
        result = agent.solve_task(task_id, task_data)
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"Success: {result['success']}")
        print(f"Success Rate: {result['best_success_rate']:.2%}")
        print(f"Attempts: {result['attempts_made']}")
        print(f"LLM Calls: {agent.llm.call_count}")
        
    except Exception as e:
        print(f"Error running detailed logging: {e}")
        import traceback
        traceback.print_exc()

def create_mermaid_diagram():
    """Create a Mermaid diagram file for the workflow."""
    mermaid_code = """
graph TD
    START([START]) --> GEN[generate_code<br/>🤖 LLM Call #1]
    GEN --> TEST[test_code<br/>⚡ Execute & Test]
    TEST --> DECISION{Decision}
    
    DECISION -->|success_rate >= 100%| FINAL[finalize<br/>📋 Format Results]
    DECISION -->|attempts < max_attempts| REFINE[refine_code<br/>🤖 LLM Call #N]
    DECISION -->|max_attempts reached| FINAL
    
    REFINE --> TEST
    FINAL --> END([END])
    
    %% Styling
    classDef llmNode fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef execNode fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef decisionNode fill:#fff3e0,stroke:#e65100,stroke-width:2px
    
    class GEN,REFINE llmNode
    class TEST,FINAL execNode
    class DECISION decisionNode
"""
    
    try:
        os.makedirs("output", exist_ok=True)
        with open("output/workflow_diagram.mmd", "w") as f:
            f.write(mermaid_code)
        
        print("✅ Mermaid diagram saved to: output/workflow_diagram.mmd")
        print("🌐 View online at: https://mermaid.live/")
        print("📋 Or paste the content into any Mermaid viewer")
        
    except Exception as e:
        print(f"Error creating Mermaid diagram: {e}")

def main():
    print("🎨 ARC LangGraph Enhanced Visualizer\n")
    
    print("1. Setting up tracing...")
    setup_langsmith_tracing()
    
    print("\n2. Creating workflow diagram...")
    create_mermaid_diagram()
    
    print("\n3. Want to run with detailed logging? (y/n): ", end="")
    choice = input().lower().strip()
    
    if choice in ['y', 'yes']:
        run_with_detailed_logging()
    else:
        print("Skipping detailed run.")
    
    print("\n🎯 Visualization Complete!")
    print("\n📚 Available Files:")
    print("- output/workflow_diagram.mmd (Mermaid diagram)")
    print("- output/workflow_description.md (Detailed description)")
    print("- output/langgraph_test_results.json (Latest run results)")

if __name__ == "__main__":
    main()