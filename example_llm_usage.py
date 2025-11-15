#!/usr/bin/env python3
"""
Example usage of the ARC LangGraph Agent with different LLMs.

This script demonstrates how to initialize and use the LangGraph agent
with various language models for solving ARC problems.
"""

import os
import json
from typing import Dict, Any

# Import the agent
try:
    from arc_langgraph_agent import ARCLangGraphAgent
    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"LangGraph agent not available: {e}")
    AGENT_AVAILABLE = False


def example_with_openai():
    """Example using OpenAI's GPT models."""
    print("=== OpenAI Example ===")
    
    try:
        from langchain_openai import ChatOpenAI
        
        # Initialize LLM (requires OPENAI_API_KEY environment variable)
        llm = ChatOpenAI(
            model="gpt-4o-mini",  # or "gpt-4", "gpt-3.5-turbo"
            temperature=0
        )
        
        # Initialize agent
        agent = ARCLangGraphAgent(llm, max_attempts=3)
        
        print("✓ OpenAI agent initialized successfully")
        return agent
        
    except ImportError:
        print("✗ OpenAI not available. Install with: pip install langchain-openai")
        return None
    except Exception as e:
        print(f"✗ Error initializing OpenAI: {e}")
        return None


def example_with_anthropic():
    """Example using Anthropic's Claude models."""
    print("\\n=== Anthropic Example ===")
    
    try:
        from langchain_anthropic import ChatAnthropic
        
        # Initialize LLM (requires ANTHROPIC_API_KEY environment variable)
        llm = ChatAnthropic(
            model="claude-3-sonnet-20240229",  # or "claude-3-haiku-20240307"
            temperature=0
        )
        
        # Initialize agent
        agent = ARCLangGraphAgent(llm, max_attempts=3)
        
        print("✓ Anthropic agent initialized successfully")
        return agent
        
    except ImportError:
        print("✗ Anthropic not available. Install with: pip install langchain-anthropic")
        return None
    except Exception as e:
        print(f"✗ Error initializing Anthropic: {e}")
        return None


def example_with_ollama():
    """Example using local Ollama models."""
    print("\\n=== Ollama Example ===")
    
    try:
        from langchain_ollama import ChatOllama
        
        # Initialize LLM (requires Ollama running locally)
        llm = ChatOllama(
            model="llama2",  # or "mistral", "codellama", etc.
            temperature=0
        )
        
        # Initialize agent
        agent = ARCLangGraphAgent(llm, max_attempts=3)
        
        print("✓ Ollama agent initialized successfully")
        return agent
        
    except ImportError:
        print("✗ Ollama not available. Install with: pip install langchain-ollama")
        return None
    except Exception as e:
        print(f"✗ Error initializing Ollama: {e}")
        return None


def create_simple_arc_task():
    """Create a simple ARC task for testing."""
    return {
        "train": [
            {
                "input": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                "output": [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
            },
            {
                "input": [[0, 0], [1, 0]],
                "output": [[1, 1], [0, 1]]
            }
        ],
        "test": [
            {
                "input": [[0, 1], [0, 0]],
                "output": [[1, 0], [1, 1]]  # This would be the expected answer
            }
        ]
    }


def run_example_task(agent: ARCLangGraphAgent):
    """Run the agent on a simple example task."""
    print("\\n=== Running Example Task ===")
    
    # Create simple task
    task_data = create_simple_arc_task()
    task_id = "example_task"
    
    print(f"Task: {task_id}")
    print(f"Training examples: {len(task_data['train'])}")
    
    # Solve task
    result = agent.solve_task(task_id, task_data)
    
    # Print results
    print(f"\\nResults:")
    print(f"  Success: {result['success']}")
    print(f"  Success Rate: {result['best_success_rate']:.2%}")
    print(f"  Attempts: {result['attempts_made']}")
    print(f"  Time: {result['execution_time']:.2f}s")
    
    # Show generated code
    if result['final_solution']:
        print(f"\\n  Generated Code:")
        code_lines = result['final_solution']['main_code'].split('\\n')
        for i, line in enumerate(code_lines[:10], 1):  # Show first 10 lines
            print(f"    {i:2}: {line}")
        if len(code_lines) > 10:
            print(f"    ... ({len(code_lines)-10} more lines)")
    
    return result


def main():
    """Main function demonstrating different LLM configurations."""
    if not AGENT_AVAILABLE:
        print("Error: LangGraph agent not available")
        return
    
    print("ARC LangGraph Agent - LLM Configuration Examples")
    print("=" * 60)
    
    # Try different LLM providers
    agents = []
    
    # OpenAI
    openai_agent = example_with_openai()
    if openai_agent:
        agents.append(("OpenAI", openai_agent))
    
    # Anthropic
    anthropic_agent = example_with_anthropic()  
    if anthropic_agent:
        agents.append(("Anthropic", anthropic_agent))
    
    # Ollama
    ollama_agent = example_with_ollama()
    if ollama_agent:
        agents.append(("Ollama", ollama_agent))
    
    # Test with first available agent
    if agents:
        provider_name, agent = agents[0]
        print(f"\\n=== Testing with {provider_name} ===")
        result = run_example_task(agent)
        
        # Export result
        json_result = agent.export_solution_to_json(result)
        print(f"\\nExported solution format:")
        print(f"  Helper functions: {len(json_result['helper_python_functions'])}")
        print(f"  Steps: {len(json_result['step_by_step_transformations'])}")
        print(f"  Code lines: {len(json_result['python_code'])}")
    else:
        print("\\n✗ No LLM providers available. Please install and configure at least one:")
        print("  - OpenAI: pip install langchain-openai (set OPENAI_API_KEY)")
        print("  - Anthropic: pip install langchain-anthropic (set ANTHROPIC_API_KEY)")
        print("  - Ollama: pip install langchain-ollama (run Ollama locally)")


def usage_instructions():
    """Print usage instructions."""
    print("""
=== LLM Configuration Instructions ===

1. OpenAI Setup:
   pip install langchain-openai
   export OPENAI_API_KEY="your-api-key"

2. Anthropic Setup:
   pip install langchain-anthropic
   export ANTHROPIC_API_KEY="your-api-key"

3. Ollama Setup (Local):
   pip install langchain-ollama
   # Install and run Ollama: https://ollama.ai
   ollama pull llama2

=== Basic Usage ===

from langchain_openai import ChatOpenAI
from arc_langgraph_agent import ARCLangGraphAgent

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create agent
agent = ARCLangGraphAgent(llm, max_attempts=3)

# Solve task
result = agent.solve_task(task_id, task_data)

# Export solution
solution = agent.export_solution_to_json(result)
""")


if __name__ == "__main__":
    import sys
    
    if "--help" in sys.argv or "-h" in sys.argv:
        usage_instructions()
    else:
        main()