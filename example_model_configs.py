#!/usr/bin/env python3
"""
Example script showing how to use the ARC LangGraph Agent with models from model_configs.py
"""

import os
from model_configs import MODEL_CONFIGS, DEFAULT_MODEL, find_model_key

def initialize_llm_from_config(model_name: str):
    """Initialize LLM based on model_configs.py configuration."""
    
    # Find the model key
    model_key = find_model_key(model_name)
    if not model_key:
        print(f"Unknown model: {model_name}")
        print(f"Available models: {', '.join(MODEL_CONFIGS.keys())}")
        return None
    
    # Get model config
    config = MODEL_CONFIGS[model_key]
    
    # Handle aliases
    if "alias_of" in config:
        config = MODEL_CONFIGS[config["alias_of"]]
    
    provider = config.get("provider")
    description = config.get("description", model_key)
    
    print(f"Initializing {description} (provider: {provider})")
    
    try:
        if provider == "google" or provider == "learnlm":
            # Google Gemini models
            try:
                from langchain_google_vertexai import ChatVertexAI
                return ChatVertexAI(
                    model_name=model_key,
                    temperature=0,
                    max_output_tokens=2000
                )
            except ImportError:
                # Fallback to langchain-google-genai if vertexai not available
                from langchain_google_genai import ChatGoogleGenerativeAI
                api_key = os.getenv('GEMINI_API_KEY')
                if not api_key:
                    print("GEMINI_API_KEY not set for Google models")
                    return None
                return ChatGoogleGenerativeAI(
                    model=model_key,
                    google_api_key=api_key,
                    temperature=0
                )
        
        elif provider == "ollama":
            # Ollama local models
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=model_key,
                temperature=0
            )
        
        else:
            print(f"Unsupported provider: {provider}")
            return None
            
    except ImportError as e:
        print(f"Error importing {provider} client: {e}")
        provider_package = "langchain-google-genai" if provider in ["google", "learnlm"] else f"langchain-{provider}"
        print(f"Please install: pip install {provider_package}")
        return None
    except Exception as e:
        print(f"Error initializing {provider} model '{model_key}': {e}")
        return None


def main():
    """Demonstrate LangGraph agent with different models from model_configs.py"""
    
    # Import LangGraph agent (only if available)
    try:
        from arc_langgraph_agent import ARCLangGraphAgent
        LANGGRAPH_AVAILABLE = True
    except ImportError as e:
        print(f"LangGraph agent not available: {e}")
        print("Please install: pip install langgraph langchain-core")
        LANGGRAPH_AVAILABLE = False
        return 1
    
    print("=== ARC LangGraph Agent - Model Configuration Demo ===\\n")
    
    # Show available models
    print("Available models from model_configs.py:")
    for model_name, config in MODEL_CONFIGS.items():
        provider = config.get("provider", "unknown")
        description = config.get("description", model_name)
        if "alias_of" in config:
            print(f"  {model_name}: Alias of {config['alias_of']}")
        else:
            print(f"  {model_name}: {description} ({provider})")
    
    print(f"\\nDefault model: {DEFAULT_MODEL}\\n")
    
    # Try to initialize the default model
    print("Testing LLM initialization...")
    llm = initialize_llm_from_config(DEFAULT_MODEL)
    
    if llm is None:
        print("\\nFailed to initialize default model. Please check your configuration.")
        return 1
    
    print(f"✓ Successfully initialized {DEFAULT_MODEL}\\n")
    
    # Create agent
    print("Creating ARC LangGraph Agent...")
    agent = ARCLangGraphAgent(llm, max_attempts=2)
    print("✓ Agent created successfully\\n")
    
    # Example task for testing
    example_task_data = {
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
            {"input": [[0, 1, 0]]}
        ]
    }
    
    print("Testing agent on example task...")
    print("Task: Invert 0s and 1s")
    print("Training examples:")
    for i, example in enumerate(example_task_data["train"], 1):
        print(f"  Example {i}: {example['input']} -> {example['output']}")
    
    # Solve the task
    try:
        result = agent.solve_task("example_invert", example_task_data)
        
        print(f"\\n=== Results ===")
        print(f"Success: {result['success']}")
        print(f"Success Rate: {result['best_success_rate']:.2%}")
        print(f"Attempts: {result['attempts_made']}")
        print(f"Execution Time: {result['execution_time']:.2f}s")
        
        if result['final_solution']:
            solution = result['final_solution']
            print(f"\\n=== Generated Code ===")
            print(solution['main_code'])
            
            # Test the solution
            test_results = agent.test_solution_on_examples(result, example_task_data)
            print(f"\\n=== Test Results ===")
            print(f"Overall Success Rate: {test_results['overall_success_rate']:.2%}")
            print(f"Successful Examples: {test_results['successful_examples']}/{test_results['total_examples']}")
        
        print("\\n✓ Agent test completed successfully!")
        
    except Exception as e:
        print(f"Error running agent: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())