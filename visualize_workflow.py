#!/usr/bin/env python3
"""
Workflow and prompt visualization tool for the LangGraph ARC agent.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arc_langgraph_agent.workflow import compile_workflow, create_initial_state
from arc_langgraph_agent.agent import ARCLangGraphAgent
from arc_langgraph_agent.nodes import *
from model_configs import MODEL_CONFIGS
import json

def visualize_workflow():
    """Display the workflow structure and node information."""
    print("="*80)
    print("LANGGRAPH ARC AGENT WORKFLOW VISUALIZATION")
    print("="*80)
    
    print("\n1. WORKFLOW NODES:")
    print("-" * 50)
    
    # Get all node functions from the nodes module
    node_functions = [
        ("start", "Entry point - initializes the workflow"),
        ("generate_solution", "Generates initial code solution using LLM"),
        ("test_solution", "Tests generated code against training examples"),
        ("should_continue", "Decision node - continue or finish based on results"),
        ("refine_solution", "Refines failed solution using reflection"),
        ("extract_helpers", "Extracts helper functions for reuse"),
        ("end", "Terminal node - workflow completion")
    ]
    
    for i, (node_name, description) in enumerate(node_functions, 1):
        print(f"  {i}. {node_name.upper()}")
        print(f"     └─ {description}")
    
    print("\n2. WORKFLOW FLOW:")
    print("-" * 50)
    print("  start")
    print("    ↓")
    print("  generate_solution")
    print("    ↓")
    print("  test_solution")
    print("    ↓")
    print("  should_continue ──┐")
    print("    ↓             │ (if max attempts reached)")
    print("  refine_solution  │")
    print("    ↓             │")
    print("  extract_helpers  │")
    print("    ↓             │")
    print("    └─────────────┘")
    print("    ↓")
    print("  end")
    
    print("\n3. STATE SCHEMA:")
    print("-" * 50)
    from arc_langgraph_agent.schema import AgentState
    print("  AgentState contains:")
    print("  - task_id: str")
    print("  - task_data: Dict[str, Any]")
    print("  - solutions: List[CodeSolution]")
    print("  - test_results: List[TestResult]")
    print("  - reflection_history: List[Dict]")
    print("  - helper_functions: List[HelperFunction]")
    print("  - attempts: int")
    print("  - max_attempts: int")
    print("  - success: bool")
    print("  - execution_time: float")

def show_sample_prompts():
    """Display sample prompts used in the workflow."""
    print("\n\n" + "="*80)
    print("SAMPLE PROMPTS")
    print("="*80)
    
    print("\n1. INITIAL GENERATION PROMPT:")
    print("-" * 50)
    print("```")
    print("You are an expert at solving Abstract Reasoning Corpus (ARC) tasks.")
    print("Your goal is to understand the pattern and write Python code to transform input to output.")
    print("")
    print("TASK EXAMPLES:")
    print("Training Example 1")
    print("Input: [grid representation]")
    print("Output: [expected output grid]")
    print("...")
    print("")
    print("HELPER FUNCTIONS AVAILABLE:")
    print("- rotate_90: Rotate grid 90 degrees clockwise")
    print("- find_objects: Find connected components in grid")
    print("...")
    print("")
    print("PYTHON CODE REQUIREMENTS:")
    print("1. Write a function called 'transform(input_grid)' that returns the output grid")
    print("2. Use clear, readable Python code with proper error handling")
    print("3. Focus on the core transformation logic - be precise and efficient")
    print("4. Use available helper functions when appropriate")
    print("5. Add comments explaining the transformation steps")
    print("6. Return ONLY executable Python code, no explanations")
    print("")
    print("Generate the transform function:")
    print("```")
    
    print("\n2. ARC-STYLE REFLECTION PROMPT:")
    print("-" * 50)
    print("```")
    print("You are an expert in reasoning about Abstract Reasoning Corpus (ARC) puzzles.")
    print("You previously attempted to solve this task but your solution was incorrect.")
    print("")
    print("TASK REFLECTION AND DEEP ANALYSIS")
    print("")
    print("Your goal:")
    print("Analyze your previous attempt deeply, understand why it failed, and provide")
    print("a CORRECTED transformation that:")
    print("1. Correctly maps every training input to its output")
    print("2. Is general and intuitive (no memorization or hard-coded values)")
    print("3. Is logical, reproducible, and object-level")
    print("")
    print("Original Training Examples")
    print("[training examples with input/output grids]")
    print("")
    print("Your Previous Solution")
    print("[previous failed code]")
    print("")
    print("Detailed Failure Analysis")
    print("Training Example 1 - FAILED")
    print("Expected size: 3x3, Predicted size: 4x3")
    print("Overlap: 45.0%")
    print("IOU (Intersection over Union): 30.0%")
    print("Error: Index out of range")
    print("")
    print("Deep Reflection Instructions")
    print("First, analyze what went wrong inside ```reasoning ``` block:")
    print("1. PATTERN MISINTERPRETATION: What pattern did you miss?")
    print("2. LOGIC ERRORS: Where exactly did your transformation logic fail?")
    print("3. EDGE CASES: What cases did you not handle properly?")
    print("4. OBJECT-LEVEL THINKING: How should you think about objects/shapes?")
    print("5. CORE INSIGHT: What is the single most important insight missing?")
    print("")
    print("Then provide a COMPLETELY REWRITTEN solution...")
    print("```")

def show_metrics_info():
    """Display information about the metrics calculated."""
    print("\n\n" + "="*80)
    print("METRICS & OUTPUT FORMAT")
    print("="*80)
    
    print("\n1. EVALUATION METRICS:")
    print("-" * 50)
    print("  OVERLAP:")
    print("  - Percentage of matching cells within intersection area")
    print("  - Formula: matching_cells / intersection_area * 100")
    print("  - Example: If 9 out of 9 overlapping cells match → 100%")
    print("")
    print("  IOU (Intersection over Union):")
    print("  - Percentage of matching cells relative to total union area")
    print("  - Formula: matching_cells / union_area * 100")
    print("  - Union = pred_area + expected_area - intersection_area")
    print("  - Example: 3x4 vs 4x3 grids with 9 matches → 9/15 = 60%")
    print("")
    print("  SIZE INFORMATION:")
    print("  - input_size: 'HxW' format for input dimensions")
    print("  - output_size: 'HxW' format for expected output dimensions")
    print("  - predict_size: 'HxW' format for predicted output dimensions")
    
    print("\n2. OUTPUT FILE STRUCTURE:")
    print("-" * 50)
    print("  output/")
    print("  └── YYYY-MM-DDTHH-MM-SS-ffffff/")
    print("      ├── params.json              # Run parameters")
    print("      ├── summary.json             # Overall statistics")
    print("      ├── training_task_ids.txt    # Available training tasks")
    print("      ├── evaluation_task_ids.txt  # Available evaluation tasks")
    print("      ├── {task_id}.json           # Individual task results")
    print("      └── ...")
    
    print("\n3. TASK RESULT FORMAT:")
    print("-" * 50)
    print("  {")
    print("    'task_id': '25ff71a9',")
    print("    'trains': [")
    print("      {")
    print("        'input': ['111', '000', '000'],")
    print("        'output': ['000', '111', '000'],")
    print("        'predict': ['000', '000', '111'],")
    print("        'iou': 33.3,")
    print("        'overlap': 100.0,")
    print("        'correct': false,")
    print("        'input_size': '3x3',")
    print("        'output_size': '3x3',")
    print("        'predict_size': '3x3'")
    print("      }")
    print("    ],")
    print("    'tests': [...],")
    print("    'success': true,")
    print("    'attempts': 3,")
    print("    'execution_time': 25.4")
    print("  }")

def show_usage_examples():
    """Show usage examples for running the agent."""
    print("\n\n" + "="*80)
    print("USAGE EXAMPLES")
    print("="*80)
    
    print("\n1. SINGLE TASK TEST:")
    print("-" * 50)
    print("  # Test specific task")
    print("  python run_langgraph_agent.py --model gemini-2.0-flash --mode single --task-id 25ff71a9")
    print("")
    print("  # Test by index")
    print("  python run_langgraph_agent.py --model gpt-4o-mini --mode single --task-index 0")
    print("")
    print("  # Random task")
    print("  python run_langgraph_agent.py --model claude-3-5-sonnet --mode single")
    
    print("\n2. BATCH TESTING:")
    print("-" * 50)
    print("  # Test 10 random tasks")
    print("  python run_langgraph_agent.py --model gemini-2.0-flash --mode batch --num-tasks 10")
    print("")
    print("  # With custom max attempts")
    print("  python run_langgraph_agent.py --model gpt-4o --mode batch --num-tasks 5 --max-attempts 3")
    
    print("\n3. AVAILABLE MODELS:")
    print("-" * 50)
    available_models = list(MODEL_CONFIGS.keys())[:10]  # Show first 10
    for model in available_models:
        provider = MODEL_CONFIGS[model].get('provider', 'unknown')
        print(f"  - {model} ({provider})")
    print(f"  ... and {len(MODEL_CONFIGS) - 10} more")

def main():
    """Main function to display all visualizations."""
    try:
        visualize_workflow()
        show_sample_prompts()
        show_metrics_info() 
        show_usage_examples()
        
        print("\n\n" + "="*80)
        print("QUICK START")
        print("="*80)
        print("1. Run a single task:")
        print("   python run_langgraph_agent.py --model gemini-2.0-flash --mode single")
        print("")
        print("2. Check output directory for detailed results")
        print("")
        print("3. View this help anytime:")
        print("   python visualize_workflow.py")
        print("="*80)
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()