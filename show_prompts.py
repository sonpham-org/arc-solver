#!/usr/bin/env python3
"""
Prompt inspector tool - shows actual prompts being used during execution.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arc_langgraph_agent.nodes import *
import json

def show_live_prompts():
    """Generate and display real prompts using sample data."""
    print("="*80)
    print("LIVE PROMPT EXAMPLES")
    print("="*80)
    
    # Sample task data
    sample_task = {
        'train': [
            {
                'input': [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                'output': [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
            },
            {
                'input': [[0, 0, 0], [1, 1, 1], [0, 0, 0]],
                'output': [[0, 0, 0], [0, 0, 0], [1, 1, 1]]
            }
        ],
        'test': [
            {'input': [[0, 1, 0], [0, 0, 0], [0, 0, 0]]}
        ]
    }
    
    # Sample helper functions
    sample_helpers = [
        {
            'name': 'rotate_90',
            'description': 'Rotate grid 90 degrees clockwise',
            'code': 'def rotate_90(grid): return [[grid[j][i] for j in range(len(grid))] for i in range(len(grid[0]))]'
        },
        {
            'name': 'find_objects',
            'description': 'Find connected components in grid',
            'code': 'def find_objects(grid): # implementation here'
        }
    ]
    
    print("\n1. INITIAL GENERATION PROMPT (ACTUAL):")
    print("-" * 60)
    try:
        prompt = build_python_focused_prompt(sample_task['train'], sample_helpers, [])
        print(prompt[:1500] + "..." if len(prompt) > 1500 else prompt)
    except Exception as e:
        print(f"Error generating initial prompt: {e}")
    
    print("\n\n2. REFLECTION PROMPT (ACTUAL):")
    print("-" * 60)
    
    # Sample failed solution
    failed_solution = {
        'main_code': '''def transform(input_grid):
    # Move all 1s down by one row
    result = [[0] * len(input_grid[0]) for _ in range(len(input_grid))]
    for i in range(len(input_grid)):
        for j in range(len(input_grid[0])):
            if input_grid[i][j] == 1:
                if i + 1 < len(result):
                    result[i + 1][j] = 1
    return result''',
        'helper_code': '',
        'attempt_number': 1
    }
    
    # Sample test results
    failed_tests = [
        {
            'example_index': 0,
            'predicted_output': [[0, 0, 0], [1, 1, 1], [0, 0, 0]],
            'overlap_percentage': 100.0,
            'iou_percentage': 100.0,
            'error_message': None
        },
        {
            'example_index': 1,
            'predicted_output': [[0, 0, 0], [0, 0, 0], [1, 1, 1]], 
            'overlap_percentage': 100.0,
            'iou_percentage': 100.0,
            'error_message': None
        }
    ]
    
    try:
        reflection_prompt = build_arc_style_reflection_prompt(
            failed_solution, 
            failed_tests, 
            sample_task['train'], 
            []
        )
        print(reflection_prompt[:2000] + "..." if len(reflection_prompt) > 2000 else reflection_prompt)
    except Exception as e:
        print(f"Error generating reflection prompt: {e}")

def show_helper_extraction_prompt():
    """Show helper function extraction prompt."""
    print("\n\n3. HELPER EXTRACTION PROMPT (ACTUAL):")
    print("-" * 60)
    
    sample_solutions = [
        {
            'main_code': '''def transform(input_grid):
    def rotate_grid(grid):
        return [[grid[j][i] for j in range(len(grid))] for i in range(len(grid[0]))]
    
    def find_pattern(grid):
        # Find where 1s are located
        ones = []
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    ones.append((i, j))
        return ones
    
    # Main logic
    rotated = rotate_grid(input_grid)
    return rotated''',
            'attempt_number': 1
        }
    ]
    
    existing_helpers = [
        {
            'name': 'count_colors',
            'description': 'Count occurrences of each color in grid',
            'code': 'def count_colors(grid): ...'
        }
    ]
    
    try:
        helper_prompt = extract_helper_functions_with_llm(sample_solutions, existing_helpers)
        print("Sample Helper Extraction Prompt:")
        print("-" * 40)
        print("You are an expert Python programmer.")
        print("Extract useful helper functions from the following code solutions...")
        print("[Solutions would be listed here]")
        print("")
        print("Existing helper functions to avoid duplicating:")
        print("- count_colors: Count occurrences of each color in grid")
        print("")
        print("Return only new, reusable helper functions in JSON format:")
        print('[{"name": "function_name", "description": "what it does", "code": "def function_name(): ..."}]')
    except Exception as e:
        print(f"Error generating helper extraction example: {e}")

def show_prompt_debugging_tips():
    """Show tips for debugging prompts."""
    print("\n\n" + "="*80)
    print("PROMPT DEBUGGING TIPS")
    print("="*80)
    
    print("\n1. VIEWING PROMPTS DURING EXECUTION:")
    print("-" * 50)
    print("  Add debug prints in nodes.py:")
    print("  ```python")
    print("  def generate_solution_node(state: AgentState) -> AgentState:")
    print("      prompt = build_initial_generation_prompt(...)")
    print("      print(f'\\n=== GENERATION PROMPT ===\\n{prompt}\\n')")
    print("      # rest of function...")
    print("  ```")
    
    print("\n2. PROMPT LENGTH MONITORING:")
    print("-" * 50)
    print("  Check token counts:")
    print("  ```python")
    print("  prompt = build_reflection_prompt(...)")
    print("  print(f'Prompt length: {len(prompt)} chars, ~{len(prompt)//4} tokens')")
    print("  ```")
    
    print("\n3. LOGGING TO FILE:")
    print("-" * 50)
    print("  Add to your code:")
    print("  ```python")
    print("  import logging")
    print("  logging.basicConfig(filename='prompts.log', level=logging.DEBUG)")
    print("  logging.debug(f'Prompt: {prompt}')")
    print("  ```")
    
    print("\n4. INTERACTIVE INSPECTION:")
    print("-" * 50)
    print("  Run this script to see sample prompts:")
    print("  python show_prompts.py")
    print("")
    print("  Or import and use directly:")
    print("  ```python")
    print("  from arc_langgraph_agent.nodes import build_python_focused_prompt")
    print("  prompt = build_python_focused_prompt(task_data['train'], helpers, previous)")
    print("  print(prompt)")
    print("  ```")

def main():
    """Main function to display all prompt examples."""
    try:
        show_live_prompts()
        show_helper_extraction_prompt()
        show_prompt_debugging_tips()
        
        print("\n\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("1. Run the agent and check output files for results")
        print("2. Add debug prints to nodes.py to see live prompts")
        print("3. Use this script anytime: python show_prompts.py")
        print("="*80)
        
    except Exception as e:
        print(f"Error during prompt display: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()