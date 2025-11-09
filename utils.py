"""
Utility functions for ARC task processing and manipulation.
"""

import random
import string
import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any


def sanitize_task(task_data: Dict[str, Any], task_id: str) -> Tuple[Dict[str, Any], str]:
    """
    Sanitize an ARC task by randomizing the task ID and mapping values to random characters.
    
    This function takes an ARC task and creates a sanitized version where:
    1. The task ID is replaced with a random string
    2. All values in input/output grids are mapped to random characters consistently
    
    Args:
        task_data (Dict[str, Any]): The original task data containing 'train' and 'test' keys
        task_id (str): The original task identifier
    
    Returns:
        Tuple[Dict[str, Any], str]: A tuple containing:
            - sanitized_task_data: Task data with values mapped to random characters
            - sanitized_task_id: A randomized task identifier
    """
    # Generate a random task ID
    sanitized_task_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    
    # Create a deep copy of the task data to avoid modifying the original
    sanitized_task_data = copy.deepcopy(task_data)
    
    # Find all unique values used in the task
    unique_values = set()
    
    # Collect values from training examples
    for example in sanitized_task_data.get('train', []):
        for row in example.get('input', []):
            unique_values.update(row)
        for row in example.get('output', []):
            unique_values.update(row)
    
    # Collect values from test examples
    for example in sanitized_task_data.get('test', []):
        for row in example.get('input', []):
            unique_values.update(row)
        # Test examples might not have outputs
        if 'output' in example:
            for row in example.get('output', []):
                unique_values.update(row)
    
    # Create a mapping from values to random characters
    # Use a mix of letters, digits, and symbols for variety
    available_chars = (
        list(string.ascii_letters) + 
        list(string.digits) + 
        ['@', '#', '$', '%', '&', '*', '+', '-', '=', '?']
    )
    
    # Ensure we have enough characters for all unique values
    if len(unique_values) > len(available_chars):
        # If we need more characters, add more symbols
        additional_chars = ['!', '^', '~', '|', '<', '>', '/', '\\', ':', ';']
        available_chars.extend(additional_chars)
    
    # Create random mapping
    unique_values_list = sorted(list(unique_values))  # Sort for consistent ordering
    random.shuffle(available_chars)  # Randomize the character assignment
    
    value_to_char_mapping = {}
    for i, value in enumerate(unique_values_list):
        if i < len(available_chars):
            value_to_char_mapping[value] = available_chars[i]
        else:
            # Fallback: use original value if we run out of characters
            value_to_char_mapping[value] = str(value)
    
    # Apply the mapping to all grids
    def map_grid(grid: List[List[Any]]) -> List[List[str]]:
        """Convert a grid of numbers/characters to a grid of characters using the mapping."""
        return [[value_to_char_mapping.get(cell, str(cell)) for cell in row] for row in grid]
    
    # Apply mapping to training examples
    for example in sanitized_task_data.get('train', []):
        example['input'] = map_grid(example['input'])
        example['output'] = map_grid(example['output'])
    
    # Apply mapping to test examples
    for example in sanitized_task_data.get('test', []):
        example['input'] = map_grid(example['input'])
        # Test examples might not have outputs
        if 'output' in example:
            example['output'] = map_grid(example['output'])
    
    return sanitized_task_data, sanitized_task_id


def get_value_to_char_mapping(task_data: Dict[str, Any]) -> Dict[Any, str]:
    """
    Get the mapping that would be used by sanitize_task without actually sanitizing.
    
    This is useful for debugging or understanding what mapping would be applied.
    
    Args:
        task_data (Dict[str, Any]): The task data to analyze
    
    Returns:
        Dict[Any, str]: Mapping from original values to characters that would be used
    """
    # Find all unique values (same logic as sanitize_task)
    unique_values = set()
    
    for example in task_data.get('train', []):
        for row in example.get('input', []):
            unique_values.update(row)
        for row in example.get('output', []):
            unique_values.update(row)
    
    for example in task_data.get('test', []):
        for row in example.get('input', []):
            unique_values.update(row)
        for row in example.get('output', []):
            unique_values.update(row)
    
    # Create the same character pool
    available_chars = (
        list(string.ascii_letters) + 
        list(string.digits) + 
        ['@', '#', '$', '%', '&', '*', '+', '-', '=', '?']
    )
    
    if len(unique_values) > len(available_chars):
        additional_chars = ['!', '^', '~', '|', '<', '>', '/', '\\', ':', ';']
        available_chars.extend(additional_chars)
    
    # Create mapping (note: this will be different each time due to randomization)
    unique_values_list = sorted(list(unique_values))
    random.shuffle(available_chars)
    
    mapping = {}
    for i, value in enumerate(unique_values_list):
        if i < len(available_chars):
            mapping[value] = available_chars[i]
        else:
            mapping[value] = str(value)
    
    return mapping


def reverse_sanitize_grid(sanitized_grid: List[List[str]], char_to_number_mapping: Dict[str, int]) -> List[List[int]]:
    """
    Convert a sanitized grid back to numbers using a character-to-number mapping.
    
    Args:
        sanitized_grid (List[List[str]]): Grid with character values
        char_to_number_mapping (Dict[str, int]): Mapping from characters back to numbers
    
    Returns:
        List[List[int]]: Grid with original number values
    """
    return [[char_to_number_mapping.get(cell, int(cell) if cell.isdigit() else 0) 
             for cell in row] for row in sanitized_grid]


def calculate_results(results_folder_path: str) -> Dict[str, Any]:
    """
    Calculate summary statistics for ARC test results in a folder.
    
    Analyzes all .json files in the results folder and creates a summary.json file
    with aggregated statistics including total costs and correctness rates.
    
    Args:
        results_folder_path (str): Path to the folder containing task result JSON files
    
    Returns:
        Dict[str, Any]: Summary statistics dictionary that was saved to summary.json
    """
    results_folder = Path(results_folder_path)
    
    if not results_folder.exists():
        raise ValueError(f"Results folder not found: {results_folder_path}")
    
    # Find all JSON files (excluding summary.json and params.json)
    task_files = [f for f in results_folder.glob("*.json") 
                  if f.name not in ["summary.json", "params.json"]]
    
    if not task_files:
        raise ValueError(f"No task JSON files found in {results_folder_path}")
    
    # Initialize counters
    total_tasks = 0
    completely_correct_tasks = 0
    total_tokens = 0
    total_estimated_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    
    task_details = []
    
    # Process each task file
    for task_file in task_files:
        try:
            with open(task_file, 'r') as f:
                task_data = json.load(f)
            
            task_id = task_file.stem
            total_tasks += 1
            
            # Extract token and cost information
            task_tokens = task_data.get('total_tokens', 0)
            task_input_tokens = task_data.get('input_tokens', 0)
            task_output_tokens = task_data.get('output_tokens', 0)
            task_cost = task_data.get('estimated_cost', 0.0)
            
            total_tokens += task_tokens
            total_input_tokens += task_input_tokens
            total_output_tokens += task_output_tokens
            total_estimated_cost += task_cost
            
            # Check if all test cases are completely correct
            test_results = task_data.get('tests', [])
            task_completely_correct = False  # Default to false
            
            if test_results:
                # A task is completely correct if ALL test cases are correct
                task_completely_correct = all(test_case.get('correct', False) for test_case in test_results)
            
            if task_completely_correct:
                completely_correct_tasks += 1
            
            # Store task details for debugging
            task_details.append({
                'task_id': task_id,
                'completely_correct': task_completely_correct,
                'tokens': task_tokens,
                'cost': task_cost,
                'test_cases': len(test_results)
            })
            
        except Exception as e:
            print(f"Warning: Error processing {task_file}: {e}")
            continue
    
    # Calculate summary statistics
    correctness_percentage = (completely_correct_tasks / total_tasks * 100) if total_tasks > 0 else 0.0
    
    # Create summary data
    summary = {
        'total_tasks': total_tasks,
        'completely_correct_tasks': completely_correct_tasks,
        'correctness_percentage': round(correctness_percentage, 2),
        'total_tokens': total_tokens,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'total_estimated_cost': round(total_estimated_cost, 6),
        'average_tokens_per_task': round(total_tokens / total_tasks, 2) if total_tasks > 0 else 0,
        'average_cost_per_task': round(total_estimated_cost / total_tasks, 6) if total_tasks > 0 else 0,
        'task_details': task_details
    }
    
    # Save summary to summary.json
    summary_file = results_folder / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_file}")
    print(f"Total tasks: {total_tasks}")
    print(f"Completely correct: {completely_correct_tasks} ({correctness_percentage:.2f}%)")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total estimated cost: ${total_estimated_cost:.6f}")
    
    return summary