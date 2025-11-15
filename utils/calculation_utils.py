"""Results calculation and analysis utilities."""

import json
from pathlib import Path
from typing import Dict, Any, List


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