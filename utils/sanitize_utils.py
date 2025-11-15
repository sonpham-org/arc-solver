"""Sanitization utilities for ARC tasks."""

import random
import string
import copy
from typing import Dict, List, Any, Tuple


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