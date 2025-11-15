"""Task-related utility functions."""

import json
import random


def load_arc_tasks(filepath):
    """Load ARC tasks from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_task_by_index(tasks, index=None):
    """Get a task by index or task ID, or random if None."""
    if index is None:
        # Select random task
        task_ids = list(tasks.keys())
        task_id = random.choice(task_ids)
        task_index = task_ids.index(task_id)
        return task_id, task_index, tasks[task_id]
    
    # Check if index is a string (task ID)
    if isinstance(index, str):
        if index in tasks:
            task_ids = list(tasks.keys())
            task_index = task_ids.index(index)
            return index, task_index, tasks[index]
        else:
            raise ValueError(f"Task ID '{index}' not found in tasks")
    
    # Handle integer index
    task_ids = list(tasks.keys())
    if 0 <= index < len(task_ids):
        task_id = task_ids[index]
        return task_id, index, tasks[task_id]
    else:
        raise ValueError(f"Task index {index} out of range [0, {len(task_ids)-1}]")