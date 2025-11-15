"""Utilities package for ARC solver."""

from .task_utils import load_arc_tasks, get_task_by_index
from .json_utils import extract_json_from_response, validate_json_structure
from .grid_utils import grid_to_string_lines, calculate_grid_overlap, calculate_grid_iou, is_valid_prediction
from .display_utils import print_colored
from .langchain_utils import run_with_langchain
from .sanitize_utils import sanitize_task, get_value_to_char_mapping, reverse_sanitize_grid
from .calculation_utils import calculate_results

__all__ = [
    # Task utilities
    'load_arc_tasks',
    'get_task_by_index',
    # JSON utilities
    'extract_json_from_response',
    'validate_json_structure',
    # Grid utilities
    'grid_to_string_lines',
    'calculate_grid_overlap',
    'calculate_grid_iou',
    'is_valid_prediction',
    # Display utilities
    'print_colored',
    # LangChain utilities
    'run_with_langchain',
    # Sanitization utilities
    'sanitize_task',
    'get_value_to_char_mapping',
    'reverse_sanitize_grid',
    # Calculation utilities
    'calculate_results',
]