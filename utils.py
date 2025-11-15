"""
Backward compatibility utilities for ARC task processing and manipulation.

This module provides imports from the organized utils package for backward compatibility.
New code should import directly from the utils package modules.
"""

# Backward compatibility imports
from utils.sanitize_utils import sanitize_task, get_value_to_char_mapping, reverse_sanitize_grid
from utils.calculation_utils import calculate_results

# Re-export everything for backward compatibility
__all__ = [
    'sanitize_task',
    'get_value_to_char_mapping', 
    'reverse_sanitize_grid',
    'calculate_results',
]