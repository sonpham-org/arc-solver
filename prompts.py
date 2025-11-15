# Legacy prompts.py - now redirects to prompts module
# 
# This file maintains backward compatibility while the actual prompt functions 
# have been moved to the prompts/ directory for better organization.

# Legacy constants (kept for backward compatibility)
intro = """
Explain the ARC Challenge in one sentence.
"""

reasoning = """
Describe how an intelligent system might approach solving an ARC problem using pattern recognition.
"""

task_example = """
Given an ARC task with colored grids, describe how to infer the transformation rule
from input to output examples.
"""

creative = """
Propose a novel strategy to improve ARC reasoning using multimodal embeddings.
"""

# Import all prompt functions from the new prompts module
from prompts import (
    build_arc_prompt,
    build_apply_prompt,
    build_apply_prompt_2,
    build_reflection_prompt,
    build_code_repair_prompt,
    build_arc_reflection_prompt,
    build_arc_baseline_prompt
)

# Re-export for backward compatibility
__all__ = [
    'intro', 'reasoning', 'task_example', 'creative',
    'build_arc_prompt',
    'build_apply_prompt',
    'build_apply_prompt_2', 
    'build_reflection_prompt',
    'build_code_repair_prompt',
    'build_arc_reflection_prompt',
    'build_arc_baseline_prompt'
]
