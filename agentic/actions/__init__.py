"""
Actions module for ARC agent - backward compatibility layer.

This __init__.py re-exports all functions from the focused modules
to maintain backward compatibility with existing imports.
"""

# Reasoning functions
from .reasoning import (
    generate_reasoning_trace,
    generate_distilled_reasoning,
    extract_reasoning_content,
    extract_reasoning_from_reflection,
    extract_key_insight_from_reasoning,
    analyze_training_examples,
)

# Transformation functions
from .transformation import (
    generate_transformation_steps,
)

# Refinement-specific transformation function
from .refine import (
    generate_refined_transformation_steps,
    generate_reflection_reasoning_trace,
)

# Fusion-specific transformation function
from .fuse import (
    generate_fused_transformation_steps,
    generate_fused_reasoning_trace,
)

# Utility functions for transformations
from .utilities import (
    parse_transformation_steps,
    build_steps_text_from_transformation_steps,
)

# Code generation functions
from .code_generation import (
    generate_code_from_reasoning,
    generate_code_from_reasoning_and_transformations,
    generate_fallback_code_from_steps,
    extract_python_solutions,
    ensure_imports_in_code,
)

# Code execution functions
from .code_execution import (
    execute_transformation_code,
    evaluate_example,
    calculate_grid_results,
    test_and_fix_code_from_trial_run,
)

# Creation functions
from .create import (
    create_solutions_with_reasoning,
)

# Fusion functions
from .fuse import (
    fuse_solutions_with_reasoning,
    result_comparison_text,
)

# Refinement functions
from .refine import (
    refine_solutions_with_reasoning,
    analyze_failures,
)

# RAG functions
from .rag import (
    generate_embedding_from_distilled_reasoning,
    store_record,
    retrieve_similar_distillations,
    extract_helpers_from_python_codes,
)

# Utility functions
from .utilities import (
    format_grid_for_prompt,
    format_grid_for_analysis,
    format_difference_map,
    _grid_to_image_bytes,
    generate_llm_predicted_output,
    _flatten_content,
)

__all__ = [
    # Reasoning
    'generate_reasoning_trace',
    'generate_reflection_reasoning_trace',
    'generate_fused_reasoning_trace',
    'generate_distilled_reasoning',
    'extract_reasoning_content',
    'extract_reasoning_from_reflection',
    'extract_key_insight_from_reasoning',
    'analyze_training_examples',
    # Transformation
    'generate_transformation_steps',
    'generate_refined_transformation_steps',
    'generate_fused_transformation_steps',
    'parse_transformation_steps',
    'build_steps_text_from_transformation_steps',
    # Code generation
    'generate_code_from_reasoning',
    'generate_code_from_reasoning_and_transformations',
    'generate_fallback_code_from_steps',
    'extract_python_solutions',
    'ensure_imports_in_code',
    # Code execution
    'execute_transformation_code',
    'evaluate_example',
    'calculate_grid_results',
    'test_and_fix_code_from_trial_run',
    # Fusion
    'fuse_solutions_with_reasoning',
    'create_solutions_with_reasoning',
    'result_comparison_text',
    # Refinement
    'refine_solutions_with_reasoning',
    'analyze_failures',
    # RAG
    'generate_embedding_from_distilled_reasoning',
    'store_record',
    'retrieve_similar_distillations',
    'extract_helpers_from_python_codes',
    # Utilities
    'format_grid_for_prompt',
    'format_grid_for_analysis',
    'format_difference_map',
    '_grid_to_image_bytes',
    'generate_llm_predicted_output',
    '_flatten_content',
]
