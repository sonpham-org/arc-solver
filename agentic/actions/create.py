"""
Solution creation functions for ARC agent.
"""

import uuid
import base64
from typing import List, Dict, Tuple, Optional

from ..schema import ReasoningTraceRecord
from ..debug import print_prompt_and_response
from .utilities import _grid_to_image_bytes
from .reasoning import generate_distilled_reasoning, generate_reasoning_trace
from .transformation import generate_transformation_steps
from .code_generation import generate_code_from_reasoning, generate_code_from_reasoning_and_transformations
from .code_execution import test_and_fix_code_from_trial_run
from .rag import (
    store_record, 
    generate_embedding_from_distilled_reasoning,
    extract_helpers_from_python_codes
)


def create_solutions_with_reasoning(llm, transformation_llm, code_llm, 
                                      training_examples: List[Dict], num_solutions: int,
                                      enable_visual_cue: bool = False,
                                      enable_rag_hint: bool = False,
                                      num_inreasoning_augmentations: int = 0,
                                      memory_context: str = "") -> Tuple[List[str], str, List[Dict]]:
    """Generate Python transformation code using reasoning-first approach.

    When `enable_visual_cue` is True, this function will render training
    input/output pairs to PNG images, encode them as base64 and include them
    in the LLM invocation (if the LLM driver supports image messages).

    Returns:
        Tuple of (python_codes_list, reasoning_trace, transformation_solutions_list, rag_entry, reasoning_retries, transformation_retries)
    """
    visual_cues = []
    if enable_visual_cue:
        # Build visual cues: for each training example, create a small image
        # that shows the input and expected output stacked vertically.
        for i, ex in enumerate(training_examples):
            inp = ex.get('input') or []
            out = ex.get('output') or []
            inp_bytes = _grid_to_image_bytes(inp)
            out_bytes = _grid_to_image_bytes(out)
            b64_in = base64.b64encode(inp_bytes).decode('utf-8')
            b64_out = base64.b64encode(out_bytes).decode('utf-8')
            visual_cues.append({
                'example_index': i,
                'input_b64': b64_in,
                'output_b64': b64_out,
            })

    # Step 1: Generate reasoning trace
    reasoning_trace, reasoning_retries = generate_reasoning_trace(
        llm, 
        training_examples, 
        visual_cues=visual_cues if enable_visual_cue else None,
        num_inreasoning_augmentations=num_inreasoning_augmentations,
        memory_context=memory_context
    )

    # Step 2: Extract step-by-step transformation from reasoning
    transformation_solutions_list, transformation_retries = generate_transformation_steps(transformation_llm, reasoning_trace, training_examples, num_solutions)

    # Step 3: Generate Python code(s) based on reasoning and steps
    # Note: `generate_code_from_reasoning` may return multiple candidate code strings.
    if not transformation_solutions_list:
        python_codes_list, _ = generate_code_from_reasoning(code_llm, reasoning_trace, training_examples, num_solutions)
    else:
        python_codes_list = generate_code_from_reasoning_and_transformations(code_llm, reasoning_trace, transformation_solutions_list,
                                                                             training_examples)
    # Trial-run + automatic fix: run candidates on a probe example and
    # request fixes from the code LLM if needed. This logic is encapsulated
    # in `test_and_fix_code_from_trial_run` which returns possibly-updated
    # candidates and the trial run diagnostics.
    try:
        result = test_and_fix_code_from_trial_run(code_llm, python_codes_list, training_examples)
        if result is not None:
            python_codes_list, trial_run_results = result
    except Exception as e:
        print(f"Warning: test_and_fix_code_from_trial_run failed: {e}")
    
    # Step 4: Create rag entry if enabled
    if enable_rag_hint:
        distilled_reasoning = generate_distilled_reasoning(llm, reasoning_trace, transformation_solutions_list, python_codes_list)
        distilled_text = f"Strategy: {distilled_reasoning.get('strategy', '')}\nConcepts: {', '.join(distilled_reasoning.get('concepts', []))}"
        embedding = generate_embedding_from_distilled_reasoning(distilled_text)
        helpers = extract_helpers_from_python_codes(python_codes_list)
        rag_entry = ReasoningTraceRecord(
            id=str(uuid.uuid4()),
            reasoning_text=reasoning_trace,
            reasoning_summary=distilled_reasoning.get('strategy', ''),
            concepts=distilled_reasoning.get('concepts', []),
            helpers=helpers,
            vector=embedding,
        )

        # Best-effort: store the distilled reasoning into the Qdrant vector store
        # If qdrant is not available this will be a no-op and will not raise.
        try:
            stored = store_record(rag_entry)
            if not stored:
                # Quietly continue if storing was skipped or unavailable
                pass
        except Exception as e:
            print(f"Warning: store_record raised an exception: {e}")
    else:
        rag_entry = None

    # Attach visual cue data onto each transformation dict (best-effort)
    if enable_visual_cue and visual_cues:
        # Try to attach example-level cues to the first solution dict to be saved later
        for sol in transformation_solutions_list:
            sol['_visual_cues'] = visual_cues

    # Return the list of candidate codes, plus reasoning and steps.
    return python_codes_list, reasoning_trace, transformation_solutions_list, rag_entry, reasoning_retries, transformation_retries
