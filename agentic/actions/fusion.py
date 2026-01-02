"""
Solution fusion functions for ARC agent.
"""

import copy
import uuid
from typing import List, Dict, Tuple

from ..schema import CodeSolution, ExampleResult, ReasoningTraceRecord
from .utilities import format_grid_for_prompt, format_difference_map, parse_transformation_steps, build_steps_text_from_transformation_steps
from .reasoning import generate_distilled_reasoning
from .rag import retrieve_similar_distillations, store_record, generate_embedding_from_distilled_reasoning


def create_solutions_with_reasoning(llm, transformation_llm, code_llm, 
                                      training_examples: List[Dict], num_solutions: int,
                                      enable_visual_cue: bool = False,
                                      enable_rag_hint: bool = False) -> Tuple[List[str], str, List[Dict]]:
    """Generate Python transformation code using reasoning-first approach.

    When `enable_visual_cue` is True, this function will render training
    input/output pairs to PNG images, encode them as base64 and include them
    in the LLM invocation (if the LLM driver supports image messages).

    Returns:
        Tuple of (python_code_list, reasoning_trace, transformation_steps)
    """
    from .reasoning import generate_reasoning_trace
    from .transformation import generate_transformation_steps
    from .code_generation import generate_code_from_reasoning, generate_code_from_reasoning_and_transformations
    from .code_execution import test_and_fix_code_from_trial_run
    from .rag import extract_helpers_from_python_codes
    from .utilities import _grid_to_image_bytes
    
    visual_cues = []
    if enable_visual_cue:
        # Build visual cues: for each training example, create a small image
        # that shows the input and expected output stacked vertically.
        import base64
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
    reasoning_trace, reasoning_retries = (generate_reasoning_trace(llm, training_examples) if not enable_visual_cue 
                                          else generate_reasoning_trace(llm, training_examples, visual_cues=visual_cues))

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
        python_codes_list, trial_run_results = test_and_fix_code_from_trial_run(code_llm, python_codes_list, training_examples)
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


def fuse_solutions_with_reasoning(llm,
                                  transformation_llm,
                                  code_llm,
                                  sola: CodeSolution,
                                  solb: CodeSolution,
                                  training_examples: List[Dict],
                                  num_fused_solutions: int,
                                  enable_visual_cue: bool = False,
                                  enable_rag_hint: bool = False) -> Tuple[List[str], str, List[Dict]]:
    """Attempt to fuse two CodeSolution candidates into a stronger combined solution.

    Returns a tuple: (python_codes_list, fused_reasoning_trace, fused_transformation_solutions_list)
    """
    from .reasoning import generate_fused_reasoning_trace
    from .transformation import generate_fused_transformation_steps
    from .code_generation import generate_code_from_reasoning, generate_code_from_reasoning_and_transformations
    from .rag import extract_helpers_from_python_codes
    
    # Build visual cues if requested
    if enable_visual_cue:
        # TODO: Implement visual cue generation for fused solutions if needed
        pass

    # Merge training_results from both solutions (concatenate, allowing duplicates)
    tra = sola.get('training_results') or []
    trb = solb.get('training_results') or []

    # 1) Generate fused reasoning trace
    fused_reasoning, reasoning_retries = generate_fused_reasoning_trace(llm, sola, solb, tra, trb, training_examples, enable_rag_hint)

    # 2) Generate fused transformation steps
    fused_transformation_solutions, transformation_retries = generate_fused_transformation_steps(transformation_llm, fused_reasoning, sola, solb, tra, trb, training_examples, num_fused_solutions)

    # 3) Generate candidate Python implementations from fused reasoning and steps
    if not fused_transformation_solutions:
        python_codes_list = generate_code_from_reasoning(code_llm, fused_reasoning, training_examples)
    else:
        python_codes_list = generate_code_from_reasoning_and_transformations(code_llm, fused_reasoning, fused_transformation_solutions, training_examples)

    # Step 4: Create rag entry if enabled
    rag_entry = None
    if enable_rag_hint:
        distilled_reasoning = generate_distilled_reasoning(llm, fused_reasoning, fused_transformation_solutions, python_codes_list)
        distilled_text = f"Strategy: {distilled_reasoning.get('strategy', '')}\nConcepts: {', '.join(distilled_reasoning.get('concepts', []))}"
        embedding = generate_embedding_from_distilled_reasoning(distilled_text)
        helpers = extract_helpers_from_python_codes(python_codes_list)
        rag_entry = ReasoningTraceRecord(
            id=str(uuid.uuid4()),
            reasoning_text=fused_reasoning,
            reasoning_summary=distilled_reasoning.get('strategy', ''),
            concepts=distilled_reasoning.get('concepts', []),
            helpers=helpers,
            vector=embedding,
        )
        # Best-effort: store the distilled reasoning into the Qdrant vector store
        # If qdrant is not available this will be a no-op and will not raise.
        try:
            stored = store_record(rag_entry)
            if stored:
                print(f"✓ Stored fused RAG entry (concepts: {len(rag_entry.concepts)}, helpers: {len(rag_entry.helpers)})")
        except Exception as e:
            print(f"Warning: store_record raised an exception: {e}")

    return python_codes_list, fused_reasoning, fused_transformation_solutions, rag_entry, reasoning_retries, transformation_retries


def result_comparison_text(training_results1: List[ExampleResult],
                           training_results2: List[ExampleResult]) -> str:
    """Generate a comparison text of training results between two solutions."""

    training_results_comparison = []
    for i, (tr1, tr2) in enumerate(zip(training_results1, training_results2)):
        size_match_a = tr1.get('matching_size', False)
        overlap_a = tr1.get('overlap_percentage', 0.0)
        size_match_b = tr2.get('matching_size', False)
        overlap_b = tr2.get('overlap_percentage', 0.0)
        training_results_comparison.extend([
            f"Example {i+1}:",
            f"  Input:\n{format_grid_for_prompt(tr1.get('input', []), indent=4)}",
            f"  Expected Output:\n{format_grid_for_prompt(tr1.get('expected_output', []), indent=4)}",
            f"  Solution A - size match: {size_match_a}, overlap: {overlap_a:.1f}%",
            f"  Solution A - predicted Output:\n{format_grid_for_prompt(tr1.get('predicted_output', []), indent=4)}",
            f"  Solution A - difference:\n{format_difference_map(tr1.get('predicted_output', []), tr1.get('expected_output', []), indent=4)}",
            f"  Solution B - size match: {size_match_b}, overlap: {overlap_b:.1f}%",
            f"  Solution B - predicted Output:\n{format_grid_for_prompt(tr2.get('predicted_output', []), indent=4)}",
            f"  Solution B - difference:\n{format_difference_map(tr2.get('predicted_output', []), tr2.get('expected_output', []), indent=4)}",
            ""
        ])
    training_results_text = "\n".join(training_results_comparison).strip()
    return training_results_text


def generate_fused_transformation_steps(llm,
                                        reasoning_trace: str,
                                        sola: Dict,
                                        solb: Dict,
                                        training_results_a: List[Dict],
                                        training_results_b: List[Dict],
                                        training_examples: List[Dict],
                                        num_solutions: int,
                                        max_retries: int = 3) -> Tuple[List[Dict], int]:
    """Generate candidate fused transformation step sequences from a fused reasoning prompt.

    Returns:
        Tuple of (solution_objects_list, num_retries_used)
        Where solution_objects_list is [{"solution_number": int, "transformation_steps": [str, ...]}, ...]
    """
    def build_fused_transformation_steps_prompt() -> str:
        steps_text_a = build_steps_text_from_transformation_steps(sola.get('step_by_step_transformation') or [])
        code_a = sola.get('main_code') or "(no code)"

        steps_text_b = build_steps_text_from_transformation_steps(solb.get('step_by_step_transformation') or [])
        code_b = solb.get('main_code') or "(no code)"

        training_results_text = result_comparison_text(training_results_a, training_results_b)

        parts = [
            "You are an expert ARC solver. Based on the reasoning that represents fusion of two solutions below, produce multiple candidate fused transformation rules.",
            "Each candidate should be a clear ordered list of transformation steps that can be implemented programmatically.",
            "",
            "",
            "----------",
            "SOLUTION A",
            "----------",
            "",
            "TRANSFORMATION STEPS A:",
            f"{steps_text_a}",
            "",
            "CODE A:",
            f"{code_a}",
            "",
            "----------",
            "SOLUTION B",
            "----------",
            "",
            "TRANSFORMATION STEPS B:",
            f"{steps_text_b}",
            "",
            "CODE B:",
            f"{code_b}",
            "",
            "---------------------",
            "RESULTS COMPARISON",
            "---------------------",
            "",
            f"{training_results_text}",
            "",
            "---------------------",
            "FUSED REASONING TRACE",
            "---------------------",
            "",
            f"{reasoning_trace}",
            "",
            "------------",
            "INSTRUCTIONS",
            "------------",
            f"Produce {num_solutions} different candidate solutions based on the above fused reasoning which attempts to combine the strengths of both solutions.",
            "Try to pick different aspects from each solution to create diverse candidates while still addressing the failures of individual solutions.",
            "Each solution should be concise and concrete so it can be executed programmatically.",
            "",
            "RESPONSE FORMAT (JSON):",
            f"Return a JSON array in a json block containing {num_solutions} solution objects. Each object should have two keys:",
            f"- \"solution_number\": an integer (1..{num_solutions})",
            "- \"transformation_steps\": a JSON array of strings, each string being a single transformation step (in order)",
            "",
            "Example response structure:",
            "```json",
            "[",
            "  {",
            "    \"solution_number\": 1,",
            "    \"transformation_steps\": [\"Step 1 text\", \"Step 2 text\"]",
            "  },",
            "  {",
            "    \"solution_number\": 2,",
            "    \"transformation_steps\": [\"Step 1 text\", \"Step 2 text\"]",
            "  },",
            "  ...",
            "]",
            "```",
            f"Do NOT output any additional text outside the ```json``` block. Generate the {num_solutions} solutions now."
        ]

        for i, ex in enumerate(training_examples, 1):
            parts.append(f"Example {i} Input:\n{format_grid_for_prompt(ex.get('input', []))}")
            parts.append(f"Example {i} Output:\n{format_grid_for_prompt(ex.get('output', []))}")
            parts.append("")

        parts.extend([
            "",
            "INSTRUCTIONS:",
            f"Produce {num_solutions} candidate fused solutions. Return them as a single JSON array inside a ```json``` fenced block. Each object should have keys: 'solution_number' (int) and 'transformation_steps' (array of strings).",
            "Do NOT include extra commentary. Generate the JSON array now."
        ])

        return "\n".join(parts)

    prompt = build_fused_transformation_steps_prompt()
    
    # Retry up to max_retries times if parsing fails
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt, temperature=0.7)
            response_text = response.content if hasattr(response, 'content') else str(response)
            # print_prompt_and_response(prompt, response_text)
            solutions = parse_transformation_steps(response_text)
            
            if solutions:
                return solutions, attempt
                
        except Exception as e:
            print(f"Warning: Failed to generate fused transformation steps (attempt {attempt + 1}/{max_retries}). Error: {e}")
        
        # If this isn't the last attempt, log and retry
        if attempt < max_retries - 1:
            print(f"Warning: Failed to parse fused transformation steps (attempt {attempt + 1}/{max_retries}). Retrying...")
    
    # After all retries failed, return empty list
    return [], max_retries
