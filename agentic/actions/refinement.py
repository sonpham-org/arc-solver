"""
Solution refinement functions for ARC agent.
"""

import copy
import uuid
from typing import List, Dict, Tuple

from ..schema import CodeSolution, ExampleResult, ReasoningTraceRecord
from .reasoning import generate_reflection_reasoning_trace, generate_distilled_reasoning
from .code_generation import generate_code_from_reasoning, generate_code_from_reasoning_and_transformations
from .rag import generate_embedding_from_distilled_reasoning, extract_helpers_from_python_codes, store_record
from .utilities import format_grid_for_prompt, format_difference_map, parse_transformation_steps


def generate_refined_transformation_steps(llm, reasoning_trace: str, sol: Dict, training_examples: List[Dict], num_solutions: int, max_retries: int = 3) -> Tuple[List[Dict], int]:
    """Extract refined step-by-step transformation from reasoning trace and previous solution failures.

    Returns:
        Tuple of (solution_objects_list, num_retries_used)
        Where solution_objects_list is [{"solution_number": int, "transformation_steps": [str, ...]}, ...]
    """
    def build_refined_transformation_steps_prompt() -> str:
        """Build a prompt that includes reasoning, the previous solution (and its failures), and training examples."""
        # Get relevant information out
        training_results = sol["training_results"]

        # Build detailed failure analysis
        failure_analysis = []
        for test in training_results:
            example_idx = test.get("example_index", 0)
            if example_idx < len(training_examples):
                example = training_examples[example_idx]
                
                analysis = f"Training Example {example_idx + 1} - FAILED\\n"
                analysis += "--\\n"
                analysis += f"Input:\\n{format_grid_for_prompt(example['input'])}\\n\\n"
                analysis += f"Expected Output:\\n{format_grid_for_prompt(example['output'])}\\n\\n"
                
                predicted = test.get("predicted_output")
                if predicted:
                    analysis += f"Your Predicted Output:\\n{format_grid_for_prompt(predicted)}\\n\\n"
                    # Calculate sizes
                    pred_h, pred_w = len(predicted), len(predicted[0]) if predicted else 0
                    exp_h, exp_w = len(example['output']), len(example['output'][0]) if example['output'] else 0
                    analysis += f"Expected size: {exp_h}x{exp_w}, Predicted size: {pred_h}x{pred_w}\n"
                    # Add a visual difference map ('.' match, 'X' mismatch; non-overlap = X)
                    try:
                        diff_map = format_difference_map(predicted, example['output'])
                        analysis += f"Difference:\n{diff_map}\n\n"
                    except Exception:
                        analysis += "Difference: (could not compute difference map)\n\n"
                else:
                    analysis += "Your Predicted Output: No output generated\\n\\n"
                    exp_h, exp_w = len(example['output']), len(example['output'][0]) if example['output'] else 0
                    analysis += f"Expected size: {exp_h}x{exp_w}, Predicted size: 0x0\\n"
                    # When there's no predicted output, mark the entire expected area as mismatches
                    try:
                        diff_map = format_difference_map(None, example['output'])
                        analysis += f"Difference:\n{diff_map}\n\n"
                    except Exception:
                        analysis += "Difference: (could not compute difference map)\n\n"
                
                analysis += f"Overlap: {test.get('overlap_percentage', 0):.1f}%\\n"
                analysis += f"IOU (Intersection over Union): {test.get('iou_percentage', 0):.1f}%\\n"
                
                error_msg = test.get("error_message")
                if error_msg:
                    analysis += f"Error: {error_msg}\\n"
                
                failure_analysis.append(analysis)
            
        failures_block = "\\n".join(failure_analysis) 

        prompt_parts = [
            "You are an expert mathematician, logistician and pattern recognizier who is solving the Abstract Reasoning Corpus (ARC) problems.",
            "You previously attempted a solution which failed to solve the task. You are asked to use the failure information to generate refined candidate transformation steps.",
            "",
            "PREVIOUS SOLUTION & FAILURES:",
            f"{failures_block}",
            ""
            "REFLECTION ON PAST FAILURE:",
            f"{reasoning_trace}",
            "",
            "INSTRUCTIONS:",
            f"Produce {num_solutions} different candidate solutions. Each solution should be a numbered sequence of clear, actionable transformation steps.",
            "Give special attention to correcting the failure modes shown in the previous solution summary.",
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

        prompt = "\n".join(prompt_parts)
        return prompt

    prompt = build_refined_transformation_steps_prompt()
    
    # Retry up to max_retries times if parsing fails
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt, temperature=0.7)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # print_prompt_and_response(prompt, response_text)

            solutions = parse_transformation_steps(response_text)
            if solutions:
                return solutions, attempt
            
            # If parsing failed and this isn't the last attempt, log and retry
            if attempt < max_retries - 1:
                print(f"Warning: Failed to parse refined transformation steps (attempt {attempt + 1}/{max_retries}). Retrying...")
                
        except Exception as e:
            print(f"Error extracting refined transformation steps (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
    
    # After all retries failed, return empty list
    return [], max_retries


def refine_solutions_with_reasoning(llm,
                                    transformation_llm,
                                    code_llm,
                                    current_solution: CodeSolution,
                                    training_examples: List[Dict],
                                    num_refined_solutions: int,
                                    enable_visual_cue: bool = False,
                                    enable_rag_hint: bool = False) -> Tuple[List[str], str, List[Dict]]:
    """Refine a CodeSolution using LLM reflection, transformation extraction, and code regeneration.

    Steps:
    1. Identify failed/partial training examples from `current_solution['training_results']`.
    2. Ask the LLM to reflect on differences between expected vs predicted and produce a
       corrected `reasoning_trace` (via `generate_reflection_reasoning_trace`).
    3. Extract concrete `transformation_steps` from the reasoning.
    4. Generate candidate Python implementations from the reasoning + steps.
    5. Pick a candidate, evaluate on training examples, and return an updated CodeSolution
       with updated `main_code`, `reasoning_trace`, `step_by_step_transformation`, and metrics.

    Returns an updated CodeSolution dict (may be same as input if refinement fails).
    """
    from .utilities import _grid_to_image_bytes
    
    # Defensive copy of solution to avoid mutating the caller's object
    sol = copy.deepcopy(current_solution)

    # Create visual cuesif needed
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

    training_results: List[ExampleResult] = sol.get('training_results', []) or []

    # Step 1: Generate reflection reasoning that focuses on what went wrong
    reasoning_trace, reasoning_retries = generate_reflection_reasoning_trace(llm, sol, training_results, training_examples, enable_rag_hint)
    sol['reasoning_trace'] = reasoning_trace

    # Step 2: Extract transformation steps from the reflection reasoning
    transformation_solutions_list, transformation_retries = generate_refined_transformation_steps(transformation_llm, reasoning_trace, sol, training_examples, num_refined_solutions)
    
    # Step 3: Generate candidate code implementations
    if not transformation_solutions_list:
        python_codes_list, _ = generate_code_from_reasoning(code_llm, reasoning_trace, training_examples, num_refined_solutions)
    else:
        python_codes_list = generate_code_from_reasoning_and_transformations(code_llm, reasoning_trace, transformation_solutions_list, training_examples)
    
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
            if stored:
                print(f"✓ Stored refined RAG entry (concepts: {len(rag_entry.concepts)}, helpers: {len(rag_entry.helpers)})")
        except Exception as e:
            print(f"Warning: store_record raised an exception: {e}")
    else:
        rag_entry = None

    # Attach visual cue data onto each transformation dict (best-effort)
    if enable_visual_cue:
        # TODO: Think about a way to add the visual cues here
        pass

    # Return the list of candidate codes, plus reasoning and steps.
    return python_codes_list, reasoning_trace, transformation_solutions_list, rag_entry, reasoning_retries, transformation_retries


def analyze_failures(failed_tests: List[ExampleResult], training_examples: List[Dict]) -> Dict[str, any]:
    """Analyze the pattern of failures to understand what went wrong."""
    analysis = {
        "num_failures": len(failed_tests),
        "error_types": [],
        "size_mismatches": [],
        "color_issues": []
    }
    
    for test in failed_tests:
        if test["error_message"]:
            analysis["error_types"].append(test["error_message"])
        
        if test["predicted_output"] and test["expected_output"]:
            pred_shape = (len(test["predicted_output"]), len(test["predicted_output"][0]) if test["predicted_output"] else 0)
            exp_shape = (len(test["expected_output"]), len(test["expected_output"][0]) if test["expected_output"] else 0)
            
            if pred_shape != exp_shape:
                analysis["size_mismatches"].append({
                    "predicted": pred_shape,
                    "expected": exp_shape
                })
    
    return analysis
