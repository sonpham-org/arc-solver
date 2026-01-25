"""
Solution refinement functions for ARC agent.
"""

import copy
import uuid
import base64
import re
from typing import List, Dict, Tuple, Optional

from ..schema import CodeSolution, ExampleResult, ReasoningTraceRecord
from ..augmentation import augment_task_data
from ..debug import print_prompt_and_response
from .reasoning import generate_distilled_reasoning, extract_reasoning_content, _flatten_content
from .code_generation import generate_code_from_reasoning, generate_code_from_reasoning_and_transformations
from .code_execution import test_code_on_examples
from .rag import (
    retrieve_similar_distillations, 
    store_record, 
    generate_embedding_from_distilled_reasoning,
    extract_helpers_from_python_codes
)
from .utilities import (
    format_grid_for_prompt, 
    format_difference_map, 
    parse_transformation_steps, 
    build_steps_text_from_transformation_steps,
    _grid_to_image_bytes
)


def generate_reflection_reasoning_trace(llm,
                                        current_solution: CodeSolution,
                                        training_results: List[ExampleResult],
                                        training_examples: List[Dict],
                                        enable_rag_hint: bool,
                                        num_inreasoning_augmentations: int = 0,
                                        max_retries: int = 3,
                                        memory_context: str = "") -> Tuple[str, int]:
    """Generate a reflection-focused reasoning trace using the ARC-style reflection prompt.

    This is intended for refinement: it asks the model to analyze failures, explain
    what went wrong, and produce a reasoning trace focused on correcting the logic.
    
    Args:
        llm: The language model to use for generation
        current_solution: The current solution being refined
        training_results: Results of testing the current solution on training examples
        training_examples: The original training examples
        enable_rag_hint: Whether to include RAG hints in the prompt
        num_inreasoning_augmentations: Number of augmented examples to generate and test during reflection
        max_retries: Maximum number of retries if extraction fails
        memory_context: Optional context from previous task attempts
    
    Returns:
        Tuple of (reasoning_trace, num_retries_used)
    """
    # Generate in-reasoning augmentations if requested
    augmented_results = []
    augmented_examples = []
    if num_inreasoning_augmentations > 0:
        print(f"🔄 [Reflection] Generating {num_inreasoning_augmentations} in-reasoning augmentations with random seeds...")
        
        # Create augmented data with different random seeds each time
        aug_data = augment_task_data(
            {"train": training_examples},
            num_augmentations=num_inreasoning_augmentations
        )
        
        if aug_data and "train" in aug_data:
            augmented_examples = aug_data["train"]
            print(f"✓ Created {len(augmented_examples)} augmented examples")
            
            # Run current solution code on augmented examples
            try:
                aug_results = test_code_on_examples(
                    current_solution["main_code"],
                    augmented_examples,
                    timeout_seconds=5
                )
                augmented_results = aug_results
                
                # Log augmentation results
                success_count = sum(1 for r in aug_results if r.get('code_success', False))
                print(f"✓ Tested code on augmented examples: {success_count}/{len(aug_results)} passed")
            except Exception as e:
                print(f"⚠️  Error testing code on augmented examples: {e}")
                augmented_results = []
    
    # Combine original and augmented results
    all_training_results = training_results + augmented_results
    all_training_examples = training_examples + augmented_examples
    
    def build_refinement_reasoning_prompt() -> str:
        """Build reflection prompt based on ARC reflection prompt style for deep analysis."""
        
        # Format previous solution
        previous_code = current_solution["main_code"]
        transformation_steps = current_solution["step_by_step_transformation"]
        reasoning_trace = current_solution["reasoning_trace"]

        # Retrieve the relevant concepts based on RAG hints if enabled
        rag_concepts = set()
        rag_hints_parts = []
        if enable_rag_hint:
            vector = current_solution.get('vector')
            entries = retrieve_similar_distillations(vector=vector, top_k=5)

            rag_concepts = set()
            for entry in entries:
                payload = entry.get('payload', {})
                concepts = payload.get('concepts') or []
                if isinstance(concepts, str):
                    concepts = [c.strip() for c in re.split(r'[;,\n]', concepts) if c.strip()]
                elif not isinstance(concepts, (list, tuple)):
                    concepts = []
                for c in concepts:
                    rag_concepts.add(c)
        
        if rag_concepts:
            rag_hints_parts = [
                "---------------------",
                "RELATED CONCEPT HINTS",
                "---------------------",
                "The following concepts were found in similar prior solutions. Feel free to consider them in your analysis:",
                "\n".join(f"- {c}" for c in rag_concepts),
                "",
            ]
        # Format transformation steps
        steps_text = build_steps_text_from_transformation_steps(transformation_steps)
        
        # Build detailed failure analysis (now includes augmented examples)
        failure_analysis = []
        for test in all_training_results:
            example_idx = test.get("example_index", 0)
            if example_idx < len(all_training_examples):
                example = all_training_examples[example_idx]
                
                # Mark if this is an augmented example
                original_count = len(training_examples)
                is_augmented = example_idx >= original_count
                prefix = "[AUGMENTED] " if is_augmented else ""
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
        
        # Add augmentation note if applicable
        aug_note = ""
        if num_inreasoning_augmentations > 0 and augmented_examples:
            aug_note = f"\nNote: {len(augmented_examples)} augmented examples were generated using random transformations (rotations, flips, color permutations) to test your solution's robustness.\n"
        
        # Build training examples block (includes augmented)
        examples_block = ""
        for i, example in enumerate(all_training_examples, 1):
            is_augmented = i > len(training_examples)
            prefix = "[AUGMENTED] " if is_augmented else ""
            examples_block += f"Training Example {i}\\n--\\n"
            examples_block += f"Input:\\n{format_grid_for_prompt(example['input'])}\\n\\n"
            examples_block += f"Output:\\n{format_grid_for_prompt(example['output'])}\\n\\n"
        
        # Add memory context if provided
        memory_context_parts = []
        if memory_context:
            memory_context_parts = [
                "### Previous Task Context ###",
                "The following is context from your previous attempts at this specific task.",
                "Use it to avoid repeating mistakes and build upon previous insights.",
                memory_context,
                "##############################",
                ""
            ]

        prompt_parts = [
            "You are an expert mathematician, logistician and pattern recognizier who is solving the"
            "Abstract Reasoning Corpus (ARC) problems.",
            "You previously attempted to solve this task but your solution was incorrect on some training examples."
            "",
            *memory_context_parts,
            "---------"
            "YOUR GOAL",
            "---------"
            "Analyze your previous attempt deeply, understand why it failed"
            "- Was there an issue with the logic of the code that led to the failure?",
            "- If the code succeeds but the output doesn't match up, what are the difference between the intended output and your predicted output?",
            "- What is missing from your reasoning and solution that leads to these differences?",
            "- How to modify your reasoning and code to correct for these errors and ensure it solves the task fully?",
            aug_note,
            "------------------"
            "YOUR PREVIOUS CODE",
            "------------------",
            f"{previous_code}",
            ""
            "-----------------------"
            "YOUR PREVIOUS REASONING",
            "-----------------------",
            f"{reasoning_trace}",
            "",
            "----------------------------------"
            "YOUR PREVIOUS TRANSFORMATION RULES",
            "----------------------------------",
            f"{steps_text}",
            "",
            "-------------------------",
            "DETAILED FAILURE ANALYSIS",
            "-------------------------",
            f"{failures_block}",
            ""] + rag_hints_parts if rag_hints_parts else [] + [
            "---------------------",
            "ANALYSIS INSTRUCTIONS",
            "---------------------",
            "Provide a ```reasoning``` block that contains your detailed analysis.",
        ]

        prompt = "\n".join(prompt_parts)
        return prompt

    prompt = build_refinement_reasoning_prompt()
    
    # Retry up to max_retries times if extraction fails
    for attempt in range(max_retries):
        response = llm.invoke(prompt)
        if hasattr(response, 'content'):
            resp_content = response.content
        else:
            resp_content = response

        response_text = _flatten_content(resp_content)

        # Prefer structured reflection extraction first
        reasoning = extract_reasoning_content(response_text)
        if reasoning and reasoning != "Unable to generate reasoning trace":
            return reasoning, attempt
        
        # If this isn't the last attempt, log and retry
        if attempt < max_retries - 1:
            print(f"Warning: Failed to extract reflection reasoning content (attempt {attempt + 1}/{max_retries}). Retrying...")
    
    # After all retries failed
    return "Unable to generate reasoning trace", max_retries


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

            print_prompt_and_response(prompt, response_text)

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
                                    enable_rag_hint: bool = False,
                                    num_inreasoning_augmentations: int = 0,
                                    memory_context: str = "") -> Tuple[List[str], str, List[Dict], Optional[ReasoningTraceRecord], int, int]:
    """Refine a CodeSolution using LLM reflection, transformation extraction, and code regeneration.

    Steps:
    1. Identify failed/partial training examples from `current_solution['training_results']`.
    2. Ask the LLM to reflect on differences between expected vs predicted and produce a
       corrected `reasoning_trace` (via `generate_reflection_reasoning_trace`).
    3. Extract concrete `transformation_steps` from the reasoning.
    4. Generate candidate Python implementations from the reasoning + steps.
    5. Pick a candidate, evaluate on training examples, and return an updated CodeSolution
       with updated `main_code`, `reasoning_trace`, `step_by_step_transformation`, and metrics.

    Returns:
        Tuple of (python_codes_list, reasoning_trace, transformation_solutions_list, rag_entry, reasoning_retries, transformation_retries)
    """
    # Defensive copy of solution to avoid mutating the caller's object
    sol = copy.deepcopy(current_solution)

    # Create visual cuesif needed
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

    training_results: List[ExampleResult] = sol.get('training_results', []) or []

    # Step 1: Generate reflection reasoning that focuses on what went wrong
    reasoning_trace, reasoning_retries = generate_reflection_reasoning_trace(
        llm, 
        sol, 
        training_results, 
        training_examples, 
        enable_rag_hint, 
        num_inreasoning_augmentations=num_inreasoning_augmentations,
        memory_context=memory_context
    )
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
