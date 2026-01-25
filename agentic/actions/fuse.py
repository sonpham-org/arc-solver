"""
Solution fusion functions for ARC agent.
"""

import copy
import uuid
import base64
import random
import re
from typing import List, Dict, Tuple, Optional

from ..schema import CodeSolution, ExampleResult, ReasoningTraceRecord
from ..augmentation import augment_task_data
from ..debug import print_prompt_and_response
from .utilities import (
    format_grid_for_prompt, 
    format_difference_map, 
    parse_transformation_steps, 
    build_steps_text_from_transformation_steps,
    _grid_to_image_bytes
)
from .reasoning import generate_distilled_reasoning, extract_reasoning_content
from .transformation import generate_transformation_steps
from .code_generation import generate_code_from_reasoning, generate_code_from_reasoning_and_transformations
from .code_execution import test_and_fix_code_from_trial_run, test_code_on_examples
from .rag import (
    retrieve_similar_distillations, 
    store_record, 
    generate_embedding_from_distilled_reasoning,
    extract_helpers_from_python_codes
)


def generate_fused_reasoning_trace(llm,
                                   sola: Dict,
                                   solb: Dict,
                                   training_results1: List[ExampleResult],
                                   training_results2: List[ExampleResult],
                                   training_examples: List[Dict],
                                   enable_rag_hint: bool,
                                   num_inreasoning_augmentations: int = 0,
                                   max_retries: int = 3,
                                   memory_context: str = "") -> Tuple[str, int]:
    """Generate a fused reasoning trace that reconciles two candidate solutions.

    The prompt includes both solutions' reasoning, transformation steps, code (if available),
    and the training results. The LLM is asked to produce a single, coherent reasoning
    trace that explains how to combine their strengths and address their failure modes.
    
    Args:
        llm: The language model to use for generation
        sola: First candidate solution
        solb: Second candidate solution
        training_results1: Results for solution A on training examples
        training_results2: Results for solution B on training examples
        training_examples: The original training examples
        enable_rag_hint: Whether to include RAG hints in the prompt
        num_inreasoning_augmentations: Number of augmented examples to generate and test during fusion
        max_retries: Maximum number of retries if extraction fails
        memory_context: Optional context from previous task attempts
    
    Returns:
        Tuple of (reasoning_trace, num_retries_used)
    """
    # Generate in-reasoning augmentations if requested
    augmented_results1 = []
    augmented_results2 = []
    augmented_examples = []
    if num_inreasoning_augmentations > 0:
        print(f"🔄 [Fusion] Generating {num_inreasoning_augmentations} in-reasoning augmentations with random seeds...")
        
        # Create augmented data with different random seeds each time
        aug_data = augment_task_data(
            {"train": training_examples},
            num_augmentations=num_inreasoning_augmentations
        )
        
        if aug_data and "train" in aug_data:
            augmented_examples = aug_data["train"]
            print(f"✓ Created {len(augmented_examples)} augmented examples")
            
            # Run both solutions on augmented examples
            # Test solution A
            try:
                aug_results1 = test_code_on_examples(
                    sola.get("main_code", ""),
                    augmented_examples,
                    timeout_seconds=5
                )
                augmented_results1 = aug_results1
                success_count = sum(1 for r in aug_results1 if r.get('code_success', False))
                print(f"✓ Solution A on augmented: {success_count}/{len(aug_results1)} passed")
            except Exception as e:
                print(f"⚠️  Error testing Solution A on augmented examples: {e}")
                augmented_results1 = []
            
            # Test solution B
            try:
                aug_results2 = test_code_on_examples(
                    solb.get("main_code", ""),
                    augmented_examples,
                    timeout_seconds=5
                )
                augmented_results2 = aug_results2
                success_count = sum(1 for r in aug_results2 if r.get('code_success', False))
                print(f"✓ Solution B on augmented: {success_count}/{len(aug_results2)} passed")
            except Exception as e:
                print(f"⚠️  Error testing Solution B on augmented examples: {e}")
                augmented_results2 = []
    
    # Combine original and augmented results
    all_training_results1 = training_results1 + augmented_results1
    all_training_results2 = training_results2 + augmented_results2
    all_training_examples = training_examples + augmented_examples
    
    def build_fused_reasoning_trace_prompt():
        reasoning_a = sola.get('reasoning_trace') or "(no reasoning)"
        steps_text_a = build_steps_text_from_transformation_steps(sola.get('step_by_step_transformation') or [])
        code_a = sola.get('main_code') or "(no code)"

        reasoning_b = solb.get('reasoning_trace') or "(no reasoning)"
        steps_text_b = build_steps_text_from_transformation_steps(solb.get('step_by_step_transformation') or [])
        code_b = solb.get('main_code') or "(no code)"

        # Use augmented results in comparison
        training_results_text = result_comparison_text(all_training_results1, all_training_results2)
        
        # Add augmentation note if applicable
        aug_note = ""
        if num_inreasoning_augmentations > 0 and augmented_examples:
            aug_note = f"\nNote: {len(augmented_examples)} augmented examples (using rotations, flips, color permutations) were generated to test both solutions' robustness.\n"

        rag_concepts = set()
        rag_hints_parts = []

        if enable_rag_hint:
            vectora = sola.get('vector')
            vectorb = solb.get('vector')
            entries_a = retrieve_similar_distillations(vector=vectora, top_k=5)
            if entries_a:
                print("✓ Retrieved RAG entries for Solution A")
            entries_b = retrieve_similar_distillations(vector=vectorb, top_k=5)
            if entries_b:
                print("✓ Retrieved RAG entries for Solution B")

            rag_concepts = set()
            for entry in entries_a + entries_b:
                payload = entry.get('payload', {})
                concepts = payload.get('concepts') or []
                if isinstance(concepts, str):
                    concepts = [c.strip() for c in re.split(r'[;,\n]', concepts) if c.strip()]
                elif not isinstance(concepts, (list, tuple)):
                    concepts = []
                for c in concepts:
                    rag_concepts.add(c)
            if rag_concepts:
                print(f"✓ Found {len(rag_concepts)} RAG concepts for fused reasoning prompt")
        
        if rag_concepts:
            rag_hints_parts = [
                "---------------------",
                "RELATED CONCEPT HINTS",
                "---------------------",
                "The following concepts were found in similar prior solutions. Feel free to consider them in your analysis:",
                "\n".join(f"- {c}" for c in rag_concepts),
                ""
            ]

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

        parts = [
            "You are an expert ARC solver. Two candidate solutions were produced for the same task.",
            "Your job is to reconcile them into a single solution that combines their strengths and remedies their weaknesses.",
            aug_note,
            *memory_context_parts,
            "----------",
            "SOLUTION A",
            "----------",
            "",
            "REASONING TRACE A:",
            f"{reasoning_a}",
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
            "REASONING TRACE B:",
            f"{reasoning_b}",
            "",
            "TRANSFORMATION STEPS B:",
            f"{steps_text_b}",
            "",
            "CODE B:",
            f"{code_b}",
            "",
            "---------------------------",
            "TRAINING RESULTS COMPARISON",
            "---------------------------",
            "For each training example, show the performance of each solution in terms of size match and overlap percentage.",
            "",
            f"{training_results_text}",
            ""] + rag_hints_parts + [
            "---------------------",
            "ANALYSIS INSTRUCTIONS",
            "---------------------",
            "Produce a single ```reasoning``` block that: "
            "- Explains how the two solutions related to the final solution",
            "- The strengths and weaknesses of each solution, and how they complement each other",
            "- Proposes a fused general rule that combines the two"
        ]
        return "\n".join(parts)
    
    prompt = build_fused_reasoning_trace_prompt()
    # If visual cues are provided and the llm driver supports image messages,
    # send a structured message containing the images (base64 data URLs).
    
    # Retry up to max_retries times if extraction fails
    for attempt in range(max_retries):
        response = llm.invoke(prompt)

        # Extract reasoning from response
        response_text = response.content if hasattr(response, 'content') else str(response)
        print_prompt_and_response(prompt, response_text)

        reasoning = extract_reasoning_content(response_text)
        if reasoning and reasoning != "Unable to generate reasoning trace":
            return reasoning, attempt
        
        # If this isn't the last attempt, log and retry
        if attempt < max_retries - 1:
            print(f"Warning: Failed to extract fused reasoning content (attempt {attempt + 1}/{max_retries}). Retrying...")
    
    # After all retries failed
    return "Unable to generate reasoning trace", max_retries


def fuse_solutions_with_reasoning(llm,
                                  transformation_llm,
                                  code_llm,
                                  sola: CodeSolution,
                                  solb: CodeSolution,
                                  training_examples: List[Dict],
                                  num_fused_solutions: int,
                                  enable_visual_cue: bool = False,
                                  enable_rag_hint: bool = False,
                                  num_inreasoning_augmentations: int = 0,
                                  memory_context: str = "") -> Tuple[List[str], str, List[Dict], Optional[ReasoningTraceRecord], int, int]:
    """Attempt to fuse two CodeSolution candidates into a stronger combined solution.

    Returns:
        Tuple of (python_codes_list, fused_reasoning_trace, fused_transformation_solutions_list, rag_entry, reasoning_retries, transformation_retries)
    """
    # Build visual cues if requested
    if enable_visual_cue:
        # TODO: Implement visual cue generation for fused solutions if needed
        pass

    # Merge training_results from both solutions (concatenate, allowing duplicates)
    tra = sola.get('training_results') or []
    trb = solb.get('training_results') or []

    # 1) Generate fused reasoning trace
    fused_reasoning, reasoning_retries = generate_fused_reasoning_trace(
        llm, 
        sola, 
        solb, 
        tra, 
        trb, 
        training_examples, 
        enable_rag_hint, 
        num_inreasoning_augmentations,
        memory_context=memory_context
    )

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
            print_prompt_and_response(prompt, response_text)
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
