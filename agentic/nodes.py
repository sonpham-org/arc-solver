"""
Nodes for the ARC LangGraph Agent workflow.

This module should contain only the node functions. All helper and
action-related utilities have been moved to `actions.py`.
"""

import copy
from typing import List, Dict, Optional, Any
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import re


# Import schema and node-facing types
from .debug import print_python_code
from .schema import AgentState, CodeSolution, ExampleResult, Grid
from langchain_core.messages import HumanMessage, AIMessage


# Import actions (helpers, prompts, execution, refinement)
from .actions import (
    fuse_solutions_with_reasoning,
    create_solutions_with_reasoning,
    refine_solutions_with_reasoning,
    execute_transformation_code,
    generate_llm_predicted_output,
    calculate_grid_results,
    evaluate_example
)
from .augmentation import augment_task_data


def _grids_same_shape(a: Optional[Grid], b: Optional[Grid]) -> bool:
    if a is None or b is None:
        return False
    if not a or not b:
        return False
    if len(a) != len(b):
        return False
    return all(len(row_a) == len(row_b) for row_a, row_b in zip(a, b))


def _overlap_percentage(a: Optional[Grid], b: Optional[Grid]) -> float:
    """Return percentage overlap (0-100) between two grids; 0.0 if shapes differ or missing."""
    if not _grids_same_shape(a, b):
        return 0.0
    total = 0
    equal = 0
    for ra, rb in zip(a, b):
        for va, vb in zip(ra, rb):
            total += 1
            if va == vb:
                equal += 1
    return (equal / total) * 100.0 if total > 0 else 0.0


def _exampleresult_from_dict(d: Dict[str, Any]) -> ExampleResult:
    """Normalize a result-dict (from `actions.evaluate_example`) into ExampleResult TypedDict."""
    return ExampleResult(
        example_index=int(d.get("example_index", -1)),
        input=d.get("input"),
        expected_output=d.get("expected_output"),
        predicted_output=d.get("predicted_output"),
        matching_size=bool(d.get("matching_size", False)),
        overlap_percentage=float(d.get("overlap_percentage", 0.0)),
        error_message=d.get("error_message"),
        code_success=bool(d.get("code_success", False)),
        llm_predicted_output=d.get("llm_predicted_output"),
        llm_matching_size=d.get("llm_matching_size"),
        llm_overlap_percentage=d.get("llm_overlap_percentage"),
        llm_error_message=d.get("llm_error_message"),
        llm_success=bool(d.get("llm_success", False)),
    )


def _calculate_stats_from_results(results: List[ExampleResult]) -> Dict[str, float]:
    n = len(results)
    if n == 0:
        return {
            "success_rate": 0.0,
            "overlap_average": 0.0,
            "error_rate": 0.0,
            "llm_success_rate": 0.0,
            "llm_overlap_average": 0.0,
        }

    code_success_count = sum(1 for r in results if r.get("code_success"))
    llm_success_count = sum(1 for r in results if r.get("llm_success"))

    # Overlap averages (results from actions are percentages 0-100)
    code_overlaps = [float(r.get("overlap_percentage", 0.0)) for r in results if r.get("predicted_output") is not None]
    llm_overlaps = [float(r.get("llm_overlap_percentage", 0.0)) for r in results if r.get("llm_predicted_output") is not None]

    overlap_avg = float(sum(code_overlaps) / len(code_overlaps)) if code_overlaps else 0.0
    llm_overlap_avg = float(sum(llm_overlaps) / len(llm_overlaps)) if llm_overlaps else 0.0

    success_rate = float(code_success_count) / float(n)
    llm_success_rate = float(llm_success_count) / float(n)
    error_rate = 1.0 - success_rate

    return {
        "success_rate": success_rate,
        "overlap_average": overlap_avg,
        "error_rate": error_rate,
        "llm_success_rate": llm_success_rate,
        "llm_overlap_average": llm_overlap_avg,
    }


def calculate_solution_statistics(solution: CodeSolution, num_original_train: Optional[int] = None) -> CodeSolution:
    """Populate statistic fields for a `CodeSolution` (mutates in-place).

    Uses the typed `training_results`/`testing_results` lists to compute
    success rates, overlap averages and error rates for code and LLM predictions.

    If `num_original_train` is provided, it splits the `training_results` list
    into original training results and augmented training results.
    """
    total_train_results: List[ExampleResult] = solution.get("training_results", []) or []
    test_results: List[ExampleResult] = solution.get("testing_results", []) or []

    if num_original_train is not None and len(total_train_results) > num_original_train:
        train_results = total_train_results[:num_original_train]
        augment_results = total_train_results[num_original_train:]
        
        # Calculate stats for original training
        train_stats = _calculate_stats_from_results(train_results)
        
        # Calculate stats for augmented training
        augment_stats = _calculate_stats_from_results(augment_results)
        
        solution["augment_training_results"] = augment_results
        solution["augment_training_success_rate"] = augment_stats["success_rate"]
        solution["augment_training_overlap_average"] = augment_stats["overlap_average"]
        solution["augment_training_error_rate"] = augment_stats["error_rate"]
    else:
        train_results = total_train_results
        train_stats = _calculate_stats_from_results(train_results)
        
        solution["augment_training_results"] = []
        solution["augment_training_success_rate"] = 0.0
        solution["augment_training_overlap_average"] = 0.0
        solution["augment_training_error_rate"] = 0.0

    test_stats = _calculate_stats_from_results(test_results)

    solution["training_results"] = train_results # Update to only include original
    solution["training_success_rate"] = train_stats["success_rate"]
    solution["training_overlap_average"] = train_stats["overlap_average"]
    solution["training_error_rate"] = train_stats["error_rate"]
    solution["llm_training_success_rate"] = train_stats["llm_success_rate"]
    solution["llm_training_overlap_average"] = train_stats["llm_overlap_average"]

    solution["testing_success_rate"] = test_stats["success_rate"]
    solution["testing_overlap_average"] = test_stats["overlap_average"]
    solution["testing_error_rate"] = test_stats["error_rate"]
    solution["llm_testing_success_rate"] = test_stats["llm_success_rate"]
    solution["llm_testing_overlap_average"] = test_stats["llm_overlap_average"]
    return solution


def generate_code_node(state: AgentState, llm, transformation_llm, code_llm) -> AgentState:
    """
    Generate Python code to solve the ARC problem using reasoning-first approach.

    This node uses the available helper functions and analyzes the training
    examples to generate a solution using the provided language model.
    """
    tid = state.get('task_id', 'unknown')
    print(f"Task {tid} [generate_code_node]: Generating initial code solutions...")

    # Deep copy the state to avoid mutating caller's object
    new_state = copy.deepcopy(state)
    task_data = state["task_data"]
    num_initial_solutions = state["num_initial_solutions"]

    # Handle preloop randomization: if in preloop mode, generate fresh augmentations
    num_augmentations = state.get("num_augmentations", 0)
    mode = state.get("augmentation_randomization_mode", "pretask")
    if num_augmentations > 0 and mode == "preloop":
        print(f"Task {tid} [generate_code_node]: Dynamic augmentation (preloop). Generating {num_augmentations} fresh examples...")
        new_state['augment_data'] = augment_task_data(task_data, num_augmentations)

    # Analyze the training examples (kept for node-level logging/analysis)
    training_examples = task_data["train"]
    
    # Add augmented data if available (from new_state in case of preloop update)
    augment_data = new_state.get('augment_data')
    if augment_data and 'train' in augment_data:
        training_examples = training_examples + augment_data['train']
        print(f"Task {tid} [generate_code_node]: Using {len(task_data['train'])} original + {len(augment_data['train'])} augmented = {len(training_examples)} total training examples")

    # Generate code using the reasoning-first approach from actions
    # Read visual cue flag from node state and pass through to generation
    enable_visual_cue = state.get('enable_visual_cue', False)
    num_inreasoning_augmentations = state.get("num_inreasoning_augmentations", 0)
    
    # NEW: Handle LLM Memory
    memory_type = state.get("llm_memory_type", "none")
    memory_context = ""
    
    if memory_type == "message_history":
        messages = state.get("messages", [])
        if messages:
            memory_context = "\n\n### Previous Task Context ###\n"
            for m in messages:
                role = "Assistant" if isinstance(m, AIMessage) else "User"
                memory_context += f"[{role}]: {m.content}\n"
    elif memory_type == "summary":
        summary = state.get("task_memory_summary", "")
        if summary:
            memory_context = f"\n\n### Task Learning Summary ###\n{summary}\n"

    python_codes, reasoning_trace, transformation_solutions_list, rag_entry, reasoning_retries, transformation_retries = create_solutions_with_reasoning(
        llm,
        transformation_llm,
        code_llm,
        training_examples,
        num_solutions=num_initial_solutions,
        enable_visual_cue=enable_visual_cue,
        num_inreasoning_augmentations=num_inreasoning_augmentations,
        memory_context=memory_context  # Pass memory context
    )

    # After generation, Update memory if using message_history
    if memory_type == "message_history":
        # We add the first solution's reasoning as representative history
        new_state["messages"] = [
            HumanMessage(content=f"Attempt {state.get('current_loop', 0)}: solve ARC task"),
            AIMessage(content=reasoning_trace)
        ]

    solutions_list = []
    for transformation, code in zip(transformation_solutions_list, python_codes):
        solution = {
            "main_code": code,
            "reasoning_trace": reasoning_trace,
            "reasoning_summary": rag_entry.reasoning_summary if rag_entry else "",
            "concepts": rag_entry.concepts if rag_entry else [],
            "vector": rag_entry.vector if rag_entry else [],
            "step_by_step_transformation": transformation,
        }
        solutions_list.append(solution)

    # Update state
    new_state["seed_solutions_list"] = solutions_list
    new_state["fused_solutions_list"] = []
    new_state["mutated_solutions_list"] = []
    new_state["num_retries"] += reasoning_retries + transformation_retries
    return new_state


def evolve_code_node(state, llm, transformation_llm, code_llm):
    """Module-level evolve node: increment generation and re-run generator."""

    tid = state.get('task_id', 'unknown')
    print(f"Task {tid} [evolve_code_node]: Evolving code solutions for generation {state.get('current_generation')}...")

    # Deep copy the state to avoid mutating caller's object
    new_state = copy.deepcopy(state)
    task_data = state["task_data"]
    
    # Handle preloop randomization: if in preloop mode, generate fresh augmentations
    num_augmentations = state.get("num_augmentations", 0)
    mode = state.get("augmentation_randomization_mode", "pretask")
    if num_augmentations > 0 and mode == "preloop":
        print(f"Task {tid} [evolve_code_node]: Dynamic augmentation (preloop). Generating {num_augmentations} fresh examples...")
        new_state['augment_data'] = augment_task_data(task_data, num_augmentations)

    # Get necessary variables
    training_examples = task_data["train"]
    
    # Add augmented data if available (from new_state in case of preloop update)
    augment_data = new_state.get('augment_data')
    if augment_data and 'train' in augment_data:
        training_examples = training_examples + augment_data['train']
        print(f"Task {tid} [evolve_code_node]: Using {len(task_data['train'])} original + {len(augment_data['train'])} augmented = {len(training_examples)} total training examples")
    
    enable_visual_cue = state.get("enable_visual_cue", False)
    enable_rag_hint = state.get("enable_rag_hint", False)
    num_seed_solutions = state["num_seed_solutions"]
    num_refinements = state["num_refinements"]
    num_solutions_per_refinement = state["num_solutions_per_refinement"]
    num_fusions = state["num_fusions"]
    num_solutions_per_fusion = state["num_solutions_per_fusion"]
    num_inreasoning_augmentations = state["num_inreasoning_augmentations"]

    # Make a deep copy of the incoming state to avoid mutating caller's object
    new_state["current_loop"] = state.get("current_loop") + 1

    # 1) Extract seed solutions from the current solutions_list
    seed_solutions: List[Dict[str, Any]] = copy.deepcopy(state.get("solutions_list") or [])

    # 2) Archive current generation into `generations`
    current_generation = state.get("current_generation")
    generation_entry = {
        "generation": current_generation,
        "solutions_list": copy.deepcopy(state.get("solutions_list") or []),
        "average_training_success_rate": sum(sol.get("training_success_rate", 0.0) for sol in state.get("solutions_list") or []) / max(len(state.get("solutions_list") or []), 1),
        "average_training_overlap_score": sum(sol.get("training_overlap_average", 0.0) for sol in state.get("solutions_list") or []) / max(len(state.get("solutions_list") or []), 1),
        "average_training_error_rate": sum(sol.get("training_error_rate", 0.0) for sol in state.get("solutions_list") or []) / max(len(state.get("solutions_list") or []), 1),
        "max_training_success_rate": max((sol.get("training_success_rate", 0.0) for sol in state.get("solutions_list") or []), default=0.0),
        "max_training_overlap_score": max((sol.get("training_overlap_average", 0.0) for sol in state.get("solutions_list") or []), default=0.0),
        "max_training_error_rate": max((sol.get("training_error_rate", 0.0) for sol in state.get("solutions_list") or []), default=0.0),

        "average_testing_success_rate": sum(sol.get("testing_success_rate", 0.0) for sol in state.get("solutions_list") or []) / max(len(state.get("solutions_list") or []), 1),
        "average_testing_overlap_score": sum(sol.get("testing_overlap_average", 0.0) for sol in state.get("solutions_list") or []) / max(len(state.get("solutions_list") or []), 1),
        "average_testing_error_rate": sum(sol.get("testing_error_rate", 0.0) for sol in state.get("solutions_list") or []) / max(len(state.get("solutions_list") or []), 1),
        "max_testing_success_rate": max((sol.get("testing_success_rate", 0.0) for sol in state.get("solutions_list") or []), default=0.0),
        "max_testing_overlap_score": max((sol.get("testing_overlap_average", 0.0) for sol in state.get("solutions_list") or []), default=0.0),
        "max_testing_error_rate": max((sol.get("testing_error_rate", 0.0) for sol in state.get("solutions_list") or []), default=0.0),
    }
    new_state["generations"].append(generation_entry)

    # increment generation counter and clear solutions_list for next generation
    new_state["current_generation"] = current_generation + 1
    new_state["solutions_list"] = []

    # 3) Sort seed solutions by training success rate then training overlap average (descending)
    # Use safe defaults if fields are missing
    def _sort_key(sol: Dict[str, Any]):
        weight = state.get('augmented_example_weight', 0.5)
        s_rate = sol.get("training_success_rate", 0.0)
        a_rate = sol.get("augment_training_success_rate", 0.0)
        s_overlap = sol.get("training_overlap_average", 0.0)
        a_overlap = sol.get("augment_training_overlap_average", 0.0)
        
        has_augment = sol.get("augment_training_results") is not None and len(sol.get("augment_training_results")) > 0
        if has_augment:
            # Combined weighted score
            weighted_success = (s_rate + a_rate * weight) / (1.0 + weight)
            weighted_overlap = (s_overlap + a_overlap * weight) / (1.0 + weight)
        else:
            weighted_success = s_rate
            weighted_overlap = s_overlap
            
        return (weighted_success, weighted_overlap)
    # Sort by score (descending). If scores are equal, use a random
    # tie-breaker so equal-scoring solutions are ordered randomly.
    seed_solutions.sort(key=lambda s: (_sort_key(s), random.random()), reverse=True)

    # Create fusion and mutation from seed solutions
    seed_solutions = seed_solutions[:num_seed_solutions]
    fused_solutions: List[Dict[str, Any]] = []
    mutated_solutions: List[Dict[str, Any]] = []

    # NEW: Memory Update for 'summary' mode if we just failures
    memory_type = state.get("llm_memory_type", "none")
    if memory_type == "summary":
        # Extract lessons from previous generation failures
        prev_sols = state.get("solutions_list", [])
        if prev_sols:
            best_prev = max(prev_sols, key=_sort_key)
            if best_prev.get("training_success_rate", 0.0) < 1.0:
                summary_prompt = f"""Review the previous failed solution and summarize what was learned.
                Reasoning: {best_prev.get('reasoning_trace', '')}
                Success Rate: {best_prev.get('training_success_rate', 0.0):.1%}
                
                Keep the summary extremely concise (max 3 bullet points) focusing on what failed and what to try next."""
                try:
                    summary_resp = llm.invoke(summary_prompt)
                    new_summary = f"{state.get('task_memory_summary', '')}\nLoop {state.get('current_loop')}: {summary_resp.content}"
                    new_state["task_memory_summary"] = new_summary
                except Exception:
                    pass

    if seed_solutions:
        # Prepare memory context for fusion/refinement
        memory_context = ""
        if memory_type == "message_history":
            messages = state.get("messages", [])
            for m in messages:
                role = "Assistant" if isinstance(m, AIMessage) else "User"
                memory_context += f"[{role}]: {m.content}\n"
        elif memory_type == "summary":
            memory_context = state.get("task_memory_summary", "")

        # NEW: Update message history with best previous results for next step
        if memory_type == "message_history" and state.get("solutions_list"):
            best_prev = max(state["solutions_list"], key=_sort_key)
            new_state["messages"] = [
                HumanMessage(content=f"Loop {state.get('current_loop', 0)} best result: {best_prev.get('training_success_rate', 0.0):.1%} success"),
                AIMessage(content=best_prev.get('reasoning_trace', ''))
            ]

        for _ in range(num_fusions):
            sol_arr = []

            # Randomly select two distinct partner solutions. Sample without replacement
            sola, solb = random.sample(seed_solutions, 2)
            python_codes, reasoning_trace, transformation_solutions_list, rag_entry, reasoning_retries, transformation_retries = fuse_solutions_with_reasoning(
                llm,
                transformation_llm,
                code_llm,
                sola,
                solb,
                training_examples,
                num_fused_solutions=num_solutions_per_fusion,
                enable_visual_cue=enable_visual_cue,
                enable_rag_hint=enable_rag_hint,
                num_inreasoning_augmentations=num_inreasoning_augmentations,
                memory_context=memory_context) # Pass memory context

            for transformation, code in zip(transformation_solutions_list, python_codes):
                solution = {
                    "main_code": code,
                    "reasoning_trace": reasoning_trace,
                    "step_by_step_transformation": transformation,
                    "reasoning_summary": rag_entry.reasoning_summary if rag_entry else "",
                    "concepts": rag_entry.concepts if rag_entry else [],
                    "vector": rag_entry.vector if rag_entry else [],
                }
                sol_arr.append(solution)
            fused_solutions.append(sol_arr)
            new_state["num_retries"] += reasoning_retries + transformation_retries

        # 6) Mutation (self-reflection)
        for i, solution in enumerate(seed_solutions[:num_refinements]):
            sol_arr = []
            python_codes, reasoning_trace, transformation_solutions_list, rag_entry, reasoning_retries, transformation_retries = refine_solutions_with_reasoning(
                llm,
                transformation_llm,
                code_llm,
                solution,
                training_examples,
                num_solutions_per_refinement,
                enable_visual_cue=enable_visual_cue,
                enable_rag_hint=enable_rag_hint,
                num_inreasoning_augmentations=num_inreasoning_augmentations,
                memory_context=memory_context, # Pass memory context
            )

            for transformation, code in zip(transformation_solutions_list, python_codes):
                solution = {
                    "main_code": code,
                    "reasoning_trace": reasoning_trace,
                    "step_by_step_transformation": transformation,
                    "reasoning_summary": rag_entry.reasoning_summary if rag_entry else "",
                    "concepts": rag_entry.concepts if rag_entry else [],
                    "vector": rag_entry.vector if rag_entry else [],
                }
                sol_arr.append(solution)
            mutated_solutions.append(sol_arr)
            new_state["num_retries"] += reasoning_retries + transformation_retries

    # 7) Assemble new solutions_list: original seed_solutions + fused + mutated
    new_state["seed_solutions_list"] = seed_solutions
    new_state["fused_solutions_list"] = fused_solutions
    new_state["mutated_solutions_list"] = mutated_solutions
    return new_state


def test_code_node(state: AgentState, llm, transformation_llm, code_llm) -> AgentState:
    """
    Test the generated code against training examples.

    This node executes the current solution on all training examples
    and calculates the success rate.
    """

    tid = state.get('task_id', 'unknown')
    print(f"Task {tid} [test_code_node]: Testing code solutions...")

    # Deep copy the new state, and shove in the training results to previous_training_results
    # And then reset training results
    new_state = copy.deepcopy(state)
    # Read runtime flags from the state (propagated by the agent)
    enable_code_predict = new_state.get("enable_code_predict", True)
    enable_llm_predict = new_state.get("enable_llm_predict", True)
    
    # Solutions list
    seed_solutions_list = new_state.get("seed_solutions_list", []) or []
    fused_solutions_list = new_state.get("fused_solutions_list", []) or []
    mutated_solutions_list = new_state.get("mutated_solutions_list", []) or []

    # Task information
    task_data = new_state.get("task_data", {})
    training_examples = task_data.get("train", [])
    num_original_train = len(training_examples)
    
    # Add augmented data if available
    augment_data = new_state.get('augment_data')
    if augment_data and 'train' in augment_data:
        training_examples = training_examples + augment_data['train']
        print(f"Task {tid} [test_code_node]: Using {num_original_train} original + {len(augment_data['train'])} augmented = {len(training_examples)} total training examples")
    
    testing_examples = task_data.get("test", [])
    enable_parallel = new_state.get("enable_parallel_eval", False)

    # Build a unified temporary list to iterate over (do NOT overwrite
    # `new_state["solutions_list"]` — that is constructed elsewhere).
    # - `seed_solutions_list` is a flat list of solutions
    # - `fused_solutions_list` and `mutated_solutions_list` are lists of lists
    temporary_solutions_list = []
    # add seed solutions (may be empty)
    if seed_solutions_list:
        temporary_solutions_list.extend(seed_solutions_list)

    # fused and mutated are groups of solution-arrays; flatten them
    if fused_solutions_list:
        for group in fused_solutions_list:
            if isinstance(group, list):
                temporary_solutions_list.extend(group)
            elif group:
                temporary_solutions_list.append(group)

    if mutated_solutions_list:
        for group in mutated_solutions_list:
            if isinstance(group, list):
                temporary_solutions_list.extend(group)
            elif group:
                temporary_solutions_list.append(group)

    # Evaluation each solutions
    for solution in temporary_solutions_list:
        main_code = solution.get("main_code", "")
        if solution.get("evaluated", False):
            continue  # skip already evaluated solutions
        main_code = solution.get("main_code", "")
        training_results: List[ExampleResult] = []
        testing_results: List[ExampleResult] = []
        # Per-solution local accumulators
        for i, example in enumerate(training_examples):
            input_grid = example["input"]
            expected_output = example["output"]
        # Use centralized helper to evaluate training examples (possibly parallel)
        transformation_steps = solution.get("step_by_step_transformation", []) if solution else []
        if enable_parallel:
            tasks = []
            with ThreadPoolExecutor() as ex:
                for i, example in enumerate(training_examples):
                    tasks.append(ex.submit(
                        lambda idx, ex_in, ex_out: (idx, evaluate_example(
                            llm,
                            main_code,
                            transformation_steps,
                            ex_in,
                            ex_out,
                            enable_code_predict=enable_code_predict,
                            enable_llm_predict=enable_llm_predict,
                        )), i, example["input"], example["output"]))

                for future in as_completed(tasks):
                    try:
                        # futures return a tuple (idx, result)
                        idx, result = future.result()
                    except Exception as e:
                        # create a failure result
                        idx = -1
                        result = {
                            "input": None,
                            "expected_output": None,
                            "predicted_output": None,
                            "matching_size": False,
                            "overlap_percentage": 0.0,
                            "error_message": str(e),
                            "code_success": False,
                            "llm_predicted_output": None,
                            "llm_matching_size": False,
                            "llm_overlap_percentage": 0.0,
                            "llm_error_message": str(e),
                            "llm_success": False,
                        }

                    # Attach example index and store (convert to ExampleResult)
                    result["example_index"] = idx
                    training_results.append(_exampleresult_from_dict(result))
        else:
            for i, example in enumerate(training_examples):
                input_grid = example["input"]
                expected_output = example["output"]
                result = evaluate_example(
                    llm,
                    main_code,
                    transformation_steps,
                    input_grid,
                    expected_output,
                    enable_code_predict=enable_code_predict,
                    enable_llm_predict=enable_llm_predict,
                )
                result["example_index"] = i
                training_results.append(_exampleresult_from_dict(result))
        # training/testing stats will be computed by `calculate_solution_statistics`
        
        # Evaluate testing examples (possibly parallel)
        if enable_parallel:
            tasks = []
            with ThreadPoolExecutor() as ex:
                for i, example in enumerate(testing_examples):
                    tasks.append(ex.submit(
                        lambda idx, ex_in, ex_out: (idx, evaluate_example(
                            llm,
                            main_code,
                            transformation_steps,
                            ex_in,
                            ex_out,
                            enable_code_predict=enable_code_predict,
                            enable_llm_predict=enable_llm_predict,
                        )), i, example["input"], example.get("output", None)))

                for future in as_completed(tasks):
                    try:
                        idx, result = future.result()
                    except Exception as e:
                        idx = -1
                        result = {
                            "input": None,
                            "expected_output": None,
                            "predicted_output": None,
                            "matching_size": False,
                            "overlap_percentage": 0.0,
                            "error_message": str(e),
                            "code_success": False,
                            "llm_predicted_output": None,
                            "llm_matching_size": False,
                            "llm_overlap_percentage": 0.0,
                            "llm_error_message": str(e),
                            "llm_success": False,
                        }

                    result["example_index"] = idx
                    testing_results.append(_exampleresult_from_dict(result))
        else:
            for i, example in enumerate(testing_examples):
                input_grid = example["input"]
                expected_output = example.get("output", None)
                result = evaluate_example(
                    llm,
                    main_code,
                    transformation_steps,
                    input_grid,
                    expected_output,
                    enable_code_predict=enable_code_predict,
                    enable_llm_predict=enable_llm_predict,
                )
                result["example_index"] = i
                testing_results.append(_exampleresult_from_dict(result))

        # Update state: attach result lists and compute statistics
        solution["training_results"] = training_results
        solution["testing_results"] = testing_results
        calculate_solution_statistics(solution, num_original_train=num_original_train)
        solution["evaluated"] = True

    # Build the canonical `solutions_list` for this state:
    # 1) include all seed solutions
    # 2) for each group in fused_solutions_list (list-of-lists), pick the best solution
    # 3) for each group in mutated_solutions_list (list-of-lists), pick the best solution
    def _sort_key(sol: Dict[str, Any]):
        weight = new_state.get('augmented_example_weight', 0.5)
        s_rate = sol.get("training_success_rate", 0.0)
        a_rate = sol.get("augment_training_success_rate", 0.0)
        s_overlap = sol.get("training_overlap_average", 0.0)
        a_overlap = sol.get("augment_training_overlap_average", 0.0)
        
        has_augment = sol.get("augment_training_results") is not None and len(sol.get("augment_training_results")) > 0
        if has_augment:
            # Combined weighted score
            weighted_success = (s_rate + a_rate * weight) / (1.0 + weight)
            weighted_overlap = (s_overlap + a_overlap * weight) / (1.0 + weight)
        else:
            weighted_success = s_rate
            weighted_overlap = s_overlap
            
        return (weighted_success, weighted_overlap)

    assembled_solutions: List[CodeSolution] = []

    # 1) all seed solutions
    if seed_solutions_list:
        assembled_solutions.extend(seed_solutions_list)

    # 2) best from each fused group
    if fused_solutions_list:
        for group in fused_solutions_list:
            if not group:
                continue
            # if group is a list of solution dicts, pick the best by the sort key
            if isinstance(group, list):
                try:
                    best = max(group, key=_sort_key)
                except Exception:
                    # Fallback: use first
                    best = group[0]
                assembled_solutions.append(best)
            else:
                assembled_solutions.append(group)

    # 3) best from each mutated/refined group
    if mutated_solutions_list:
        for group in mutated_solutions_list:
            if not group:
                continue
            if isinstance(group, list):
                try:
                    best = max(group, key=_sort_key)
                except Exception:
                    best = group[0]
                assembled_solutions.append(best)
            else:
                assembled_solutions.append(group)

    new_state["solutions_list"] = assembled_solutions
    return new_state

def finalize_node(state: AgentState) -> AgentState:
    """
    Finalize the workflow and prepare the final output.
    """
    print(f"Task {state.get('task_id', 'unknown')} [finalize_node]: Finalizing workflow state...")
    new_state = copy.deepcopy(state)
    return new_state


def save_state_node(state: AgentState) -> AgentState:
    """Persist the latest workflow state to `latest_state.json` inside
    the folder specified by `state['task_folder']`.

    This node is tolerant of missing folders and exceptions; it logs
    failures and returns the original (or possibly mutated) state.
    """

    tid = state.get('task_id', 'unknown')
    print(f"Task {tid} [save_state_node]: Saving state to disk...")

    task_folder = state.get('task_folder')
    if not task_folder:
        print("save_state_node: no task_folder in state; skipping save.")
        return state
    try:
        os.makedirs(task_folder, exist_ok=True)
        path = os.path.join(task_folder, "latest_state.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as e:
        print(f"save_state_node: failed to save state to {task_folder}: {e}")
    return state

def _all_examples_success(results):
    """Helper to check if all examples succeeded."""
    if not results:
        return False
    return all(bool(r.get('code_success', False)) for r in results)


def get_perfect_solutions(state: AgentState) -> List[Dict]:
    """Extract solutions with 100% training accuracy."""
    solutions = state.get('solutions_list', [])
    perfect = []
    
    for sol in solutions:
        results = sol.get('training_results', [])
        if _all_examples_success(results):
            perfect.append(sol)
    
    return perfect


def verify_solutions_node(state: AgentState, llm, transformation_llm, code_llm) -> AgentState:
    """Multi-stage verification of perfect-scoring solutions.
    
    Pipeline:
    1. Basic training verification (always enabled)
    2. Chain of Verification (CoVE) if enabled
    3. LLM-as-judge if enabled
    4. Adversarial testing if enabled
    5. Augmentation testing (always if augmentation enabled)
    6. Final confidence scoring and decision
    """
    task_id = state.get('task_id', 'unknown')
    print(f"\n{'='*60}")
    print(f"Task {task_id} [VERIFICATION STAGE]")
    print(f"{'='*60}\n")
    
    # Get configuration
    cove_enabled = state.get('cove_verification', False)
    llm_judge_enabled = state.get('llm_as_judge_verification', False)
    adversarial_enabled = state.get('adversarial_verification', False)
    confidence_threshold = state.get('verification_confidence_threshold', 0.75)
    num_augmentations = state.get('verification_num_augmentations', 10)
    
    # Get all perfect solutions
    perfect_solutions = get_perfect_solutions(state)
    
    if not perfect_solutions:
        print("[verify] No perfect solutions found, continuing evolution...")
        state['verification_passed'] = False
        state['verification_confidence'] = 0.0
        return state
    
    print(f"[verify] Found {len(perfect_solutions)} perfect solution(s)")
    print(f"[verify] Verification enabled: CoVE={cove_enabled}, LLM-Judge={llm_judge_enabled}, Adversarial={adversarial_enabled}")
    
    # Run verification on each perfect solution
    verification_results = []
    for i, solution in enumerate(perfect_solutions):
        print(f"\n--- Verifying Solution {i+1}/{len(perfect_solutions)} ---")
        
        result = {
            'solution': solution,
            'training_score': 1.0,  # Already perfect on training
            'cove_score': None,
            'llm_judge_score': None,
            'adversarial_pass_rate': None,
            'augmentation_pass_rate': None,
            'overall_confidence': 0.0,
            'concerns': [],
            'recommendation': ''
        }
        
        # Stage 1: Chain of Verification
        if cove_enabled:
            print("  [1/4] Running Chain of Verification...")
            cove_result = run_chain_of_verification(state, solution, llm)
            result['cove_score'] = cove_result['score']
            result['concerns'].extend(cove_result.get('concerns', []))
            print(f"  → CoVE Score: {result['cove_score']:.2%}")
        
        # Stage 2: LLM-as-Judge
        if llm_judge_enabled:
            print("  [2/4] Running LLM-as-Judge verification...")
            judge_result = run_llm_as_judge(state, solution, llm)
            result['llm_judge_score'] = judge_result['score']
            result['concerns'].extend(judge_result.get('concerns', []))
            print(f"  → LLM-Judge Score: {result['llm_judge_score']:.2%}")
        
        # Stage 3: Adversarial Testing
        if adversarial_enabled:
            print("  [3/4] Running adversarial edge case testing...")
            adv_result = run_adversarial_testing(state, solution, llm)
            result['adversarial_pass_rate'] = adv_result['pass_rate']
            result['concerns'].extend(adv_result.get('concerns', []))
            print(f"  → Adversarial Pass Rate: {result['adversarial_pass_rate']:.2%}")
        
        # Stage 4: Augmentation Testing (if num_augmentations > 0)
        if num_augmentations > 0:
            print("  [4/4] Testing on augmented examples...")
            aug_result = run_augmentation_testing(state, solution, num_augmentations)
            result['augmentation_pass_rate'] = aug_result['pass_rate']
            if aug_result['pass_rate'] < 0.8:
                result['concerns'].append(f"Low augmentation pass rate: {aug_result['pass_rate']:.1%}")
            print(f"  → Augmentation Pass Rate: {result['augmentation_pass_rate']:.2%}")
        
        # Calculate overall confidence
        result['overall_confidence'] = calculate_overall_confidence(
            result, cove_enabled, llm_judge_enabled, adversarial_enabled, num_augmentations > 0
        )
        result['recommendation'] = make_recommendation(result['overall_confidence'], confidence_threshold)
        
        print(f"  → Overall Confidence: {result['overall_confidence']:.2%}")
        print(f"  → Recommendation: {result['recommendation']}")
        
        verification_results.append(result)
    
    # Select best verified solution
    best = max(verification_results, key=lambda x: x['overall_confidence'])
    
    # Store results in state
    state['verification_results'] = verification_results
    state['best_verified_solution'] = best['solution']
    state['verification_confidence'] = best['overall_confidence']
    state['verification_concerns'] = best['concerns']
    
    # Decision: pass if confidence meets threshold
    if best['overall_confidence'] >= confidence_threshold:
        print(f"\n✅ VERIFICATION PASSED (confidence: {best['overall_confidence']:.2%})")
        state['verification_passed'] = True
    else:
        print(f"\n⚠️  VERIFICATION UNCERTAIN (confidence: {best['overall_confidence']:.2%})")
        print(f"   Concerns: {', '.join(best['concerns'][:3])}")
        state['verification_passed'] = False
        
        # Add feedback for next evolution round
        state['verification_feedback'] = {
            'concerns': best['concerns'],
            'recommendation': 'continue_search'
        }
    
    return state


def run_chain_of_verification(state: AgentState, solution: Dict, llm) -> Dict:
    """Implement Chain of Verification (CoVE).
    
    Process:
    1. Generate verification questions about the solution
    2. Answer each question independently
    3. Cross-check answers for consistency
    """
    task_data = state.get('task_data', {})
    code = solution.get('code', '')
    
    # Step 1: Generate verification questions
    question_prompt = f"""You have a solution that scores 100% on training examples for an ARC task.
Generate 5 critical verification questions to test if this solution truly generalizes.

Training Examples: {json.dumps(task_data.get('train', [])[:2])}  # Only first 2 for brevity

Solution Code:
```python
{code}
```

Generate questions that probe:
1. What is the core transformation rule?
2. What assumptions does the code make?
3. What edge cases might break it?
4. Is it specific to training or general?
5. How would it handle variations?

Return ONLY a JSON object:
{{"questions": ["Question 1?", "Question 2?", ...]}}
"""
    
    try:
        questions_response = llm.invoke(question_prompt)
        questions = parse_json_response(questions_response, 'questions', [])
        
        if not questions:
            return {'score': 0.5, 'concerns': ['Could not generate verification questions']}
        
        # Step 2: Answer each question
        answers = []
        for q in questions[:5]:  # Limit to 5 questions
            answer_prompt = f"""Answer this verification question about the ARC solution:

Question: {q}

Solution Code:
```python
{code}
```

Provide a brief, analytical answer."""
            
            answer_response = llm.invoke(answer_prompt)
            answer_text = answer_response if isinstance(answer_response, str) else getattr(answer_response, 'content', str(answer_response))
            answers.append({'question': q, 'answer': answer_text})
        
        # Step 3: Cross-check for confidence
        consistency_prompt = f"""Review these Q&A pairs about an ARC solution.
Rate your confidence that this solution will generalize well.

Q&A:
{json.dumps(answers, indent=2)}

Return JSON:
{{
  "confidence_score": 0.0-1.0,
  "concerns": ["concern 1", ...]
}}
"""
        
        consistency_response = llm.invoke(consistency_prompt)
        consistency = parse_json_response(consistency_response)
        
        return {
            'score': consistency.get('confidence_score', 0.5),
            'concerns': consistency.get('concerns', []),
            'questions': answers
        }
    
    except Exception as e:
        print(f"  Warning: CoVE failed: {e}")
        return {'score': 0.5, 'concerns': [f'CoVE error: {str(e)}']}


def run_llm_as_judge(state: AgentState, solution: Dict, llm) -> Dict:
    """Use LLM to judge solution quality holistically."""
    task_data = state.get('task_data', {})
    code = solution.get('code', '')
    training_results = solution.get('training_results', [])
    
    prompt = f"""You are an expert judge evaluating an ARC solution.

Task: {json.dumps(task_data.get('train', []))}

Solution Code:
```python
{code}
```

Training Results: All {len(training_results)} examples passed ✓

Evaluate this solution on:
1. Generalization potential (0-1)
2. Code quality and clarity (0-1)
3. Robustness to variations (0-1)
4. Likelihood it's overfitted (0-1, where 1=definitely overfitted)

Return JSON:
{{
  "generalization": 0.0-1.0,
  "quality": 0.0-1.0,
  "robustness": 0.0-1.0,
  "overfit_risk": 0.0-1.0,
  "concerns": ["..."]
}}
"""
    
    try:
        response = llm.invoke(prompt)
        result = parse_json_response(response)
        
        # Calculate composite score (lower overfit_risk is better)
        score = (
            result.get('generalization', 0.5) * 0.4 +
            result.get('quality', 0.5) * 0.2 +
            result.get('robustness', 0.5) * 0.3 +
            (1.0 - result.get('overfit_risk', 0.5)) * 0.1
        )
        
        return {
            'score': score,
            'concerns': result.get('concerns', [])
        }
    
    except Exception as e:
        print(f"  Warning: LLM-as-Judge failed: {e}")
        return {'score': 0.5, 'concerns': [f'Judge error: {str(e)}']}


def run_adversarial_testing(state: AgentState, solution: Dict, llm) -> Dict:
    """Generate and test adversarial edge cases."""
    task_data = state.get('task_data', {})
    code = solution.get('code', '')
    
    # Generate edge cases
    prompt = f"""Generate 3-5 adversarial test cases for this ARC solution.

Training Examples: {json.dumps(task_data.get('train', [])[:2])}

Solution Code:
```python
{code}
```

Generate edge cases with different:
- Grid sizes
- Color patterns
- Sparse/dense layouts

Return JSON:
{{"edge_cases": [{{"input": [[...]], "rationale": "..."}}]}}
"""
    
    try:
        response = llm.invoke(prompt)
        edge_cases = parse_json_response(response, 'edge_cases', [])
        
        if not edge_cases:
            return {'pass_rate': 0.5, 'concerns': ['Could not generate edge cases']}
        
        # Test each edge case
        passed = 0
        failed_cases = []
        
        for case in edge_cases[:5]:  # Limit to 5
            input_grid = case.get('input')
            if not input_grid:
                continue
            
            try:
                output, error = execute_transformation_code(code, input_grid, timeout_seconds=5)
                if error is None and output is not None:
                    passed += 1
                else:
                    failed_cases.append(case.get('rationale', 'Unknown'))
            except:
                failed_cases.append(case.get('rationale', 'Execution error'))
        
        total = len(edge_cases[:5])
        pass_rate = passed / total if total > 0 else 0.0
        
        concerns = []
        if pass_rate < 0.6:
            concerns.append(f"Failed {len(failed_cases)} adversarial cases: {', '.join(failed_cases[:2])}")
        
        return {'pass_rate': pass_rate, 'concerns': concerns}
    
    except Exception as e:
        print(f"  Warning: Adversarial testing failed: {e}")
        return {'pass_rate': 0.5, 'concerns': [f'Adversarial error: {str(e)}']}


def run_augmentation_testing(state: AgentState, solution: Dict, num_augmentations: int) -> Dict:
    """Test solution on augmented examples."""
    task_data = state.get('task_data', {})
    code = solution.get('code', '')
    
    try:
        # Generate augmented examples
        augmented = augment_task_data(task_data, num_augmentations)
        aug_examples = augmented.get('train', [])
        
        if not aug_examples:
            return {'pass_rate': 0.5, 'concerns': ['No augmented examples generated']}
        
        # Test on augmented examples
        passed = 0
        for example in aug_examples:
            input_grid = example.get('input')
            expected_output = example.get('output')
            
            try:
                predicted_output, error = execute_transformation_code(code, input_grid, timeout_seconds=5)
                
                if predicted_output and expected_output:
                    match = (predicted_output == expected_output)
                    if match:
                        passed += 1
            except:
                pass
        
        pass_rate = passed / len(aug_examples)
        
        return {'pass_rate': pass_rate, 'concerns': []}
    
    except Exception as e:
        print(f"  Warning: Augmentation testing failed: {e}")
        return {'pass_rate': 0.5, 'concerns': [f'Augmentation error: {str(e)}']}


def calculate_overall_confidence(result: Dict, cove_enabled: bool, llm_judge_enabled: bool, 
                                 adversarial_enabled: bool, augmentation_enabled: bool) -> float:
    """Calculate weighted confidence score from all enabled verifications."""
    scores = []
    weights = []
    
    # Training score (always counted)
    scores.append(result['training_score'])
    weights.append(0.3)
    
    # CoVE score
    if cove_enabled and result['cove_score'] is not None:
        scores.append(result['cove_score'])
        weights.append(0.25)
    
    # LLM-as-judge score
    if llm_judge_enabled and result['llm_judge_score'] is not None:
        scores.append(result['llm_judge_score'])
        weights.append(0.25)
    
    # Adversarial score
    if adversarial_enabled and result['adversarial_pass_rate'] is not None:
        scores.append(result['adversarial_pass_rate'])
        weights.append(0.15)
    
    # Augmentation score
    if augmentation_enabled and result['augmentation_pass_rate'] is not None:
        scores.append(result['augmentation_pass_rate'])
        weights.append(0.15)
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight == 0:
        return result['training_score']
    
    normalized_weights = [w / total_weight for w in weights]
    
    # Calculate weighted average
    confidence = sum(s * w for s, w in zip(scores, normalized_weights))
    return confidence


def make_recommendation(confidence: float, threshold: float) -> str:
    """Make recommendation based on confidence."""
    if confidence >= threshold:
        return "accept"
    elif confidence >= threshold - 0.15:
        return "refine"
    else:
        return "continue_search"


def parse_json_response(response, key=None, default=None):
    """Parse JSON from LLM response."""
    content = response if isinstance(response, str) else getattr(response, 'content', str(response))
    
    # Try to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
    if json_match:
        content = json_match.group(1)
    
    try:
        data = json.loads(content)
        if key:
            return data.get(key, default)
        return data
    except:
        if key:
            return default
        return {}
