"""
Nodes for the ARC LangGraph Agent workflow.

This module should contain only the node functions. All helper and
action-related utilities have been moved to `actions.py`.
"""

import copy
from typing import List, Dict, Optional, Any
import random
from concurrent.futures import ThreadPoolExecutor, as_completed


# Import schema and node-facing types
from .debug import print_python_code
from .schema import AgentState, CodeSolution, ExampleResult, Grid

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


def calculate_solution_statistics(solution: CodeSolution) -> CodeSolution:
    """Populate statistic fields for a `CodeSolution` (mutates in-place).

    Uses the typed `training_results`/`testing_results` lists to compute
    success rates, overlap averages and error rates for code and LLM predictions.
    """
    train_results: List[ExampleResult] = solution.get("training_results", []) or []
    test_results: List[ExampleResult] = solution.get("testing_results", []) or []

    train_stats = _calculate_stats_from_results(train_results)
    test_stats = _calculate_stats_from_results(test_results)

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
    task_data = state["task_data"]
    num_initial_solutions = state["num_initial_solutions"]

    # Analyze the training examples (kept for node-level logging/analysis)
    training_examples = task_data["train"]

    # Generate code using the reasoning-first approach from actions
    # Read visual cue flag from node state and pass through to generation
    enable_visual_cue = state.get('enable_visual_cue', False)

    python_codes, reasoning_trace, transformation_solutions_list, rag_entry = create_solutions_with_reasoning(
        llm,
        transformation_llm,
        code_llm,
        training_examples,
        num_solutions=num_initial_solutions,
        enable_visual_cue=enable_visual_cue
    )

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
    new_state = copy.deepcopy(state)
    new_state["seed_solutions_list"] = solutions_list
    new_state["fused_solutions_list"] = []
    new_state["mutated_solutions_list"] = []
    return new_state


def evolve_code_node(state, llm, transformation_llm, code_llm):
    """Module-level evolve node: increment generation and re-run generator."""
    # Get necessary variables
    training_examples = state["task_data"]["train"]
    enable_visual_cue = state.get("enable_visual_cue", False)
    enable_rag_hint = state.get("enable_rag_hint", False)
    num_seed_solutions = state["num_seed_solutions"]
    num_refinements = state["num_refinements"]
    num_solutions_per_refinement = state["num_solutions_per_refinement"]
    num_fusions = state["num_fusions"]
    num_solutions_per_fusion = state["num_solutions_per_fusion"]

    # Make a deep copy of the incoming state to avoid mutating caller's object
    new_state = copy.deepcopy(state)
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
        "average_testing_success_rate": sum(sol.get("testing_success_rate", 0.0) for sol in state.get("solutions_list") or []) / max(len(state.get("solutions_list") or []), 1),
        "average_testing_overlap_score": sum(sol.get("testing_overlap_average", 0.0) for sol in state.get("solutions_list") or []) / max(len(state.get("solutions_list") or []), 1),
        "average_testing_error_rate": sum(sol.get("testing_error_rate", 0.0) for sol in state.get("solutions_list") or []) / max(len(state.get("solutions_list") or []), 1),
    }
    generations = new_state.get("generations", [])
    generations.append(generation_entry)
    new_state["generations"] = generations

    # increment generation counter and clear solutions_list for next generation
    new_state["current_generation"] = current_generation + 1
    new_state["solutions_list"] = []

    # 3) Sort seed solutions by training success rate then training overlap average (descending)
    # Use safe defaults if fields are missing
    def _sort_key(sol: Dict[str, Any]):
        return (
            sol.get("training_success_rate", 0.0),
            sol.get("training_overlap_average", 0.0),
        )
    # Sort by score (descending). If scores are equal, use a random
    # tie-breaker so equal-scoring solutions are ordered randomly.
    seed_solutions.sort(key=lambda s: (_sort_key(s), random.random()), reverse=True)

    # Create fusion and mutation from seed solutions
    seed_solutions = seed_solutions[:num_seed_solutions]
    fused_solutions: List[Dict[str, Any]] = []
    mutated_solutions: List[Dict[str, Any]] = []

    if seed_solutions:
        for _ in range(num_fusions):
            sol_arr = []

            # Randomly select two distinct partner solutions. Sample without replacement
            sola, solb = random.sample(seed_solutions, 2)
            python_codes, reasoning_trace, transformation_solutions_list, rag_entry = fuse_solutions_with_reasoning(
                llm,
                transformation_llm,
                code_llm,
                sola,
                solb,
                training_examples,
                num_fused_solutions=num_solutions_per_fusion,
                enable_visual_cue=enable_visual_cue,
                enable_rag_hint=enable_rag_hint,
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
            fused_solutions.append(sol_arr)

        # 6) Mutation (self-reflection)
        for i, solution in enumerate(seed_solutions[:num_refinements]):
            sol_arr = []
            python_codes, reasoning_trace, transformation_solutions_list, rag_entry = refine_solutions_with_reasoning(
                llm,
                transformation_llm,
                code_llm,
                solution,
                training_examples,
                num_solutions_per_refinement,
                enable_visual_cue=enable_visual_cue,
                enable_rag_hint=enable_rag_hint,
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
        calculate_solution_statistics(solution)
        solution["evaluated"] = True

    # Build the canonical `solutions_list` for this state:
    # 1) include all seed solutions
    # 2) for each group in fused_solutions_list (list-of-lists), pick the best solution
    # 3) for each group in mutated_solutions_list (list-of-lists), pick the best solution
    def _sort_key(sol: Dict[str, Any]):
        return (
            sol.get("training_success_rate", 0.0),
            sol.get("training_overlap_average", 0.0),
        )

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
    new_state = copy.deepcopy(state)
    return new_state