"""
Reasoning generation functions for ARC agent.
"""

import re
import textwrap
from typing import List, Dict, Optional, Any, Tuple

from ..schema import AgentState, CodeSolution, ExampleResult
from .utilities import format_grid_for_prompt, format_grid_for_analysis, format_difference_map, _flatten_content, build_steps_text_from_transformation_steps
from agentic.debug import print_prompt_and_response

def generate_reasoning_trace(llm, training_examples: List[Dict], visual_cues: Optional[List[Dict]] = None, num_inloop_augmentations: int = 0, max_retries: int = 3) -> Tuple[str, int]:
    """Generate detailed reasoning trace analyzing ARC patterns.
    
    Args:
        num_inloop_augmentations: Number of augmented examples to generate during reasoning
    
    Returns:
        Tuple of (reasoning_trace, num_retries_used)
    """
    from ..augmentation import augment_task_data
    
    # Generate in-loop augmentations if requested
    augmented_examples = []
    if num_inloop_augmentations > 0:
        print(f"🔄 [Reasoning] Generating {num_inloop_augmentations} in-loop augmentations with random seeds...")
        
        # Create augmented data with different random seeds each time
        aug_data = augment_task_data(
            {"train": training_examples},
            num_augmentations=num_inloop_augmentations
        )
        
        if aug_data and "train" in aug_data:
            augmented_examples = aug_data["train"]
            print(f"✓ Created {len(augmented_examples)} augmented examples")
    
    # Combine original and augmented examples
    all_training_examples = training_examples + augmented_examples

    def build_initial_reasoning_prompt(training_examples: List[Dict]) -> str:
        """Build prompt for generating detailed reasoning about ARC patterns."""
        
        # Add augmentation note if applicable
        aug_note = ""
        if num_inloop_augmentations > 0 and augmented_examples:
            aug_note = f"\nNote: {len(augmented_examples)} augmented examples were generated using random transformations (rotations, flips, color permutations) to help identify the core pattern.\n"
        
        prompt_parts = [
            "You are an expert mathematician, logistician and pattern recognizier who is solving the"
            "Abstract Reasoning Corpus (ARC) problems.",
            "Your task is to deeply analyze the input-output examples and understand the underlying pattern.",
            "Focus on identifying the core transformation rule that maps inputs to outputs.",
            aug_note,
            "YOUR GOAL:",
            "Given the training pairs and test inputs, infer a general transformation rule that:",
            "- Correctly maps every training input to its output.",
            "- Is general and intuitive (no memorization or hard-coded values).",
            "- Is logical, reproducible, and object-level.",

            "GUIDELINES:",
            "- The SAME rule must successfully transform all training pairs.",
            "- Treat all grid values (numbers/characters) as categorical labels, not magnitudes. Do not use arithmetic operations.",
            "- Avoid rules that depend on specific values or characters.",
            "- Make rules in a general manner using object-level reasoning (movements, shapes, fills, mirrors, rotations, bounding boxes, duplicates, etc.).",
            "- Take as many rules as you need to achieve your goals.",
            "",
            "TRAINING EXAMPLES:"
        ]
        
        # Add training examples with detailed formatting (includes augmented)
        for i, example in enumerate(training_examples):
            is_augmented = i >= len(training_examples) - len(augmented_examples)
            prefix = "[AUGMENTED] " if is_augmented else ""
            prompt_parts.append(f"\n{prefix}Example {i+1}:")
            prompt_parts.append(f"Input Grid ({len(example['input'])}x{len(example['input'][0]) if example['input'] else 0}):")
            prompt_parts.append(format_grid_for_analysis(example['input']))
            prompt_parts.append(f"Output Grid ({len(example['output'])}x{len(example['output'][0]) if example['output'] else 0}):")
            prompt_parts.append(format_grid_for_analysis(example['output']))
        
        prompt_parts.extend([
            "",
            "ANALYSIS INSTRUCTIONS:",
            "Provide a ```reasoning``` block that contains your detailed analysis.",
        ])
        
        return "\n".join(prompt_parts)
    

    prompt = build_initial_reasoning_prompt(all_training_examples)
    # If visual cues are provided and the llm driver supports image messages,
    # send a structured message containing the images (base64 data URLs).
    if visual_cues:
        pass

    # Retry up to max_retries times if extraction fails
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            # Flatten content to handle list/dict/string responses
            if hasattr(response, 'content'):
                resp_content = response.content
            else:
                resp_content = response
            
            response_text = _flatten_content(resp_content)
            print_prompt_and_response(prompt, response_text)

            # Extract reasoning from response
            reasoning = extract_reasoning_content(response_text)
            if reasoning and reasoning != "Unable to generate reasoning trace":
                return reasoning, attempt
            
            # If this isn't the last attempt, log and retry
            if attempt < max_retries - 1:
                print(f"Warning: Failed to extract reasoning content (attempt {attempt + 1}/{max_retries}). Retrying...")
        
        except Exception as e:
            print(f"Warning: Error in generate_reasoning_trace (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
            else:
                import traceback
                traceback.print_exc()
    
    # After all retries failed
    return "Unable to generate reasoning trace", max_retries


def generate_reflection_reasoning_trace(llm,
                                        current_solution: CodeSolution,
                                        training_results: List[ExampleResult],
                                        training_examples: List[Dict],
                                        enable_rag_hint: bool,
                                        num_inloop_augmentations: int = 0,
                                        max_retries: int = 3) -> Tuple[str, int]:
    """Generate a reflection-focused reasoning trace using the ARC-style reflection prompt.

    This is intended for refinement: it asks the model to analyze failures, explain
    what went wrong, and produce a reasoning trace focused on correcting the logic.
    
    Args:
        num_inloop_augmentations: Number of augmented examples to generate and test during reflection
    
    Returns:
        Tuple of (reasoning_trace, num_retries_used)
    """
    from .rag import retrieve_similar_distillations
    from ..augmentation import augment_task_data
    from .code_execution import test_code_on_examples
    import random
    
    # Generate in-loop augmentations if requested
    augmented_results = []
    augmented_examples = []
    if num_inloop_augmentations > 0:
        print(f"🔄 [Reflection] Generating {num_inloop_augmentations} in-loop augmentations with random seeds...")
        
        # Create augmented data with different random seeds each time
        aug_data = augment_task_data(
            {"train": training_examples},
            num_augmentations=num_inloop_augmentations
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
        if num_inloop_augmentations > 0 and augmented_examples:
            aug_note = f"\nNote: {len(augmented_examples)} augmented examples were generated using random transformations (rotations, flips, color permutations) to test your solution's robustness.\n"
        
        # Build training examples block (includes augmented)
        examples_block = ""
        for i, example in enumerate(all_training_examples, 1):
            is_augmented = i > len(training_examples)
            prefix = "[AUGMENTED] " if is_augmented else ""
            examples_block += f"Training Example {i}\\n--\\n"
            examples_block += f"Input:\\n{format_grid_for_prompt(example['input'])}\\n\\n"
            examples_block += f"Output:\\n{format_grid_for_prompt(example['output'])}\\n\\n"
        
        prompt_parts = [
            "You are an expert mathematician, logistician and pattern recognizier who is solving the"
            "Abstract Reasoning Corpus (ARC) problems.",
            "You previously attempted to solve this task but your solution was incorrect on some training examples."
            "",
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


def generate_fused_reasoning_trace(llm,
                                   sola: Dict,
                                   solb: Dict,
                                   training_results1: List[ExampleResult],
                                   training_results2: List[ExampleResult],
                                   training_examples: List[Dict],
                                   enable_rag_hint: bool,
                                   num_inloop_augmentations: int = 0,
                                   max_retries: int = 3) -> Tuple[str, int]:
    """Generate a fused reasoning trace that reconciles two candidate solutions.

    The prompt includes both solutions' reasoning, transformation steps, code (if available),
    and the training results. The LLM is asked to produce a single, coherent reasoning
    trace that explains how to combine their strengths and address their failure modes.
    
    Args:
        num_inloop_augmentations: Number of augmented examples to generate and test during fusion
    
    Returns:
        Tuple of (reasoning_trace, num_retries_used)
    """
    from .fusion import result_comparison_text
    from .rag import retrieve_similar_distillations
    from ..debug import print_prompt_and_response
    from ..augmentation import augment_task_data
    from .code_execution import test_code_on_examples
    import random
    
    # Generate in-loop augmentations if requested
    augmented_results1 = []
    augmented_results2 = []
    augmented_examples = []
    if num_inloop_augmentations > 0:
        print(f"🔄 [Fusion] Generating {num_inloop_augmentations} in-loop augmentations with random seeds...")
        
        # Create augmented data with different random seeds each time
        aug_data = augment_task_data(
            {"train": training_examples},
            num_augmentations=num_inloop_augmentations
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
        if num_inloop_augmentations > 0 and augmented_examples:
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

        parts = [
            "You are an expert ARC solver. Two candidate solutions were produced for the same task.",
            "Your job is to reconcile them into a single solution that combines their strengths and remedies their weaknesses.",
            aug_note,
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


def generate_distilled_reasoning(llm, reasoning_trace, transformation_steps, python_codes):
    """Distill a detailed `reasoning_trace` into a JSON-like structure.

    Returns a dict with keys:
      - 'strategy': concise single-paragraph summary (<=150 words)
      - 'concepts': list of short concept strings

    The LLM is instructed to return ONLY valid JSON. This function will try
    to robustly parse common response shapes (```json``` fenced block, bare
    JSON, or a JSON-like substring). On failure it will produce a best-effort
    dict with empty `concepts`.
    """
    import json

    def build_distill_reasoning_prompt() -> str:
        prompt_parts = [
            "---------------",
            "REASONING TRACE",
            "---------------",
            reasoning_trace,
            "",
            "------------",
            "INSTRUCTIONS",
            "------------",
            "Read ONLY the reasoning trace above and produce a JSON object with exactly two fields:\n",
            "1) \"strategy\": a concise single-paragraph summary (<=150 words) describing the high-level strategy used to solve the task;\n",
            "2) \"concepts\": an array of short strings naming the key operations or concepts used (e.g., \"connected components\", \"symmetry\", \"color mapping\").\n",
            "Return ONLY valid JSON within a ```json```. Do NOT include any additional text, commentary, or markdown.\n",
            "Example output:",
            '```json',
            '{"strategy": "Brief summary...", "concepts": ["symmetry", "fill", "mirror"]}',
            '```',
            "Perform the distillation now:"
        ]
        return "\n".join(prompt_parts)

    prompt = build_distill_reasoning_prompt()

    try:
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        return {"strategy": f"(LLM error during distillation: {e})", "concepts": []}

    # Extract JSON candidate
    json_candidate = None
    m = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
    if m:
        json_candidate = m.group(1).strip()
    else:
        m2 = re.search(r'\{.*?\}', response_text, re.DOTALL)
        if m2 and 'strategy' in m2.group(0):
            json_candidate = m2.group(0)

    if json_candidate:
        try:
            parsed = json.loads(json_candidate)
            strategy = str(parsed.get('strategy', '')).strip()
            concepts = parsed.get('concepts') or parsed.get('concept') or []
            if isinstance(concepts, str):
                concepts = [c.strip() for c in re.split(r'[;,\n]', concepts) if c.strip()]
            elif not isinstance(concepts, (list, tuple)):
                concepts = []

            words = strategy.split()
            if len(words) > 150:
                strategy = " ".join(words[:150]) + "..."

            return {"strategy": strategy, "concepts": concepts}
        except Exception:
            pass

    # Forgiving fallback
    text = textwrap.dedent(str(response_text)).strip()
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    strategy = paragraphs[0] if paragraphs else text
    words = strategy.split()
    if len(words) > 150:
        strategy = " ".join(words[:150]) + "..."

    concepts = []
    for line in paragraphs[1:4]:
        if len(line) < 200 and (',' in line or ';' in line or line.lower().startswith('concepts') or len(line.split()) <= 6):
            cand = re.sub(r'^(concepts?:\s*)', '', line, flags=re.IGNORECASE)
            parts = [c.strip() for c in re.split(r'[;,\n]', cand) if c.strip()]
            for p in parts:
                if 2 <= len(p) <= 60:
                    concepts.append(p)
        if concepts:
            break

    return {"strategy": strategy, "concepts": concepts}


def extract_reasoning_content(response_text: str) -> str:
    """Extract reasoning content from LLM response."""
    
    # Ensure we have a string to work with
    if not isinstance(response_text, str):
        try:
            # Try to convert to string if it's not already
            if isinstance(response_text, (list, dict)):
                response_text = str(response_text)
            else:
                response_text = str(response_text)
        except Exception:
            return "Unable to generate reasoning trace"
    
    if not response_text or not response_text.strip():
        return "Unable to generate reasoning trace"
    
    # Look for reasoning block
    try:
        reasoning_match = re.search(r'```reasoning\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            return reasoning_match.group(1).strip()
    except (TypeError, AttributeError) as e:
        print(f"Warning: regex search failed in extract_reasoning_content: {e}")
        # Continue to fallback extraction methods
    
    # Fallback: look for structured content
    patterns = [
        r'PATTERN OBSERVATION:(.*?)(?=TRANSFORMATION HYPOTHESIS:|$)',
        r'TRANSFORMATION HYPOTHESIS:(.*?)(?=VERIFICATION:|$)',
        r'VERIFICATION:(.*?)(?=CORE INSIGHT:|$)',
        r'CORE INSIGHT:(.*?)(?=$)'
    ]
    
    reasoning_parts = []
    try:
        for pattern in patterns:
            match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if match:
                reasoning_parts.append(match.group(1).strip())
        
        if reasoning_parts:
            return '\n\n'.join(reasoning_parts)
    except (TypeError, AttributeError) as e:
        print(f"Warning: pattern matching failed in extract_reasoning_content: {e}")
    
    # Ultimate fallback: return first substantial paragraph
    try:
        lines = response_text.split('\n')
        substantial_lines = [line.strip() for line in lines if len(line.strip()) > 20]
        
        return '\n'.join(substantial_lines[:10]) if substantial_lines else response_text[:500]
    except Exception:
        return response_text[:500] if len(response_text) > 500 else response_text


def extract_reasoning_from_reflection(response_content: str) -> str:
    """Extract reasoning section from ARC-style reflection response."""
    
    # Look for reasoning block
    reasoning_match = re.search(r'```reasoning\s*(.*?)\s*```', response_content, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        return reasoning_match.group(1).strip()
    
    # Fallback: look for analysis patterns
    patterns = [
        r'PATTERN MISINTERPRETATION:(.*?)(?=\d\.|\n\n|$)',
        r'LOGIC ERRORS:(.*?)(?=\d\.|\n\n|$)',
        r'EDGE CASES:(.*?)(?=\d\.|\n\n|$)',
        r'CORE INSIGHT:(.*?)(?=\d\.|\n\n|$)'
    ]
    
    reasoning_parts = []
    for pattern in patterns:
        match = re.search(pattern, response_content, re.DOTALL | re.IGNORECASE)
        if match:
            reasoning_parts.append(match.group(1).strip())
    
    if reasoning_parts:
        return '; '.join(reasoning_parts)
    
    # Ultimate fallback
    return "No structured reasoning found in response"


def extract_key_insight_from_reasoning(reasoning: str) -> str:
    """Extract the key insight from reasoning text."""
    # Look for core insight patterns
    
    patterns = [
        r'(?:CORE INSIGHT|key insight|main insight|crucial insight)[:\s]+(.*?)(?:\n|$)',
        r'(?:The pattern is|Pattern:|Main pattern)[:\s]+(.*?)(?:\n|$)',
        r'(?:I need to|Should|Must)[:\s]+(.*?)(?:\n|$)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, reasoning, re.IGNORECASE)
        if match:
            insight = match.group(1).strip()
            # Clean up and limit length
            insight = re.sub(r'\s+', ' ', insight)
            return insight[:200] + '...' if len(insight) > 200 else insight
    
    # Fallback: take first meaningful sentence
    sentences = re.split(r'[.!?]+', reasoning)
    for sentence in sentences:
        if len(sentence.strip()) > 20:  # Skip very short sentences
            cleaned = re.sub(r'\s+', ' ', sentence.strip())
            return cleaned[:200] + '...' if len(cleaned) > 200 else cleaned
    
    return "Pattern recognition issue identified"


def analyze_training_examples(training_examples: List[Dict]) -> str:
    """Analyze training examples to understand the pattern."""
    if not training_examples:
        return "No training examples provided."
    
    analysis = []
    analysis.append(f"Found {len(training_examples)} training examples.")
    
    # Analyze input/output dimensions
    input_sizes = [(len(ex["input"]), len(ex["input"][0]) if ex["input"] else 0) 
                   for ex in training_examples]
    output_sizes = [(len(ex["output"]), len(ex["output"][0]) if ex["output"] else 0) 
                    for ex in training_examples]
    
    analysis.append(f"Input sizes: {input_sizes}")
    analysis.append(f"Output sizes: {output_sizes}")
    
    # Check if sizes are consistent
    if len(set(input_sizes)) == 1:
        analysis.append("All inputs have the same size.")
    else:
        analysis.append("Input sizes vary.")
    
    if len(set(output_sizes)) == 1:
        analysis.append("All outputs have the same size.")
    else:
        analysis.append("Output sizes vary.")
    
    return "\n".join(analysis)
