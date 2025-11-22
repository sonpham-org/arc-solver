"""
Action helpers for the ARC LangGraph Agent workflow.

This module contains helper functions, prompt builders, LLM-driven
generation utilities, execution helpers, and refinement logic.
"""

import json
import copy
from typing import List, Dict, Optional, Any, TypedDict, Tuple, Union
import traceback
import sys
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
import traceback
import re

# Import schema and tools
from .schema import AgentState, CodeSolution, ExampleResult
from .tools import FUNCTION_MAP
from .debug import print_prompt_and_response, print_python_code


def generate_llm_predicted_output(llm,
                                  transformation_steps: Dict[str, Any],
                                  input_grid: List[List[int]]) -> Tuple[Optional[List[List[int]]], Optional[str]]:
    """Use the LLM to apply the step-by-step transformation to a single input grid.

    The LLM is instructed to return the transformed grid as a JSON array
    (list of lists of integers). Returns (grid, None) on success, or
    (None, error_message) on failure.
    """
    try:
        steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(transformation_steps)) if transformation_steps else "(no steps provided)"

        prompt_parts = [
            "You are an expert that can execute step-by-step grid transformations by following instructions",
            "Given the following input grid and transformation steps, you are tasked with applying the steps and return the resulting grid.",
            
            "Do NOT return any other text after that block.",
            "",
            "INPUT GRID:",
            format_grid_for_prompt(input_grid),
            "",
            "TRANSFORMATION STEPS:",
            steps_text,
            "",
            "Follow the transformation steps carefully, show your detailed step-by-step transformation"
            "After finishing all the steps, show the 2D grid inside a fenced block labeled ```llm_predicted_output``` containing only the grid rows as lines of numbers (space-separated or contiguous digits)."
        ]

        prompt = "\n".join(prompt_parts)

        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        # print_prompt_and_response(prompt, response)

        # Prefer a fenced block labelled ```llm_predicted_output``` containing
        # the grid as lines of numbers (space-separated or run-together digits).
        import re, json
        block_match = re.search(r'```llm_predicted_output\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
        if block_match:
            block = block_match.group(1).strip()
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            parsed_grid = []
            for line in lines:
                # Split on whitespace; if no whitespace, split into single chars
                if re.search(r'\s+', line):
                    parts = re.split(r'\s+', line.strip())
                else:
                    parts = list(line.strip())
                row = []
                for p in parts:
                    try:
                        row.append(int(p))
                    except Exception:
                        # If conversion fails, try to strip non-digits then int
                        digits = re.findall(r'-?\d+', p)
                        if digits:
                            row.append(int(digits[0]))
                        else:
                            # Give up and return error with raw block
                            return None, f"Non-numeric token in llm_predicted_output block: '{p}'"
                parsed_grid.append(row)
            return parsed_grid, None

        # Fallback: try to find a JSON array in the response
        json_match = re.search(r'(\[\s*\[.*?\]\s*\])', response_text, re.DOTALL)
        if json_match:
            candidate = json_match.group(1)
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list) and all(isinstance(row, list) for row in parsed):
                    norm = []
                    for row in parsed:
                        new_row = []
                        for cell in row:
                            try:
                                new_row.append(int(cell))
                            except Exception:
                                new_row.append(cell)
                        norm.append(new_row)
                    return norm, None
            except Exception:
                pass

        # Fallback: try to parse any Python-style list literal
        try:
            parsed2 = eval(response_text, {"__builtins__": {}}, {})
            if isinstance(parsed2, list) and all(isinstance(r, list) for r in parsed2):
                return parsed2, None
        except Exception:
            pass

        # If response contains an explicit error line, return it as error
        if isinstance(response_text, str) and ("error" in response_text.lower() or "cannot" in response_text.lower() or "failed" in response_text.lower()):
            return None, response_text.strip()

        return None, f"Could not parse LLM response as grid. Raw response: {response_text[:1000]}"

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return None, f"Exception calling LLM for predicted output: {e}\n{tb}"


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


def generate_solutions_with_reasoning(llm, code_llm, training_examples: List[Dict]) -> Tuple[List[str], str, List[Dict]]:
    """Generate Python transformation code using reasoning-first approach.
    
    Returns:
        Tuple of (python_code, reasoning_trace, transformation_steps)
    """
    
    # Step 1: Generate reasoning trace
    reasoning_trace = generate_reasoning_trace(llm, training_examples)
    
    # Step 2: Extract step-by-step transformation from reasoning
    transoformation_solutions_list = generate_transformation_steps(llm, reasoning_trace, training_examples)
    
    # Step 3: Generate Python code(s) based on reasoning and steps
    # Note: `generate_code_from_reasoning` may return multiple candidate code strings.
    python_codes_list = generate_code_from_reasoning(llm, code_llm, reasoning_trace, transoformation_solutions_list,
                                                     training_examples)

    

    # Return the list of candidate codes, plus reasoning and steps.
    return python_codes_list, reasoning_trace, transoformation_solutions_list

def generate_reasoning_trace(llm, training_examples: List[Dict]) -> str:
    """Generate detailed reasoning trace analyzing ARC patterns."""

    def build_initial_reasoning_prompt(training_examples: List[Dict]) -> str:
        """Build prompt for generating detailed reasoning about ARC patterns."""
        
        prompt_parts = [
            "You are an expert mathematician, logistician and pattern recognizier who is solving the"
            "Abstract Reasoning Corpus (ARC) problems.",
            "Your task is to deeply analyze the input-output examples and understand the underlying pattern.",
            "Focus on identifying the core transformation rule that maps inputs to outputs.",

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
        
        # Add training examples with detailed formatting
        for i, example in enumerate(training_examples):
            prompt_parts.append(f"\nExample {i+1}:")
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
    
    try:
        prompt = build_initial_reasoning_prompt(training_examples)
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        # print_prompt_and_response(prompt, response)

        # Extract reasoning from response
        reasoning = extract_reasoning_content(response_text)
        return reasoning if reasoning else "Unable to generate reasoning trace"
        
    except Exception as e:
        print(f"Error generating reasoning trace: {e}")
        return f"Error in reasoning generation: {str(e)}"


def generate_reflection_reasoning_trace(llm,
                                       current_solution: CodeSolution,
                                       training_results: List[ExampleResult],
                                       training_examples: List[Dict]) -> str:
    """Generate a reflection-focused reasoning trace using the ARC-style reflection prompt.

    This is intended for refinement: it asks the model to analyze failures, explain
    what went wrong, and produce a reasoning trace focused on correcting the logic.
    """
    def build_refinement_reasoning_prompt(current_solution: CodeSolution,
                                        failed_tests: List[ExampleResult],
                                        training_examples: List[Dict]) -> str:
        """Build reflection prompt based on ARC reflection prompt style for deep analysis."""
        
        # Format previous solution
        previous_code = current_solution["main_code"]
        
        # Build detailed failure analysis
        failure_analysis = []
        for test in failed_tests:
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
                    analysis += f"Expected size: {exp_h}x{exp_w}, Predicted size: {pred_h}x{pred_w}\\n"
                else:
                    analysis += "Your Predicted Output: No output generated\\n\\n"
                    exp_h, exp_w = len(example['output']), len(example['output'][0]) if example['output'] else 0
                    analysis += f"Expected size: {exp_h}x{exp_w}, Predicted size: 0x0\\n"
                
                analysis += f"Overlap: {test.get('overlap_percentage', 0):.1f}%\\n"
                analysis += f"IOU (Intersection over Union): {test.get('iou_percentage', 0):.1f}%\\n"
                
                error_msg = test.get("error_message")
                if error_msg:
                    analysis += f"Error: {error_msg}\\n"
                
                failure_analysis.append(analysis)
        
        failures_block = "\\n".join(failure_analysis)
        
        # Build training examples block
        examples_block = ""
        for i, example in enumerate(training_examples, 1):
            examples_block += f"Training Example {i}\\n--\\n"
            examples_block += f"Input:\\n{format_grid_for_prompt(example['input'])}\\n\\n"
            examples_block += f"Output:\\n{format_grid_for_prompt(example['output'])}\\n\\n"
        
        prompt_parts = [
            "You are an expert mathematician, logistician and pattern recognizier who is solving the"
            "Abstract Reasoning Corpus (ARC) problems.",
            "Your task is to deeply analyze the input-output examples and understand the underlying pattern.",
            "You previously attempted to solve this task but your solution was incorrect on some training examples."

            "YOUR GOAL:",
            "Analyze your previous attempt deeply, understand why it failed, and provide a CORRECTED reasoning rule that:",
            "1. Correctly maps every training input to its output",
            "2. Is general and intuitive (no memorization or hard-coded values)",
            "3. Is logical, reproducible, and object-level",
            "",
            "ORIGINAL TRAINING EXAMPLES",
            f"{examples_block}",
            "",
            "YOUR PREVIOUS SOLUTION",
            "```python",
            f"{previous_code}",
            "```",
            "",
            "DETAILED FAILURE ANALYSIS",
            f"{failures_block}",
            "",
            "ANALYSIS INSTRUCTIONS:",
            "Provide a ```reasoning``` block that contains your detailed analysis.",
        ]

        prompt = "\n".join(prompt_parts)

        return prompt
    try:
        prompt = build_refinement_reasoning_prompt(current_solution, training_results, training_examples)
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        # print_prompt_and_response(prompt, response)

        # Prefer structured reflection extraction first
        reasoning = extract_reasoning_content(response_text)
        if reasoning and reasoning != "No structured reasoning found in response":
            return reasoning

    except Exception as e:
        print(f"Error generating reflection reasoning trace: {e}")
        return f"Error in reflection reasoning generation: {str(e)}"


def format_grid_for_analysis(grid: List[List[int]]) -> str: 
    """Format grid for detailed analysis in reasoning prompts."""
    if not grid:
        return "(empty grid)"
    
    formatted_rows = []
    for row in grid:
        formatted_rows.append("".join(str(cell) for cell in row))
    
    return "\n".join(formatted_rows)


def extract_reasoning_content(response_text: str) -> str:
    """Extract reasoning content from LLM response."""
    import re
    
    # Look for reasoning block
    reasoning_match = re.search(r'```reasoning\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        return reasoning_match.group(1).strip()
    
    # Fallback: look for structured content
    patterns = [
        r'PATTERN OBSERVATION:(.*?)(?=TRANSFORMATION HYPOTHESIS:|$)',
        r'TRANSFORMATION HYPOTHESIS:(.*?)(?=VERIFICATION:|$)',
        r'VERIFICATION:(.*?)(?=CORE INSIGHT:|$)',
        r'CORE INSIGHT:(.*?)(?=$)'
    ]
    
    reasoning_parts = []
    for pattern in patterns:
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            reasoning_parts.append(match.group(1).strip())
    
    if reasoning_parts:
        return '\n\n'.join(reasoning_parts)
    
    # Ultimate fallback: return first substantial paragraph
    lines = response_text.split('\n')
    substantial_lines = [line.strip() for line in lines if len(line.strip()) > 20]
    
    return '\n'.join(substantial_lines[:10]) if substantial_lines else response_text[:500]


def generate_transformation_steps(llm, reasoning_trace: str, training_examples: List[Dict]) -> List[Dict]:
    """Extract clear step-by-step transformation from reasoning trace.

    Returns a list of solution objects: [{"solution_number": int, "transformation_steps": [str, ...]}, ...]
    """
    
    prompt = build_transformation_steps_prompt(reasoning_trace, training_examples)
    
    try:
        response = llm.invoke(prompt, temperature=0.7)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Print prompt and response with ANSI colors for easier reading in terminals
        # print_prompt_and_response(prompt, response)

        solutions = parse_transformation_steps(response_text)
        if solutions:
            return solutions
        return [{"solution_number": 1, "transformation_steps": ["Unable to extract transformation steps"]}]
        
    except Exception as e:
        print(f"Error extracting transformation steps: {e}")
        return [{"solution_number": 1, "transformation_steps": [f"Error in step extraction: {str(e)}"]}]


def build_transformation_steps_prompt(reasoning_trace: str, training_examples: List[Dict]) -> str:
    """Build prompt for extracting clear transformation steps."""
    # Build training examples block
    examples_block = ""
    for i, example in enumerate(training_examples, 1):
        examples_block += f"Training Example {i}\\n--\\n"
        examples_block += f"Input:\\n{format_grid_for_prompt(example['input'])}\\n\\n"
        examples_block += f"Output:\\n{format_grid_for_prompt(example['output'])}\\n\\n"
    prompt_parts = [
        "You are an expert mathematician, logistician and pattern recognizier who is solving the Abstract Reasoning Corpus (ARC) problems.",
        "Based on the following reasoning analysis, extract clear step-by-step transformation instructions.",

        "------------------",
        "TRAINING EXAMPLES",
        "------------------",
        f"{examples_block}",
        "",
        "------------------",
        "REASONING ANALYSIS",
        "------------------",
        f"{reasoning_trace}",
        "",
        "------------------",
        "INSTRUCTIONS",
        "------------------",
        "Produce 10 different candidate solutions. Each solution should be a numbered sequence of clear, actionable transformation steps.",
        "Be creative: try different, plausible interpretations of the reasoning so the set of solutions explores diverse approaches (use different object-level operations, orders, or heuristics).",
        "Each solution should be concise and concrete so it can be executed programmatically.",
        "",
        "RESPONSE FORMAT (JSON):",
        "Return a JSON array in a json block containing 10 solution objects. Each object should have two keys:",
        "- \"solution_number\": an integer (1..10)",
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
        "Do NOT output any additional text outside the ```json``` block. Generate the 10 solutions now."
    ]

    prompt = "\n".join(prompt_parts)

    return prompt


def parse_transformation_steps(response_text: str) -> List[str]:
    """Parse transformation steps from LLM response."""
    import re, json

    # Attempt 1: prefer a fenced ```json``` block containing the JSON array
    try:
        json_block_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
        candidate = None
        if json_block_match:
            candidate = json_block_match.group(1).strip()
        else:
            # Fallback: try to find a bare JSON array anywhere in the text
            start = response_text.find('[')
            end = response_text.rfind(']')
            if start != -1 and end != -1 and end > start:
                candidate = response_text[start:end+1]

        if candidate:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                solutions = []
                for item in parsed:
                    if isinstance(item, dict):
                        sol_num = item.get('solution_number') or item.get('solution') or item.get('solution_number')
                        steps = item.get('transformation_steps') or item.get('steps') or []
                        # Normalize steps to list of strings
                        if isinstance(steps, str):
                            step_lines = [ln.strip() for ln in steps.splitlines() if ln.strip()]
                            steps = [re.sub(r'^\d+\.\s*', '', ln).strip() for ln in step_lines]
                        elif isinstance(steps, list):
                            steps = [str(s).strip() for s in steps if str(s).strip()]
                        else:
                            steps = []

                        solutions.append({
                            "solution_number": int(sol_num) if (sol_num is not None and str(sol_num).isdigit()) else sol_num,
                            "transformation_steps": steps
                        })

                if solutions:
                    return solutions
    except Exception:
        pass

    # Fallback 1: look for fenced 'steps' block
    steps_match = re.search(r'```steps\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
    if steps_match:
        steps_content = steps_match.group(1).strip()
    else:
        steps_content = response_text

    # Try to split into multiple "Solution X" sections
    sol_splits = re.split(r'\bSolution\s*(\d+)\b', steps_content, flags=re.IGNORECASE)
    # re.split returns [before, num1, block1, num2, block2, ...] if matches
    if len(sol_splits) > 1:
        solutions = []
        # iterate pairs
        it = iter(sol_splits)
        pre = next(it)
        for token in it:
            try:
                num = token
                block = next(it)
            except StopIteration:
                break
            # extract numbered steps within block
            step_pattern = r'\d+\.\s*(.+)'
            matches = re.findall(step_pattern, block)
            steps = [m.strip() for m in matches]
            solutions.append({"solution_number": int(num) if num.isdigit() else num, "transformation_steps": steps})

        if solutions:
            return solutions

    # Fallback 2: extract any numbered steps across the text as a single solution
    step_pattern = r'(\d+\.\s*(.+))'
    matches = re.findall(step_pattern, steps_content, re.MULTILINE)
    if matches:
        steps = [match[1].strip() for match in matches]
        return [{"solution_number": 1, "transformation_steps": steps}]

    # Ultimate fallback: split long lines and return as single solution entries
    lines = [line.strip() for line in steps_content.split('\n') if line.strip()]
    steps = []
    for line in lines:
        cleaned_line = re.sub(r'^\d+\.\s*|^-\s*|^\*\s*', '', line).strip()
        if len(cleaned_line) > 5:
            steps.append(cleaned_line)

    return [{"solution_number": 1, "transformation_steps": steps[:50]}]


def generate_code_from_reasoning(llm, code_llm, reasoning_trace: str, transformation_steps: List[str],
                                 training_examples: List[Dict]) -> str:
    """Generate Python code based on reasoning trace and transformation steps.

    This function will request code from the LLM, then immediately try to execute
    the generated `transform(input_grid)` on the first training example (if present).
    If execution fails and a `code_llm` is provided, it will invoke that
    LLM up to 3 times to refine the main `transform` function and retry execution.
    The function returns the (possibly refined) Python source for the transform
    function (or a fallback template on failure).
    """

    def build_code_from_reasoning_prompt(reasoning_trace: str, transformation_steps: List[Dict],
                                         training_examples: List[Dict]) -> str:
        """Build prompt for generating Python code from reasoning and steps."""
        # Only support the new structured format: a list of solution dicts
        if not (transformation_steps and isinstance(transformation_steps, list) and isinstance(transformation_steps[0], dict)):
            raise ValueError("transformation_steps must be a list of solution dicts of the form {'solution_number': int, 'transformation_steps': [str,...]}")

        parts = []
        for sol in transformation_steps:
            sol_num = sol.get('solution_number', '?')
            parts.append(f"Solution {sol_num}:")
            for i, s in enumerate(sol.get('transformation_steps', []) or [], 1):
                parts.append(f"{i}. {s}")
            parts.append("")
        steps_text = '\n'.join(parts).strip()

        prompt_parts = [
            "You are a Python expert implementing ARC transformations.",
            "",
            "Given the following reasoning analysis and step-by-step transformation, implement a Python function.",
            "",
            "REASONING ANALYSIS:",
            f"{reasoning_trace}",
            "",
            "TRAINING EXAMPLES:",
            f"{len(training_examples)} input-output example pairs are provided for validation.",
            "",
            "",
            "TRANSFORMATION STEP SOLUTIONS:",
            f"{steps_text}",
            "",
            "IMPLEMENTATION REQUIREMENTS:",
            "For each solutions"
            "1. Write a function called 'transform(input_grid)' that takes a 2D list of integers as input and returns a transformed 2D list of integers",
            "2. Implement each transformation step clearly and precisely",
            "3. Import any necessary standard libraries at the top for EACH solution",
            "4. Include helper functions where necessary.",
            "5. DO NOT ADD ANY EXPLANATIONS OR COMMENTS IN THE CODE",
            "6. Address any error cases or edge conditions mentioned in the reasoning to ensure correctness and robustness",
            "7. Return ONLY executable Python code",
            "The <count> will help you keep track of what-th solution are you at. Make sure you have all solutions implemented."
            "",
            "Example structure:",
            "<count>1</count>",
            "<solution>",
            "from typing import List",
            "import ... # Import ANY necessary standard libraries to run the code here",
            "import ... # Import ANY necessary standard libraries to run the code here",
            "def helper_function_1(...):",
            "    # Add helper functions if neeeded",
            "def helper_function_2(...):",
            "    # Add helper functions if needed",
            "def transform(input_grid):",
            "    [implementations of transformation steps]",
            "    return transformed_grid",
            "</solution>",
            "<count>2</count>",
            "<solution>...</solution>",
            "..."
            "",
            "Generate the Python code now:"
        ]

        prompt = "\n".join(prompt_parts)

        return prompt

    prompt = build_code_from_reasoning_prompt(reasoning_trace, transformation_steps, 
                                              training_examples)

    try:
        response = llm.invoke(prompt, temperature=0.3)
        response_text = response.content if hasattr(response, 'content') else str(response)
        # print_prompt_and_response(prompt, response)

        # Extract candidate python solutions (may be multiple)
        candidate_codes = extract_python_solutions(response_text)
        # Ensure common imports are present in each candidate code block so
        # the probe can execute without trivial missing-import errors.
        candidate_codes = [ensure_imports_in_code(c) for c in candidate_codes]

        # No test input available — return all candidates
        return candidate_codes

    except Exception as e:
        print(f"Error generating code from reasoning: {e}")
        return [generate_fallback_code_from_steps(transformation_steps)]


def generate_fallback_code_from_steps(transformation_steps: List[Union[str, Dict]]) -> str:
    """Generate fallback Python code template from transformation steps."""
    
    code_lines = ["def transform(input_grid):"]
    code_lines.append("    # Copy input grid to work with")
    code_lines.append("    result = copy_grid(input_grid)")
    code_lines.append("    height, width = get_grid_dimensions(input_grid)")
    code_lines.append("")
    # If the new structured format (list of solution dicts) is provided,
    # use the first solution's steps for the fallback template.
    steps_list = []
    try:
        if transformation_steps and isinstance(transformation_steps[0], dict):
            steps_list = transformation_steps[0].get('transformation_steps', []) or []
        else:
            steps_list = transformation_steps or []
    except Exception:
        steps_list = transformation_steps or []

    for i, step in enumerate(steps_list, 1):
        s_text = str(step)
        code_lines.append(f"    # Step {i}: {s_text[:80]}{'...' if len(s_text) > 80 else ''}")
        code_lines.append(f"    # TODO: Implement step {i}")
        code_lines.append("")
    
    code_lines.append("    return result")
    
    return "\n".join(code_lines)


def ensure_imports_in_code(code: str) -> str:
    """Ensure common imports exist at top of a generated Python code string.

    This scans the provided `code` for usage of common modules and typing
    names and prepends import lines that are missing. The function performs
    a conservative, best-effort check and only adds imports for a small set
    of common utilities (typing, json, re, copy, itertools, collections,
    math, numpy).

    The function attempts to combine typing imports into a single
    `from typing import ...` line.
    """
    import re as _re

    if not code or not isinstance(code, str):
        return code

    # Find already-present import lines to avoid duplicates
    existing_imports = set()
    for m in _re.finditer(r'^\s*(?:from|import)\s+([^\n]+)', code, _re.MULTILINE):
        existing_imports.add(m.group(0).strip())

    # Map usage tokens -> import statements (conservative set)
    typing_tokens = {tok for tok in ("List", "Dict", "Any", "Tuple", "Optional", "Set") if _re.search(r'\b%s\b' % tok, code)}

    imports_needed = []
    if typing_tokens:
        typing_line = f"from typing import {', '.join(sorted(typing_tokens))}"
        if not any(l.startswith('from typing') for l in existing_imports):
            imports_needed.append(typing_line)

    token_map = [
        (r'\bjson\b', 'import json'),
        (r'\bre\b', 'import re'),
        (r'\bcopy\b', 'import copy'),
        (r'\bitertools\b', 'import itertools'),
        (r'\bdefaultdict\b|\bCounter\b', 'from collections import defaultdict, Counter'),
        (r'\bcollections\b', 'import collections'),
        (r'\bmath\b', 'import math'),
        (r'\bnp\.', 'import numpy as np'),
        (r'\bnumpy\b', 'import numpy as np'),
        (r'\bdataclass\b', 'from dataclasses import dataclass'),
    ]

    for pattern, imp in token_map:
        if _re.search(pattern, code) and not any(imp in s for s in existing_imports):
            imports_needed.append(imp)

    # If there are no imports to add, return original code
    if not imports_needed:
        return code

    # Prepend imports to the code, keeping a blank line separation
    header = "\n".join(imports_needed) + "\n\n"
    return header + code


def extract_python_solutions(response_text: str) -> List[str]:
    """Extract Python solution code blocks from LLM response and return a list of code strings.

    The LLM response is expected to contain multiple solutions in the following structure:

    <count>1</count>
    <solution>...</solution>
    <count>2</count>
    <solution>...</solution>
    ...

    This function returns a list of the extracted solution bodies (strings). If a
    `<solution>` block contains fenced python code, it will extract the inner
    python; otherwise the raw block text is returned (trimmed).
    """
    import re

    solutions: List[str] = []

    # 1) Prefer explicit <solution>...</solution> blocks
    sol_blocks = re.findall(r'<solution>(.*?)</solution>', response_text, re.DOTALL | re.IGNORECASE)
    if sol_blocks:
        for blk in sol_blocks:
            blk = blk.strip()
            # Use the raw content inside the <solution>...</solution> tags without further parsing.
            # This keeps the original block exactly as the LLM returned it.
            solutions.append(blk)

        return [s for s in solutions if s]


def execute_transformation_code(main_code: str,
                               input_grid: List[List[int]]) -> Tuple[Optional[List[List[int]]], Optional[str]]:
    """Execute the transformation code on an input grid.

    Returns:
        (result_grid, error_message)

    - `result_grid` is the transformed grid when execution succeeds, otherwise `None`.
    - `error_message` is `None` on success, otherwise contains the exception traceback or
      a short error description useful for refinement.
    """
    try:
        # Create execution namespace
        namespace = {"__builtins__": __builtins__}

        # Execute the main code
        # Use the shared debug helper to print the code (prints in YELLOW)
        # print_python_code(main_code)
        # Execute the main code
        exec(main_code, namespace)

        # Call the transform function
        if "transform" in namespace:
            try:
                result = namespace["transform"](input_grid)
                return result, None
            except Exception as inner_e:
                tb = traceback.format_exc()
                print(f"Error while running transform(): {inner_e}\n{tb}")
                return None, str(inner_e) + "\n" + tb
        else:
            err = "transform function not found in executed code"
            print(err)
            return None, err

    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error executing transformation code: {e}\n{tb}")
        return None, str(e) + "\n" + tb


def calculate_grid_results(predicted: List[List[int]], expected: List[List[int]]) -> Tuple[bool, float]:
    """Compare two 2D grids and return (size_match, value_match_percent).

    - size_match: True iff the predicted grid has the same dimensions as the
      expected grid (dimensions only; values are not considered).
    - value_match_percent: Percentage (0.0-100.0) of cells that match between
      the predicted and expected grids. The percentage is calculated relative
      to the expected grid's total cells. If the predicted grid is smaller or
      larger, non-overlapping cells count as mismatches.

    Args:
        predicted: 2D list representing the predicted output grid.
        expected: 2D list representing the expected output grid.

    Returns:
        (size_match, value_match_percent)
    """
    # Compute dimensions
    pred_h = len(predicted) if predicted is not None else 0
    pred_w = len(predicted[0]) if pred_h > 0 and predicted[0] else 0
    exp_h = len(expected) if expected is not None else 0
    exp_w = len(expected[0]) if exp_h > 0 and expected[0] else 0

    # First return value: size match (dimensions only)
    size_match = (pred_h == exp_h and pred_w == exp_w)

    # Calculate value match percentage relative to expected grid area
    total_cells = exp_h * exp_w
    if total_cells == 0:
        return (size_match, 0.0)

    matching_cells = 0
    for i in range(exp_h):
        for j in range(exp_w):
            if i < pred_h and j < pred_w:
                try:
                    if predicted[i][j] == expected[i][j]:
                        matching_cells += 1
                except Exception:
                    # Treat any comparison error as mismatch
                    pass
            else:
                # Out-of-range predicted cell counts as mismatch
                pass

    value_match_percent = (matching_cells / total_cells) * 100.0

    return (size_match, value_match_percent)


def evaluate_example(llm,
                     main_code: str,
                     transformation_steps: List[str],
                     input_grid: List[List[int]],
                     expected_output: List[List[int]],
                     enable_code_predict: bool = True,
                     enable_llm_predict: bool = True) -> Dict[str, Any]:
    """Evaluate a single training/test example.

    Runs the provided `main_code` on `input_grid`, computes grid comparison
    metrics against `expected_output`, and asks the LLM to apply the
    `transformation_steps` to the `input_grid` for a comparison baseline.

    Returns a result dict compatible with `nodes.test_code_node` usage.
    """
    # Execute the code only if enabled
    exec_predicted_output = None
    exec_error = None
    matching_size = False
    overlap_percentage = 0.0
    error_message = None
    code_success = False

    if enable_code_predict:
        try:
            exec_predicted_output, exec_error = execute_transformation_code(main_code, input_grid)
        except Exception as e:
            exec_predicted_output = None
            exec_error = str(e)

        # Compute code metrics (if execution produced an output)
        if exec_predicted_output is not None and exec_error is None:
            matching_size, overlap_percentage = calculate_grid_results(exec_predicted_output, expected_output)
        else:
            matching_size, overlap_percentage = False, 0.0
        error_message = exec_error or None
        code_success = bool(matching_size) and (overlap_percentage == 100.0)

    else:
        # Not executing code — leave defaults (no prediction)
        exec_predicted_output = None
        exec_error = None
        matching_size = False
        overlap_percentage = 0.0
        error_message = None
        code_success = False

    # Ask the LLM to apply the step-by-step transformation to the input only if enabled
    llm_predicted_output = None
    llm_error = None
    llm_matching_size = False
    llm_overlap_percentage = 0.0
    llm_error_message = None
    llm_success = False

    if enable_llm_predict:
        try:
            # transformation_steps is expected to be a dict with key 'transformation_steps' in our flow
            steps_for_llm = transformation_steps["transformation_steps"] if isinstance(transformation_steps, dict) and "transformation_steps" in transformation_steps else transformation_steps
            llm_predicted_output, llm_error = generate_llm_predicted_output(llm, steps_for_llm, input_grid)
        except Exception as e:
            llm_predicted_output = None
            llm_error = str(e)

        # Compute LLM-specific metrics (if LLM produced an output)
        if llm_predicted_output is not None and llm_error is None:
            llm_matching_size, llm_overlap_percentage = calculate_grid_results(llm_predicted_output, expected_output)
        else:
            llm_matching_size, llm_overlap_percentage = False, 0.0
        llm_error_message = llm_error or None
        llm_success = bool(llm_matching_size) and (llm_overlap_percentage == 100.0)
    else:
        llm_predicted_output = None
        llm_error = None
        llm_matching_size = False
        llm_overlap_percentage = 0.0
        llm_error_message = None
        llm_success = False

    result = {
        "input": input_grid,
        "expected_output": expected_output,
        "predicted_output": exec_predicted_output,
        "matching_size": matching_size,
        "overlap_percentage": overlap_percentage,
        "error_message": error_message,
        "code_success": code_success,
        "llm_predicted_output": llm_predicted_output,
        "llm_matching_size": llm_matching_size,
        "llm_overlap_percentage": llm_overlap_percentage,
        "llm_error_message": llm_error_message,
        "llm_success": llm_success,
    }

    return result


def analyze_failures(failed_tests: List[ExampleResult], training_examples: List[Dict]) -> Dict[str, Any]:
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


# def refine_solution_based_on_failures(llm, code_llm, 
#                                       current_solution: CodeSolution,
#                                       training_results: List[ExampleResult],
#                                       training_examples: List[Dict], helper_functions: Dict[str, HelperFunction]) -> CodeSolution:
#     """Refine the solution based on test failures using LLM with ARC-style reflection."""

#     # 1) Get reflection reasoning trace (structured analysis only)
#     reasoning = generate_reflection_reasoning_trace(llm, current_solution, training_results, training_examples)

#     # 2) Extract transformation steps from the reasoning
#     transformation_steps = generate_transformation_steps(llm, reasoning, training_examples)

#     # 3) Generate refined code from reasoning + steps
#     refined_codes = generate_code_from_reasoning(llm, code_llm, reasoning, transformation_steps, training_examples, helper_functions)

#     # refined_codes may be a list of candidates; pick the first candidate that
#     # looks like a valid transform function, otherwise fall back to the first.
#     chosen_code = None
#     if isinstance(refined_codes, list):
#         for c in refined_codes:
#             if isinstance(c, str) and 'def transform(' in c:
#                 chosen_code = c
#                 break
#         if chosen_code is None and refined_codes:
#             chosen_code = refined_codes[0]
#     else:
#         chosen_code = refined_codes

#     # If code generation succeeded, return the refined solution
#     if chosen_code and isinstance(chosen_code, str) and 'def transform(' in chosen_code:
#         refined_solution = {
#             "main_code": chosen_code,
#             "helper_functions": helper_functions,
#             "reasoning_trace": reasoning,
#             "step_by_step_transformation": transformation_steps or ["Refined based on failure analysis"],
#         }

#         return refined_solution
#     else:
#         # Return the original solution unchanged
#         return current_solution


def extract_reasoning_from_reflection(response_content: str) -> str:
    """Extract reasoning section from ARC-style reflection response."""
    import re
    
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
    import re
    
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


def format_grid_for_prompt(grid: List[List[int]]) -> str:
    """Format grid for display in prompts."""
    return "\\n".join(" ".join(map(str, row)) for row in grid)