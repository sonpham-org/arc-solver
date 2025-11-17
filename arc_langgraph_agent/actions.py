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
from .schema import AgentState, CodeSolution, ExampleResult, HelperFunction
from .tools import FUNCTION_MAP
from .debug import print_prompt_and_response, print_python_code


def generate_llm_predicted_output(llm,
                                  transformation_steps: List[str],
                                  input_grid: List[List[int]]) -> Tuple[Optional[List[List[int]]], Optional[str]]:
    """Use the LLM to apply the step-by-step transformation to a single input grid.

    The LLM is instructed to return the transformed grid as a JSON array
    (list of lists of integers). Returns (grid, None) on success, or
    (None, error_message) on failure.
    """
    try:
        steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(transformation_steps)) if transformation_steps else "(no steps provided)"

        prompt_parts = [
            "You are an expert that can execute step-by-step grid transformations.",
            "Given the following input grid and transformation steps, apply the steps and return the resulting grid.",
            "Return your application in a ```transformation``` block and the 2D grid inside a fenced block labelled ```llm_predicted_output``` containing only the grid rows as lines of numbers (space-separated or contiguous digits).",
            "Example:",
            "```llm_predicted_output",
            "1 0 0",
            "0 2 2",
            "```",
            "Do NOT return any other text after that block.",
            "",
            "INPUT GRID:",
            format_grid_for_prompt(input_grid),
            "",
            "TRANSFORMATION STEPS:",
            steps_text,
            "",
        ]

        prompt = "\n".join(prompt_parts)

        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        print_prompt_and_response(prompt, response)

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


def generate_helper_functions(training_examples: List[Dict], existing_helpers: List[HelperFunction]) -> List[HelperFunction]:
    """Generate helper functions that might be useful for the task."""
    return []  # Simplified - no longer generating new helper functions


def extract_helper_functions(llm, code_llm, main_code: str, training_examples: List[Dict], 
                            existing_library: List[HelperFunction]) -> List[HelperFunction]:
    """Use LLM to extract useful helper functions from generated code."""
    
    prompt = build_helper_extraction_prompt(main_code, training_examples, existing_library)
    
    try:
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        print_prompt_and_response(prompt, response)
        
        # Parse extracted helpers from response
        extracted_helpers = parse_helper_functions_response(response_text)

        # Try executing each extracted helper to ensure it runs. If execution fails
        # and a `code_llm` is provided, try to refine the helper using the
        # error traceback and re-attempt execution (up to 3 attempts).
        for helper in extracted_helpers:
            helper_code = helper.get("code", "")
            helper_name = helper.get("name", "<unknown>")
            success = False
            attempts = 0
            last_err = None

            while attempts < 3 and not success:
                attempts += 1
                try:
                    exec_namespace = {"__builtins__": __builtins__}
                    exec(helper_code, exec_namespace)
                    # If execution succeeds, keep the (possibly corrected) code
                    # helper["_exec_namespace"] = exec_namespace
                    success = True
                except Exception:
                    tb = traceback.format_exc()
                    last_err = tb
                    print(f"Execution of helper '{helper_name}' failed on attempt {attempts}: {tb}")

                    # If we don't have a code_llm to refine, stop retrying
                    if code_llm is None:
                        break

                    # Ask the `code_llm` to fix the helper code. Provide the
                    # original helper and the traceback. Request only corrected
                    # Python code in the response.
                    refine_prompt = (
                        f"The following Python helper function (or helper code) failed to execute when run.\n"
                        f"Helper name: {helper_name}\n\n"
                        f"ORIGINAL HELPER CODE:\n```python\n{helper_code}\n```\n\n"
                        f"ERROR TRACEBACK:\n{tb}\n\n"
                        "Please provide a corrected version of the helper code only (no explanations). "
                        "Include any missing imports and keep function names where possible. Return only valid Python code."
                    )

                    try:
                        response = code_llm.invoke(refine_prompt)
                        response_text = response.content if hasattr(response, 'content') else str(response)
                        print_prompt_and_response(refine_prompt, response)

                        # Try to extract python from the LLM response, fall back to raw
                        fixed_code = extract_python_code(response_text) or response_text

                        # Update helper code for next attempt
                        helper_code = fixed_code
                        helper["code"] = helper_code
                    except Exception as llm_e:
                        print(f"Error invoking code_llm for helper refinement: {llm_e}")
                        break

            if not success:
                print(f"Failed to execute helper '{helper_name}' after {attempts} attempts. Last error:\n{last_err}")

        return extracted_helpers
        
    except Exception as e:
        print(f"Error extracting helper functions: {e}")
        return []


def build_helper_extraction_prompt(main_code: str, training_examples: List[Dict],
                                 existing_library: List[HelperFunction]) -> str:
    """Build prompt for extracting reusable helper functions."""
    
    existing_names = [h["name"] for h in existing_library]
    examples_summary = f"Based on {len(training_examples)} training examples involving grid transformations."

    prompt = (
            f"""You are an expert at creating reusable helper functions for ARC problem solving.

Analyze this transformation code and extract 2-4 useful helper functions that could be reused across different ARC tasks.

GENERATED CODE:
```python
{main_code}
```

CONTEXT:
{examples_summary}

EXISTING HELPER LIBRARY:
{', '.join(existing_names) if existing_names else 'None yet'}

EXTRACTION GUIDELINES:
1. Extract logical units that could be useful across different ARC tasks
2. Focus on grid operations, pattern detection, geometric transformations
3. Avoid helpers that are too specific to this exact task
4. Don't duplicate existing helpers: {existing_names}
5. Make functions self-contained with necessary imports

Return your response as a JSON list:
[
    {{
        "name": "function_name",
        "code": "def function_name(params):\n    # implementation\n    return result",
        "description": "What this function does and when to use it",
        "parameters": ["param1", "param2"]
    }}
]

Extract 2-4 the most valuable helper functions:""")
    return prompt


def parse_helper_functions_response(response_text: str) -> List[HelperFunction]:
    """Parse helper functions from LLM response."""
    import json
    import re
    
    try:
        # Look for JSON array in response
        json_match = re.search(r'\[\s*\{.*?\}\s*\]', response_text, re.DOTALL)
        if json_match:
            helpers_data = json.loads(json_match.group())
            
            helpers = []
            for helper_data in helpers_data:
                if all(key in helper_data for key in ["name", "code", "description"]):
                    helper = {
                        "name": helper_data["name"],
                        "code": helper_data["code"],
                        "description": helper_data["description"],
                        "parameters": helper_data.get("parameters", [])
                    }
                    helpers.append(helper)
            
            return helpers
    
    except Exception as e:
        print(f"Error parsing helper functions: {e}")
    
    return []


def generate_python_transformation_code_with_reasoning(llm, code_llm, training_examples: List[Dict], 
                                                       helper_functions: List[HelperFunction],
                                                       previous_solutions: List[CodeSolution]) -> Tuple[str, str, List[str]]:
    """Generate Python transformation code using reasoning-first approach.
    
    Returns:
        Tuple of (python_code, reasoning_trace, transformation_steps)
    """
    
    # Step 1: Generate reasoning trace
    reasoning_trace = generate_reasoning_trace(llm, training_examples, helper_functions, previous_solutions)
    
    # Step 2: Extract step-by-step transformation from reasoning
    transformation_steps = generate_transformation_steps(llm, reasoning_trace, training_examples)
    
    # Step 3: Generate Python code based on reasoning and steps
    python_code = generate_code_from_reasoning(llm, code_llm, reasoning_trace, transformation_steps, 
                                                training_examples, helper_functions)
    
    return python_code, reasoning_trace, transformation_steps


def generate_reasoning_trace(llm, training_examples: List[Dict], 
                           helper_functions: List[HelperFunction],
                           previous_solutions: List[CodeSolution]) -> str:
    """Generate detailed reasoning trace analyzing ARC patterns."""

    def build_initial_reasoning_prompt(training_examples: List[Dict],
                                   helper_functions: Dict[str, HelperFunction]) -> str:
        """Build prompt for generating detailed reasoning about ARC patterns."""
        
        prompt_parts = [
            "You are an expert mathematician, logistician and pattern recognizier who is solving the"
            "Abstract Reasoning Corpus (ARC) problems.",
            "Your task is to deeply analyze the input-output examples and understand the underlying pattern.",
            "Focus on identifying the core transformation rule that maps inputs to outputs.",
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
            "Provide a ```reasoning``` block that contains the following information.",
            "",
            "```reasoning",
            "PATTERN OBSERVATION:",
            "- What do you notice about the relationship between inputs and outputs?",
            "- What patterns in movement, color changes, shape tranformations, layering, rotation, create, delete (etc.) that stand out?",
            "- What changes between input and output grids?",
            "- Are there consistent elements, colors, shapes, or spatial relationships?",
            "",
            "TRANSFORMATION HYPOTHESIS:",
            "- What is the core rule that transforms input to output?",
            "- How do objects move, transform shape, layer, rotate. How do colors change? How are patterns form and delete.",
            "- Are there geometric transformations (rotation, reflection, scaling)? Are there visual transformations? Are there physical transformations?",
            "",
            "VERIFICATION:",
            "- Does your hypothesis work for ALL training examples?",
            "- What edge cases or special conditions exist?",
            "",
            "CORE INSIGHT:",
            "- What is the single most important insight about this transformation?",
            "- How would you explain this pattern in simple terms?",
            "```",
        ])
        
        return "\n".join(prompt_parts)
    
    try:
        prompt = build_initial_reasoning_prompt(training_examples, helper_functions)
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        print_prompt_and_response(prompt, response)

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

            "Your goal:",
            "Analyze your previous attempt deeply, understand why it failed, and provide a CORRECTED reasoning rule that:",
            "1. Correctly maps every training input to its output",
            "2. Is general and intuitive (no memorization or hard-coded values)",
            "3. Is logical, reproducible, and object-level",
            "",
            "ORIGINAL TRAINING EXAMPLES",
            "",
            f"{examples_block}",
            "",
            "YOUR PREVIOUS SOLUTION",
            "```python",
            f"{previous_code}",
            "```",
            "",
            "DETAILED FAILURE ANALYSIS",
            "",
            f"{failures_block}",
            "",
            "",
            "ANALYSIS INSTRUCTIONS:",
            "",
            "Provide a ```reasoning``` block that contains the following information.",
            "",
            "```reasoning",
            "LOGIC ERRORS",
            "- Where exactly did your transformation logic fail?",
            "- Is your failure just due to missing edge cases or flawed steps? Or is there a fundamental misunderstanding of the pattern?",
            ""
            "CODE ERRORS",
            "- Were there any coding mistakes, bugs, or mis-implementations that caused incorrect outputs or buggy error messages?",
            "",
            "PATTERN MISINTERPRETATION",
            "- What pattern did you miss or misunderstand?",
            "- What objects, shapes, colors, spatial or movement relationships did you fail to account for?",
            "",
            "CORE INSIGHT: "
            "What is the insight you're missing?",
            "```",
            "",
            "Return ONLY the ```reasoning``` block. DO NOT include executable Python code in your response."
        ]

        prompt = "\n".join(prompt_parts)

        return prompt
    try:
        prompt = build_refinement_reasoning_prompt(current_solution, training_results, training_examples)
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        print_prompt_and_response(prompt, response)

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


def generate_transformation_steps(llm, reasoning_trace: str, training_examples: List[Dict]) -> List[str]:
    """Extract clear step-by-step transformation from reasoning trace."""
    
    prompt = build_transformation_steps_prompt(reasoning_trace, training_examples)
    
    try:
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Print prompt and response with ANSI colors for easier reading in terminals
        print_prompt_and_response(prompt, response)

        steps = parse_transformation_steps(response_text)
        return steps if steps else ["Unable to extract transformation steps"]
        
    except Exception as e:
        print(f"Error extracting transformation steps: {e}")
        return [f"Error in step extraction: {str(e)}"]


def build_transformation_steps_prompt(reasoning_trace: str, training_examples: List[Dict]) -> str:
    """Build prompt for extracting clear transformation steps."""
    prompt_parts = [
        "You are an expert mathematician, logistician and pattern recognizier who is solving the Abstract Reasoning Corpus (ARC) problems.",
        "Based on the following reasoning analysis, extract clear step-by-step transformation instructions.",
        "------------------",
        "REASONING ANALYSIS",
        "------------------",
        f"{reasoning_trace}",
        "",
        "------------------",
        "INSTRUCTIONS",
        "------------------",
        "Extract the transformation as a numbered list of clear, actionable steps.",
        "Each step should be implementable in code and describe exactly what to do.",
        "",
        "Format your response as:",
        "```steps",
        "1. [First transformation step]",
        "2. [Second transformation step]",
        "3. [Third transformation step]",
        "...",
        "```",
        "",
        "Generate the step-by-step transformation:"
    ]

    prompt = "\n".join(prompt_parts)

    return prompt


def parse_transformation_steps(response_text: str) -> List[str]:
    """Parse transformation steps from LLM response."""
    import re
    
    # Look for steps block
    steps_match = re.search(r'```steps\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
    if steps_match:
        steps_content = steps_match.group(1).strip()
    else:
        steps_content = response_text
    
    # Extract numbered steps
    step_pattern = r'(\d+\.\s*(.+))'
    matches = re.findall(step_pattern, steps_content, re.MULTILINE)
    
    if matches:
        return [match[1].strip() for match in matches]
    
    # Fallback: split by lines and look for step-like content
    lines = [line.strip() for line in steps_content.split('\n') if line.strip()]
    steps = []
    
    for line in lines:
        # Remove leading numbers/bullets
        cleaned_line = re.sub(r'^\d+\.\s*|^-\s*|^\*\s*', '', line).strip()
        if len(cleaned_line) > 10:  # Filter out very short lines
            steps.append(cleaned_line)
    
    return steps[:10]  # Limit to reasonable number of steps


def generate_code_from_reasoning(llm, code_llm, reasoning_trace: str, transformation_steps: List[str],
                                 training_examples: List[Dict], helper_functions: Dict[str, HelperFunction]) -> str:
    """Generate Python code based on reasoning trace and transformation steps.

    This function will request code from the LLM, then immediately try to execute
    the generated `transform(input_grid)` on the first training example (if present).
    If execution fails and a `code_llm` is provided, it will invoke that
    LLM up to 3 times to refine the main `transform` function and retry execution.
    The function returns the (possibly refined) Python source for the transform
    function (or a fallback template on failure).
    """

    def build_code_from_reasoning_prompt(reasoning_trace: str, transformation_steps: List[str],
                                   training_examples: List[Dict], helper_functions: Dict[str, HelperFunction]) -> str:
        """Build prompt for generating Python code from reasoning and steps."""
        steps_text = '\n'.join(f"{i+1}. {step}" for i, step in enumerate(transformation_steps))

        available_helpers = ""
        if helper_functions:
            available_helpers = "\nAVAILABLE HELPER FUNCTIONS:\n"
            for func in list(helper_functions.values())[:10]:
                available_helpers += f"- {func['name']}: {func['description']}\n"

        prompt_parts = [
            "You are a Python expert implementing ARC transformations.",
            "",
            "Given the following reasoning analysis and step-by-step transformation, implement a Python function.",
            "",
            "REASONING ANALYSIS:",
            f"{reasoning_trace}",
            "",
            "TRANSFORMATION STEPS:",
            f"{steps_text}",
            "",
            "AVAILABLE HELPERS:",
            f"{available_helpers}",
            "",
            "TRAINING EXAMPLES:",
            f"{len(training_examples)} input-output example pairs are provided for validation.",
            "",
            "IMPLEMENTATION REQUIREMENTS:",
            "1. Write a function called 'transform(input_grid)' that takes a 2D list of integers as input and returns a transformed 2D list of integers",
            "2. Implement each transformation step clearly and precisely",
            "3. Import any necessary standard libraries at the top",
            "4. Use available helper functions where appropriate. If needed, define new helper functions within the code.",
            "5. DO NOT ADD ANY EXPLANATIONS OR COMMENTS IN THE CODE",
            "6. Address any error cases or edge conditions mentioned in the reasoning to ensure correctness and robustness",
            "7. Return ONLY executable Python code",
            "",
            "Example structure:",
            "```python",
            "",
            "import ...",
            "",
            "def helper_function_1(...):",
            "    # Add helper functions if neeeded",
            "",
            "def helper_function_2(...):",
            "    # Add helper functions if needed",
            "",
            "def transform(input_grid: List[List[int]]) -> List[List[int]]:",
            "    [implementations of transformation steps]",
            "    return transformed_grid",
            "```",
            "",
            "Generate the Python code implementing the transformation:"
        ]

        prompt = "\n".join(prompt_parts)

        return prompt

    prompt = build_code_from_reasoning_prompt(reasoning_trace, transformation_steps, 
                                            training_examples, helper_functions)

    try:
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        print_prompt_and_response(prompt, response)

        # Extract and validate Python code
        python_code = extract_and_validate_python_code(response_text)
        print_python_code(python_code)

        if not python_code:
            return generate_fallback_code_from_steps(transformation_steps)

        # Prepare a test input (first training example) if available
        test_input = None
        if training_examples and len(training_examples) > 0:
            test_input = training_examples[0].get('input')

        # Normalize helper functions to a list
        helper_list = []
        if helper_functions:
            if isinstance(helper_functions, dict):
                helper_list = list(helper_functions.values())
            elif isinstance(helper_functions, list):
                helper_list = helper_functions
            else:
                try:
                    helper_list = list(helper_functions)
                except Exception:
                    helper_list = []

        # Execute generated code once; if it fails and a code_refinement_llm is provided,
        # attempt up to 3 refinement iterations to fix the main transform function.
        if test_input is not None:
            result, err = execute_transformation_code(python_code, test_input, helper_list)
            if err is None:
                return python_code

            print(f"Initial execution failed: {err}")

            if code_llm is not None:
                last_err = None
                for attempt in range(1, 4):
                    try:
                        prompt = (
                            "The following Python transformation function `transform(input_grid)` failed when run.\n"
                            "Please produce a corrected version of the `transform` function only (no explanations),\n"
                            "preserving the function name and intent where possible. Include any missing imports.\n\n"
                            "--- ORIGINAL TRANSFORM FUNCTION ---\n"
                            f"{python_code}\n\n"
                            "--- HELPER FUNCTIONS (for context) ---\n"
                            f"{('\n\n'.join(h.get('code','') for h in helper_list))}\n\n"
                            "--- ERROR TRACEBACK ---\n"
                            f"{err}\n\n"
                            f"Attempt: {attempt} of 3.\n\n"
                            "Return ONLY the fixed `transform` function wrapped in triple backticks (```)."
                        )

                        response = code_llm.invoke(prompt)
                        response_text = response.content if hasattr(response, 'content') else str(response)

                        new_code = extract_python_code(response_text)
                        if new_code:
                            # Try executing the refined code
                            result2, err2 = execute_transformation_code(new_code, test_input, helper_list)
                            if err2 is None:
                                print("Code refinement succeeded on attempt", attempt)
                                print_python_code(new_code)
                                return new_code
                            else:
                                print(f"Refined code attempt {attempt} failed: {err2}")
                                last_err = err2
                                python_code = new_code  # try using refined code in next attempt
                                continue
                        else:
                            print(f"code_llm attempt {attempt} did not return code. Response:\n{response_text}")
                            last_err = f"No code returned on attempt {attempt}."
                            continue

                    except Exception as llm_e:
                        tb_llm = traceback.format_exc()
                        print(f"Error calling code_llm on attempt {attempt}: {llm_e}\n{tb_llm}")
                        last_err = str(llm_e) + "\n" + tb_llm
                        continue

                # After retries
                print(f"Code refinement failed after 3 attempts: {last_err}")
                return python_code

            else:
                return python_code

        # No test input available — just return the generated code
        return python_code

    except Exception as e:
        print(f"Error generating code from reasoning: {e}")
        return generate_fallback_code_from_steps(transformation_steps)


def generate_fallback_code_from_steps(transformation_steps: List[str]) -> str:
    """Generate fallback Python code template from transformation steps."""
    
    code_lines = ["def transform(input_grid):"]
    code_lines.append("    # Copy input grid to work with")
    code_lines.append("    result = copy_grid(input_grid)")
    code_lines.append("    height, width = get_grid_dimensions(input_grid)")
    code_lines.append("")
    
    for i, step in enumerate(transformation_steps, 1):
        code_lines.append(f"    # Step {i}: {step[:80]}{'...' if len(step) > 80 else ''}")
        code_lines.append(f"    # TODO: Implement step {i}")
        code_lines.append("")
    
    code_lines.append("    return result")
    
    return "\n".join(code_lines)


def extract_and_validate_python_code(response_text: str) -> str:
    """Extract and validate Python code from LLM response."""
    import re
    
    # First try to find code in markdown blocks
    code_blocks = re.findall(r'```python\n(.*?)```', response_text, re.DOTALL)
    if code_blocks:
        code = code_blocks[0].strip()
        if 'def transform(' in code:
            return code
    
    # Try to find code without markdown
    code_blocks = re.findall(r'```\n(.*?)```', response_text, re.DOTALL)
    if code_blocks:
        code = code_blocks[0].strip()
        if 'def transform(' in code:
            return code
    
    # Look for function definition directly
    lines = response_text.split('\n')
    code_lines = []
    in_function = False
    
    for line in lines:
        if 'def transform(' in line:
            in_function = True
            code_lines.append(line)
        elif in_function:
            if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                break  # End of function
            code_lines.append(line)
    
    if code_lines and 'def transform(' in code_lines[0]:
        return '\n'.join(code_lines)
    
    return ""


def extract_python_code(response_text: str) -> str:
    """Extract Python code from LLM response."""
    import re
    
    # Look for code blocks with python tag
    code_blocks = re.findall(r'```python\n(.*?)```', response_text, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    
    # Look for code blocks without tag
    code_blocks = re.findall(r'```\n(.*?)```', response_text, re.DOTALL)
    if code_blocks:
        code = code_blocks[0].strip()
        if 'def transform(' in code:
            return code
    
    # Look for inline code blocks
    code_blocks = re.findall(r'```(.*?)```', response_text, re.DOTALL)
    if code_blocks:
        for code in code_blocks:
            if 'def transform(' in code:
                return code.strip()
    
    # Look for function definition directly in text
    lines = response_text.split('\n')
    code_lines = []
    in_function = False
    indent_level = 0
    
    for line in lines:
        if 'def transform(' in line:
            in_function = True
            indent_level = len(line) - len(line.lstrip())
            code_lines.append(line)
        elif in_function:
            current_indent = len(line) - len(line.lstrip())
            # Continue if line is indented more than function definition or is empty
            if line.strip() == '' or current_indent > indent_level:
                code_lines.append(line)
            elif line.strip() and current_indent <= indent_level:
                # End of function
                break
    
    if code_lines:
        return '\n'.join(code_lines)
    
    return ""


def execute_transformation_code(main_code: str,
                               input_grid: List[List[int]],
                               helper_functions: Union[List[HelperFunction], Dict[str, HelperFunction]]) -> Tuple[Optional[List[List[int]]], Optional[str]]:
    """Execute the transformation code on an input grid.

    Returns:
        (result_grid, error_message)

    - `result_grid` is the transformed grid when execution succeeds, otherwise `None`.
    - `error_message` is `None` on success, otherwise contains the exception traceback or
      a short error description useful for refinement.
    """
    try:
        # Normalize helper_functions to a list of helper dicts or code strings
        helper_list = []
        if helper_functions is None:
            helper_list = []
        elif isinstance(helper_functions, dict):
            try:
                helper_list = list(helper_functions.values())
            except Exception:
                helper_list = []
        elif isinstance(helper_functions, list):
            helper_list = helper_functions
        else:
            try:
                helper_list = list(helper_functions)
            except Exception:
                helper_list = []

        # Create execution namespace
        namespace = {"__builtins__": __builtins__}

        # Add helper functions to namespace. Support helpers as:
        # - dict-like with a "code" key
        # - objects with a .get or attribute access
        # - raw strings containing Python code
        for helper in helper_list:
            try:
                # If helper is a string, assume it's Python source
                if isinstance(helper, str):
                    exec(helper, namespace)
                # If helper is a dict-like with a "code" key
                elif isinstance(helper, dict) and "code" in helper:
                    exec(helper["code"], namespace)
                # If helper has a 'code' attribute (e.g., TypedDict-like object)
                elif hasattr(helper, "get") and helper.get("code"):
                    exec(helper.get("code"), namespace)
                elif hasattr(helper, "code"):
                    exec(getattr(helper, "code"), namespace)
                else:
                    # Last resort: try to exec the helper directly
                    exec(helper, namespace)
            except Exception:
                # Continue loading other helpers even if one fails; the
                # error will surface later when running transform.
                tb = traceback.format_exc()
                print(f"Error executing helper during setup: {tb}")

        # Add built-in helper functions
        for name, func in FUNCTION_MAP.items():
            namespace[name] = func

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


def refine_solution_based_on_failures(llm, code_llm, 
                                      current_solution: CodeSolution,
                                      training_results: List[ExampleResult],
                                      training_examples: List[Dict], helper_functions: Dict[str, HelperFunction]) -> CodeSolution:
    """Refine the solution based on test failures using LLM with ARC-style reflection."""

    # 1) Get reflection reasoning trace (structured analysis only)
    reasoning = generate_reflection_reasoning_trace(llm, current_solution, training_results, training_examples)

    # 2) Extract transformation steps from the reasoning
    transformation_steps = generate_transformation_steps(llm, reasoning, training_examples)

    # 3) Generate refined code from reasoning + steps
    refined_code = generate_code_from_reasoning(llm, code_llm, reasoning, transformation_steps, training_examples, helper_functions)

    # If code generation succeeded, return the refined solution
    if refined_code and 'def transform(' in refined_code:
        refined_solution = {
            "main_code": refined_code,
            "helper_functions": helper_functions,
            "reasoning_trace": reasoning,
            "step_by_step_transformation": transformation_steps or ["Refined based on failure analysis"],
        }

        return refined_solution
    else:
        # Return the original solution unchanged
        return current_solution


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