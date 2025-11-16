"""
Action helpers for the ARC LangGraph Agent workflow.

This module contains helper functions, prompt builders, LLM-driven
generation utilities, execution helpers, and refinement logic.
"""

import json
import copy
from typing import List, Dict, Optional, Any, TypedDict, Tuple
import traceback
import sys
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
import re

# Import schema and tools
from .schema import AgentState, CodeSolution, ExampleResult, HelperFunction
from .tools import FUNCTION_MAP

# ANSI color codes for terminal output (used for debugging LLM prompts/responses)
BLUE = "\033[34m"
RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"


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


def extract_helper_functions(llm, main_code: str, training_examples: List[Dict], 
                            existing_library: List[HelperFunction]) -> List[HelperFunction]:
    """Use LLM to extract useful helper functions from generated code."""
    
    prompt = build_helper_extraction_prompt(main_code, training_examples, existing_library)
    
    try:
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        print(f"{BLUE}--- Helper Extraction LLM Prompt ---{RESET}")
        print(f"{BLUE}{prompt}{RESET}")
        print(f"{GREEN}--- Helper Extraction LLM Response ---{RESET}")
        print(f"{GREEN}{response_text}{RESET}")
        
        # Parse extracted helpers from response
        extracted_helpers = parse_helper_functions_response(response_text)
        
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


def generate_python_transformation_code_with_reasoning(llm, training_examples: List[Dict], 
                                                       helper_functions: List[HelperFunction],
                                                       previous_solutions: List[CodeSolution]) -> Tuple[str, str, List[str]]:
    """Generate Python transformation code using reasoning-first approach.
    
    Returns:
        Tuple of (python_code, reasoning_trace, transformation_steps)
    """
    
    try:
        # Step 1: Generate reasoning trace
        reasoning_trace = generate_reasoning_trace(llm, training_examples, helper_functions, previous_solutions)
        
        # Step 2: Extract step-by-step transformation from reasoning
        transformation_steps = extract_transformation_steps(llm, reasoning_trace, training_examples)
        
        # Step 3: Generate Python code based on reasoning and steps
        python_code = generate_code_from_reasoning(llm, reasoning_trace, transformation_steps, 
                                                  training_examples, helper_functions)
        
        return python_code, reasoning_trace, transformation_steps
        
    except Exception as e:
        print(f"Error generating code with reasoning: {e}")
        # Fallback to original approach
        fallback_code = generate_main_transformation_code(training_examples, helper_functions, previous_solutions)
        return fallback_code, "Error in reasoning generation", ["Fallback transformation"]


def generate_reasoning_trace(llm, training_examples: List[Dict], 
                           helper_functions: List[HelperFunction],
                           previous_solutions: List[CodeSolution]) -> str:
    """Generate detailed reasoning trace analyzing ARC patterns."""
    
    try:
        prompt = build_initial_reasoning_prompt(training_examples, helper_functions)
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)

        # Print prompt and response with ANSI colors for easier reading in terminals
        try:
            print(f"{BLUE}--- LLM Prompt ---{RESET}")
            print(f"{BLUE}{prompt}{RESET}")
            print(f"{GREEN}--- LLM Response ---{RESET}")
            print(f"{GREEN}{response_text}{RESET}")
        except Exception:
            # If printing fails for any reason, silently continue
            pass
        
        # Extract reasoning from response
        reasoning = extract_reasoning_content(response_text)
        
        return reasoning if reasoning else "Unable to generate reasoning trace"
        
    except Exception as e:
        print(f"Error generating reasoning trace: {e}")
        return f"Error in reasoning generation: {str(e)}"


def generate_reflection_reasoning_trace(llm,
                                       current_solution: CodeSolution,
                                       failed_tests: List[ExampleResult],
                                       training_examples: List[Dict],
                                       reflection_history: List[Dict] = None) -> str:
    """Generate a reflection-focused reasoning trace using the ARC-style reflection prompt.

    This is intended for refinement: it asks the model to analyze failures, explain
    what went wrong, and produce a reasoning trace focused on correcting the logic.
    """
    try:
        prompt = build_refinement_reasoning_prompt(current_solution, failed_tests, training_examples, reflection_history or [])
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)

        # Prefer structured reflection extraction first
        reasoning = extract_reasoning_from_reflection(response_text)
        if reasoning and reasoning != "No structured reasoning found in response":
            return reasoning

        # Fallback to extracting a substantial analysis from the response
        reasoning = extract_reasoning_content(response_text)
        return reasoning if reasoning else "Unable to generate reflection reasoning trace"

    except Exception as e:
        print(f"Error generating reflection reasoning trace: {e}")
        return f"Error in reflection reasoning generation: {str(e)}"


def build_initial_reasoning_prompt(training_examples: List[Dict],
                                   helper_functions: Dict[str, HelperFunction]) -> str:
    """Build prompt for generating detailed reasoning about ARC patterns."""
    
    prompt_parts = [
        "You are an expert at analyzing Abstract Reasoning Corpus (ARC) problems.",
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
        "Provide a detailed reasoning trace in the following structure:",
        "",
        "```reasoning",
        "PATTERN OBSERVATION:",
        "- What do you notice about the relationship between inputs and outputs?",
        "- What changes between input and output grids?",
        "- Are there consistent elements, colors, shapes, or spatial relationships?",
        "",
        "TRANSFORMATION HYPOTHESIS:",
        "- What is the core rule that transforms input to output?",
        "- How do objects, colors, or patterns change?",
        "- Are there geometric transformations (rotation, reflection, scaling)?",
        "",
        "VERIFICATION:",
        "- Does your hypothesis work for ALL training examples?",
        "- What edge cases or special conditions exist?",
        "",
        "CORE INSIGHT:",
        "- What is the single most important insight about this transformation?",
        "- How would you explain this pattern in simple terms?",
        "```",
        "",
        "Generate your detailed reasoning trace now:"
    ])
    
    return "\n".join(prompt_parts)


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


def extract_transformation_steps(llm, reasoning_trace: str, training_examples: List[Dict]) -> List[str]:
    """Extract clear step-by-step transformation from reasoning trace."""
    
    prompt = build_transformation_steps_prompt(reasoning_trace, training_examples)
    
    try:
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Print prompt and response with ANSI colors for easier reading in terminals
        try:
            print(f"{BLUE}--- Transformation Steps Prompt ---{RESET}")
            print(f"{BLUE}{prompt}{RESET}")
            print(f"{GREEN}--- Transformation Steps Response ---{RESET}")
            print(f"{GREEN}{response_text}{RESET}")
        except Exception:
            pass

        steps = parse_transformation_steps(response_text)
        
        return steps if steps else ["Unable to extract transformation steps"]
        
    except Exception as e:
        print(f"Error extracting transformation steps: {e}")
        return [f"Error in step extraction: {str(e)}"]


def build_transformation_steps_prompt(reasoning_trace: str, training_examples: List[Dict]) -> str:
    """Build prompt for extracting clear transformation steps."""
    
    prompt = f"""Based on the following reasoning analysis, extract clear step-by-step transformation instructions.

REASONING ANALYSIS:
{reasoning_trace}

TRAINING EXAMPLES SUMMARY:
We have {len(training_examples)} examples of input-output transformations.

INSTRUCTIONS:
Extract the transformation as a numbered list of clear, actionable steps.
Each step should be implementable in code and describe exactly what to do.

Format your response as:
```steps
1. [First transformation step]
2. [Second transformation step]
3. [Third transformation step]
...
```

Generate the step-by-step transformation:"""
    
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


def generate_code_from_reasoning(llm, reasoning_trace: str, transformation_steps: List[str],
                               training_examples: List[Dict], helper_functions: List[HelperFunction]) -> str:
    """Generate Python code based on reasoning trace and transformation steps."""
    
    prompt = build_code_from_reasoning_prompt(reasoning_trace, transformation_steps, 
                                            training_examples, helper_functions)
    
    try:
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        print(f"{BLUE}--- LLM Prompt ---{RESET}")
        print(f"{BLUE}{prompt}{RESET}")
        print(f"{GREEN}--- LLM Response ---{RESET}")
        print(f"{GREEN}{response_text}{RESET}")
        
        # Extract and validate Python code
        python_code = extract_and_validate_python_code(response_text)
        print(f"{RED}--- Extracted Python Code ---{RESET}")
        print(f"{RED}{python_code}{RESET}")
        
        if python_code:
            return python_code
        else:
            return generate_fallback_code_from_steps(transformation_steps)
        
    except Exception as e:
        print(f"Error generating code from reasoning: {e}")
        return generate_fallback_code_from_steps(transformation_steps)


def build_code_from_reasoning_prompt(reasoning_trace: str, transformation_steps: List[str],
                                   training_examples: List[Dict], helper_functions: List[HelperFunction]) -> str:
    """Build prompt for generating Python code from reasoning and steps."""
    
    steps_text = '\n'.join(f"{i+1}. {step}" for i, step in enumerate(transformation_steps))
    
    available_helpers = ""
    if helper_functions:
        available_helpers = "\nAVAILABLE HELPER FUNCTIONS:\n"
        for func in helper_functions[:10]:
            available_helpers += f"- {func['name']}: {func['description']}\n"
    
    prompt = f"""You are a Python expert implementing ARC transformations.

Given the following reasoning analysis and step-by-step transformation, implement a Python function.

REASONING ANALYSIS:
{reasoning_trace}

TRANSFORMATION STEPS:
{steps_text}

AVAILABLE HELPERS:
{available_helpers}

TRAINING EXAMPLES:
{len(training_examples)} input-output example pairs are provided for validation.

IMPLEMENTATION REQUIREMENTS:
1. Write a function called 'transform(input_grid)' that returns the output grid
2. Implement each transformation step clearly and precisely
3. Use available helper functions where appropriate
4. Add clear comments mapping to the transformation steps
5. Handle edge cases and ensure robustness
6. Return ONLY executable Python code

Example structure:
```python

import ...

def helper_function_1(...):
    # Add helper functions if neeeded

def helper_function_2(...):
    # Add helper functions if needed

def transform(input_grid: List[List[int]]) -> List[List[int]]:
    # Step 1: [Short comment comment describing first step]
    [implementation of step 1]
    
    # Step 2: [Short comment comment describing second step]
    [implementation of step 2]
    
    # Continue for all steps...
    return transformed_grid
```

Generate the Python code implementing the transformation:"""
    
    return prompt


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


def build_python_focused_prompt(training_examples: List[Dict],
                               helper_functions: List[HelperFunction],
                               previous_solutions: List[CodeSolution]) -> str:
    """Build prompt focused on generating high-quality Python transformation code."""
    
    prompt_parts = [
        "You are a Python expert specializing in ARC problem solving.",
        "Your PRIMARY goal is to write clean, efficient Python code that transforms input grids to output grids.",
        "",
        "TRAINING EXAMPLES:"
    ]
    
    # Add training examples with clear input/output format
    for i, example in enumerate(training_examples):
        prompt_parts.append(f"\nExample {i+1}:")
        prompt_parts.append(f"Input:  {example['input']}")
        prompt_parts.append(f"Output: {example['output']}")
    
    # Add available helper functions
    if helper_functions:
        prompt_parts.append("\nAVAILABLE HELPER FUNCTIONS:")
        for func in helper_functions[:10]:  # Limit to avoid token overflow
            prompt_parts.append(f"- {func['name']}: {func['description']}")
    
    # Add context from previous attempts
    if previous_solutions:
        prompt_parts.append("\nPREVIOUS ATTEMPTS (failed):")
        for i, sol in enumerate(previous_solutions[-2:], 1):
            prompt_parts.append(f"Attempt {i}: {sol['main_code'][:100]}...")
    
    prompt_parts.extend([
        "",
        "PYTHON CODE REQUIREMENTS:",
        "1. Write a function called 'transform(input_grid)' that returns the output grid",
        "2. Use clear, readable Python code with proper error handling", 
        "3. Focus on the core transformation logic - be precise and efficient",
        "4. Use available helper functions or create new helper functions when appropriate",
        "5. Add comments explaining the transformation steps",
        "6. Return ONLY executable Python code, no explanations",
        "7. No TODO comments - implement the logic fully",
        "",
        "Generate the transform function:"
    ])
    
    return "\n".join(prompt_parts)


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


def build_code_generation_prompt(training_examples: List[Dict],
                                helper_functions: List[HelperFunction],
                                previous_solutions: List[CodeSolution]) -> str:
    """Build a prompt for LLM to generate transformation code."""
    
    prompt_parts = [
        "You are an expert at solving ARC (Abstraction and Reasoning Corpus) problems.",
        "Your task is to analyze the input-output examples and write a Python function called 'transform' that solves the pattern.",
        "",
        "TRAINING EXAMPLES:"
    ]
    
    # Add training examples
    for i, example in enumerate(training_examples):
        prompt_parts.append(f"\nExample {i+1}:")
        prompt_parts.append(f"Input: {example['input']}")
        prompt_parts.append(f"Output: {example['output']}")
    
    # Add available helper functions
    if helper_functions:
        prompt_parts.append("\nAVAILABLE HELPER FUNCTIONS:")
        helper_names = [func['name'] for func in helper_functions]
        prompt_parts.append(f"You can use these functions: {', '.join(helper_names)}")
        
        # Add function descriptions
        for func in helper_functions[:5]:  # Limit to avoid token overflow
            prompt_parts.append(f"- {func['name']}: {func.get('description', 'Helper function')}")
    
    # Add information about previous attempts if any
    if previous_solutions:
        prompt_parts.append("\nPREVIOUS ATTEMPTS (that failed):")
        for i, sol in enumerate(previous_solutions[-2:], 1):  # Show last 2 attempts
            prompt_parts.append(f"Attempt {i}: {sol.get('main_code', '')[:200]}...")
    
    prompt_parts.extend([
        "",
        "REQUIREMENTS:",
        "1. Write a function called 'transform(input_grid)' that takes a 2D list and returns a 2D list",
        "2. The function should work for ALL the training examples shown above",
        "3. Use the available helper functions when possible",
        "4. Focus on finding the core pattern or rule that transforms input to output",
        "5. Return ONLY the Python code, no explanations",
        "",
        "Example format:",
        "```python",
        "def transform(input_grid):",
        "    # Your transformation logic here",
        "    return result_grid",
        "```",
        "",
        "Generate the transform function now:"
    ])
    
    return "\n".join(prompt_parts)


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


def generate_main_transformation_code(training_examples: List[Dict],
                                    helper_functions: List[HelperFunction],
                                    previous_solutions: List[CodeSolution]) -> str:
    """Generate the main transformation code."""
    
    if not training_examples:
        return "def transform(input_grid):\n    return input_grid"
    
    # Analyze the first example to get some insights
    first_example = training_examples[0]
    input_grid = first_example["input"]
    output_grid = first_example["output"]
    
    # Basic template based on pattern analysis
    if len(input_grid) == len(output_grid) and len(input_grid[0]) == len(output_grid[0]):
        # Same size - might be color transformation or pattern replacement
        code = """def transform(input_grid):
    # Copy the input grid
    result = copy_grid(input_grid)
    height, width = get_grid_dimensions(input_grid)
    
    # Apply transformation logic
    for i in range(height):
        for j in range(width):
            # TODO: Add specific transformation logic based on pattern analysis
            # For now, just copy the input
            pass
    
    return result"""
    else:
        # Different size - might be scaling or cropping
        code = """def transform(input_grid):
    height, width = get_grid_dimensions(input_grid)
    
    # TODO: Determine output dimensions and create result grid
    # For now, return a copy of input
    result = copy_grid(input_grid)
    
    return result"""
    
    return code


def get_solution_reasoning(solution: CodeSolution) -> str:
    """Get reasoning trace from solution, with fallback for compatibility."""
    return solution.get("reasoning_trace", "No reasoning trace available")


def get_solution_steps(solution: CodeSolution) -> List[str]:
    """Get transformation steps from solution, with fallback for compatibility."""
    # Try new field first, then fall back to legacy field
    steps = solution.get("step_by_step_transformation", [])
    if not steps:
        steps = solution.get("step_by_step_description", [])
    return steps if steps else ["No transformation steps available"]


def generate_step_by_step_description(training_examples: List[Dict], main_code: str) -> List[str]:
    """Generate a step-by-step description of the transformation."""
    return []


def calculate_confidence_score(training_examples: List[Dict], 
                              main_code: str, 
                              helper_functions: List[HelperFunction]) -> float:
    """Calculate a confidence score for the generated solution."""
    return 0.5  # Simplified - confidence score no longer used


def execute_transformation_code(main_code: str,
                               input_grid: List[List[int]],
                               helper_functions: List[HelperFunction]) -> Tuple[Optional[List[List[int]]], Optional[str]]:
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

        # Add helper functions to namespace
        for helper in helper_functions:
            exec(helper["code"], namespace)

        # Add built-in helper functions
        for name, func in FUNCTION_MAP.items():
            namespace[name] = func

        # Execute the main code
        exec(main_code, namespace)

        # Call the transform function
        if "transform" in namespace:
            try:
                print("Code successfully executed. Running transform function...")
                result = namespace["transform"](input_grid)
                print("Transform function executed successfully.")
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


def refine_code_based_on_analysis(original_code: str, 
                                 failure_analysis: Dict[str, Any],
                                 helper_functions: List[HelperFunction]) -> str:
    """Refine the code based on failure analysis using intelligent analysis."""
    
    # Try to intelligently refine the code based on failure patterns
    if not original_code or "pass" in original_code:
        # If original code is empty or has placeholders, return a basic template
        return """def transform(input_grid):
    result = copy_grid(input_grid)
    height, width = get_grid_dimensions(result)
    
    # TODO: Implement actual transformation logic
    # This is a basic template - the LLM should provide real logic
    for i in range(height):
        for j in range(width):
            # Pattern matching and transformation logic needed here
            if result[i][j] == 0:  # Example: transform zeros
                # Add your transformation logic
                pass
    
    return result"""
    
    # If we have actual code, try to preserve and enhance it
    refined_lines = []
    original_lines = original_code.split('\n')
    
    for line in original_lines:
        # Remove placeholder comments and passes
        if 'pass' in line and ('TODO' in line or 'Enhanced' in line or 'Refined' in line):
            refined_lines.append(line.replace('pass', '# Add transformation logic here'))
        else:
            refined_lines.append(line)
    
    return '\n'.join(refined_lines)


def refine_solution_based_on_failures(llm,
                                      current_solution: CodeSolution,
                                      failed_tests: List[ExampleResult],
                                      training_examples: List[Dict],
                                      reflection_history: List[Dict] = None) -> Tuple[CodeSolution, Dict]:
    """Refine the solution based on test failures using LLM with ARC-style reflection."""
    
    if reflection_history is None:
        reflection_history = []
    
    try:
        # Use the reflection-first flow: generate a reasoning trace, extract steps, then generate code
        helper_functions = current_solution.get("helper_functions", [])

        # 1) Get reflection reasoning trace (structured analysis only)
        reasoning = generate_reflection_reasoning_trace(llm, current_solution, failed_tests, training_examples, reflection_history)

        # 2) Extract transformation steps from the reasoning
        transformation_steps = extract_transformation_steps(llm, reasoning, training_examples)

        # 3) Generate refined code from reasoning + steps
        refined_code = generate_code_from_reasoning(llm, reasoning, transformation_steps, training_examples, helper_functions)

        # If code generation succeeded, return the refined solution
        if refined_code and 'def transform(' in refined_code:
            refined_solution = {
                "main_code": refined_code,
                "helper_functions": helper_functions,
                "reasoning_trace": reasoning,
                "step_by_step_transformation": transformation_steps or ["Refined based on failure analysis"],
            }

            reflection_record = {
                "attempt_number": len(reflection_history) + 1,
                "reasoning": reasoning,
                "key_insight": extract_key_insight_from_reasoning(reasoning),
                "failed_examples": [t.get("example_index", 0) for t in failed_tests],
                "refinement_type": "arc_reflection"
            }

            return refined_solution, reflection_record
        else:
            # LLM produced no usable refined code. Do NOT fallback to rule-based refinement.
            reflection_record = {
                "attempt_number": len(reflection_history) + 1,
                "reasoning": reasoning if reasoning else "LLM failed to produce valid refined code",
                "key_insight": extract_key_insight_from_reasoning(reasoning) if reasoning else "LLM code-generation failed",
                "failed_examples": [t.get("example_index", 0) for t in failed_tests],
                "refinement_type": "no_refinement"
            }
            # Return the original solution unchanged along with the reflection record
            return current_solution, reflection_record
            
    except Exception as e:
        print(f"Error refining code with LLM: {e}")
        # Do not perform rule-based fallback here; return original solution and an error reflection record
        reflection_record = {
            "attempt_number": len(reflection_history) + 1,
            "reasoning": f"Error during LLM refinement: {str(e)}",
            "key_insight": "Technical error prevented LLM refinement",
            "failed_examples": [t.get("example_index", 0) for t in failed_tests],
            "refinement_type": "llm_error"
        }
        return current_solution, reflection_record


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


def build_refinement_reasoning_prompt(current_solution: CodeSolution,
                                     failed_tests: List[ExampleResult],
                                     training_examples: List[Dict],
                                     reflection_history: List[Dict]) -> str:
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
    
    # Add reflection context
    # TODO: Think about this reflection context for now. It is a little bit of an odd-ball at the moment.
    reflection_context = ""
    if reflection_history:
        reflection_context = f"\\nPrevious reflection attempts: {len(reflection_history)}\\n"
        reflection_context += "Key insights from previous attempts:\\n"
        for i, refl in enumerate(reflection_history[-2:], 1):  # Show last 2
            insight = refl.get("key_insight", "No insight recorded")
            reflection_context += f"{i}. {insight}\\n"
    
    prompt = f"""You are an expert in reasoning about Abstract Reasoning Corpus (ARC) puzzles.
You previously attempted to solve this task but your solution was incorrect on some training examples.

====================
TASK REFLECTION AND DEEP ANALYSIS
====================

Your goal:
Analyze your previous attempt deeply, understand why it failed, and provide a CORRECTED transformation that:
1. Correctly maps every training input to its output
2. Is general and intuitive (no memorization or hard-coded values)
3. Is logical, reproducible, and object-level

====================
Original Training Examples
====================
{examples_block}

====================
Your Previous Solution
====================
```python
{previous_code}
```

====================
Detailed Failure Analysis
====================
{failures_block}

====================
Deep Reflection Instructions
====================
    First, analyze what went wrong inside a ```reasoning``` block:
    1. PATTERN MISINTERPRETATION: What pattern did you miss or misunderstand?
    2. LOGIC ERRORS: Where exactly did your transformation logic fail?
    3. EDGE CASES: What cases did you not handle properly?
    4. OBJECT-LEVEL THINKING: How should you think about this in terms of objects, shapes, movements?
    5. CORE INSIGHT: What is the single most important insight you're missing?

    After the analysis, produce a clear, numbered ACTION PLAN (no executable code):
    - Provide a concise algorithmic description of the corrected transformation (1-3 paragraphs).
    - List any helper functions that should be added or adjusted (name and brief signature only).
    - Enumerate edge cases and how to handle them.
    - Suggest specific, focused tests or checks that would validate the fix.

    Return ONLY the requested analysis and action plan inside clearly ```reasoning``` block Do NOT include executable Python code in your response.
"""
    
    return prompt


def format_grid_for_prompt(grid: List[List[int]]) -> str:
    """Format grid for display in prompts."""
    return "\\n".join(" ".join(map(str, row)) for row in grid)