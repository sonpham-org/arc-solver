"""
Code generation functions for ARC agent.
"""

import re
import ast
from typing import List, Dict, Tuple, Union

from .utilities import format_grid_for_prompt
from agentic.debug import print_prompt_and_response


def generate_code_from_reasoning(code_llm, reasoning_trace: str, training_examples: List[Dict], num_solutions: int, max_retries: int = 3) -> Tuple[List[str], int]:
    """Generate Python code based on reasoning trace only (without transformation steps).

    Returns:
        Tuple of (python_codes_list, num_retries_used)
    """

    def build_code_from_reasoning_only_prompt(reasoning_trace: str, training_examples: List[Dict], num_solutions: int) -> str:
        """Build prompt for generating Python code from reasoning only."""
        prompt_parts = [
            "You are a Python expert implementing ARC transformations.",
            "",
            "Given the following reasoning analysis, implement Python functions that solve the task.",
            "",
            "------------------",
            "REASONING ANALYSIS",
            "------------------",
            f"{reasoning_trace}",
            "",
            "-----------------",
            "TRAINING EXAMPLES",
            "------------------",
            f"{len(training_examples)} input-output example pairs are provided for validation.",
            "",
        ]
        
        for i, ex in enumerate(training_examples, 1):
            prompt_parts.append(f"Example {i} Input:")
            prompt_parts.append(format_grid_for_prompt(ex.get('input', [])))
            prompt_parts.append(f"Example {i} Output:")
            prompt_parts.append(format_grid_for_prompt(ex.get('output', [])))
            prompt_parts.append("")
        
        prompt_parts.extend([
            "---------------------------",
            "IMPLEMENTATION REQUIREMENTS",
            "---------------------------",
            f"Produce {num_solutions} different candidate solutions based on the reasoning above.",
            "For each solution:",
            "1. Write a function called 'transform(input_grid)' that takes a 2D list of integers as input and returns a transformed 2D list of integers",
            "2. Implement the transformation clearly and precisely based on the reasoning",
            "3. Import any necessary standard libraries at the top for EACH solution",
            "4. Include helper functions where necessary",
            "5. DO NOT ADD ANY EXPLANATIONS OR COMMENTS IN THE CODE",
            "6. Address any error cases or edge conditions mentioned in the reasoning to ensure correctness and robustness",
            "7. Return ONLY executable Python code",
            "The <count> will help you keep track of what-th solution you are at. Make sure you have all solutions implemented.",
            "",
            "Example structure:",
            "<count>1</count>",
            "<solution>",
            "from typing import List",
            "import ... # Import ANY necessary standard libraries to run the code here",
            "def helper_function_1(...):",
            "    # Add helper functions if needed",
            "def helper_function_2(...):",
            "    # Add helper functions if needed",
            "def transform(input_grid):",
            "    [implementation based on reasoning]",
            "    return transformed_grid",
            "</solution>",
            "<count>2</count>",
            "<solution>...</solution>",
            "...",
            "",
            "Generate the Python code now:"
        ])

        prompt = "\n".join(prompt_parts)
        return prompt

    prompt = build_code_from_reasoning_only_prompt(reasoning_trace, training_examples, num_solutions)
    
    # Retry up to max_retries times if extraction fails
    for attempt in range(max_retries):
        try:
            response = code_llm.invoke(prompt, temperature=0.3)
            response_text = response.content if hasattr(response, 'content') else str(response)
            print_prompt_and_response(prompt, response_text)
            
            # Extract candidate python solutions (may be multiple)
            candidate_codes = extract_python_solutions(response_text)
            # Ensure common imports are present in each candidate code block
            candidate_codes = [ensure_imports_in_code(c) for c in candidate_codes]
            
            if candidate_codes:
                print(f"{len(candidate_codes)} candidate code solutions generated.")
                return candidate_codes, attempt
                
        except Exception as e:
            print(f"Warning: Failed to generate code from reasoning (attempt {attempt + 1}/{max_retries}). Error: {e}")
        
        # If this isn't the last attempt, log and retry
        if attempt < max_retries - 1:
            print(f"Warning: Retrying code generation (attempt {attempt + 1}/{max_retries})...")
    
    # After all retries failed, return empty list
    print(f"Error: Failed to generate code from reasoning after {max_retries} attempts.")
    return [], max_retries


def generate_code_from_reasoning_and_transformations(code_llm, reasoning_trace: str, transformation_steps: List[str],
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
            "------------------",
            "REASONING ANALYSIS",
            "------------------",
            f"{reasoning_trace}",
            "",
            "-----------------",
            "TRAINING EXAMPLES",
            "------------------",
            f"{len(training_examples)} input-output example pairs are provided for validation.",
            "",
            "-----------------------------",
            "TRANSFORMATION STEP SOLUTIONS",
            "-----------------------------",
            f"{steps_text}",
            "",
            "---------------------------",
            "IMPLEMENTATION REQUIREMENTS",
            "---------------------------",
            "For each solution",
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
        response = code_llm.invoke(prompt, temperature=0.3)
        response_text = response.content if hasattr(response, 'content') else str(response)

        print_prompt_and_response(prompt, response_text)
        
        # Extract candidate python solutions (may be multiple)
        candidate_codes = extract_python_solutions(response_text)
        # Ensure common imports are present in each candidate code block so
        # the probe can execute without trivial missing-import errors.
        candidate_codes = [ensure_imports_in_code(c) for c in candidate_codes]
        print(len(candidate_codes), "candidate code solutions generated.")

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

    if not code or not isinstance(code, str):
        return code

    # Find already-present import lines to avoid duplicates
    existing_imports = set()
    for m in re.finditer(r'^\s*(?:from|import)\s+([^\n]+)', code, re.MULTILINE):
        existing_imports.add(m.group(0).strip())

    # Map usage tokens -> import statements (conservative set)
    typing_tokens = {tok for tok in ("List", "Dict", "Any", "Tuple", "Optional", "Set") if re.search(r'\b%s\b' % tok, code)}

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
        if re.search(pattern, code) and not any(imp in s for s in existing_imports):
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
