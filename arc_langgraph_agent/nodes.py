"""
Nodes for the ARC LangGraph Agent workflow.

This module contains the individual nodes that make up the agent workflow,
including code generation, testing, and refinement.
"""

import json
import copy
from typing import List, Dict, Optional, Any, TypedDict, Tuple
import traceback
import sys
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

# Import schema and tools
from .schema import AgentState, CodeSolution, TestResult, HelperFunction
from .tools import FUNCTION_MAP


# Helper functions for the nodes

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


def extract_helpers_node(state: AgentState, llm) -> AgentState:
    """
    Extract potential useful helper functions from the generated solution
    and add them to the growing tool set.
    """
    current_solution = state.get("current_solution")
    if not current_solution:
        return state
    
    training_examples = state["task_data"]["train"]
    global_library = state.get("global_helper_library", [])
    
    # Extract new helper functions from the current code
    extracted_helpers = extract_helper_functions_with_llm(
        llm,
        current_solution["main_code"],
        training_examples,
        global_library
    )
    
    # Update state with new helpers
    new_state = copy.deepcopy(state)
    new_state["extracted_helpers"] = extracted_helpers
    
    # Add to available helpers for this task
    for helper in extracted_helpers:
        if not any(h["name"] == helper["name"] for h in new_state["available_helpers"]):
            new_state["available_helpers"].append(helper)
    
    # Add to global library (growing across tasks)
    for helper in extracted_helpers:
        if not any(h["name"] == helper["name"] for h in new_state["global_helper_library"]):
            new_state["global_helper_library"].append(helper)
    
    # CRITICAL: Update current solution's helper_functions for execution
    if new_state.get("current_solution"):
        new_state["current_solution"]["helper_functions"] = extracted_helpers
    
    return new_state


def extract_helper_functions_with_llm(llm, main_code: str, training_examples: List[Dict], 
                                     existing_library: List[HelperFunction]) -> List[HelperFunction]:
    """Use LLM to extract useful helper functions from generated code."""
    
    prompt = build_helper_extraction_prompt(main_code, training_examples, existing_library)
    
    try:
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
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
    
    prompt = f"""You are an expert at creating reusable helper functions for ARC problem solving.

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
5. Make functions general and well-documented

Return your response as a JSON list:
[{{
  "name": "function_name",
  "code": "def function_name(params):\\n    # implementation\\n    return result",
  "description": "What this function does and when to use it",
  "parameters": ["param1", "param2"]
}}]

Extract 2-4 the most valuable helper functions:"""
    
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


def generate_python_transformation_code(llm, training_examples: List[Dict], 
                                      helper_functions: List[HelperFunction],
                                      previous_solutions: List[CodeSolution]) -> str:
    """Generate Python transformation code with primary focus on code quality."""
    
    # Use the new reasoning-first approach but only return the code for compatibility
    python_code, _, _ = generate_python_transformation_code_with_reasoning(
        llm, training_examples, helper_functions, previous_solutions)
    
    return python_code


def generate_reasoning_trace(llm, training_examples: List[Dict], 
                           helper_functions: List[HelperFunction],
                           previous_solutions: List[CodeSolution]) -> str:
    """Generate detailed reasoning trace analyzing ARC patterns."""
    
    prompt = build_reasoning_prompt(training_examples, helper_functions, previous_solutions)
    
    try:
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Extract reasoning from response
        reasoning = extract_reasoning_content(response_text)
        
        return reasoning if reasoning else "Unable to generate reasoning trace"
        
    except Exception as e:
        print(f"Error generating reasoning trace: {e}")
        return f"Error in reasoning generation: {str(e)}"


def build_reasoning_prompt(training_examples: List[Dict],
                         helper_functions: List[HelperFunction],
                         previous_solutions: List[CodeSolution]) -> str:
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
    
    # Add context from previous attempts
    if previous_solutions:
        prompt_parts.append("\nPREVIOUS FAILED REASONING:")
        for i, sol in enumerate(previous_solutions[-2:], 1):
            if hasattr(sol, 'get') and sol.get('reasoning_trace'):
                prompt_parts.append(f"Attempt {i}: {sol['reasoning_trace'][:200]}...")
    
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
        formatted_rows.append(" ".join(str(cell) for cell in row))
    
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
        
        # Extract and validate Python code
        python_code = extract_and_validate_python_code(response_text)
        
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
{steps_text}{available_helpers}

TRAINING EXAMPLES:
{len(training_examples)} input-output example pairs are provided for validation.

IMPLEMENTATION REQUIREMENTS:
1. Write a function called 'transform(input_grid)' that returns the output grid
2. Implement each transformation step clearly and precisely
3. Use available helper functions where appropriate
4. Add clear comments mapping to the transformation steps
5. Handle edge cases and ensure robustness
6. Return ONLY executable Python code, no explanations

Example structure:
```python
def transform(input_grid):
    # Step 1: [comment describing first step]
    # [implementation of step 1]
    
    # Step 2: [comment describing second step]
    # [implementation of step 2]
    
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
        "4. Use available helper functions or create new helepr functions when appropriate",
        "5. Add comments explaining the transformation steps",
        "6. Return ONLY executable Python code, no explanations",
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
                               helper_functions: List[HelperFunction]) -> Optional[List[List[int]]]:
    """Execute the transformation code on an input grid."""
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
            result = namespace["transform"](input_grid)
            return result
        else:
            return None
            
    except Exception as e:
        print(f"Error executing transformation code: {e}")
        return None


def calculate_grid_overlap(predicted: List[List[int]], expected: List[List[int]]) -> float:
    """Calculate the percentage overlap between two grids."""
    if not predicted or not expected:
        return 0.0
    
    if len(predicted) != len(expected):
        return 0.0
    
    total_cells = 0
    matching_cells = 0
    
    for i, (pred_row, exp_row) in enumerate(zip(predicted, expected)):
        if len(pred_row) != len(exp_row):
            return 0.0
        
        for j, (pred_cell, exp_cell) in enumerate(zip(pred_row, exp_row)):
            total_cells += 1
            if pred_cell == exp_cell:
                matching_cells += 1
    
    return (matching_cells / total_cells) * 100.0 if total_cells > 0 else 0.0


def analyze_failures(failed_tests: List[TestResult], training_examples: List[Dict]) -> Dict[str, Any]:
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


def refine_solution_based_on_failures_with_llm(llm,
                                              current_solution: CodeSolution,
                                              failed_tests: List[TestResult],
                                              training_examples: List[Dict],
                                              reflection_history: List[Dict] = None) -> Tuple[CodeSolution, Dict]:
    """Refine the solution based on test failures using LLM with ARC-style reflection."""
    
    if reflection_history is None:
        reflection_history = []
    
    # Build a refinement prompt with reflection history
    prompt = build_arc_style_reflection_prompt(
        current_solution, failed_tests, training_examples, reflection_history
    )
    
    try:
        # Call the LLM to generate refined code
        response = llm.invoke(prompt)
        
        # Extract response content
        if hasattr(response, 'content'):
            response_content = response.content
        else:
            response_content = str(response)
        
        # Extract reasoning and code from response
        reasoning = extract_reasoning_from_reflection(response_content)
        refined_code = extract_python_code(response_content)
        
        if refined_code and 'def transform(' in refined_code and len(refined_code.strip()) > 100:
            # Create new solution with LLM-generated refinement
            refined_solution = {
                "main_code": refined_code,
                "helper_functions": current_solution.get("helper_functions", []),
                "reasoning_trace": reasoning,
                "step_by_step_transformation": ["Refined based on failure analysis"],
                "step_by_step_description": ["Refined based on failure analysis"],  # For compatibility
                "confidence_score": 0.5
            }
            
            # Create reflection record
            reflection_record = {
                "attempt_number": len(reflection_history) + 1,
                "reasoning": reasoning,
                "key_insight": extract_key_insight_from_reasoning(reasoning),
                "failed_examples": [t.get("example_index", 0) for t in failed_tests],
                "refinement_type": "arc_reflection"
            }
            
            return refined_solution, reflection_record
        else:
            # Fallback to original refinement method
            fallback_solution = refine_solution_based_on_failures(current_solution, failed_tests, training_examples)
            reflection_record = {
                "attempt_number": len(reflection_history) + 1,
                "reasoning": "Fallback to rule-based refinement due to LLM parsing issues",
                "key_insight": "LLM response could not be parsed properly",
                "failed_examples": [t.get("example_index", 0) for t in failed_tests],
                "refinement_type": "fallback"
            }
            return fallback_solution, reflection_record
            
    except Exception as e:
        print(f"Error refining code with LLM: {e}")
        # Fallback to original refinement method
        fallback_solution = refine_solution_based_on_failures(current_solution, failed_tests, training_examples)
        reflection_record = {
            "attempt_number": len(reflection_history) + 1,
            "reasoning": f"Error during LLM refinement: {str(e)}",
            "key_insight": "Technical error prevented LLM refinement",
            "failed_examples": [t.get("example_index", 0) for t in failed_tests],
            "refinement_type": "error_fallback"
        }
        return fallback_solution, reflection_record


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


def build_refinement_prompt(current_solution: CodeSolution,
                           failed_tests: List[TestResult],
                           training_examples: List[Dict]) -> str:
    """Build a prompt for LLM to refine the solution based on failures."""
    return build_arc_style_reflection_prompt(
        current_solution, failed_tests, training_examples, []
    )


def build_arc_style_reflection_prompt(current_solution: CodeSolution,
                                     failed_tests: List[TestResult],
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
{examples_block}====================
Your Previous Solution
====================
```python
{previous_code}
```

====================
Detailed Failure Analysis
====================
{failures_block}{reflection_context}====================
Deep Reflection Instructions
====================
First, analyze what went wrong inside ```reasoning ``` block:
1. PATTERN MISINTERPRETATION: What pattern did you miss or misunderstand?
2. LOGIC ERRORS: Where exactly did your transformation logic fail?
3. EDGE CASES: What cases did you not handle properly?
4. OBJECT-LEVEL THINKING: How should you think about this in terms of objects, shapes, movements?
5. CORE INSIGHT: What is the single most important insight you're missing?

Then provide a COMPLETELY REWRITTEN solution that addresses ALL failures:
```python
def transform(input_grid):
    # Completely rewritten transformation
    # Address all identified issues
    # Focus on the core pattern you discovered
    return transformed_grid
```

Make your reflection deep and thorough. Push yourself to truly understand the pattern."""
    
    return prompt


def format_grid_for_prompt(grid: List[List[int]]) -> str:
    """Format grid for display in prompts."""
    return "\\n".join(" ".join(map(str, row)) for row in grid)


def refine_solution_based_on_failures(current_solution: CodeSolution,
                                     failed_tests: List[TestResult],
                                     training_examples: List[Dict]) -> CodeSolution:
    """Refine the solution based on test failures."""
    
    # Analyze failure patterns
    failure_analysis = analyze_failures(failed_tests, training_examples)
    
    # Generate refined code based on analysis
    refined_code = refine_code_based_on_analysis(
        current_solution["main_code"], 
        failure_analysis,
        current_solution.get("helper_functions", [])
    )
    
    # Create new solution with all required fields
    refined_solution = {
        "main_code": refined_code,
        "helper_functions": current_solution.get("helper_functions", []),
        "reasoning_trace": current_solution.get("reasoning_trace", "") + "\n\nRefined based on failure analysis",
        "step_by_step_transformation": current_solution.get("step_by_step_transformation", []),
        "step_by_step_description": current_solution.get("step_by_step_description", []),
        "confidence_score": 0.5
    }
    
    return refined_solution


# Main workflow nodes
def generate_code_node(state: AgentState, llm) -> AgentState:
    """
    Generate Python code to solve the ARC problem using reasoning-first approach.
    
    This node uses the available helper functions and analyzes the training
    examples to generate a solution using the provided language model.
    """
    task_data = state["task_data"]
    attempt_number = state["attempt_number"]
    available_helpers = state.get("available_helpers", [])
    previous_solutions = state.get("previous_solutions", [])
    
    # Analyze the training examples
    training_examples = task_data["train"]
    
    # Create a prompt for code generation based on examples
    analysis_text = analyze_training_examples(training_examples)
    
    # Generate code using reasoning-first approach
    main_code, reasoning_trace, transformation_steps = generate_python_transformation_code_with_reasoning(
        llm,
        training_examples, 
        available_helpers,
        previous_solutions
    )
    
    # Create the solution with reasoning traces
    solution = {
        "main_code": main_code,
        "helper_functions": [],
        "reasoning_trace": reasoning_trace,
        "step_by_step_transformation": transformation_steps,
        "step_by_step_description": transformation_steps,  # For compatibility
        "confidence_score": 0.5  # Default value
    }
    
    # Update state
    new_state = copy.deepcopy(state)
    new_state["current_solution"] = solution
    new_state["should_continue"] = True
    
    return new_state


def test_code_node(state: AgentState) -> AgentState:
    """
    Test the generated code against training examples.
    
    This node executes the current solution on all training examples
    and calculates the success rate.
    """
    current_solution = state["current_solution"]
    task_data = state["task_data"]
    
    if not current_solution:
        # No solution to test
        new_state = copy.deepcopy(state)
        new_state["test_results"] = []
        new_state["success_rate"] = 0.0
        new_state["should_continue"] = False
        return new_state
    
    training_examples = task_data["train"]
    test_results = []
    successful_tests = 0
    
    # Prepare the execution environment
    helper_functions = current_solution["helper_functions"]
    main_code = current_solution["main_code"]
    
    for i, example in enumerate(training_examples):
        input_grid = example["input"]
        expected_output = example["output"]
        
        try:
            # Execute the code
            predicted_output = execute_transformation_code(
                main_code, input_grid, helper_functions
            )
            
            if predicted_output is not None:
                # Calculate overlap percentage
                overlap = calculate_grid_overlap(predicted_output, expected_output)
                success = overlap >= 100.0  # Perfect match required
                
                test_result = {
                    "example_index": i,
                    "success": success,
                    "predicted_output": predicted_output,
                    "expected_output": expected_output,
                    "overlap_percentage": overlap,
                    "error_message": None
                }
                
                if success:
                    successful_tests += 1
            else:
                test_result = {
                    "example_index": i,
                    "success": False,
                    "predicted_output": None,
                    "expected_output": expected_output,
                    "overlap_percentage": 0.0,
                    "error_message": "Code execution returned None"
                }
                
        except Exception as e:
            test_result = {
                "example_index": i,
                "success": False,
                "predicted_output": None,
                "expected_output": expected_output,
                "overlap_percentage": 0.0,
                "error_message": str(e)
            }
        
        test_results.append(test_result)
    
    success_rate = successful_tests / len(training_examples) if training_examples else 0.0
    
    # Update state
    new_state = copy.deepcopy(state)
    new_state["test_results"] = test_results
    new_state["success_rate"] = success_rate
    
    return new_state


def refinement_node(state: AgentState, llm) -> AgentState:
    """
    Refine the solution based on test failures.
    
    This node analyzes failed test cases and attempts to improve the solution.
    """
    test_results = state.get("test_results", [])
    current_solution = state["current_solution"]
    task_data = state["task_data"]
    attempt_number = state["attempt_number"]
    max_attempts = state["max_attempts"]
    
    # Analyze failures
    failed_tests = [t for t in test_results if not t["success"]]
    
    if not failed_tests or attempt_number >= max_attempts:
        # No failures or max attempts reached
        new_state = copy.deepcopy(state)
        new_state["should_continue"] = False
        return new_state
    
    # Generate refinement based on failures using LLM with reflection
    reflection_history = state.get("reflection_history", [])
    refined_solution, reflection_record = refine_solution_based_on_failures_with_llm(
        llm, current_solution, failed_tests, task_data["train"], reflection_history
    )
    
    # Update state for next attempt
    new_state = copy.deepcopy(state)
    new_state["previous_solutions"].append(current_solution)
    new_state["current_solution"] = refined_solution
    new_state["attempt_number"] = attempt_number + 1
    new_state["should_continue"] = True
    
    # Update reflection history
    new_reflection_history = reflection_history + [reflection_record]
    new_state["reflection_history"] = new_reflection_history
    
    return new_state


def should_continue_predicate(state: AgentState) -> str:
    """
    Determine if the workflow should continue or end.
    
    Returns:
        "test" if we should test the current solution
        "refine" if we should refine based on failures
        "end" if we should end the workflow
    """
    success_rate = state.get("success_rate", 0.0)
    attempt_number = state.get("attempt_number", 1)
    max_attempts = state.get("max_attempts", 5)
    current_solution = state.get("current_solution")
    
    # If no solution, end
    if not current_solution:
        return "end"
    
    # If perfect success, end
    if success_rate >= 1.0:
        return "end"
    
    # If max attempts reached, end
    if attempt_number >= max_attempts:
        return "end"
    
    # If we haven't tested yet, test
    if "test_results" not in state:
        return "test"
    
    # If we have failures and haven't reached max attempts, refine
    if success_rate < 1.0 and attempt_number < max_attempts:
        return "refine"
    
    return "end"


def finalize_node(state: AgentState) -> AgentState:
    """
    Finalize the workflow and prepare the final output.
    """
    current_solution = state.get("current_solution")
    task_data = state["task_data"]
    
    # Generate final prediction for the test case if we have a solution
    final_prediction = None
    if current_solution and task_data.get("test"):
        test_input = task_data["test"][0]["input"]  # Assume single test case
        try:
            final_prediction = execute_transformation_code(
                current_solution["main_code"],
                test_input,
                current_solution["helper_functions"]
            )
        except Exception as e:
            # Log error but continue
            state["errors"].append(f"Failed to generate final prediction: {e}")
    
    new_state = copy.deepcopy(state)
    new_state["final_prediction"] = final_prediction
    new_state["should_continue"] = False
    
    return new_state