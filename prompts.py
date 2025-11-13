# src/config.py

import json

intro = """
Explain the ARC Challenge in one sentence.
"""

reasoning = """
Describe how an intelligent system might approach solving an ARC problem using pattern recognition.
"""

task_example = """
Given an ARC task with colored grids, describe how to infer the transformation rule
from input to output examples.
"""

creative = """
Propose a novel strategy to improve ARC reasoning using multimodal embeddings.
"""

def build_arc_prompt(task_data: dict, task_id: str) -> str:
    """
    Builds a comprehensive prompt for an ARC (Abstraction and Reasoning Corpus) task.

    This function constructs a detailed prompt that instructs an AI model to reason about
    and solve an ARC puzzle. The prompt includes the task's training examples, test inputs,
    guidelines for rule inference, and a specified output format.

    Args:
        task_data (dict): A dictionary containing all ARC tasks, where keys are task IDs
            and values are task dictionaries with 'train' and 'test' keys.
        task_id (str): The unique identifier of the specific task to build a prompt for.

    Returns:
        str: A formatted string containing the complete prompt for the ARC task, including
            task description, training examples, test inputs, guidelines, and output format
            instructions.

    The prompt encourages the model to:
    - Infer a single, general transformation rule from training pairs
    - Use object-level reasoning rather than memorization
    - Output the rule in a structured JSON format within <json> tags
    """
    task = task_data[task_id]
    train, test = task.get("train", []), task.get("test", [])

    examples_block = "\n\n".join(
        f"Training Example {i}\n--\nInput:\n" +
        "\n".join(" ".join(map(str, r)) for r in ex["input"]) +
        "\n\nOutput:\n" +
        "\n".join(" ".join(map(str, r)) for r in ex["output"])
        for i, ex in enumerate(train, 1)
    )
    tests_block = "\n\n".join(
        f"Test Input {i}\n--\nInput:\n" +
        "\n".join(" ".join(map(str, r)) for r in t["input"])
        for i, t in enumerate(test, 1)
    )

    return f"""
You are an expert in reasoning about Abstract Reasoning Corpus (ARC) puzzles.

====================
Task {task_id}
====================

Your goal:
Given the training pairs and test inputs, infer a general transformation rule that:
1. Correctly maps every training input to its output.
2. Is general and intuitive (no memorization or hard-coded values).
3. Is logical, reproducible, and object-level.

====================
Guidelines
====================
- The SAME rule must successfully transform all training pairs.
- Treat all grid values (numbers/characters) as categorical labels, not magnitudes. Do not use arithmetic operations.
- Avoid rules that depend on specific values or characters. 
- Make rules in a general manner using object-level reasoning (movements, shapes, fills, mirrors, rotations, bounding boxes, duplicates, etc.).
- Take as many rules as you need to achieve your goals.

====================
Training Examples
====================
{examples_block}

====================
Output Format
====================
First, show your reasoning process inside ```reasoning ``` block:
- Analyze each training example carefully
- Look for patterns, transformations, and commonalities
- Identify the core transformation rule(s)

Be concise in your reasoning. Keep it short but informative.

Then you MUST return a JSON object inside ```json ``` block:
{{
  # Python compatible code that describes
  # any helper functions needed to implement the rule.
  # Each rule will run this code before applying the transformation code.
  "helper_python_functions": [
    "...",
  ],
  "step_by_step_transformations": [{{
      "step_number": 1,
      "description": [
        "...",
      ], # Describe the transformation conceptually
      "pseudo_code": [
      ],
  }},
  "python_code": [
    "def transform(grid):",
    "    # Complete transformation implementation",
    "    # This function must be fully executable on its own",
    "    # and return a complete transformed grid",
    "    return processed_grid"
  ]
}}""".strip()


def build_apply_prompt(final_json: dict, task: dict, task_id: str, include_examples=True) -> str:
    """
    Builds a prompt to apply an inferred transformation rule to test inputs.

    This function creates a detailed prompt that instructs an AI model to execute
    a previously inferred rule on test inputs for an ARC task. The prompt includes
    the rule summary, step-by-step instructions, and optionally the training examples
    for context.

    Args:
        final_json (dict): A dictionary containing the inferred rule with keys like
            'rule_summary', 'step_by_step_rule', and optionally 'pseudocode'.
        task (dict): A dictionary representing the ARC task, containing 'train' and
            'test' keys with their respective example lists.
        task_id (str): The unique identifier of the task.
        include_examples (bool, optional): Whether to include training examples in the
            prompt for additional context. Defaults to True.

    Returns:
        str: A formatted string containing the complete application prompt, including
            the rule details, training examples (if included), test inputs, and
            instructions for step-by-step execution with intermediate reasoning.

    The prompt requires the model to:
    - Apply the rule exactly to each test input
    - Show intermediate reasoning in <reasoning> blocks with step details
    - Provide final outputs in <output> blocks
    - Maintain valid ARC grid format (2D arrays of integers)
    """
    rule_summary = final_json.get("rule_summary", "")
    step_by_step = final_json.get("step_by_step_rule", [])
    pseudocode = final_json.get("pseudocode", "")
    examples_block = ""
    if include_examples:
        examples_block = "\n\n".join(
            f"Training Example {i}\n--\nInput:\n" +
            "\n".join(" ".join(map(str, r)) for r in ex["input"]) +
            "\n\nOutput:\n" +
            "\n".join(" ".join(map(str, r)) for r in ex["output"])
            for i, ex in enumerate(task.get("train", []), 1)
        )
    tests_block = "\n\n".join(
        f"Test Input {i}\n--\nInput:\n" +
        "\n".join(" ".join(map(str, r)) for r in t["input"])
        for i, t in enumerate(task.get("test", []), 1)
    )

    return f"""
You are an expert ARC executor.

====================
Task {task_id}
====================

Rule Summary:
{rule_summary}

Step-by-Step Rule:
{chr(10).join(step_by_step)}

Pseudocode:
{pseudocode}

====================
Training Examples
====================
{examples_block}

====================
Test Inputs
====================
{tests_block}

====================
Instructions
====================
1) Apply the rule EXACTLY, step-by-step, to each test input.
2) Show your intermediate reasoning inside a <reasoning></reasoning> block.
3) Within the reasoning block, after each step, shows the intermediate grid state in the following format:
  <step>
  {{
    "step_number": i,
    "description": "Describe what is being done in this step.",
    "before": [[...]],   // grid state before applying this step
    "after":  [[...]]    // grid state after applying this step
  }}
  </step>
4) Provide ONLY the final output grid for each test input inside <output></output>.
5) Ensure valid ARC format (2D array of integers).
6) No text outside of <reasoning> and <output> blocks.
""".strip()


def build_apply_prompt_2(final_json: dict, task: dict, task_id: str, include_examples=True) -> str:
    """
    Builds a prompt that suggests a recommended transformation ruleset to guide
    the model when applying it to test inputs.

    Unlike build_apply_prompt(), this version does not enforce exact step-by-step
    execution. Instead, it presents the ruleset as recommended reasoning patterns
    or transformation principles that should inform how the test inputs are handled.

    Args:
        final_json (dict): Contains inferred rules, with optional keys such as
            'rule_summary', 'step_by_step_rule', and 'pseudocode'.
        task (dict): The ARC task data (train/test examples).
        task_id (str): Unique identifier of the task.
        include_examples (bool, optional): Whether to include training examples
            for additional context. Defaults to True.

    Returns:
        str: A natural-language prompt emphasizing interpretation and reasoning,
             while still including reasoning and output blocks.
    """
    rule_summary = final_json.get("rule_summary", "")
    step_by_step = final_json.get("step_by_step_rule", [])
    pseudocode = final_json.get("pseudocode", "")
    examples_block = ""

    if include_examples:
        examples_block = "\n\n".join(
            f"Training Example {i}\n--\nInput:\n" +
            "\n".join(" ".join(map(str, r)) for r in ex["input"]) +
            "\n\nOutput:\n" +
            "\n".join(" ".join(map(str, r)) for r in ex["output"])
            for i, ex in enumerate(task.get("train", []), 1)
        )

    tests_block = "\n\n".join(
        f"Test Input {i}\n--\nInput:\n" +
        "\n".join(" ".join(map(str, r)) for r in t["input"])
        for i, t in enumerate(task.get("test", []), 1)
    )

    return f"""
You are an expert in visual reasoning and transformation synthesis for ARC tasks.

====================
Task {task_id}
====================

Recommended Rule Summary:
{rule_summary}

Guided Transformation Steps (for reference):
{chr(10).join(step_by_step)}

Reference Pseudocode (optional):
{pseudocode}

====================
Training Examples
====================
{examples_block}

====================
Test Inputs
====================
{tests_block}

====================
Instructions
====================
1) Use the above rules and examples as RECOMMENDED GUIDANCE — you may interpret or generalize them as needed.
2) For each test input, describe your reasoning and transformation process inside a <reasoning></reasoning> block.
3) Within the reasoning block, show intermediate reasoning and key decisions:
   <step>
   {{
      "thought": "Explain what you observed or inferred.",
      "action": "Describe the transformation applied (if any).",
      "grid_snapshot": [[...]]  // optional intermediate state
   }}
   </step>
4) Provide the final transformed grid for each test input inside <output></output>.
5) Stay consistent with the overall transformation logic demonstrated in the training examples.
6) Keep valid ARC grid formatting (2D arrays of integers).
""".strip()


def build_reflection_prompt(final_json: dict, wrong_cases: list, model_outputs: list,
                            task: dict = None, task_id: str = "") -> str:
    """
    Builds a prompt for reflecting on and refining a transformation rule based on failures.

    This function creates a detailed prompt that instructs an AI model to analyze
    incorrect outputs, identify flaws in the current rule, and propose improvements.
    It's used in an iterative refinement process for ARC task solving.

    Args:
        final_json (dict): A dictionary containing the current rule with keys like
            'rule_summary', 'step_by_step_rule', and optionally 'pseudocode'.
        wrong_cases (list): A list of tuples, where each tuple contains (input_grid, expected_output_grid)
            for cases where the current rule produced incorrect results.
        model_outputs (list): A list of the model's actual outputs for the wrong cases,
            corresponding to the wrong_cases list. Each output can be a 2D list or string.
        task (dict, optional): The full ARC task dictionary containing 'train' examples.
            Used to provide additional context. Defaults to None.
        task_id (str, optional): The unique identifier of the task. Defaults to "".

    Returns:
        str: A formatted string containing the complete reflection prompt, including
            the current rule, failure cases with inputs/expected/model outputs,
            training examples (if provided), and instructions for analysis and improvement.

    The prompt guides the model to:
    - Identify specific failures and their causes
    - Propose minimal, general fixes
    - Output an improved rule in JSON format within <json> tags
    - Show expected corrected outputs for each failure case
    - Ensure the refined rule is deterministic with tie-breakers
    """
    rule_summary = final_json.get("rule_summary", "")
    step_by_step = final_json.get("step_by_step_rule", [])
    pseudocode = final_json.get("pseudocode", "")

    wrong_block = "\n\n".join(
        f"Case {i}\n--\nInput:\n" +
        "\n".join(" ".join(map(str, r)) for r in inp) +
        "\n\nExpected Output:\n" +
        "\n".join(" ".join(map(str, r)) for r in exp) +
        "\n\nModel Output:\n" +
        ("\n".join(" ".join(map(str, r)) for r in model_outputs[i-1])
         if isinstance(model_outputs[i-1], list) else str(model_outputs[i-1]))
        for i, (inp, exp) in enumerate(wrong_cases, 1)
    )

    examples_block = ""
    if task and task.get("train"):
        examples_block = "\n\n".join(
            f"Training Example {i}\n--\nInput:\n" +
            "\n".join(" ".join(map(str, r)) for r in ex["input"]) +
            "\n\nOutput:\n" +
            "\n".join(" ".join(map(str, r)) for r in ex["output"])
            for i, ex in enumerate(task["train"], 1)
        )

    return f"""
You are a meticulous ARC reasoning assistant tasked with refining your transformation rule after observing failures.

====================
Task {task_id}
====================

Current Instruction (JSON Rule)
--------------------
Rule Summary:
{rule_summary}

Step-by-Step Rule:
{chr(10).join(step_by_step)}

Pseudocode:
{pseudocode}

====================
Observed Failures
====================
{wrong_block}

====================
Instructions for Reflection and Revision
====================
1) For each failure case, identify which assumption or omission caused the incorrect output.
2) Propose one minimal, general fix for that case.
3) Merge all fixes into a single improved JSON instruction of the same format:
  {{
    "rule_summary": "Describe the transformation conceptually.",
    "step_by_step_rule": [
      "1) ...", 
      "2) ..."]
  }}
4) Indicate which fixes correspond to which failure cases (via comments or inline notes).
5) Ensure the new rule is general, deterministic, and includes tie-breakers.
6) After the new JSON, conceptually apply it to each failed case and provide expected corrected outputs.

====================
Return Format
====================
- Step-by-step reflection for each case will be inside the <reasoning>...</reasoning> block.
- Final improved JSON inside <json>...</json>.
- For each wrong case, show expected fixed output inside <output_case_i>...</output_case_i>.
""".strip()


def build_code_repair_prompt(task_data: dict, task_id: str, previous_json: dict, execution_errors: list) -> str:
    """
    Build a code repair prompt for when JSON is valid but Python code fails to execute.
    
    Args:
        task_data: Dictionary containing all ARC tasks
        task_id: The unique identifier of the task
        previous_json: The JSON response with failing code
        execution_errors: List of execution errors for each training example
    
    Returns:
        A formatted prompt that includes the previous JSON, execution errors,
        and asks for corrected Python code.
    """
    task = task_data[task_id]
    train = task.get("train", [])
    
    # We don't need to show training examples again since errors already include input/expected output
    
    # Build execution errors block with line-specific information
    error_details = []
    for i, error_info in enumerate(execution_errors, 1):
        error_text = f"EXAMPLE {i} FAILURE:\n"
        error_text += f"Input: {error_info['input']}\n"
        error_text += f"Expected: {error_info['expected']}\n"
        
        # Add specific error information
        error_text += f"\nERROR: {error_info.get('error', 'Unknown error')}"
        
        # Add line-specific information if available
        if error_info.get('error_line'):
            error_text += f"\nFailed at: {error_info['error_line']}"
        
        if error_info.get('code_context'):
            error_text += f"\n{error_info['code_context']}"
        
        if 'predicted' in error_info and error_info['predicted'] is not None:
            error_text += f"\nActual output: {error_info['predicted']}"
        
        error_details.append(error_text)
    
    errors_block = "\n\n".join(error_details)
    
    # Format the previous JSON
    previous_json_str = json.dumps(previous_json, indent=2) if previous_json else "No previous JSON available"
    
    # Extract and format Python code with line numbers for easier reference
    python_code_with_lines = ""
    if previous_json and 'python_code' in previous_json:
        python_lines = previous_json['python_code']
        python_code_with_lines = "\n".join(f"{i+1:2d}: {line}" for i, line in enumerate(python_lines))
    
    return f"""
You are an expert Python programmer specializing in ARC puzzle transformations.
Your previous JSON response had valid structure but the Python code failed to execute on the training examples.

====================
Task {task_id} - CODE REPAIR NEEDED
====================

Your goal:
Fix the Python code in your JSON response so that it executes successfully on ALL training examples.
The JSON structure is correct, but the transformation code has execution errors that need to be resolved.

====================
Your Previous JSON (with failing code)
====================
```json
{previous_json_str}
```

====================
Current Python Code (with line numbers)
====================
{python_code_with_lines}

====================
Detailed Execution Errors
====================
{errors_block}

====================
Instructions for Code Repair
====================
Analyze the execution errors above and provide a CORRECTED JSON response with fixed Python code.

Focus on these common issues:
1. **Index out of range errors**: Check grid dimensions and array bounds (refer to line numbers above)
2. **Variable/function not defined**: Ensure all variables are properly initialized
3. **Type errors**: Make sure data types are compatible (lists, integers, etc.)
4. **Logic errors**: Verify the transformation logic matches the expected pattern
5. **Edge cases**: Handle special cases like empty grids, single values, etc.
6. **Line-specific fixes**: Pay attention to the specific lines mentioned in the error details

Provide the corrected JSON in the same format:

```json
{{
  "helper_python_functions": [
    "# Add any helper functions needed",
  ],
  "step_by_step_transformations": [{{
      "step_number": 1,
      "description": [
        "Describe what the transformation does",
      ],
      "pseudo_code": [
        "Outline the logic steps"
      ],
  }}],
  "python_code": [
    "def transform(grid):",
    "    # CORRECTED implementation that handles all the errors above",
    "    # Make sure this code executes successfully on all training examples",
    "    return processed_grid"
  ]
}}
```

CRITICAL: The Python code must execute without errors on ALL training examples. Test your logic carefully!
""".strip()


def build_arc_reflection_prompt(task_data: dict, task_id: str, previous_json: dict, train_results: list) -> str:
    """
    Build a reflection prompt for failed tasks that asks the model to analyze 
    their previous attempt and generate an improved JSON solution.
    
    Args:
        task_data: Dictionary containing all ARC tasks
        task_id: The unique identifier of the task
        previous_json: The JSON response from the previous attempt
        train_results: List of training results showing predicted vs expected outputs
    
    Returns:
        A formatted prompt that includes the previous JSON, failure analysis,
        and asks for a corrected JSON response.
    """
    task = task_data[task_id]
    train = task.get("train", [])
    
    # Build training examples block
    examples_block = "\n\n".join(
        f"Training Example {i}\n--\nInput:\n" +
        "\n".join(" ".join(map(str, r)) for r in ex["input"]) +
        "\n\nOutput:\n" +
        "\n".join(" ".join(map(str, r)) for r in ex["output"])
        for i, ex in enumerate(train, 1)
    )
    
    # Build failure analysis block
    failure_analysis = []
    for i, result in enumerate(train_results, 1):
        if not result.get('correct', False):
            predicted = result.get('predicted', 'No output generated')
            expected = result.get('expected', 'Unknown')
            error = result.get('error', 'No error reported')
            
            failure_text = f"Training Example {i} - FAILED\n"
            failure_text += f"Expected Output:\n"
            if isinstance(expected, list):
                failure_text += "\n".join(" ".join(map(str, r)) for r in expected)
            else:
                failure_text += str(expected)
            
            failure_text += f"\n\nYour Predicted Output:\n"
            if isinstance(predicted, list):
                failure_text += "\n".join(" ".join(map(str, r)) for r in predicted)
            else:
                failure_text += str(predicted)
            
            if error and error != 'No error reported':
                failure_text += f"\n\nError Encountered:\n{error}"
            
            failure_analysis.append(failure_text)
    
    failures_block = "\n\n".join(failure_analysis) if failure_analysis else "No specific failure details available"
    
    # Format the previous JSON
    previous_json_str = json.dumps(previous_json, indent=2) if previous_json else "No previous JSON available"
    
    return f"""
You are an expert in reasoning about Abstract Reasoning Corpus (ARC) puzzles. 
You previously attempted to solve this task but your solution was incorrect on some training examples.

====================
Task {task_id} - REFLECTION AND CORRECTION
====================

Your goal:
Analyze your previous attempt, understand why it failed, and provide a CORRECTED transformation rule that:
1. Correctly maps every training input to its output.
2. Is general and intuitive (no memorization or hard-coded values).
3. Is logical, reproducible, and object-level.

====================
Original Training Examples
====================
{examples_block}

====================
Your Previous JSON Response
====================
```json
{previous_json_str}
```

====================
Analysis of Failures
====================
{failures_block}

====================
Instructions for Reflection
====================
First, analyze what went wrong in your previous attempt inside ```reasoning ``` block:
- Identify which part of your logic was incorrect
- Understand why the predicted outputs didn't match the expected outputs
- Determine what pattern or rule you missed or misinterpreted
- Think about how to fix the transformation logic

Then provide a CORRECTED JSON object inside ```json ``` block with the same format:
{{
  # Python compatible code that describes
  # any helper functions needed to implement the rule.
  # Each rule will run this code before applying the transformation code.
  "helper_python_functions": [
    "...",
  ],
  "step_by_step_transformations": [{{
      "step_number": 1,
      "description": [
        "...",
      ], # Describe the transformation conceptually
      "pseudo_code": [
      ],
  }}],
  "python_code": [
    "def transform(grid):",
    "    # Complete transformation implementation",
    "    # This function must be fully executable on its own",
    "    # and return a complete transformed grid",
    "    return processed_grid"
  ]
}}

Make sure your corrected solution addresses all the failures identified above and works correctly on ALL training examples.
""".strip()


def build_arc_baseline_prompt(task_data: dict, task_id: str) -> str:
    """Build a minimal prompt that asks the model to output the test outputs.

    The prompt includes the training examples for context but then asks the model
    explicitly to return ONLY the final test output arrays in a ```json ``` block
    as a plain 2D array or list of 2D arrays (if multiple test inputs).
    """
    task = task_data[task_id]
    train, test = task.get("train", []), task.get("test", [])

    examples_block = "\n\n".join(
        f"Training Example {i}\nInput:\n" + "\n".join(" ".join(map(str, r)) for r in ex["input"]) +
        "\n\nOutput:\n" + "\n".join(" ".join(map(str, r)) for r in ex["output"]) 
        for i, ex in enumerate(train, 1)
    )

    tests_block = "\n\n".join(
        f"Test Input {i}\nInput:\n" + "\n".join(" ".join(map(str, r)) for r in t["input"]) 
        for i, t in enumerate(test, 1)
    )

    return f"""
You are an assistant that given ARC training examples should provide the
final transformed grid(s) for the TEST input(s) only.

Task: {task_id}

Training examples (for context):
{examples_block}

Test inputs:
{tests_block}

Instruction:
Return ONLY a JSON block (delimited with ```json ... ```). The JSON MUST be
either a single 2D array (if there is one test input) or a list of 2D arrays
(if there are multiple test inputs). Each 2D array should be composed of integers
matching ARC grid format. No extra text, reasoning, or explanation should be
included outside the ```json ``` block.
""".strip()