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
Given the training pairs and test inputs, infer ONE single, general transformation rule that:
1. Correctly maps every training input to its output.
2. Is general and intuitive (no memorization or hard-coded values).
3. Is logical, reproducible, and object-level.

====================
Guidelines
====================
- The SAME rule must apply consistently to all training pairs.
- Describe concepts (e.g. “background color,” “largest object”) instead of numeric IDs.
- Treat numbers as categorical labels, not magnitudes.
- Prefer object-level reasoning (shapes, fills, mirrors, rotations, bounding boxes, etc.).
- Include deterministic tie-breakers (e.g. “topmost-leftmost” for ties).
- Avoid vague phrasing or direct value substitution.

====================
Training Examples
====================
{examples_block}

====================
Test Input(s)
====================
{tests_block}

====================
Output Format
====================
Return your result as a JSON object inside <json>...</json>:
{{
  "rule_summary": "Describe the transformation conceptually.",
  "step_by_step_rule": [
    "1) ...", 
    "2) ..."]
}}

====================
Final Instruction
====================
Analyze all training pairs together.
Show your reasoning step-by-step, then output the JSON object inside <json> tags.
""".strip()


def build_apply_prompt(final_json: dict, task: dict, task_id: str, include_examples=True) -> str:
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
  
    "step_number": i,
    "description": "Describe what is being done in this step.",
    "before": [[...]],   // grid state before applying this step
    "after":  [[...]]    // grid state after applying this step
  
  </step>
4) Provide ONLY the final output grid for each test input inside <output></output>.
5) Ensure valid ARC format (2D array of integers).
6) No text outside of <reasoning> and <output> blocks.
""".strip()


def build_reflection_prompt(final_json: dict, wrong_cases: list, model_outputs: list,
                            task: dict = None, task_id: str = "") -> str:
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