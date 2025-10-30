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
    Build a clean, general ARC transformation prompt for an LLM.

    Args:
        task_data: dict in ARC format like:
            {
              "00576224": {
                "train": [{"input": [[...]], "output": [[...]]}, ...],
                "test":  [{"input": [[...]]}, ...]
              }
            }
        task_id: string key, e.g. "00576224"

    Returns:
        A complete prompt string ready to send to an LLM.
    """

    # === Extract task ===
    task = task_data[task_id]
    train = task.get("train", [])
    test = task.get("test", [])

    # === Format training examples ===
    formatted_examples = []
    for i, ex in enumerate(train, start=1):
        inp = "\n".join(" ".join(map(str, row)) for row in ex["input"])
        out = "\n".join(" ".join(map(str, row)) for row in ex["output"])
        formatted_examples.append(f"Training Example {i}\n--\nInput:\n{inp}\n\nOutput:\n{out}\n")

    examples_block = "\n".join(formatted_examples)

    # === Format test inputs into neat grids (instead of raw JSON dump) ===
    formatted_tests = []
    for i, t in enumerate(test, start=1):
        # test items are expected to be dicts with an 'input' key
        if isinstance(t, dict) and t.get("input") is not None:
            inp = "\n".join(" ".join(map(str, row)) for row in t.get("input", []))
        else:
            # fallback: stringify the test item
            inp = json.dumps(t)
        formatted_tests.append(f"Test Input {i}\n--\nInput:\n{inp}\n")

    tests_block = "\n".join(formatted_tests) if formatted_tests else json.dumps(test, indent=2)

    # === Construct prompt ===
    prompt = f"""
You are an expert in reasoning about Abstract Reasoning Corpus (ARC) puzzles.

Your task:
Given the following training examples and one or more test inputs, infer ONE single, general transformation rule that:
1. Correctly maps every training input to its output.
2. Is general and intuitive (no memorization or hard-coded numbers).
3. Is clear, step-by-step, logical, and reproducible.

====================
Guidelines
====================
- The SAME rule must apply consistently to all training pairs.
- Describe the rule conceptually: refer to roles such as “background color,” “largest object,” or “most frequent color,” not numeric IDs.
- Avoid referencing coordinates, dimensions, or color numbers.
- Treat numerical color or value identifiers as categorical labels: the exact numeric value does not matter and should be interpreted only as a distinct category (e.g., "background", "foreground", "marker").
- Prefer object-level reasoning: components, symmetries, translations, fills, mirrors, rotations, bounding boxes, etc.
- Include deterministic tie-breakers (e.g., topmost-leftmost if sizes tie).
- Avoid vague phrasing or direct number substitution.
- Be simple, intuitive, and consistent.

====================
Training Examples
====================
{examples_block}

====================
Test Input(s)
====================
{tests_block}

====================
Output format (strict JSON schema)
====================
Return a single JSON object with these fields:

{{
  "rule_summary": "2–4 sentences describing the transformation conceptually.",
  "step_by_step_rule": [
    "1) ...",
    "2) ..."
  ],
  "pseudocode": "Concise pseudocode representation of the transformation.",
}}

====================
Final instruction
====================
Analyze all training pairs together. Identify the invariant transformation pattern.
Shows your reasoning step-by-step. and at the end, provide the JSON object as specified above
inside a <json>...</json> block.
    """.strip()

    return prompt


def build_apply_prompt(final_json: dict, task: dict, task_id: str, include_examples: bool = True) -> str:
  """Build an APPLY_PROMPT that instructs the LLM to apply the extracted rule
  to the task's test input(s).

  final_json: parsed JSON produced by the model (dict with rule_summary, step_by_step_rule, pseudocode)
  task: the ARC task dict containing 'train' and 'test'
  task_id: the id string
  """
  rule_summary = final_json.get("rule_summary", "")
  step_by_step = final_json.get("step_by_step_rule", [])
  pseudocode = final_json.get("pseudocode", final_json.get("pseudocode", ""))

  # Format training examples
  formatted_examples = []
  if include_examples:
    for i, ex in enumerate(task.get("train", []), start=1):
      inp = "\n".join(" ".join(map(str, row)) for row in ex["input"]) if "input" in ex else ""
      out = "\n".join(" ".join(map(str, row)) for row in ex["output"]) if "output" in ex else ""
      formatted_examples.append(f"Training Example {i}\nInput:\n{inp}\n\nOutput:\n{out}\n")

  examples_block = "\n".join(formatted_examples)

  # Single test input block
  tests = task.get("test", [])
  formatted_tests = []
  for i, t in enumerate(tests, start=1):
    inp = "\n".join(" ".join(map(str, row)) for row in t.get("input", []))
    formatted_tests.append(f"Test Input {i}:\n{inp}\n")

  apply_prompt = f"""
You are an expert ARC executor.

Below is a concise rule summary, a step-by-step rule, and pseudocode derived from analyzing the training examples for task {task_id}.

Rule summary:
{rule_summary}

Step-by-step rule:
{chr(10).join(step_by_step)}

Pseudocode:
{pseudocode}

Training examples (all):
{examples_block}

Test inputs:
{chr(10).join(formatted_tests)}

Instructions:
1) Apply the above rule EXACTLY, line by line, to each test input.
2) Show your intermediate reasoning while applying the rule for each test input.
3) When you produce the final answer for each test input, include ONLY the output grid inside a single pair of tags: <output>...</output>
4) Ensure the output grid matches the expected ARC format (2D array of integers).
5) Do not include any other text outside the <output> tags for the final answer.

Return the reasoning in the <reasoning></reasoning> block and the final output in the <output></output> block for each test input.
""".strip()

  return apply_prompt


def build_reflection_prompt(final_json: dict, wrong_cases: list, model_outputs: list, task: dict = None, task_id: str = "") -> str:
  """Build a REFLECTION prompt that asks the LLM to introspect on failures and
  produce an improved `final_json` (rule_summary, step_by_step_rule, pseudocode).

  Args:
    final_json: dict produced previously with keys 'rule_summary',
      'step_by_step_rule', and 'pseudocode'.
    wrong_cases: list of tuples (input_grid, expected_output_grid) that the
      model got wrong when applying the rule.
    model_outputs: list of grids (or stringified grids) that the model actually
      produced for the corresponding wrong_cases.
    task: optional full task dict (with 'train' and 'test') for context.
    task_id: optional task id string for inclusion in the prompt.

  Returns:
    A string prompt instructing the LLM to analyze, reflect, and produce a
    revised JSON `final_json` that corrects the observed failures.
  """

  # Defensive defaults
  rule_summary = final_json.get("rule_summary", "")
  step_by_step = final_json.get("step_by_step_rule", [])
  pseudocode = final_json.get("pseudocode", "")

  # Format wrong cases
  formatted_wrong = []
  for i, (inp, expected) in enumerate(wrong_cases, start=1):
    inp_str = "\n".join(" ".join(map(str, row)) for row in inp) if inp else ""
    exp_str = "\n".join(" ".join(map(str, row)) for row in expected) if expected else ""
    model_out = model_outputs[i-1] if i-1 < len(model_outputs) else None
    model_out_str = "\n".join(" ".join(map(str, row)) for row in model_out) if isinstance(model_out, list) else str(model_out)
    formatted_wrong.append(
      f"Case {i}\n--\nInput:\n{inp_str}\n\nExpected Output:\n{exp_str}\n\nModel Output:\n{model_out_str}\n"
    )

  wrong_block = "\n\n".join(formatted_wrong)

  # Optional training examples for context
  examples_block = ""
  if task and task.get("train"):
    formatted_examples = []
    for i, ex in enumerate(task.get("train", []), start=1):
      inp = "\n".join(" ".join(map(str, row)) for row in ex.get("input", []))
      out = "\n".join(" ".join(map(str, row)) for row in ex.get("output", []))
      formatted_examples.append(f"Training Example {i}\nInput:\n{inp}\n\nOutput:\n{out}\n")
    examples_block = "\n".join(formatted_examples)

  prompt = f"""
You are a meticulous ARC reasoning assistant with the goal of improving your own
instruction (the JSON rule) after observing failures.

Context:
Task ID: {task_id}

Current extracted instruction (the JSON produced earlier):
Rule summary:
{rule_summary}

Step-by-step rule:
{chr(10).join(step_by_step)}

Pseudocode:
{pseudocode}

Observed failures (cases where your instruction produced an incorrect output):
{wrong_block}

Instructions for reflection and revision:
1) For each failure case above, analyze step-by-step how the current
   Step-by-step rule and Pseudocode would be applied to the Input. Show the
   exact intermediate steps and where they diverge from the Expected Output.
2) For each case, explicitly state which assumption, omission, or tie-breaker
   in the current rule caused the incorrect Model Output. Be concrete (e.g.,
   "rule assumed background was color X", or "pseudocode filled only single
   objects instead of grouped objects").
3) Propose one concise change to the Step-by-step rule or pseudocode that would
   fix that specific failure. Keep changes minimal and general (no dataset
   memorization). Show the revised step(s) only for each case.
4) After analyzing all failure cases, merge the per-case fixes into a single
   improved JSON instruction. The new JSON must have the same schema as before:

   {{
     "rule_summary": "2-4 sentences",
     "step_by_step_rule": ["1) ...","2) ..."],
     "pseudocode": "concise pseudocode"
   }}

5) In the new JSON, highlight (inline, in the pseudocode or an added comment)
   which of the proposed fixes correspond to which failure case numbers.
6) Ensure the new JSON is general, deterministic, and includes tie-breakers
   for ambiguous choices.
7) Finally, for verification, apply the new JSON (conceptually) to each
   previously-wrong Input and show the expected Output grid produced by the
   revised rule. If any case still fails, explain why and propose further
   modification.

Return format (strict):
1) Your step-by-step reflection and analysis for each failure case.
2) The final improved JSON object inside a single <json>...</json> block.
3) After the JSON, for each wrong case, a small block showing the expected
   output if the new JSON were applied, inside <output_case_i>...</output_case_i>
   tags (replace i with the case number). Do not include extra text outside
   these required sections.

Be concise but thorough. Focus on fixable rule-level changes rather than
 ad-hoc exceptions.
""".strip()

  return prompt