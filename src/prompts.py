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
{json.dumps(test, indent=2)}

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
  "self_check": [
    "✔ Works on all training pairs.",
    "✔ Uses roles, not absolute values.",
    "✔ Defines tie-breaking rules.",
    "✔ Likely to generalize."
  ],
}}

====================
Final instruction
====================
Analyze all training pairs together. Identify the invariant transformation pattern.
Return ONLY the JSON object described above — nothing else.
    """.strip()

    return prompt