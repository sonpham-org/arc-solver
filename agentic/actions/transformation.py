"""
Transformation extraction and parsing functions for ARC agent.
"""

import re
import json
from typing import List, Dict, Tuple

from .utilities import format_grid_for_prompt, parse_transformation_steps
from agentic.debug import print_prompt_and_response


def generate_transformation_steps(llm, reasoning_trace: str, training_examples: List[Dict], num_solutions: int, max_retries: int = 3) -> Tuple[List[Dict], int]:
    """Extract clear step-by-step transformation from reasoning trace.

    Returns:
        Tuple of (solution_objects_list, num_retries_used)
        Where solution_objects_list is [{"solution_number": int, "transformation_steps": [str, ...]}, ...]
    """

    def build_transformation_steps_prompt() -> str:
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
            "",
            "TRAINING EXAMPLES",
            f"{examples_block}",
            "",
            "REASONING ANALYSIS",
            f"{reasoning_trace}",
            "",
            "INSTRUCTIONS",
            f"Produce {num_solutions} different candidate solutions. Each solution should be a numbered sequence of clear, actionable transformation steps.",
            "Be creative: try different, plausible interpretations of the reasoning so the set of solutions explores diverse approaches (use different object-level operations, orders, or heuristics).",
            "Each solution should be concise and concrete so it can be executed programmatically.",
            "",
            "RESPONSE FORMAT (JSON):",
            f"Return a JSON array in a json block containing {num_solutions} solution objects. Each object should have two keys:",
            f"- \"solution_number\": an integer (1..{num_solutions})",
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
            f"Do NOT output any additional text outside the ```json``` block. Generate the {num_solutions} solutions now."
        ]

        prompt = "\n".join(prompt_parts)
        return prompt
        
    prompt = build_transformation_steps_prompt()
    
    # Retry up to max_retries times if parsing fails
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt, temperature=0.7)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            print_prompt_and_response(prompt, response_text)

            solutions = parse_transformation_steps(response_text)
            if solutions:
                return solutions, attempt
            
            # If parsing failed and this isn't the last attempt, log and retry
            if attempt < max_retries - 1:
                print(f"Warning: Failed to parse transformation steps (attempt {attempt + 1}/{max_retries}). Retrying...")
                
        except Exception as e:
            print(f"Error extracting transformation steps (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
    
    # After all retries failed, return empty list
    return [], max_retries
