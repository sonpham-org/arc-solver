"""
Reasoning generation functions for ARC agent.
"""

import re
import textwrap
import random
import json
import traceback
from typing import List, Dict, Optional, Any, Tuple

from ..schema import AgentState, CodeSolution, ExampleResult
from ..augmentation import augment_task_data
from ..debug import print_prompt_and_response
from .utilities import (
    format_grid_for_prompt, 
    format_grid_for_analysis, 
    format_difference_map, 
    _flatten_content, 
    build_steps_text_from_transformation_steps
)
from .rag import retrieve_similar_distillations
from .code_execution import test_code_on_examples

def generate_reasoning_trace(llm, training_examples: List[Dict], visual_cues: Optional[List[Dict]] = None, num_inreasoning_augmentations: int = 0, max_retries: int = 3, memory_context: str = "") -> Tuple[str, int]:
    """Generate detailed reasoning trace analyzing ARC patterns.
    
    Args:
        llm: The language model to use for generation
        training_examples: List of ARC training examples (input/output grids)
        visual_cues: Optional list of visual cues for the examples
        num_inreasoning_augmentations: Number of augmented examples to generate during reasoning
        max_retries: Maximum number of retries if extraction fails
        memory_context: Previous task context/summary for memory
    
    Returns:
        Tuple of (reasoning_trace, num_retries_used)
    """
    # Generate in-reasoning augmentations if requested
    augmented_examples = []
    if num_inreasoning_augmentations > 0:
        print(f"🔄 [Reasoning] Generating {num_inreasoning_augmentations} in-reasoning augmentations with random seeds...")
        
        # Create augmented data with different random seeds each time
        aug_data = augment_task_data(
            {"train": training_examples},
            num_augmentations=num_inreasoning_augmentations
        )
        
        if aug_data and "train" in aug_data:
            augmented_examples = aug_data["train"]
            print(f"✓ Created {len(augmented_examples)} augmented examples")
    
    # Combine original and augmented examples
    all_training_examples = training_examples + augmented_examples

    def build_initial_reasoning_prompt(training_examples: List[Dict], memory_context: str = "") -> str:
        """Build prompt for generating detailed reasoning about ARC patterns."""
        
        # Add augmentation note if applicable
        aug_note = ""
        if num_inreasoning_augmentations > 0 and augmented_examples:
            aug_note = f"\nNote: {len(augmented_examples)} augmented examples were generated using random transformations (rotations, flips, color permutations) to help identify the core pattern.\n"
        
        # Add memory context if provided
        mem_sec = ""
        if memory_context:
            mem_sec = f"\n{memory_context}\n"

        prompt_parts = [
            "You are an expert mathematician, logistician and pattern recognizier who is solving the"
            "Abstract Reasoning Corpus (ARC) problems.",
            "Your task is to deeply analyze the input-output examples and understand the underlying pattern.",
            mem_sec,
            "Focus on identifying the core transformation rule that maps inputs to outputs.",
            aug_note,
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
        
        # Add training examples with detailed formatting (includes augmented)
        for i, example in enumerate(training_examples):
            is_augmented = i >= len(training_examples) - len(augmented_examples)
            prefix = "[AUGMENTED] " if is_augmented else ""
            prompt_parts.append(f"\n{prefix}Example {i+1}:")
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
    

    prompt = build_initial_reasoning_prompt(all_training_examples, memory_context=memory_context)
    # If visual cues are provided and the llm driver supports image messages,
    # send a structured message containing the images (base64 data URLs).
    if visual_cues:
        pass

    # Retry up to max_retries times if extraction fails
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            # Flatten content to handle list/dict/string responses
            if hasattr(response, 'content'):
                resp_content = response.content
            else:
                resp_content = response
            
            response_text = _flatten_content(resp_content)
            print_prompt_and_response(prompt, response_text)

            # Extract reasoning from response
            reasoning = extract_reasoning_content(response_text)
            if reasoning and reasoning != "Unable to generate reasoning trace":
                return reasoning, attempt
            
            # If this isn't the last attempt, log and retry
            if attempt < max_retries - 1:
                print(f"Warning: Failed to extract reasoning content (attempt {attempt + 1}/{max_retries}). Retrying...")
        
        except Exception as e:
            print(f"Warning: Error in generate_reasoning_trace (attempt {attempt + 1}/{max_retries}): {e}")
            # After all retries failed
    return "Unable to generate reasoning trace", max_retries


def generate_distilled_reasoning(llm, reasoning_trace, transformation_steps, python_codes):
    """Distill a detailed `reasoning_trace` into a JSON-like structure.

    Args:
        llm: The language model to use for distillation
        reasoning_trace: The detailed reasoning trace to distill
        transformation_steps: The transformation steps associated with the solution
        python_codes: The Python code associated with the solution

    Returns:
        A dict with keys:
          - 'strategy': concise single-paragraph summary (<=150 words)
          - 'concepts': list of short concept strings

    The LLM is instructed to return ONLY valid JSON. This function will try
    To robustly parse common response shapes (```json``` fenced block, bare
    JSON, or a JSON-like substring). On failure it will produce a best-effort
    dict with empty `concepts`.
    """
    def build_distill_reasoning_prompt() -> str:
        prompt_parts = [
            "---------------",
            "REASONING TRACE",
            "---------------",
            reasoning_trace,
            "",
            "------------",
            "INSTRUCTIONS",
            "------------",
            "Read ONLY the reasoning trace above and produce a JSON object with exactly two fields:\n",
            "1) \"strategy\": a concise single-paragraph summary (<=150 words) describing the high-level strategy used to solve the task;\n",
            "2) \"concepts\": an array of short strings naming the key operations or concepts used (e.g., \"connected components\", \"symmetry\", \"color mapping\").\n",
            "Return ONLY valid JSON within a ```json```. Do NOT include any additional text, commentary, or markdown.\n",
            "Example output:",
            '```json',
            '{"strategy": "Brief summary...", "concepts": ["symmetry", "fill", "mirror"]}',
            '```',
            "Perform the distillation now:"
        ]
        return "\n".join(prompt_parts)

    prompt = build_distill_reasoning_prompt()

    try:
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        return {"strategy": f"(LLM error during distillation: {e})", "concepts": []}

    # Extract JSON candidate
    json_candidate = None
    m = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
    if m:
        json_candidate = m.group(1).strip()
    else:
        m2 = re.search(r'\{.*?\}', response_text, re.DOTALL)
        if m2 and 'strategy' in m2.group(0):
            json_candidate = m2.group(0)

    if json_candidate:
        try:
            parsed = json.loads(json_candidate)
            strategy = str(parsed.get('strategy', '')).strip()
            concepts = parsed.get('concepts') or parsed.get('concept') or []
            if isinstance(concepts, str):
                concepts = [c.strip() for c in re.split(r'[;,\n]', concepts) if c.strip()]
            elif not isinstance(concepts, (list, tuple)):
                concepts = []

            words = strategy.split()
            if len(words) > 150:
                strategy = " ".join(words[:150]) + "..."

            return {"strategy": strategy, "concepts": concepts}
        except Exception:
            pass

    # Forgiving fallback
    text = textwrap.dedent(str(response_text)).strip()
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    strategy = paragraphs[0] if paragraphs else text
    words = strategy.split()
    if len(words) > 150:
        strategy = " ".join(words[:150]) + "..."

    concepts = []
    for line in paragraphs[1:4]:
        if len(line) < 200 and (',' in line or ';' in line or line.lower().startswith('concepts') or len(line.split()) <= 6):
            cand = re.sub(r'^(concepts?:\s*)', '', line, flags=re.IGNORECASE)
            parts = [c.strip() for c in re.split(r'[;,\n]', cand) if c.strip()]
            for p in parts:
                if 2 <= len(p) <= 60:
                    concepts.append(p)
        if concepts:
            break

    return {"strategy": strategy, "concepts": concepts}


def extract_reasoning_content(response_text: str) -> str:
    """Extract reasoning content from LLM response."""
    
    # Ensure we have a string to work with
    if not isinstance(response_text, str):
        try:
            # Try to convert to string if it's not already
            if isinstance(response_text, (list, dict)):
                response_text = str(response_text)
            else:
                response_text = str(response_text)
        except Exception:
            return "Unable to generate reasoning trace"
    
    if not response_text or not response_text.strip():
        return "Unable to generate reasoning trace"
    
    # Look for reasoning block
    try:
        reasoning_match = re.search(r'```reasoning\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            return reasoning_match.group(1).strip()
    except (TypeError, AttributeError) as e:
        print(f"Warning: regex search failed in extract_reasoning_content: {e}")
        # Continue to fallback extraction methods
    
    # Fallback: look for structured content
    patterns = [
        r'PATTERN OBSERVATION:(.*?)(?=TRANSFORMATION HYPOTHESIS:|$)',
        r'TRANSFORMATION HYPOTHESIS:(.*?)(?=VERIFICATION:|$)',
        r'VERIFICATION:(.*?)(?=CORE INSIGHT:|$)',
        r'CORE INSIGHT:(.*?)(?=$)'
    ]
    
    reasoning_parts = []
    try:
        for pattern in patterns:
            match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if match:
                reasoning_parts.append(match.group(1).strip())
        
        if reasoning_parts:
            return '\n\n'.join(reasoning_parts)
    except (TypeError, AttributeError) as e:
        print(f"Warning: pattern matching failed in extract_reasoning_content: {e}")
    
    # Ultimate fallback: return first substantial paragraph
    try:
        lines = response_text.split('\n')
        substantial_lines = [line.strip() for line in lines if len(line.strip()) > 20]
        
        return '\n'.join(substantial_lines[:10]) if substantial_lines else response_text[:500]
    except Exception:
        return response_text[:500] if len(response_text) > 500 else response_text


def extract_reasoning_from_reflection(response_content: str) -> str:
    """Extract reasoning section from ARC-style reflection response."""
    
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
