#!/usr/bin/env python3
"""
ARC Task Visualizer

Generates side-by-side images of each training example's input (left)
and output (right) for ARC tasks and saves them to `output_vis/<task_id>/`.

Usage:
  python run_task_visualizer.py --year 2024 --training-tasks-json data/arc-2024/arc-agi_training_challenges.json --limit 50

The `--limit` argument limits the total number of images produced.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import shutil
from typing import Any, Dict, List, Optional
from run_multi_solution_langgraph_agent import initialize_llm_from_config
import io
import base64
import subprocess
import tempfile
import resource
import traceback
import re

# ANSI colors for console output
BLUE = "\033[34m"
GREEN = "\033[32m"
RESET = "\033[0m"

try:
    from PIL import Image, ImageDraw
except Exception as e:
    print("Pillow is required to run this script. Install with: pip install pillow")
    raise

# Default configuration
YEAR = 2024
TRAINING_TASKS_JSON = f"data/arc-{YEAR}/arc-agi_training_challenges.json"

# Simple ARC-like palette for integers 0-9
ARC_PALETTE = {
    0: (0, 0, 0),
    1: (0, 0, 255),
    2: (255, 0, 0),
    3: (0, 255, 0),
    4: (255, 255, 0),
    5: (255, 0, 255),
    6: (0, 255, 255),
    7: (255, 165, 0),
    8: (128, 128, 128),
    9: (255, 255, 255),
}


def load_arc_tasks(file_path: str) -> Dict[str, Dict]:
    with open(file_path, "r") as f:
        return json.load(f)


def value_to_color(v: Any) -> tuple:
    """Map a cell value to an RGB color.

    If the value is an int and in 0-9, use ARC palette. Otherwise hash to a color.
    """
    try:
        if isinstance(v, int):
            return ARC_PALETTE.get(v, (200, 200, 200))
        # Strings that represent small ints sometimes appear
        if isinstance(v, str) and v.isdigit():
            vi = int(v)
            return ARC_PALETTE.get(vi, (200, 200, 200))
    except Exception:
        pass

    # Fallback deterministic color based on hash
    h = abs(hash(str(v)))
    r = (h & 0xFF0000) >> 16
    g = (h & 0x00FF00) >> 8
    b = h & 0x0000FF
    return (r % 230 + 10, g % 230 + 10, b % 230 + 10)


def draw_grid(image_draw: ImageDraw.Draw, origin_x: int, origin_y: int, grid: List[List[Any]], cell_size: int = 24, line_color=(50,50,50)):
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]
            color = value_to_color(val)
            x0 = origin_x + c * cell_size
            y0 = origin_y + r * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            image_draw.rectangle([x0, y0, x1, y1], fill=color)
            # grid lines
            image_draw.rectangle([x0, y0, x1, y1], outline=line_color)


def make_side_by_side_image(input_grid: List[List[Any]], output_grid: List[List[Any]], cell_size: int = 24, gap: int = 10) -> Image.Image:
    # Create an image for a single input/output pair and return it.
    in_rows = len(input_grid)
    in_cols = len(input_grid[0]) if in_rows else 0
    out_rows = len(output_grid)
    out_cols = len(output_grid[0]) if out_rows else 0

    height = max(in_rows, out_rows) * cell_size
    width = (in_cols + out_cols) * cell_size + gap

    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw input at left
    origin_x = 0
    origin_y = 0
    draw_grid(draw, origin_x, origin_y, input_grid, cell_size)

    # Draw output at right
    origin_x = in_cols * cell_size + gap
    origin_y = 0
    draw_grid(draw, origin_x, origin_y, output_grid, cell_size)

    return img


def make_task_subplot_image(train_examples: List[Dict[str, Any]], cell_size: int = 24, gap: int = 10, max_examples: Optional[int] = None) -> Optional[Image.Image]:
    """Compose multiple input/output pairs into one vertical subplot image.

    Returns an Image or None if no valid examples.
    """
    pairs: List[Image.Image] = []
    count = 0
    for ex in train_examples:
        if max_examples is not None and count >= max_examples:
            break
        inp = ex.get("input")
        out = ex.get("output")
        if inp is None or out is None:
            continue
        try:
            pair_img = make_side_by_side_image(inp, out, cell_size=cell_size, gap=gap)
            pairs.append(pair_img)
            count += 1
        except Exception:
            continue

    if not pairs:
        return None

    # compute canvas size
    max_w = max(p.width for p in pairs)
    total_h = sum(p.height for p in pairs) + (len(pairs) - 1) * gap

    # Add top margin for headers
    header_h = 30
    canvas = Image.new("RGB", (max_w, total_h + header_h), color=(255, 255, 255))
    y = header_h
    draw = ImageDraw.Draw(canvas)

    # Draw headers for Input and Output near the top
    try:
        draw.text((10, 6), "Input", fill=(0, 0, 0))
        draw.text((max_w // 2 + 10, 6), "Output", fill=(0, 0, 0))
    except Exception:
        pass

    for p in pairs:
        # center horizontally
        x = (max_w - p.width) // 2
        canvas.paste(p, (x, y))
        y += p.height + gap

    return canvas


def extract_story_from_response(response: str) -> Optional[str]:
    if not response:
        return None
    import re
    pattern = r'```story\s*\n(.*?)\n```'
    m = re.search(pattern, response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # fallback: return whole response
    return response.strip()


def grid_to_ascii(grid: Optional[List[List[Any]]]) -> str:
    """Render a small grid as compact ASCII for prompts.

    Example:
      0 1 2
      0 0 1
    """
    if not grid:
        return "<empty>"
    try:
        lines: List[str] = []
        for row in grid:
            # represent each cell as its short repr
            parts = []
            for v in row:
                if v is None:
                    parts.append(".")
                else:
                    parts.append(str(v))
            lines.append(" ".join(parts))
        return "\n".join(lines)
    except Exception:
        return str(grid)


def grid_to_block(grid: Optional[List[List[Any]]]) -> str:
    """Render a grid as compact block without separators, e.g.
    012
    120
    000
    """
    if not grid:
        return "<empty>"
    try:
        lines: List[str] = []
        for row in grid:
            parts = []
            for v in row:
                if v is None:
                    parts.append('.')
                else:
                    parts.append(str(v))
            lines.append("".join(parts))
        return "\n".join(lines)
    except Exception:
        return str(grid)


def call_llm_for_task(llm, train_examples: List[Dict[str, Any]], task_id: str, temperature: float = 1.2, embed_image_b64: Optional[str] = None, embed_summary: Optional[str] = None, embed_image_b64s: Optional[List[Optional[str]]] = None) -> Dict[str, Optional[str]]:
    """Call the LLM with a prompt containing all training examples and optional embedded image.
    Returns dict with 'story' (extracted) and 'raw' (raw response string).
    """
    # Build prompt listing all examples
    body = (
        "You are a creative storyteller. Given a set of training examples (INPUT -> OUTPUT),\n"
        "write an extremely creative, metaphor-rich narrative that explains the pattern or transformation across these examples.\n"
        "Relate the transformations to real-world phenomena (physics, ecology, culture, etc.).\n"
        "For clarity, include a markdown code block labeled 'story' containing the narrative.\n\n"
    )

    for i, ex in enumerate(train_examples):
        inp = ex.get('input')
        out = ex.get('output')
        # Include both a compact block visual and the JSON for precision
        block_in = grid_to_block(inp)
        block_out = grid_to_block(out)
        body += (
            f"EXAMPLE {i}:\n"
            "BEFORE (INPUT) - compact block:\n"
            f"{block_in}\n\n"
            "AFTER (OUTPUT) - compact block:\n"
            f"{block_out}\n\n"
            "(JSON for reference)\n"
            f"INPUT_JSON:\n{json.dumps(inp)}\nOUTPUT_JSON:\n{json.dumps(out)}\n\n"
        )
        # attach per-example embedded image if provided
        if embed_image_b64s and i < len(embed_image_b64s) and embed_image_b64s[i]:
            try:
                body += f"VISUAL_EXAMPLE_{i}:\n![pair](data:image/png;base64,{embed_image_b64s[i]})\n"
            except Exception:
                body += f"\nVISUAL_EXAMPLE_{i}_PRESENT\n"

    # Tell the model to use its visual capabilities for the embedded thumbnail
    if embed_image_b64:
        try:
            body += f"VISUAL (use your visual capabilities):\n![task](data:image/png;base64,{embed_image_b64})\n"
        except Exception:
            body += "\nVISUAL_BASE64_PRESENT\n"

    if embed_summary:
        body += f"\nVISUAL_SUMMARY:\n{embed_summary}\n"

    # Force a single story block that explains the common transformation across examples.
    body += (
        "\nINSTRUCTION: Assume these examples all follow the same transformation.\n"
        "Produce exactly one markdown code block labeled 'story' that contains a single, self-contained narrative explaining the shared transformation.\n"
        "Do NOT answer SAME or DIFFERENT — always provide the single 'story' block.\n"
        "Use your visual capabilities to ground descriptions in the embedded image and the compact ASCII grids above.\n\n"
    )

    # Print prompt in blue
    try:
        print(f"{BLUE}Prompt for task {task_id}:\n{body}{RESET}")
    except Exception:
        pass

    try:
        if hasattr(llm, 'temperature'):
            try:
                setattr(llm, 'temperature', float(temperature))
            except Exception:
                pass

        resp = llm.invoke(body)
        content = getattr(resp, 'content', resp if isinstance(resp, str) else str(resp))
        try:
            print(f"{GREEN}LLM response for task {task_id}:\n{content}{RESET}")
        except Exception:
            pass

        story = extract_story_from_response(content)
        return {"story": story or "", "raw": content}
    except Exception as e:
        err = f"LLM error: {e}"
        try:
            print(f"{GREEN}{err}{RESET}")
        except Exception:
            pass
        return {"story": "", "raw": err}


def call_llm_for_pair(llm, input_grid: List[List[Any]], output_grid: List[List[Any]], task_id: str, ex_idx: int, temperature: float = 1.2) -> str:
    # Build a concise prompt asking for a creative story in a ```story``` block
    block_in = grid_to_block(input_grid)
    block_out = grid_to_block(output_grid)
    prompt = (
        "You are a creative storyteller. Given an INPUT grid and an OUTPUT grid,\n"
        "write an extremely creative, metaphor-rich story explaining how the INPUT transforms into the OUTPUT.\n"
        "Relate the transformation to real-world phenomena (physics, ecology, culture, etc.).\n"
        "Be vivid and imaginative. Output only a markdown code block labeled 'story' containing the narrative.\n\n"
        f"INPUT (block):\n{block_in}\n\nOUTPUT (block):\n{block_out}\n\n"
    )

    try:
        # Try to set temperature if the llm object supports it
        if hasattr(llm, 'temperature'):
            try:
                setattr(llm, 'temperature', float(temperature))
            except Exception:
                pass
        resp = llm.invoke(prompt)
        content = getattr(resp, 'content', resp if isinstance(resp, str) else str(resp))
        story = extract_story_from_response(content)
        return story or ""
    except Exception as e:
        return f"LLM error: {e}"


def extract_code_from_response(response: str) -> Optional[str]:
    if not response:
        return None
    # extract first python code block
    m = re.search(r"```(?:python)?\s*\n(.*?)\n```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def generate_code_from_story(llm, story: str, task_id: str, examples: List[Dict[str, Any]], temperature: float = 0.8) -> Optional[str]:
    prompt = (
        "You are a Python programmer. Given the narrative story that explains how to transform INPUT grids to OUTPUT grids, "
        "write a self-contained Python module that implements a function `transform(grid)` which performs the described transformation. "
        "Include any necessary imports and helper functions. The function should accept the same nested-list representation as the ARC examples and return the transformed nested-list.\n\n"
        "Here is the story (for reference):\n" + story + "\n\n"
        "Provide only the Python code inside a ```python``` code block. The code should not read or write files and must define `transform(grid)`.\n"
    )
    try:
        if hasattr(llm, 'temperature'):
            try:
                setattr(llm, 'temperature', float(temperature))
            except Exception:
                pass
        resp = llm.invoke(prompt)
        content = getattr(resp, 'content', resp if isinstance(resp, str) else str(resp))
        code = extract_code_from_response(content)
        return code
    except Exception:
        return None


def evaluate_candidate_code(code_str: str, test_inputs: List[Any], timeout: int = 6) -> Dict[str, Any]:
    """Execute the provided Python `code_str` in-process using `exec` in a restricted namespace,
    then call `transform` on each test input and return outputs and status.
    """
    result: Dict[str, Any] = {'success': False, 'outputs': None, 'error': None}
    try:
        # Restrict builtins to a small safe subset
        safe_builtins = {
            'range': range, 'len': len, 'min': min, 'max': max, 'sum': sum,
            'any': any, 'all': all, 'enumerate': enumerate, 'list': list,
            'dict': dict, 'set': set, 'tuple': tuple, 'abs': abs
        }
        gl: Dict[str, Any] = {'__builtins__': safe_builtins}
        loc: Dict[str, Any] = {}
        try:
            exec(code_str, gl, loc)
        except Exception:
            # include trace in error
            result['error'] = traceback.format_exc()
            return result

        transform = gl.get('transform') or loc.get('transform')
        if not callable(transform):
            result['error'] = 'no_transform'
            return result

        outs: List[Any] = []
        for t in test_inputs:
            try:
                o = transform(t)
                outs.append(o)
            except Exception:
                outs.append({'error': traceback.format_exc()})

        result['outputs'] = outs
        result['success'] = True
    except Exception:
        result['error'] = traceback.format_exc()
    return result


def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    b = buf.getvalue()
    return base64.b64encode(b).decode('ascii')


def compare_expected_actual(expected: Any, actual: Any) -> Dict[str, Any]:
    """Compare expected and actual ARC-style grids.

    Returns a dict containing per-test metrics:
      - exact_match: bool (deep equality)
      - size_match: bool (rows and cols equal when both grids)
      - numeric_total: int (number of numeric cells in expected)
      - numeric_matches: int (how many numeric cells match in actual)
      - overlap_ratio: float (numeric_matches / max(1, numeric_total))
      - pass: bool (size_match and all numeric positions match)
    """
    out: Dict[str, Any] = {
        'exact_match': False,
        'size_match': False,
        'numeric_total': 0,
        'numeric_matches': 0,
        'overlap_ratio': 0.0,
        'pass': False,
    }
    try:
        if expected == actual:
            out['exact_match'] = True

        # If both are grids (list of lists), compute shape and numeric overlap
        if isinstance(expected, list) and expected and isinstance(expected[0], list) and isinstance(actual, list) and actual and isinstance(actual[0], list):
            er = len(expected)
            ec = len(expected[0])
            ar = len(actual)
            ac = len(actual[0])
            out['size_match'] = (er == ar and ec == ac)

            numeric_total = 0
            numeric_matches = 0
            for r in range(min(er, ar)):
                for c in range(min(ec, ac)):
                    ev = expected[r][c]
                    av = actual[r][c]
                    # consider numeric if int or digit-string
                    is_numeric = False
                    ev_int = None
                    try:
                        if isinstance(ev, int):
                            is_numeric = True
                            ev_int = int(ev)
                        elif isinstance(ev, str) and ev.isdigit():
                            is_numeric = True
                            ev_int = int(ev)
                    except Exception:
                        is_numeric = False

                    if is_numeric:
                        numeric_total += 1
                        try:
                            if isinstance(av, int) and int(av) == ev_int:
                                numeric_matches += 1
                            elif isinstance(av, str) and av.isdigit() and int(av) == ev_int:
                                numeric_matches += 1
                        except Exception:
                            pass

            out['numeric_total'] = numeric_total
            out['numeric_matches'] = numeric_matches
            out['overlap_ratio'] = float(numeric_matches) / max(1, numeric_total)

            # define pass as size match and all numeric positions match (or no numeric positions)
            out['pass'] = out['size_match'] and (numeric_total == 0 or numeric_matches == numeric_total)
        else:
            # Fallback: use equality
            out['size_match'] = True
            out['numeric_total'] = 0
            out['numeric_matches'] = 0
            out['overlap_ratio'] = 1.0 if out['exact_match'] else 0.0
            out['pass'] = out['exact_match']
    except Exception as e:
        out['error'] = str(e)

    return out


def ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def parse_args():
    p = argparse.ArgumentParser(description="ARC task visualizer: create side-by-side input/output images for training examples")
    p.add_argument("--year", type=int, default=YEAR, help=f"Year for ARC dataset (default {YEAR})")
    p.add_argument("--training-tasks-json", type=str, default=TRAINING_TASKS_JSON, help="Path to training tasks JSON")
    p.add_argument("--training-solutions-json", type=str, default=f"data/arc-{{YEAR}}/arc-agi_evaluation_solutions.json", help="Path to training solutions JSON (expected outputs). Use {YEAR} to substitute the year.")
    p.add_argument("--output-dir", type=str, default="output_vis", help="Directory to save visualization images")
    p.add_argument("--limit", type=int, default=0, help="Max number of tasks to process (0 = no limit)")
    p.add_argument("--mode", type=str, choices=["individual", "together"], default="together", help="Save images per-example (individual) or one subplot per task (together). Default: together")
    p.add_argument("--llm-model", type=str, default="gemini-2.5-flash-lite", help="Model name to initialize via initialize_llm_from_config (default: gemini-2.5-flash-lite)")
    p.add_argument("--no-llm", action="store_true", help="Do not call LLM for stories (only generate images)")
    p.add_argument("--cell-size", type=int, default=24, help="Pixel size of each grid cell")
    return p.parse_args()


def main():
    args = parse_args()

    # If user passed year but left training json default, try to replace year in default path
    tasks_json = args.training_tasks_json
    if "{YEAR}" in tasks_json:
        tasks_json = tasks_json.replace("{YEAR}", str(args.year))

    # If default path was used, update it with provided year
    if tasks_json == TRAINING_TASKS_JSON and args.year != YEAR:
        tasks_json = f"data/arc-{args.year}/arc-agi_training_challenges.json"

    if not os.path.exists(tasks_json):
        print(f"Training tasks JSON not found: {tasks_json}")
        return 1

    tasks = load_arc_tasks(tasks_json)

    # Load training solutions (expected outputs) if provided
    solutions_json = args.training_solutions_json
    if "{YEAR}" in solutions_json:
        solutions_json = solutions_json.replace("{YEAR}", str(args.year))

    solutions_map: Dict[str, Any] = {}
    if os.path.exists(solutions_json):
        try:
            with open(solutions_json, 'r') as sf:
                solutions_map = json.load(sf)
            print(f"Loaded solutions for {len(solutions_map)} tasks from {solutions_json}")
        except Exception as e:
            print(f"Warning: failed to load solutions JSON {solutions_json}: {e}")
            solutions_map = {}
    else:
        # Not fatal; continue but evaluations that expect solution data will be empty
        print(f"Solutions JSON not found: {solutions_json}. Falling back to task 'test' outputs (may be missing).")

    out_root = os.path.abspath(args.output_dir)

    # Clear the output folder before each run (remove all files/subfolders)
    if os.path.exists(out_root):
        try:
            for name in os.listdir(out_root):
                path = os.path.join(out_root, name)
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
        except Exception as e:
            print(f"Warning: couldn't fully clear output directory {out_root}: {e}")
    else:
        ensure_dir(out_root)

    tasks_processed = 0
    tasks_successful_count = 0
    task_limit = args.limit if args.limit and args.limit > 0 else None

    print(f"Loaded {len(tasks)} tasks from {tasks_json}")
    print(f"Saving images to: {out_root}")

    # Initialize LLM if requested
    llm = None
    if not getattr(args, 'no_llm', False):
        try:
            llm = initialize_llm_from_config(args.llm_model)
            if llm is None:
                print(f"Warning: could not initialize LLM '{args.llm_model}', continuing without stories")
        except Exception as e:
            print(f"Warning: error initializing LLM: {e}")
            llm = None

    for task_id, task_data in tasks.items():
        if task_limit is not None and tasks_processed >= task_limit:
            break

        train_examples = task_data.get("train", [])
        if not train_examples:
            continue

        # Determine expected outputs and test inputs for evaluation.
        # Prefer the external `solutions_map` if it contains entries for this task_id;
        # otherwise fall back to the task's `test` entries.
        if isinstance(solutions_map, dict) and task_id in solutions_map:
            expected_outputs = solutions_map.get(task_id) or []
            # Ensure expected outputs is a list of grids
            if not isinstance(expected_outputs, list):
                expected_outputs = []
        else:
            expected_outputs = [t.get('output') for t in task_data.get('test', [])]

        test_inputs = [t.get('input') for t in task_data.get('test', [])]

        task_folder = os.path.join(out_root, task_id)
        ensure_dir(task_folder)

        stories: Dict[str, str] = {}

        if args.mode == 'together':
            try:
                subplot_img = make_task_subplot_image(train_examples, cell_size=args.cell_size, gap=10, max_examples=None)
                if subplot_img is None:
                    continue

                fname = f"{task_id}_train_subplot.png"
                fpath = os.path.join(task_folder, fname)
                subplot_img.save(fpath)
                print(f"Wrote subplot {fpath}")

                wrote_any = False
                if llm is not None:
                    # First, create and save per-example pair images (no per-example LLM calls)
                    for i, ex in enumerate(train_examples):
                        inp = ex.get('input')
                        out = ex.get('output')
                        if inp is None or out is None:
                            continue
                        try:
                            pair_img = make_side_by_side_image(inp, out, cell_size=args.cell_size)
                            pair_fname = f"{task_id}_pair_{i}.png"
                            pair_path = os.path.join(task_folder, pair_fname)
                            pair_img.save(pair_path)
                            try:
                                img_b64 = image_to_base64(pair_img)
                            except Exception:
                                img_b64 = None
                            wrote_any = True
                        except Exception:
                            pair_fname = None
                            img_b64 = None

                        stories[str(i)] = {
                            'image_file': pair_fname,
                            'image_b64': img_b64
                        }

                    # Create a thumbnail of the combined subplot and embed it in one prompt
                    embed_b64 = None
                    embed_summary = None
                    thumb_fname = None
                    try:
                        thumb = subplot_img.copy()
                        thumb.thumbnail((getattr(args, 'thumbnail_size', 64), getattr(args, 'thumbnail_size', 64)))
                        embed_b64 = image_to_base64(thumb)
                        embed_summary = f"task_pairs:{len(train_examples)},size:{subplot_img.width}x{subplot_img.height},cell_size:{args.cell_size}"
                        try:
                            thumb_fname = f"{task_id}_thumb_task.png"
                            thumb_path = os.path.join(task_folder, thumb_fname)
                            thumb.save(thumb_path)
                        except Exception as e:
                            print(f"Failed to save task thumbnail for {task_id}: {e}")
                    except Exception:
                        embed_b64 = None
                        embed_summary = None
                        thumb_fname = None

                    # Call LLM once with all training examples and the combined visual
                    try:
                        llm_result = call_llm_for_task(llm, train_examples, task_id, temperature=1.4, embed_image_b64=embed_b64, embed_summary=embed_summary)
                        story = llm_result.get('story', '') if isinstance(llm_result, dict) else (llm_result or '')
                        raw = llm_result.get('raw') if isinstance(llm_result, dict) else (llm_result or None)
                    except Exception as e:
                        story = ''
                        raw = f"LLM error: {e}"

                    # Debug: show story extraction info and attempt code generation in together mode
                    try:
                        print(f"[DEBUG] Extracted story length: {len(story) if story else 0}")
                        if story:
                            print(f"[DEBUG] Story (first 800 chars):\n{story[:800]}")
                        else:
                            print("[DEBUG] No story extracted from LLM response.")
                    except Exception:
                        pass

                    # Try to generate candidate code even in together mode for debugging
                    candidate_code = None
                    candidate_fname = None
                    eval_result = None
                    if story and llm is not None:
                        try:
                            print(f"[DEBUG] Attempting to generate candidate code from story for task {task_id}...")
                            candidate_code = generate_code_from_story(llm, story, task_id, train_examples)
                            if candidate_code:
                                print(f"[DEBUG] Candidate code generated (len={len(candidate_code)}). Saving to file.")
                                candidate_fname = f"{task_id}_candidate.py"
                                candidate_path = os.path.join(task_folder, candidate_fname)
                                with open(candidate_path, 'w') as cf:
                                    cf.write(candidate_code)
                                # evaluate on test cases (if any)
                                test_examples = [t.get('input') for t in task_data.get('test', [])]
                                eval_result = evaluate_candidate_code(candidate_code, test_examples)
                                print(f"[DEBUG] Evaluation result: success={eval_result.get('success')}\n")
                            else:
                                print(f"[DEBUG] No candidate code returned by generate_code_from_story for task {task_id}.")
                        except Exception as e:
                            print(f"[DEBUG] Error generating/evaluating candidate for {task_id}: {e}")

                    raw_fname = None
                    if raw is not None:
                        try:
                            raw_fname = f"{task_id}_llm_response_task.json"
                            raw_path = os.path.join(task_folder, raw_fname)
                            with open(raw_path, 'w') as rf:
                                json.dump({'response': raw}, rf, indent=2)
                        except Exception as e:
                            print(f"Failed to save raw LLM response for task {task_id}: {e}")

                    # Attach task-level story and metadata (include candidate/eval if present)
                    stories['task'] = {
                        'story': story,
                        'thumbnail_file': thumb_fname,
                        'llm_response_file': raw_fname,
                        'candidate_code_file': candidate_fname,
                        'evaluation': eval_result
                    }

                    # Add summary info if evaluation present
                    try:
                        if isinstance(eval_result, dict):
                            score = eval_result.get('score')
                            per_test = eval_result.get('per_test')
                            passed = 0
                            total = 0
                            if isinstance(per_test, list) and per_test:
                                total = len(per_test)
                                passed = sum(1 for p in per_test if p.get('pass'))
                            elif score is not None:
                                try:
                                    total = len(expected_outputs)
                                    passed = int(round(float(score) * total))
                                except Exception:
                                    total = 0
                                    passed = 0

                            results_text = None
                            if total > 0:
                                results_text = f"Score: {score}  Passed: {passed}/{total} tests."
                            elif score is not None:
                                results_text = f"Score: {score}"
                            else:
                                results_text = "No evaluation results."

                            eval_result.setdefault('results_text', results_text)
                            if 'candidate_code' not in eval_result and candidate_code:
                                eval_result['candidate_code'] = candidate_code

                            stories['summary'] = {
                                'task_id': task_id,
                                'score': score,
                                'passed': passed,
                                'total_tests': total,
                                'results_text': results_text,
                                'candidate_code_file': candidate_fname
                            }
                    except Exception:
                        pass

                    # Save stories (including per-example entries and task-level story)
                    try:
                        with open(os.path.join(task_folder, f"{task_id}_stories.json"), 'w') as sf:
                            json.dump(stories, sf, indent=2)
                    except Exception as e:
                        print(f"Failed to save stories for {task_id}: {e}")

                    # If we evaluated candidate code, attach expected/predicted formatted blocks and print them
                    try:
                        if isinstance(eval_result, dict) and eval_result.get('outputs') is not None:
                            expected_outputs = [t.get('output') for t in task_data.get('test', [])]
                            actual_outputs = eval_result.get('outputs')
                            formatted_expected = []
                            formatted_actual = []
                            for exp, act in zip(expected_outputs, actual_outputs):
                                formatted_expected.append(grid_to_block(exp))
                                formatted_actual.append(grid_to_block(act) if isinstance(act, list) else str(act))
                            eval_result['expected_formatted'] = formatted_expected
                            eval_result['predicted_formatted'] = formatted_actual
                            # print concise per-test comparison
                            print(f"[RESULTS] Task {task_id}: {eval_result.get('results_text', 'no results')}")
                            for idx, (e_f, a_f) in enumerate(zip(formatted_expected, formatted_actual)):
                                print(f"-- Test {idx} expected:\n{e_f}\n   predicted:\n{a_f}\n")
                    except Exception:
                        pass

                if subplot_img is not None:
                    wrote_any = True if ('wrote_any' not in locals() and True) else wrote_any
                if wrote_any:
                    # count this task as processed
                    tasks_processed += 1
                    # mark success if evaluation indicates full pass
                    try:
                        if isinstance(eval_result, dict) and eval_result.get('score') is not None:
                            if float(eval_result.get('score')) >= 1.0:
                                tasks_successful_count += 1
                    except Exception:
                        pass
            except Exception as e:
                print(f"Failed to render subplot for {task_id}: {e}")

        else:
            try:
                # In individual mode we still call the LLM once per task, but include
                # each example's picture/thumbnail inline after the example in the prompt.
                wrote_any = False
                embed_image_b64s: List[Optional[str]] = []
                thumb_fnames: List[Optional[str]] = []

                # Save per-example pair images and prepare per-example thumbnails
                for i, ex in enumerate(train_examples):
                    inp = ex.get('input')
                    out = ex.get('output')
                    if inp is None or out is None:
                        embed_image_b64s.append(None)
                        thumb_fnames.append(None)
                        continue

                    try:
                        img = make_side_by_side_image(inp, out, cell_size=args.cell_size)
                        # Add small header labels
                        draw = ImageDraw.Draw(img)
                        try:
                            draw.text((5, 2), 'Input', fill=(0, 0, 0))
                            draw.text((img.width // 2 + 5, 2), 'Output', fill=(0, 0, 0))
                        except Exception:
                            pass

                        fname = f"{task_id}_train_{i}.png"
                        fpath = os.path.join(task_folder, fname)
                        img.save(fpath)
                        print(f"Wrote {fpath}")
                        wrote_any = True

                        # create and save thumbnail for embedding
                        try:
                            thumb = img.copy()
                            thumb.thumbnail((getattr(args, 'thumbnail_size', 64), getattr(args, 'thumbnail_size', 64)))
                            embed_b64 = image_to_base64(thumb)
                            thumb_fname = f"{task_id}_thumb_{i}.png"
                            thumb_path = os.path.join(task_folder, thumb_fname)
                            try:
                                thumb.save(thumb_path)
                            except Exception:
                                pass
                        except Exception:
                            embed_b64 = None
                            thumb_fname = None

                        embed_image_b64s.append(embed_b64)
                        thumb_fnames.append(thumb_fname)

                        # record basic metadata
                        try:
                            img_b64 = image_to_base64(img)
                        except Exception:
                            img_b64 = None
                        stories[str(i)] = {
                            'image_file': fname,
                            'image_b64': img_b64,
                            'thumbnail_file': thumb_fname
                        }
                    except Exception as e:
                        print(f"Failed to render {task_id} train {i}: {e}")

                # Call LLM once with all examples and per-example visuals
                if llm is not None:
                    try:
                        llm_result = call_llm_for_task(llm, train_examples, task_id, temperature=1.4, embed_image_b64s=embed_image_b64s)
                        story = llm_result.get('story', '') if isinstance(llm_result, dict) else (llm_result or '')
                        raw = llm_result.get('raw') if isinstance(llm_result, dict) else (llm_result or None)
                    except Exception as e:
                        story = ''
                        raw = f"LLM error: {e}"

                    # Save raw response
                    raw_fname = None
                    if raw is not None:
                        try:
                            raw_fname = f"{task_id}_llm_response_task.json"
                            raw_path = os.path.join(task_folder, raw_fname)
                            with open(raw_path, 'w') as rf:
                                json.dump({'response': raw}, rf, indent=2)
                        except Exception as e:
                            print(f"Failed to save raw LLM response for {task_id}: {e}")

                    # Generate Python candidate code from the story
                    candidate_code = None
                    try:
                        print(f"[DEBUG] Calling generate_code_from_story for task {task_id}...")
                        candidate_code = generate_code_from_story(llm, story, task_id, train_examples)
                        if candidate_code is None:
                            print(f"[DEBUG] generate_code_from_story returned None for task {task_id}.")
                        else:
                            print(f"[DEBUG] Candidate code length for task {task_id}: {len(candidate_code)}")
                    except Exception as e:
                        candidate_code = None
                        print(f"[DEBUG] Exception in generate_code_from_story: {e}")

                    candidate_fname = None
                    eval_result = None
                    if candidate_code:
                        try:
                            candidate_fname = f"{task_id}_candidate.py"
                            candidate_path = os.path.join(task_folder, candidate_fname)
                            with open(candidate_path, 'w') as cf:
                                cf.write(candidate_code)
                            # evaluate on test cases (if any)
                            test_examples = [t.get('input') for t in task_data.get('test', [])]
                            eval_result = evaluate_candidate_code(candidate_code, test_examples)
                            # If we have evaluation outputs and expected outputs, compute size+numeric overlap scoring
                            try:
                                expected_outputs = [t.get('output') for t in task_data.get('test', [])]
                                actual_outputs = eval_result.get('outputs') if isinstance(eval_result, dict) else None
                                if expected_outputs and actual_outputs:
                                    per_test = []
                                    pass_count = 0
                                    for exp, act in zip(expected_outputs, actual_outputs):
                                        comp = compare_expected_actual(exp, act)
                                        per_test.append(comp)
                                        if comp.get('pass'):
                                            pass_count += 1
                                    overall = float(pass_count) / max(1, len(expected_outputs))
                                    eval_result['per_test'] = per_test
                                    eval_result['score'] = overall
                                    # Also store the candidate Python code that produced these outputs
                                    try:
                                        if isinstance(eval_result, dict):
                                            eval_result['candidate_code'] = candidate_code
                                    except Exception:
                                        pass
                            except Exception as e:
                                # Keep eval_result but annotate the scoring error
                                try:
                                    if isinstance(eval_result, dict):
                                        eval_result.setdefault('score_error', str(e))
                                except Exception:
                                    pass
                        except Exception as e:
                            eval_result = {'error': str(e)}

                    # Attach task-level story and metadata
                    stories['task'] = {
                        'story': story,
                        'thumbnail_files': thumb_fnames,
                        'llm_response_file': raw_fname,
                        'candidate_code_file': candidate_fname,
                        'evaluation': eval_result
                    }

                    # Add a concise human-readable summary and embed candidate code into evaluation JSON
                    try:
                        if isinstance(eval_result, dict):
                            score = eval_result.get('score')
                            per_test = eval_result.get('per_test')
                            passed = 0
                            total = 0
                            if isinstance(per_test, list) and per_test:
                                total = len(per_test)
                                passed = sum(1 for p in per_test if p.get('pass'))
                            elif score is not None:
                                # best-effort reconstruct
                                try:
                                    total = len(expected_outputs)
                                    passed = int(round(float(score) * total))
                                except Exception:
                                    total = 0
                                    passed = 0

                            results_text = None
                            if total > 0:
                                results_text = f"Score: {score}  Passed: {passed}/{total} tests."
                            elif score is not None:
                                results_text = f"Score: {score}"
                            else:
                                results_text = "No evaluation results."

                            eval_result.setdefault('results_text', results_text)
                            # also include the candidate code string into evaluation for easy inspection
                            try:
                                if 'candidate_code' not in eval_result and candidate_code:
                                    eval_result['candidate_code'] = candidate_code
                            except Exception:
                                pass

                            # put a summary at top-level of stories (placed at end when serialized)
                            stories['summary'] = {
                                'task_id': task_id,
                                'score': score,
                                'passed': passed,
                                'total_tests': total,
                                'results_text': results_text,
                                'candidate_code_file': candidate_fname
                            }
                    except Exception:
                        pass

                    # Save stories
                    try:
                        with open(os.path.join(task_folder, f"{task_id}_stories.json"), 'w') as sf:
                            json.dump(stories, sf, indent=2)
                    except Exception as e:
                        print(f"Failed to save stories for {task_id}: {e}")

                    # Print concise summary to console as well
                    try:
                        if isinstance(stories.get('summary'), dict):
                            print(f"Task {task_id} evaluation: {stories['summary']['results_text']}")
                    except Exception:
                        pass

                if wrote_any:
                    tasks_processed += 1
                    try:
                        if isinstance(eval_result, dict) and eval_result.get('score') is not None:
                            if float(eval_result.get('score')) >= 1.0:
                                tasks_successful_count += 1
                    except Exception:
                        pass
            except Exception as e:
                print(f"Failed to render individual images for {task_id}: {e}")

        if task_limit is not None and tasks_processed >= task_limit:
            print(f"Reached limit of {task_limit} tasks. Done.")
            break

    print(f"Finished. Generated {tasks_processed} task subplots under {out_root}")
    # Print overall success percentage
    try:
        pct = float(tasks_successful_count) / tasks_processed * 100.0 if tasks_processed > 0 else 0.0
        print(f"Overall success: {tasks_successful_count}/{tasks_processed} tasks passed = {pct:.1f}%")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
