#!/usr/bin/env python3
"""
ARC Task Visualizer

Generates side-by-side images of each training example's input (left)
and output (right) for ARC tasks and saves them to `output_vis/<task_id>/`.

Usage:
  python task_visualizer.py --year 2024 --training-tasks-json data/arc-2024/arc-agi_training_challenges.json --limit 50

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


def call_llm_for_task(llm, train_examples: List[Dict[str, Any]], task_id: str, temperature: float = 1.2, embed_image_b64: Optional[str] = None, embed_summary: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Call the LLM with a prompt containing all training examples and optional embedded image.
    Returns dict with 'story' (extracted) and 'raw' (raw response string).
    """
    body = (
        "You are a creative storyteller. Given a set of training examples (INPUT -> OUTPUT),\n"
        "write an extremely creative, metaphor-rich narrative that explains the pattern or transformation across these examples.\n"
        "Relate the transformations to real-world phenomena (physics, ecology, culture, etc.).\n"
        "For clarity, include a markdown code block labeled 'story' containing the narrative.\n\n"
    )

    for i, ex in enumerate(train_examples):
        inp = ex.get('input')
        out = ex.get('output')
        body += f"EXAMPLE {i}:\nINPUT:\n{json.dumps(inp)}\nOUTPUT:\n{json.dumps(out)}\n\n"

    if embed_image_b64:
        try:
            body += f"VISUAL:\n![task](data:image/png;base64,{embed_image_b64})\n"
        except Exception:
            body += "\nVISUAL_BASE64_PRESENT\n"

    if embed_summary:
        body += f"\nVISUAL_SUMMARY:\n{embed_summary}\n"

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


def call_llm_for_pair(llm, input_grid: List[List[Any]], output_grid: List[List[Any]], task_id: str, ex_idx: int, temperature: float = 1.2, embed_image_b64: Optional[str] = None, embed_summary: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Call the LLM with a prompt, print colored prompt/response to console,
    and return a dict with keys `story` and `raw` (raw response string).
    """
    prompt = (
        "You are a creative storyteller. Given an INPUT grid and an OUTPUT grid,\n"
        "write an extremely creative, metaphor-rich story explaining how the INPUT transforms into the OUTPUT.\n"
        "Relate the transformation to real-world phenomena (physics, ecology, culture, etc.).\n"
        "Be vivid and imaginative. Output only a markdown code block labeled 'story' containing the narrative.\n\n"
        f"INPUT:\n{json.dumps(input_grid)}\n\nOUTPUT:\n{json.dumps(output_grid)}\n\n"
    )

    # If an embedded image is provided, include it in the prompt (as a data URI)
    if embed_image_b64:
        try:
            prompt += f"\nVISUAL:\n![pair](data:image/png;base64,{embed_image_b64})\n"
        except Exception:
            prompt += f"\nVISUAL_BASE64_PRESENT\n"

    if embed_summary:
        prompt += f"\nVISUAL_SUMMARY:\n{embed_summary}\n"

    # Print prompt in blue (may include embedded base64)
    try:
        print(f"{BLUE}Prompt for {task_id}#{ex_idx}:\n{prompt}{RESET}")
    except Exception:
        pass

    try:
        # Try to set temperature if the llm object supports it
        if hasattr(llm, 'temperature'):
            try:
                setattr(llm, 'temperature', float(temperature))
            except Exception:
                pass

        resp = llm.invoke(prompt)
        content = getattr(resp, 'content', resp if isinstance(resp, str) else str(resp))

        # Print LLM response in green
        try:
            print(f"{GREEN}LLM response for {task_id}#{ex_idx}:\n{content}{RESET}")
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


def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    b = buf.getvalue()
    return base64.b64encode(b).decode('ascii')


def ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def parse_args():
    p = argparse.ArgumentParser(description="ARC task visualizer: create side-by-side input/output images for training examples")
    p.add_argument("--year", type=int, default=YEAR, help=f"Year for ARC dataset (default {YEAR})")
    p.add_argument("--training-tasks-json", type=str, default=TRAINING_TASKS_JSON, help="Path to training tasks JSON")
    p.add_argument("--output-dir", type=str, default="output_vis", help="Directory to save visualization images")
    p.add_argument("--limit", type=int, default=0, help="Max number of tasks to process (0 = no limit)")
    p.add_argument("--mode", type=str, choices=["individual", "together"], default="together", help="Save images per-example (individual) or one subplot per task (together). Default: together")
    p.add_argument("--llm-model", type=str, default="gemini-2.5-flash-lite", help="Model name to initialize via initialize_llm_from_config (default: gemini-2.5-flash-lite)")
    p.add_argument("--no-llm", action="store_true", help="Do not call LLM for stories (only generate images)")
    p.add_argument("--cell-size", type=int, default=24, help="Pixel size of each grid cell")
    # Embed visual thumbnail into LLM prompt (as base64). Default: enabled
    p.add_argument("--embed-visual", dest="embed_visual", action="store_true", help="Embed a small base64 thumbnail of the pair image into the LLM prompt")
    p.add_argument("--no-embed-visual", dest="embed_visual", action="store_false", help="Do not embed visual thumbnail into the LLM prompt")
    p.set_defaults(embed_visual=True)
    p.add_argument("--thumbnail-size", type=int, default=64, help="Maximum size (px) for thumbnail embedding (default 64)")
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

        task_folder = os.path.join(out_root, task_id)
        ensure_dir(task_folder)

        # Stories collected per task
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

                # For together mode, save per-example pair images but call the LLM once for the whole task
                wrote_any = False
                if llm is not None:
                    # Save per-example pair images and record metadata
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
                        thumb.thumbnail((args.thumbnail_size, args.thumbnail_size))
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

                    raw_fname = None
                    if raw is not None:
                        try:
                            raw_fname = f"{task_id}_llm_response_task.json"
                            raw_path = os.path.join(task_folder, raw_fname)
                            with open(raw_path, 'w') as rf:
                                json.dump({'response': raw}, rf, indent=2)
                        except Exception as e:
                            print(f"Failed to save raw LLM response for {task_id}: {e}")

                    # Attach task-level story and metadata
                    stories['task'] = {
                        'story': story,
                        'thumbnail_file': thumb_fname,
                        'llm_response_file': raw_fname
                    }

                    # Save stories (including per-example entries and task-level story)
                    try:
                        with open(os.path.join(task_folder, f"{task_id}_stories.json"), 'w') as sf:
                            json.dump(stories, sf, indent=2)
                    except Exception as e:
                        print(f"Failed to save stories for {task_id}: {e}")

                # Only count this task if we wrote at least one image (either subplot or pairs)
                if subplot_img is not None:
                    wrote_any = True if ('wrote_any' not in locals() and True) else wrote_any
                if wrote_any:
                    tasks_processed += 1
            except Exception as e:
                print(f"Failed to render subplot for {task_id}: {e}")

        else:  # individual mode: create one image per example
            try:
                wrote_any = False
                for i, ex in enumerate(train_examples):
                    inp = ex.get('input')
                    out = ex.get('output')
                    if inp is None or out is None:
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

                        if llm is not None:
                            # Prepare optional embedded thumbnail and summary
                            embed_b64 = None
                            embed_summary = None
                            try:
                                if args.embed_visual and (img is not None):
                                    thumb = img.copy()
                                    thumb.thumbnail((args.thumbnail_size, args.thumbnail_size))
                                    embed_b64 = image_to_base64(thumb)
                                    embed_summary = f"pair_size:{img.width}x{img.height},cell_size:{args.cell_size}"
                                    # Save thumbnail file to output folder
                                    try:
                                        thumb_fname = f"{task_id}_thumb_{i}.png"
                                        thumb_path = os.path.join(task_folder, thumb_fname)
                                        thumb.save(thumb_path)
                                    except Exception as e:
                                        print(f"Failed to save thumbnail for {task_id}#{i}: {e}")
                            except Exception:
                                embed_b64 = None
                                embed_summary = None
                                thumb_fname = None

                            llm_result = call_llm_for_pair(llm, inp, out, task_id, i, temperature=1.4, embed_image_b64=embed_b64, embed_summary=embed_summary)
                            story = llm_result.get('story', '') if isinstance(llm_result, dict) else (llm_result or '')
                            raw = llm_result.get('raw') if isinstance(llm_result, dict) else (llm_result or None)
                            # embed image already saved as individual file above
                            try:
                                img_b64 = image_to_base64(img)
                            except Exception:
                                img_b64 = None

                            raw_fname = None
                            if raw is not None:
                                try:
                                    raw_fname = f"{task_id}_llm_response_{i}.json"
                                    raw_path = os.path.join(task_folder, raw_fname)
                                    with open(raw_path, 'w') as rf:
                                        json.dump({'response': raw}, rf, indent=2)
                                except Exception as e:
                                    print(f"Failed to save raw LLM response for {task_id}#{i}: {e}")

                            stories[str(i)] = {
                                'story': story,
                                'image_file': fname,
                                'image_b64': img_b64,
                                'thumbnail_file': thumb_fname,
                                'llm_response_file': raw_fname
                            }
                    except Exception as e:
                        print(f"Failed to render {task_id} train {i}: {e}")

                # Save stories
                if stories:
                    try:
                        with open(os.path.join(task_folder, f"{task_id}_stories.json"), 'w') as sf:
                            json.dump(stories, sf, indent=2)
                    except Exception as e:
                        print(f"Failed to save stories for {task_id}: {e}")
                # Only count task if we wrote at least one image
                if wrote_any:
                    tasks_processed += 1
            except Exception as e:
                print(f"Failed to render individual images for {task_id}: {e}")

        if task_limit is not None and tasks_processed >= task_limit:
            print(f"Reached limit of {task_limit} tasks. Done.")
            break
    print(f"Finished. Generated {tasks_processed} task subplots under {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
