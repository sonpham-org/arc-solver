"""
Pure utility functions for ARC agent.
"""

import re
import json
from typing import List, Dict, Optional, Any, Tuple
from agentic.debug import print_prompt_and_response


def format_grid_for_prompt(grid: List[List[int]], indent: int = 0) -> str:
    """Format grid for display in prompts."""
    if grid is None:
        grid = []
    indentation = " " * indent
    return "\n".join(indentation + " ".join(map(str, row)) for row in grid)


def format_grid_for_analysis(grid: List[List[int]]) -> str: 
    """Format grid for detailed analysis in reasoning prompts."""
    if not grid:
        return "(empty grid)"
    
    formatted_rows = []
    for row in grid:
        formatted_rows.append("".join(str(cell) for cell in row))
    
    return "\n".join(formatted_rows)


def format_difference_map(predicted: Optional[List[List[int]]], expected: Optional[List[List[int]]], indent: int = 0) -> str:
    """Return a visual difference map where '.' indicates a matching cell and 'X' a mismatch.

    If the predicted and expected grids have different sizes, the non-overlapping
    area is marked with 'X'. The returned string contains one line per row.
    """
    if not expected:
        return "(no expected output)"

    pred_h = len(predicted) if predicted else 0
    pred_w = len(predicted[0]) if pred_h > 0 and predicted[0] else 0
    exp_h = len(expected) if expected else 0
    exp_w = len(expected[0]) if exp_h > 0 and expected[0] else 0

    h = max(exp_h, pred_h)
    w = max(exp_w, pred_w)

    rows = []
    for i in range(h):
        cols = []
        for j in range(w):
            if i < pred_h and j < pred_w and i < exp_h and j < exp_w:
                try:
                    cols.append('.' if predicted[i][j] == expected[i][j] else 'X')
                except Exception:
                    cols.append('X')
            else:
                # Out-of-range or missing cell => mismatch
                cols.append('X')
        rows.append(''.join(cols))

    # Apply indentation to each row
    indentation = " " * indent
    rows = [indentation + row for row in rows]

    return "\n".join(rows)


def _grid_to_image_bytes(grid: List[List[int]], cell_size: int = 24, padding: int = 8) -> bytes:
    """Render a grid (list of lists of ints) to a PNG bytes buffer.

    Returns PNG bytes. If Pillow is not available, raises ImportError.
    """
    try:
        from PIL import Image, ImageDraw
    except Exception:
        raise ImportError("Pillow is required to generate visual cues (pip install pillow)")

    # Simple color map for values 0..9
    DEFAULT_COLORS = {
        0: (255, 255, 255),
        1: (230, 25, 75),
        2: (60, 180, 75),
        3: (255, 225, 25),
        4: (0, 130, 200),
        5: (245, 130, 48),
        6: (145, 30, 180),
        7: (70, 240, 240),
        8: (240, 50, 230),
        9: (210, 245, 60),
    }

    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    img_w = w * cell_size + padding * 2
    img_h = h * cell_size + padding * 2
    img = Image.new('RGB', (img_w, img_h), (240, 240, 240))
    draw = ImageDraw.Draw(img)

    for r in range(h):
        for c in range(w):
            val = grid[r][c]
            color = DEFAULT_COLORS.get(val, (200, 200, 200))
            x0 = padding + c * cell_size
            y0 = padding + r * cell_size
            x1 = x0 + cell_size - 1
            y1 = y0 + cell_size - 1
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=(100, 100, 100))

    import io
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def generate_llm_predicted_output(llm,
                                  transformation_steps: Dict[str, Any],
                                  input_grid: List[List[int]]) -> Tuple[Optional[List[List[int]]], Optional[str]]:
    """Use the LLM to apply the step-by-step transformation to a single input grid.

    The LLM is instructed to return the transformed grid as a JSON array
    (list of lists of integers). Returns (grid, None) on success, or
    (None, error_message) on failure.
    """
    try:
        steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(transformation_steps)) if transformation_steps else "(no steps provided)"

        prompt_parts = [
            "You are an expert that can execute step-by-step grid transformations by following instructions",
            "Given the following input grid and transformation steps, you are tasked with applying the steps and return the resulting grid.",
            
            "Do NOT return any other text after that block.",
            "",
            "INPUT GRID:",
            format_grid_for_prompt(input_grid),
            "",
            "TRANSFORMATION STEPS:",
            steps_text,
            "",
            "Follow the transformation steps carefully, show your detailed step-by-step transformation"
            "After finishing all the steps, show the 2D grid inside a fenced block labeled ```llm_predicted_output``` containing only the grid rows as lines of numbers (space-separated or contiguous digits)."
        ]

        prompt = "\n".join(prompt_parts)
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)

        print_prompt_and_response(prompt, response_text)

        # Prefer a fenced block labelled ```llm_predicted_output``` containing
        # the grid as lines of numbers (space-separated or run-together digits).
        block_match = re.search(r'```llm_predicted_output\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
        if block_match:
            block = block_match.group(1).strip()
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            parsed_grid = []
            for line in lines:
                # Split on whitespace; if no whitespace, split into single chars
                if re.search(r'\s+', line):
                    parts = re.split(r'\s+', line.strip())
                else:
                    parts = list(line.strip())
                row = []
                for p in parts:
                    try:
                        row.append(int(p))
                    except Exception:
                        # If conversion fails, try to strip non-digits then int
                        digits = re.findall(r'-?\d+', p)
                        if digits:
                            row.append(int(digits[0]))
                        else:
                            # Give up and return error with raw block
                            return None, f"Non-numeric token in llm_predicted_output block: '{p}'"
                parsed_grid.append(row)
            return parsed_grid, None

        # Fallback: try to find a JSON array in the response
        json_match = re.search(r'(\[\s*\[.*?\]\s*\])', response_text, re.DOTALL)
        if json_match:
            candidate = json_match.group(1)
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list) and all(isinstance(row, list) for row in parsed):
                    norm = []
                    for row in parsed:
                        new_row = []
                        for cell in row:
                            try:
                                new_row.append(int(cell))
                            except Exception:
                                new_row.append(cell)
                        norm.append(new_row)
                    return norm, None
            except Exception:
                pass

        # Fallback: try to parse any Python-style list literal
        try:
            parsed2 = eval(response_text, {"__builtins__": {}}, {})
            if isinstance(parsed2, list) and all(isinstance(r, list) for r in parsed2):
                return parsed2, None
        except Exception:
            pass

        # If response contains an explicit error line, return it as error
        if isinstance(response_text, str) and ("error" in response_text.lower() or "cannot" in response_text.lower() or "failed" in response_text.lower()):
            return None, response_text.strip()

        return None, f"Could not parse LLM response as grid. Raw response: {response_text[:1000]}"

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return None, f"Exception calling LLM for predicted output: {e}\n{tb}"


def _flatten_content(c):
    """Flatten various response content types to a single string."""
    try:
        if isinstance(c, str):
            return c
        if isinstance(c, dict):
            # Common forms: {'content': '...'} or {'choices': [...]}
            if 'content' in c and isinstance(c['content'], (str, list)):
                return _flatten_content(c['content'])
            if 'choices' in c and isinstance(c['choices'], list):
                return '\n'.join(_flatten_content(ch) for ch in c['choices'])
            return str(c)
        if isinstance(c, (list, tuple)):
            parts = []
            for item in c:
                if isinstance(item, dict):
                    # Message-like item with 'content' or 'type' fields
                    if 'content' in item:
                        parts.append(_flatten_content(item['content']))
                        continue
                    # Structured content items: {'type':'text','text':...} or {'type':'image',...}
                    if item.get('type') == 'text' and 'text' in item:
                        parts.append(str(item['text']))
                        continue
                    if item.get('type') == 'image':
                        parts.append('[IMAGE]')
                        continue
                    parts.append(str(item))
                else:
                    parts.append(str(item))
            return '\n'.join([p for p in parts if p])
        return str(c)
    except Exception:
        return str(c)


def parse_transformation_steps(response_text: str) -> List[Dict]:
    """Parse transformation steps from LLM response.
    
    Returns a list of solution objects, each with:
    - solution_number: int
    - transformation_steps: List[str]
    """
    # Attempt 1: prefer a fenced ```json``` block containing the JSON array
    try:
        json_block_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
        candidate = None
        if json_block_match:
            candidate = json_block_match.group(1).strip()
        else:
            # Fallback: try to find a bare JSON array anywhere in the text
            start = response_text.find('[')
            end = response_text.rfind(']')
            if start != -1 and end != -1 and end > start:
                candidate = response_text[start:end+1]

        if candidate:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                solutions = []
                for item in parsed:
                    if isinstance(item, dict):
                        sol_num = item.get('solution_number') or item.get('solution') or item.get('solution_number')
                        steps = item.get('transformation_steps') or item.get('steps') or []
                        # Normalize steps to list of strings
                        if isinstance(steps, str):
                            step_lines = [ln.strip() for ln in steps.splitlines() if ln.strip()]
                            steps = [re.sub(r'^\d+\.\s*', '', ln).strip() for ln in step_lines]
                        elif isinstance(steps, list):
                            steps = [str(s).strip() for s in steps if str(s).strip()]
                        else:
                            steps = []

                        solutions.append({
                            "solution_number": int(sol_num) if (sol_num is not None and str(sol_num).isdigit()) else sol_num,
                            "transformation_steps": steps
                        })

                if solutions:
                    return solutions
    except Exception:
        pass

    # Fallback 1: look for fenced 'steps' block
    steps_match = re.search(r'```steps\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
    if steps_match:
        steps_content = steps_match.group(1).strip()
    else:
        steps_content = response_text

    # Try to split into multiple "Solution X" sections
    sol_splits = re.split(r'\bSolution\s*(\d+)\b', steps_content, flags=re.IGNORECASE)
    # re.split returns [before, num1, block1, num2, block2, ...] if matches
    if len(sol_splits) > 1:
        solutions = []
        # iterate pairs
        it = iter(sol_splits)
        pre = next(it)
        for token in it:
            try:
                num = token
                block = next(it)
            except StopIteration:
                break
            # extract numbered steps within block
            step_pattern = r'\d+\.\s*(.+)'
            matches = re.findall(step_pattern, block)
            steps = [m.strip() for m in matches]
            solutions.append({"solution_number": int(num) if num.isdigit() else num, "transformation_steps": steps})

        if solutions:
            return solutions

    # Fallback 2: extract any numbered steps across the text as a single solution
    step_pattern = r'(\d+\.\s*(.+))'
    matches = re.findall(step_pattern, steps_content, re.MULTILINE)
    if matches:
        steps = [match[1].strip() for match in matches]
        return [{"solution_number": 1, "transformation_steps": steps}]

    # Ultimate fallback: split long lines and return as single solution entries
    lines = [line.strip() for line in steps_content.split('\n') if line.strip()]
    steps = []
    for line in lines:
        cleaned_line = re.sub(r'^\d+\.\s*|^-\s*|^\*\s*', '', line).strip()
        if len(cleaned_line) > 5:
            steps.append(cleaned_line)

    return [{"solution_number": 1, "transformation_steps": steps[:50]}]


def build_steps_text_from_transformation_steps(transformation_steps: List[str]) -> str:
    """Build a numbered steps text block from a list of transformation steps."""
    if not transformation_steps:
        return "(no transformation steps provided)"
    return "\n".join(f"{i+1}. {s}" for i, s in enumerate(transformation_steps))
