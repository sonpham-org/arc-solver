#!/usr/bin/env python3
"""
Gemini LLM runner that:
- Loads GEMINI_API_KEY from .env
- Imports config.py
- Accepts the name of a variable from config.py to use as the prompt
"""

import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai
import argparse
import prompts
from typing import Tuple
import asyncio
import threading
import signal
import atexit
import time
from dotenv import dotenv_values
from collections import Counter

# local DB helper
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from db import sqlite_store
from db_writer import enqueue as db_enqueue, get_global_writer

# Ensure the package's src/ directory is importable when running this file as
# a script (python src/main.py). This inserts the file's directory onto
# sys.path so we can import local modules like model_configs reliably.
sys.path.insert(0, os.path.dirname(__file__))
import model_configs
DEFAULT_MODEL = model_configs.DEFAULT_MODEL
is_known_model = model_configs.is_known_model

# Load .env from root/
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
# load simple .env values for non-sensitive runtime config
_env = dotenv_values(os.path.join(os.path.dirname(__file__), "..", ".env"))
# default concurrency if not provided in .env
DEFAULT_MAX_CONCURRENCY = int(_env.get("MAX_CONCURRENCY", 4) or 4)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Missing GEMINI_API_KEY in .env")
os.environ.setdefault("GRPC_VERBOSITY", "NONE")
os.environ.setdefault("GRPC_LOG_SEVERITY_LEVEL", "ERROR")

genai.configure(api_key=api_key)


def size_score(expected_rows, expected_cols, generated_rows, generated_cols):
    if expected_rows == generated_rows and expected_cols == generated_cols:
        return 1.0
    else:
        row_frac = min(expected_rows, generated_rows) / max(expected_rows, generated_rows)
        col_frac = min(expected_cols, generated_cols) / max(expected_cols, generated_cols)
        return row_frac * col_frac


def color_score(expected_grid, generated_grid):
    if not expected_grid or not generated_grid:
        return 0.0
    # Flatten expected_grid to get colors
    expected_colors = [cell for row in expected_grid for cell in row]
    color_counts = Counter(expected_colors)
    # Sort colors by count descending
    sorted_colors = sorted(color_counts, key=color_counts.get, reverse=True)
    # Assign weights: least popular (last in sorted) gets 0.5, others 1.0
    weights = {color: 1.0 for color in sorted_colors}
    if sorted_colors:
        weights[sorted_colors[-1]] = 0.5
    # Now, compute score
    total_weight = 0.0
    matched_weight = 0.0
    for r in range(len(expected_grid)):
        for c in range(len(expected_grid[0])):
            if r < len(generated_grid) and c < len(generated_grid[0]):
                expected_cell = expected_grid[r][c]
                generated_cell = generated_grid[r][c]
                weight = weights.get(expected_cell, 1.0)
                total_weight += weight
                if expected_cell == generated_cell:
                    matched_weight += weight
    if total_weight == 0:
        return 0.0
    return matched_weight / total_weight


def build_model(model_name: str):
    """Create and return a GenerativeModel for the given model_name.

    We keep this factory to make testing and future extension easier.
    """
    # Optionally warn if model is unknown (but still attempt to create it)
    if not is_known_model(model_name):
        print(f"Warning: model '{model_name}' is not in model_configs registry")
    return genai.GenerativeModel(model_name)

def _with_retries(func, *args, retries: int = 3, backoff: float = 0.5, **kwargs):
    """Simple retry/backoff wrapper for synchronous functions.

    Returns the function's return value or raises the last exception.
    """
    last_exc = None
    delay = backoff
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            time.sleep(delay)
            delay *= 2
    # final attempt
    return func(*args, **kwargs)


def query_gemini(prompt: str) -> str:
    """Send prompt to Gemini and return the response text.

    Note: the underlying client is synchronous. This wrapper adds a small
    retry/backoff policy. Callers should run this inside a thread via
    asyncio.to_thread when used from async code.
    """
    response = _with_retries(model.generate_content, prompt)
    # depending on SDK, response may be string-like or an object with .text
    text = getattr(response, "text", response)
    return text.strip()


def _extract_trailing_json(text: str) -> Tuple[str, str]:
    """Attempt to split text into (body, trailing_json).

    If no JSON is found at the end, returns (text, ""). This is a simple
    heuristic that searches for the last '{' or '[' and tries to parse JSON
    from there.
    """
    import json
    import re

    # 1) Prefer an explicit <json>...</json> block (use the last such block if multiple)
    matches = list(re.finditer(r"<json[^>]*>([\s\S]*?)</json>", text, flags=re.IGNORECASE))
    if matches:
        m = matches[-1]
        candidate = m.group(1).strip()
        # strip ``` fences if present inside
        if candidate.startswith("```") and candidate.count("```") >= 2:
            candidate = candidate.strip("`\n ")
        # remove leading language tag like 'json\n'
        if candidate.lower().startswith("json\n"):
            candidate = candidate.split("\n", 1)[1]
        try:
            parsed = json.loads(candidate)
            body = (text[: m.start()]).strip()
            return body, json.dumps(parsed)
        except Exception:
            # If parsing failed, fall back to the other heuristics below
            pass

    # 2) Prefer code-fenced block parsing next: look for the last pair of ``` fences
    last_fence = text.rfind("```")
    if last_fence != -1:
        open_fence = text.rfind("```", 0, last_fence)
        if open_fence != -1:
            candidate = text[open_fence + 3 : last_fence].strip()
            # if the first line is a language marker like 'json', strip it
            if "\n" in candidate:
                first, rest = candidate.split("\n", 1)
                if first.strip().lower().startswith("json"):
                    candidate = rest.strip()
            try:
                parsed = json.loads(candidate)
                body = text[:open_fence].strip()
                return body, json.dumps(parsed)
            except Exception:
                # fall through to brace-based heuristic
                pass

    # 3) fallback: find last occurrences of braces that might start JSON
    candidates = [text.rfind("{"), text.rfind("[")]
    start = max(candidates)
    if start == -1:
        return text, ""

    body = text[:start].strip()
    candidate = text[start:].strip()
    # remove any trailing closing fence (```) if present
    if "```" in candidate:
        candidate = candidate[: candidate.rfind("```")].strip()

    try:
        parsed = json.loads(candidate)
        return body, json.dumps(parsed)
    except Exception:
        return text, ""


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: split on whitespace and punctuation. This is
    not precise but good for tracking relative costs.
    """
    # Prefer exact tokenizer if available (tiktoken). Fallback to a quick
    # heuristic if not installed.
    try:
        import tiktoken

        # choose an encoding compatible enough; if model name is known we
        # could map model -> encoding. Using cl100k_base as a reasonable
        # default for many models.
        enc = tiktoken.get_encoding("cl100k_base")
        if not text:
            return 0
        return max(1, len(enc.encode(text)))
    except Exception:
        import re

        if not text:
            return 0
        words = re.findall(r"\w+", text)
        return max(1, len(words))


def _estimate_cost(tokens: int, model_name: str | None = None, kind: str = "input") -> float:
    """Estimate cost in USD for given tokens using model_configs pricing.

    kind should be 'input' or 'output'. Falls back to a simple env-based
    rate when model pricing is unavailable.
    """
    if not tokens:
        return 0.0
    try:
        # Use model_configs.estimate_cost which accepts both input and output
        # token counts; call with tokens placed in the appropriate bucket.
        if model_name:
            if kind == "input":
                return float(model_configs.estimate_cost(model_name, input_tokens=int(tokens), output_tokens=0))
            else:
                return float(model_configs.estimate_cost(model_name, input_tokens=0, output_tokens=int(tokens)))
    except Exception:
        # fallback: env-configured per-1k rate
        try:
            rate = float(os.getenv("COST_PER_1K", _env.get("COST_PER_1K", 0.001) or 0.001))
        except Exception:
            rate = 0.001
        try:
            return float(tokens) * (rate / 1000.0)
        except Exception:
            return 0.0

async def _process_generation(i, prompt, tid, challenges, solutions, selected_model_name, run_name, run_timestamp, run_output_dir):
    """Process a single generation i for task tid."""
    gen_model_calls = 0
    gen_input_tokens = 0
    gen_output_tokens = 0
    gen_total_cost = 0.0
    gen_result = {}

    # call model (synchronous SDK call wrapped in a thread)
    try:
        t0 = time.perf_counter()
        text = await asyncio.to_thread(query_gemini, prompt)
        text = text.strip()
        call_dur = time.perf_counter() - t0
        print(f"(model call time: {call_dur:.3f}s)")
        gen_model_calls += 1
    except Exception as e:
        print(f"Error querying Gemini for {tid} generation {i}: {e}")
        return None, 0, 0, 0, 0.0, {}

    # create a generation record immediately so we always capture the
    # response even if later processing fails. Try to extract any
    # trailing JSON (final_json) from the response so downstream
    # apply/reflection logic can run. Ensure local variables are
    # defined to avoid NameError in exceptional control paths.
    try:
        body, final_json = _extract_trailing_json(text)
    except Exception:
        body, final_json = (text, None)

    in_tokens = _estimate_tokens(prompt) or 0
    out_tokens = _estimate_tokens(text) or 0
    gen_cost = _estimate_cost(in_tokens, selected_model_name, kind="input") + _estimate_cost(out_tokens, selected_model_name, kind="output")
    gen_rec = {
        "generation_index": i,
        "timestamp": __import__('datetime').datetime.utcnow().isoformat() + "Z",
        "prompt_response": text.splitlines() if text else None,
        # attach prompt_text later when writing out the file
        "prompt_text": None,
        "final_json": final_json,
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "cost_estimate": gen_cost,
        "train_scores": None,
        "training_comparison": None,
        "test_score": None,
        "testing_comparison": None,
        "apply_cost": None,
        "response_row_id": None,
    }
    gen_input_tokens += in_tokens
    gen_output_tokens += out_tokens
    gen_total_cost += gen_cost

    # parsed form of final_json (if any) for apply logic below
    try:
        parsed = None
        if final_json:
            try:
                parsed = __import__("json").loads(final_json)
            except Exception:
                parsed = None
    except Exception:
        parsed = None

    if isinstance(parsed, dict):
        try:
            # First: apply the extracted rule to each training example
            train_scores = []
            train_examples = challenges[tid].get("train", [])
            for j, ex in enumerate(train_examples):
                try:
                    # build an apply prompt that applies the rule to this single training input
                    apply_prompt_train = prompts.build_apply_prompt(parsed, {"train": [ex], "test": []}, tid, include_examples=False)
                    apply_resp_train = await asyncio.to_thread(query_gemini, apply_prompt_train)
                    apply_text_train = apply_resp_train
                    # metrics for this apply call
                    gen_model_calls += 1
                    _incr_in = _estimate_tokens(apply_prompt_train) or 0
                    _incr_out = _estimate_tokens(apply_text_train) or 0
                    gen_input_tokens += _incr_in
                    gen_output_tokens += _incr_out
                    gen_total_cost += _estimate_cost(_incr_in, selected_model_name, kind="input") + _estimate_cost(_incr_out, selected_model_name, kind="output")
                except Exception:
                    apply_text_train = None

                score_train_example = 0.0
                parsed_out_t = None
                if apply_text_train:
                    import re
                    m_t = re.search(r"<output>([\s\S]*?)</output>", apply_text_train, flags=re.IGNORECASE)
                    if m_t:
                        out_blob_t = m_t.group(1).strip()
                        try:
                            try:
                                parsed_out_t = __import__("json").loads(out_blob_t)
                            except Exception:
                                lines = [ln.strip() for ln in out_blob_t.splitlines() if ln.strip()]
                                if lines:
                                    grid = []
                                    ok = True
                                    for ln in lines:
                                        nums = re.findall(r"-?\d+", ln)
                                        if not nums:
                                            ok = False
                                            break
                                        grid.append([int(x) for x in nums])
                                    if ok:
                                        parsed_out_t = grid

                            if parsed_out_t is not None and tid in challenges:
                                # compare to expected training output
                                try:
                                    expected_grid = train_examples[j]["output"]
                                    er = len(expected_grid)
                                    ec = len(expected_grid[0]) if er > 0 else 0
                                    gr = len(parsed_out_t) if isinstance(parsed_out_t, list) else 0
                                    gc = len(parsed_out_t[0]) if gr > 0 and isinstance(parsed_out_t[0], list) else 0
                                    total = er * ec if er and ec else 0
                                    if total > 0:
                                        matches = 0
                                        for r in range(er):
                                            for c in range(ec):
                                                try:
                                                    if parsed_out_t[r][c] == expected_grid[r][c]:
                                                        matches += 1
                                                except Exception:
                                                    pass
                                        gr = len(parsed_out_t) if isinstance(parsed_out_t, list) else 0
                                        gc = len(parsed_out_t[0]) if gr > 0 and isinstance(parsed_out_t[0], list) else 0
                                        gr = len(parsed_out_t) if isinstance(parsed_out_t, list) else 0
                                        gc = len(parsed_out_t[0]) if gr > 0 and isinstance(parsed_out_t[0], list) else 0
                                        size_s = size_score(er, ec, gr, gc)
                                        color_s = color_score(expected_grid, parsed_out_t)
                                        score_train_example = size_s * color_s
                                except Exception:
                                    score_train_example = 0.0
                        except Exception:
                            score_train_example = 0.0

                train_scores.append(float(score_train_example))

            # store train_scores list JSON into DB and update score_train mean
            try:
                import json as _json
                train_scores_json = _json.dumps(train_scores)
                if 'rid' in locals() and rid:
                    try:
                        db_enqueue("update_response_train_scores", rid, train_scores_json)
                        mean_train = sum(train_scores) / len(train_scores) if train_scores else 0.0
                        db_enqueue("update_response_train_score", rid, float(mean_train))
                    except Exception:
                        pass
            except Exception:
                pass

            # collect training comparison entries for this generation
            try:
                comps = []
                for j, ex in enumerate(train_examples):
                    try:
                        expected_grid = ex.get("output")
                    except Exception:
                        expected_grid = None
                    try:
                        if isinstance(expected_grid, list) and expected_grid and isinstance(expected_grid[0], list):
                            expected_lines = [" ".join(str(x) for x in row) for row in expected_grid]
                        else:
                            expected_lines = expected_grid
                    except Exception:
                        expected_lines = expected_grid
                    try:
                        generated_val = None
                        # we may have parsed_out_t from the loop for the last example only; best-effort
                    except Exception:
                        generated_val = None
                    try:
                        score_val = train_scores[j] if j < len(train_scores) else None
                    except Exception:
                        score_val = None
                    comps.append({
                        "example_index": j,
                        "expected": expected_lines,
                        "generated": None,
                        "score": score_val,
                    })
                gen_result["train_scores"] = train_scores
                gen_result["training_comparison"] = comps
            except Exception:
                pass

            # Now run apply on the test inputs
            try:
                apply_prompt = prompts.build_apply_prompt(parsed, challenges[tid], tid)
            except Exception as e:
                print(f"\033[91m[{tid} gen {i}] failed to build apply_prompt: {e}\033[0m")
                apply_text = None
            else:
                t_apply0 = time.perf_counter()
                apply_resp_text = await asyncio.to_thread(query_gemini, apply_prompt)
                apply_text = apply_resp_text
                # metrics for apply
                gen_model_calls += 1
                _incr_in = _estimate_tokens(apply_prompt) or 0
                _incr_out = _estimate_tokens(apply_text) or 0
                gen_input_tokens += _incr_in
                gen_output_tokens += _incr_out
                gen_total_cost += _estimate_cost(_incr_in, selected_model_name, kind="input") + _estimate_cost(_incr_out, selected_model_name, kind="output")
                print(f"(apply call time: {time.perf_counter() - t_apply0:.3f}s)")
                if apply_text:
                    print(f"\033[92m[{tid} gen {i}] apply_prompt executed successfully\033[0m")
                else:
                    print(f"\033[91m[{tid} gen {i}] apply_prompt execution failed\033[0m")
            
        except Exception as e:
            print(f"Warning: failed to run apply prompt for {tid} generation {i}: {e}")
            apply_text = None
        run_apply = False
        try:
            if isinstance(parsed.get("confidence", None), (int, float)):
                run_apply = parsed.get("confidence") >= 0.5
        except Exception:
            run_apply = False
        if not run_apply:
            # fallback: don't run apply for very large responses
            run_apply = _estimate_tokens(text) < 1200

        if not run_apply:
            # skip apply for this response
            return gen_rec, gen_model_calls, gen_input_tokens, gen_output_tokens, gen_total_cost, gen_result

        # First: apply the extracted rule to each training example (no examples included in prompt)
        try:
            train_scores = []
            train_examples = challenges[tid].get("train", [])
            for j, ex in enumerate(train_examples):
                try:
                    # build an apply prompt that applies the rule to this single training input
                    apply_prompt_train = prompts.build_apply_prompt(parsed, {"train": [ex], "test": []}, tid, include_examples=False)
                    apply_resp_train = await asyncio.to_thread(query_gemini, apply_prompt_train)
                    apply_text_train = apply_resp_train
                    # metrics for this apply call
                    gen_model_calls += 1
                    _incr_in = _estimate_tokens(apply_prompt_train) or 0
                    _incr_out = _estimate_tokens(apply_text_train) or 0
                    gen_input_tokens += _incr_in
                    gen_output_tokens += _incr_out
                    gen_total_cost += _estimate_cost(_incr_in, selected_model_name, kind="input") + _estimate_cost(_incr_out, selected_model_name, kind="output")
                except Exception:
                    apply_text_train = None

                score_train_example = 0.0
                if apply_text_train:
                    import re
                    m_t = re.search(r"<output>([\s\S]*?)</output>", apply_text_train, flags=re.IGNORECASE)
                    if m_t:
                        out_blob_t = m_t.group(1).strip()
                        try:
                            parsed_out_t = None
                            try:
                                parsed_out_t = __import__("json").loads(out_blob_t)
                            except Exception:
                                lines = [ln.strip() for ln in out_blob_t.splitlines() if ln.strip()]
                                if lines:
                                    grid = []
                                    ok = True
                                    for ln in lines:
                                        nums = re.findall(r"-?\d+", ln)
                                        if not nums:
                                            ok = False
                                            break
                                        grid.append([int(x) for x in nums])
                                    if ok:
                                        parsed_out_t = grid

                            if parsed_out_t is not None and tid in challenges:
                                # compare to expected training output
                                try:
                                    expected_grid = train_examples[j]["output"]
                                    er = len(expected_grid)
                                    ec = len(expected_grid[0]) if er > 0 else 0
                                    total = er * ec if er and ec else 0
                                    if total > 0:
                                        matches = 0
                                        for r in range(er):
                                            for c in range(ec):
                                                try:
                                                    if parsed_out_t[r][c] == expected_grid[r][c]:
                                                        matches += 1
                                                except Exception:
                                                    pass
                                        score_train_example = matches / total
                                except Exception:
                                    score_train_example = 0.0
                        except Exception:
                            score_train_example = 0.0

                train_scores.append(float(score_train_example))

            # store train_scores list JSON into DB and update score_train mean
            try:
                import json as _json
                train_scores_json = _json.dumps(train_scores)
                if 'rid' in locals() and rid:
                    # these are non-critical updates; enqueue them to the
                    # background DB writer rather than blocking the event loop.
                    try:
                        db_enqueue("update_response_train_scores", rid, train_scores_json)
                        mean_train = sum(train_scores) / len(train_scores) if train_scores else 0.0
                        db_enqueue("update_response_train_score", rid, float(mean_train))
                    except Exception:
                        pass
            except Exception:
                pass

            # collect training comparison entries for this generation
            try:
                comps = []
                # for each training example try to capture expected vs generated
                for j, ex in enumerate(train_examples):
                    try:
                        expected_grid = ex.get("output")
                    except Exception:
                        expected_grid = None
                    # convert expected_grid (2D array) to list of space-separated lines
                    try:
                        if isinstance(expected_grid, list) and expected_grid and isinstance(expected_grid[0], list):
                            expected_lines = [" ".join(str(x) for x in row) for row in expected_grid]
                        else:
                            expected_lines = expected_grid
                    except Exception:
                        expected_lines = expected_grid
                    # parsed_out_t may only be populated for the most recent example
                    try:
                        generated_val = locals().get("parsed_out_t", None)
                    except Exception:
                        generated_val = None
                    # convert generated_val grids to list of lines when possible
                    try:
                        if isinstance(generated_val, list) and generated_val and isinstance(generated_val[0], list):
                            generated_lines = [" ".join(str(x) for x in row) for row in generated_val]
                        else:
                            generated_lines = generated_val
                    except Exception:
                        generated_lines = generated_val
                    try:
                        score_val = train_scores[j] if j < len(train_scores) else None
                    except Exception:
                        score_val = None
                    comps.append({
                        "example_index": j,
                        "expected": expected_lines,
                        "generated": generated_lines,
                        "score": score_val,
                    })
                gen_result["train_scores"] = train_scores
                gen_result["training_comparison"] = comps
            except Exception:
                pass

            # Now run apply on the test inputs as before
                # attempt to build apply_prompt and report status
                try:
                    apply_prompt = prompts.build_apply_prompt(parsed, challenges[tid], tid)
                    print(f"\033[92m[{tid} gen {i}] apply_prompt built\033[0m")
                except Exception as e:
                    print(f"\033[91m[{tid} gen {i}] failed to build apply_prompt: {e}\033[0m")
                    # skip apply if we can't build the prompt
                    return gen_rec, gen_model_calls, gen_input_tokens, gen_output_tokens, gen_total_cost, gen_result

                t_apply0 = time.perf_counter()
                apply_resp_text = await asyncio.to_thread(query_gemini, apply_prompt)
                apply_text = apply_resp_text
                if apply_text:
                    print(f"\033[92m[{tid} gen {i}] apply_prompt executed successfully\033[0m")
                else:
                    print(f"\033[91m[{tid} gen {i}] apply_prompt execution failed\033[0m")
            # metrics for apply
            gen_model_calls += 1
            _incr_in = _estimate_tokens(apply_prompt) or 0
            _incr_out = _estimate_tokens(apply_text) or 0
            gen_input_tokens += _incr_in
            gen_output_tokens += _incr_out
            gen_total_cost += _estimate_cost(_incr_in, selected_model_name, kind="input") + _estimate_cost(_incr_out, selected_model_name, kind="output")
            print(f"(apply call time: {time.perf_counter() - t_apply0:.3f}s)")
        except Exception as e:
            print(f"Warning: failed to run apply prompt for {tid} generation {i}: {e}")
            apply_text = None

        extracted = None
        if apply_text:
            import re

            m = re.search(r"<output>([\s\S]*?)</output>", apply_text, flags=re.IGNORECASE)
            if m:
                out_blob = m.group(1).strip()
                try:
                    expected = solutions.get(tid)
                    parsed_out = None
                    try:
                        parsed_out = __import__("json").loads(out_blob)
                    except Exception:
                        lines = [ln.strip() for ln in out_blob.splitlines() if ln.strip()]
                        if lines:
                            grid = []
                            ok = True
                            for ln in lines:
                                nums = re.findall(r"-?\d+", ln)
                                if not nums:
                                    ok = False
                                    break
                                grid.append([int(x) for x in nums])
                            if ok:
                                parsed_out = grid
                except Exception:
                    parsed_out = None

                match = False
                score = 0.0
                if parsed_out is not None and tid in solutions:
                    try:
                        expected_grid = solutions[tid][0]
                        er = len(expected_grid)
                        ec = len(expected_grid[0]) if er > 0 else 0
                        gr = len(parsed_out) if isinstance(parsed_out, list) else 0
                        gc = len(parsed_out[0]) if gr > 0 and isinstance(parsed_out[0], list) else 0
                        total = er * ec if er and ec else 0
                        if total > 0:
                            matches = 0
                            for r in range(er):
                                for c in range(ec):
                                    try:
                                        if parsed_out[r][c] == expected_grid[r][c]:
                                            matches += 1
                                    except Exception:
                                        pass
                            size_s = size_score(er, ec, gr, gc)
                            color_s = color_score(expected_grid, parsed_out)
                            score = size_s * color_s
                            match = matches == total
                        else:
                            score = 0.0
                            match = False
                    except Exception:
                        match = False
                        score = 0.0

                try:
                    # find last inserted row id by selecting max id for this prompt_hash & tid & prompt_index
                    conn = __import__("sqlite3").connect(sqlite_store.DB_PATH)
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT id FROM responses WHERE prompt_hash = ? AND challenge_id = ? AND json_index = ? ORDER BY id DESC LIMIT 1",
                        (None, str(tid), i),
                    )
                    row = cur.fetchone()
                    if row:
                        rid = row[0]
                        # non-critical updates: enqueue
                        try:
                            db_enqueue("update_response_scores", rid, float(score))
                        except Exception:
                            pass
                        try:
                            if parsed_out is not None:
                                import json as _json
                                db_enqueue("update_response_apply_output", rid, _json.dumps(parsed_out))
                        except Exception:
                            pass
                    conn.close()
                    if apply_text and row:
                        apply_tokens = _estimate_tokens(apply_text)
                        try:
                            db_enqueue("update_response_tokens", rid, apply_tokens)
                        except Exception:
                            pass
                        try:
                            # compute additional monetary cost for apply_text tokens and add to the response cost
                            try:
                                apply_cost = model_configs.estimate_cost(selected_model_name, input_tokens=0, output_tokens=apply_tokens)
                            except Exception:
                                apply_cost = 0.0
                            db_enqueue("update_response_add_cost", rid, float(apply_cost))
                        except Exception:
                            pass
                    # collect testing comparison and per-generation values
                    try:
                        comps_t = []
                        try:
                            expected_grid = solutions.get(tid)[0] if (solutions and tid in solutions) else None
                        except Exception:
                            expected_grid = None
                        # convert expected and generated grids to list of space-separated lines
                        try:
                            if isinstance(expected_grid, list) and expected_grid and isinstance(expected_grid[0], list):
                                expected_lines_t = [" ".join(str(x) for x in row) for row in expected_grid]
                            else:
                                expected_lines_t = expected_grid
                        except Exception:
                            expected_lines_t = expected_grid
                        try:
                            if isinstance(parsed_out, list) and parsed_out and isinstance(parsed_out[0], list):
                                parsed_out_lines = [" ".join(str(x) for x in row) for row in parsed_out]
                            else:
                                parsed_out_lines = parsed_out
                        except Exception:
                            parsed_out_lines = parsed_out
                        comps_t.append({
                            "expected": expected_lines_t,
                            "generated": parsed_out_lines,
                            "score": score,
                        })
                        gen_result["test_score"] = score
                        gen_result["testing_comparison"] = comps_t
                        gen_result["apply_cost"] = apply_cost if 'apply_cost' in locals() else None
                        gen_result["response_row_id"] = rid
                    except Exception:
                        pass
                except Exception as e:
                    print(f"Warning: failed to update scores for {tid} generation {i}: {e}")

    return gen_rec, gen_model_calls, gen_input_tokens, gen_output_tokens, gen_total_cost, gen_result


async def _process_task(tid: str, challenges: dict, challenges_path: str, selected_model_name: str, solutions_path: str, sem: asyncio.Semaphore, run_name: str = None, run_timestamp: str = None, run_output_dir: str = None, num_initial_generations: int = 1, max_reflections: int = 3):
    """Process a single task id: build prompt, query model, store results and optionally run apply prompt.

    This function is safe to run concurrently up to the provided semaphore.
    Blocking operations (model calls and sqlite writes) are dispatched to threads.
    """
    async with sem:
        print(f"\n--- Task {tid} ---")
        try:
            prompt = prompts.build_arc_prompt(challenges, tid)
        except Exception as e:
            print(f"Error building prompt for {tid}: {e}")
            return

        start_build = time.perf_counter()
        # print a concise status instead of the full prompt to avoid flooding
        print(f"\n🧩 Built prompt for task {tid} (lines={len(prompt.splitlines())})")
        build_dur = time.perf_counter() - start_build
        print(f"(prompt build time: {build_dur:.3f}s)")

    # ensure run output dir exists (if provided) and track per-generation records
    try:
        if run_output_dir:
            os.makedirs(run_output_dir, exist_ok=True)
    except Exception:
        pass

    # track simple per-task metrics to report back to the runner
    task_model_calls = 0
    task_input_tokens = 0
    task_output_tokens = 0
    task_total_cost = 0.0
    task_start_time = time.perf_counter()

    # prepare storage for per-generation records to persist to JSON file
    gen_records = []
    # temporary storage for results computed during the run; we'll merge
    # these into gen_records after the full per-task loop completes so
    # train/test comparisons are written only once everything is done.
    gen_results = {}
    prompt_hash = None

    solutions = {}
    if solutions_path:
        try:
            with open(solutions_path, "r") as sf:
                solutions = __import__("json").load(sf)
        except Exception:
            solutions = {}

    # Send to Gemini n times
    for i in range(num_initial_generations):
            # call model (synchronous SDK call wrapped in a thread)
            try:
                t0 = time.perf_counter()
                text = await asyncio.to_thread(query_gemini, prompt)
                text = text.strip()
                call_dur = time.perf_counter() - t0
                print(f"(model call time: {call_dur:.3f}s)")
                task_model_calls += 1
            except Exception as e:
                print(f"Error querying Gemini for {tid} generation {i}: {e}")
                continue

            # create a generation record immediately so we always capture the
            # response even if later processing fails. Try to extract any
            # trailing JSON (final_json) from the response so downstream
            # apply/reflection logic can run. Ensure local variables are
            # defined to avoid NameError in exceptional control paths.
            try:
                body, final_json = _extract_trailing_json(text)
            except Exception:
                body, final_json = (text, None)

            in_tokens = _estimate_tokens(prompt) or 0
            out_tokens = _estimate_tokens(text) or 0
            gen_cost = _estimate_cost(in_tokens, selected_model_name, kind="input") + _estimate_cost(out_tokens, selected_model_name, kind="output")
            gen_rec = {
                "generation_index": i,
                "timestamp": __import__('datetime').datetime.utcnow().isoformat() + "Z",
                "prompt_response": text.splitlines() if text else None,
                # attach prompt_text later when writing out the file
                "prompt_text": None,
                "final_json": final_json,
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "cost_estimate": gen_cost,
                "train_scores": None,
                "training_comparison": None,
                "test_score": None,
                "testing_comparison": None,
                "apply_cost": None,
                "response_row_id": None,
            }
            gen_records.append(gen_rec)
            # accumulate into task-wide totals
            task_input_tokens += in_tokens
            task_output_tokens += out_tokens
            task_total_cost += gen_cost

            # parsed form of final_json (if any) for apply logic below
            try:
                parsed = None
                if final_json:
                    try:
                        parsed = __import__("json").loads(final_json)
                    except Exception:
                        parsed = None
            except Exception:
                parsed = None

            if isinstance(parsed, dict):
                    try:
                        # load solutions file if provided (optional)
                        try:
                            solutions = {}
                            if solutions_path:
                                with open(solutions_path, "r") as sf:
                                    solutions = __import__("json").load(sf)
                        except Exception:
                            solutions = {}

                        # First: apply the extracted rule to each training example
                        train_scores = []
                        train_examples = challenges[tid].get("train", [])
                        for j, ex in enumerate(train_examples):
                            try:
                                # build an apply prompt that applies the rule to this single training input
                                apply_prompt_train = prompts.build_apply_prompt(parsed, {"train": [ex], "test": []}, tid, include_examples=False)
                                apply_resp_train = await asyncio.to_thread(query_gemini, apply_prompt_train)
                                apply_text_train = apply_resp_train
                                # metrics for this apply call
                                task_model_calls += 1
                                _incr_in = _estimate_tokens(apply_prompt_train) or 0
                                _incr_out = _estimate_tokens(apply_text_train) or 0
                                task_input_tokens += _incr_in
                                task_output_tokens += _incr_out
                                task_total_cost += _estimate_cost(_incr_in, selected_model_name, kind="input") + _estimate_cost(_incr_out, selected_model_name, kind="output")
                            except Exception:
                                apply_text_train = None

                            score_train_example = 0.0
                            parsed_out_t = None
                            if apply_text_train:
                                import re
                                m_t = re.search(r"<output>([\s\S]*?)</output>", apply_text_train, flags=re.IGNORECASE)
                                if m_t:
                                    out_blob_t = m_t.group(1).strip()
                                    try:
                                        try:
                                            parsed_out_t = __import__("json").loads(out_blob_t)
                                        except Exception:
                                            lines = [ln.strip() for ln in out_blob_t.splitlines() if ln.strip()]
                                            if lines:
                                                grid = []
                                                ok = True
                                                for ln in lines:
                                                    nums = re.findall(r"-?\d+", ln)
                                                    if not nums:
                                                        ok = False
                                                        break
                                                    grid.append([int(x) for x in nums])
                                                if ok:
                                                    parsed_out_t = grid

                                        if parsed_out_t is not None and tid in challenges:
                                            # compare to expected training output
                                            try:
                                                expected_grid = train_examples[j]["output"]
                                                er = len(expected_grid)
                                                ec = len(expected_grid[0]) if er > 0 else 0
                                                total = er * ec if er and ec else 0
                                                if total > 0:
                                                    matches = 0
                                                    for r in range(er):
                                                        for c in range(ec):
                                                            try:
                                                                if parsed_out_t[r][c] == expected_grid[r][c]:
                                                                    matches += 1
                                                            except Exception:
                                                                pass
                                                    size_s = size_score(er, ec, gr, gc)
                                                color_s = color_score(expected_grid, parsed_out_t)
                                                score_train_example = size_s * color_s
                                            except Exception:
                                                score_train_example = 0.0
                                    except Exception:
                                        score_train_example = 0.0

                            train_scores.append(float(score_train_example))

                        # store train_scores list JSON into DB and update score_train mean
                        try:
                            import json as _json
                            train_scores_json = _json.dumps(train_scores)
                            if rid:
                                try:
                                    db_enqueue("update_response_train_scores", rid, train_scores_json)
                                    mean_train = sum(train_scores) / len(train_scores) if train_scores else 0.0
                                    db_enqueue("update_response_train_score", rid, float(mean_train))
                                except Exception:
                                    pass
                        except Exception:
                            pass

                        # collect training comparison entries for this generation
                        try:
                            comps = []
                            for j, ex in enumerate(train_examples):
                                try:
                                    expected_grid = ex.get("output")
                                except Exception:
                                    expected_grid = None
                                try:
                                    if isinstance(expected_grid, list) and expected_grid and isinstance(expected_grid[0], list):
                                        expected_lines = [" ".join(str(x) for x in row) for row in expected_grid]
                                    else:
                                        expected_lines = expected_grid
                                except Exception:
                                    expected_lines = expected_grid
                                try:
                                    generated_val = None
                                    # we may have parsed_out_t from the loop for the last example only; best-effort
                                except Exception:
                                    generated_val = None
                                try:
                                    score_val = train_scores[j] if j < len(train_scores) else None
                                except Exception:
                                    score_val = None
                                comps.append({
                                    "example_index": j,
                                    "expected": expected_lines,
                                    "generated": None,
                                    "score": score_val,
                                })
                            gen_results.setdefault(i, {})["train_scores"] = train_scores
                            gen_results.setdefault(i, {})["training_comparison"] = comps
                        except Exception:
                            pass

                        # Now run apply on the test inputs
                        try:
                            apply_prompt = prompts.build_apply_prompt(parsed, challenges[tid], tid)
                        except Exception as e:
                            print(f"\033[91m[{tid} gen {i}] failed to build apply_prompt: {e}\033[0m")
                            apply_text = None
                        else:
                            t_apply0 = time.perf_counter()
                            apply_resp_text = await asyncio.to_thread(query_gemini, apply_prompt)
                            apply_text = apply_resp_text
                            # metrics for apply
                            task_model_calls += 1
                            _incr_in = _estimate_tokens(apply_prompt) or 0
                            _incr_out = _estimate_tokens(apply_text) or 0
                            task_input_tokens += _incr_in
                            task_output_tokens += _incr_out
                            task_total_cost += _estimate_cost(_incr_in, selected_model_name, kind="input") + _estimate_cost(_incr_out, selected_model_name, kind="output")
                            print(f"(apply call time: {time.perf_counter() - t_apply0:.3f}s)")
                            if apply_text:
                                print(f"\033[92m[{tid} gen {i}] apply_prompt executed successfully\033[0m")
                            else:
                                print(f"\033[91m[{tid} gen {i}] apply_prompt execution failed\033[0m")
                        
                    except Exception as e:
                        print(f"Warning: failed to run apply prompt for {tid} generation {i}: {e}")
                        apply_text = None
                    run_apply = False
                    try:
                        if isinstance(parsed.get("confidence", None), (int, float)):
                            run_apply = parsed.get("confidence") >= 0.5
                    except Exception:
                        run_apply = False
                    if not run_apply:
                        # fallback: don't run apply for very large responses
                        run_apply = _estimate_tokens(text) < 1200

                    if not run_apply:
                        # skip apply for this response
                        continue

                    # First: apply the extracted rule to each training example (no examples included in prompt)
                    try:
                        train_scores = []
                        train_examples = challenges[tid].get("train", [])
                        for j, ex in enumerate(train_examples):
                            try:
                                # build an apply prompt that applies the rule to this single training input
                                apply_prompt_train = prompts.build_apply_prompt(parsed, {"train": [ex], "test": []}, tid, include_examples=False)
                                apply_resp_train = await asyncio.to_thread(query_gemini, apply_prompt_train)
                                apply_text_train = apply_resp_train
                                # metrics for this apply call
                                task_model_calls += 1
                                _incr_in = _estimate_tokens(apply_prompt_train) or 0
                                _incr_out = _estimate_tokens(apply_text_train) or 0
                                task_input_tokens += _incr_in
                                task_output_tokens += _incr_out
                                task_total_cost += _estimate_cost(_incr_in, selected_model_name, kind="input") + _estimate_cost(_incr_out, selected_model_name, kind="output")
                            except Exception:
                                apply_text_train = None

                        score_train_example = 0.0
                        if apply_text_train:
                            import re
                            m_t = re.search(r"<output>([\s\S]*?)</output>", apply_text_train, flags=re.IGNORECASE)
                            if m_t:
                                out_blob_t = m_t.group(1).strip()
                                try:
                                    parsed_out_t = None
                                    try:
                                        parsed_out_t = __import__("json").loads(out_blob_t)
                                    except Exception:
                                        lines = [ln.strip() for ln in out_blob_t.splitlines() if ln.strip()]
                                        if lines:
                                            grid = []
                                            ok = True
                                            for ln in lines:
                                                nums = re.findall(r"-?\d+", ln)
                                                if not nums:
                                                    ok = False
                                                    break
                                                grid.append([int(x) for x in nums])
                                            if ok:
                                                parsed_out_t = grid

                                    if parsed_out_t is not None and tid in challenges:
                                        # compare to expected training output
                                        try:
                                            expected_grid = train_examples[j]["output"]
                                            er = len(expected_grid)
                                            ec = len(expected_grid[0]) if er > 0 else 0
                                            gr = len(parsed_out_t) if isinstance(parsed_out_t, list) else 0
                                            gc = len(parsed_out_t[0]) if gr > 0 and isinstance(parsed_out_t[0], list) else 0
                                            total = er * ec if er and ec else 0
                                            if total > 0:
                                                matches = 0
                                                for r in range(er):
                                                    for c in range(ec):
                                                        try:
                                                            if parsed_out_t[r][c] == expected_grid[r][c]:
                                                                matches += 1
                                                        except Exception:
                                                            pass
                                                size_s = size_score(er, ec, gr, gc)
                                                color_s = color_score(expected_grid, parsed_out_t)
                                                score_train_example = size_s * color_s
                                        except Exception:
                                            score_train_example = 0.0
                                except Exception:
                                    score_train_example = 0.0

                        train_scores.append(float(score_train_example))                        # store train_scores list JSON into DB and update score_train mean
                        try:
                            import json as _json
                            train_scores_json = _json.dumps(train_scores)
                            if rid:
                                # these are non-critical updates; enqueue them to the
                                # background DB writer rather than blocking the event loop.
                                try:
                                    db_enqueue("update_response_train_scores", rid, train_scores_json)
                                    mean_train = sum(train_scores) / len(train_scores) if train_scores else 0.0
                                    db_enqueue("update_response_train_score", rid, float(mean_train))
                                except Exception:
                                    pass
                        except Exception:
                            pass

                        # collect training comparison entries for this generation
                        try:
                            comps = []
                            # for each training example try to capture expected vs generated
                            for j, ex in enumerate(train_examples):
                                try:
                                    expected_grid = ex.get("output")
                                except Exception:
                                    expected_grid = None
                                # convert expected_grid (2D array) to list of space-separated lines
                                try:
                                    if isinstance(expected_grid, list) and expected_grid and isinstance(expected_grid[0], list):
                                        expected_lines = [" ".join(str(x) for x in row) for row in expected_grid]
                                    else:
                                        expected_lines = expected_grid
                                except Exception:
                                    expected_lines = expected_grid
                                # parsed_out_t may only be populated for the most recent example
                                try:
                                    generated_val = locals().get("parsed_out_t", None)
                                except Exception:
                                    generated_val = None
                                # convert generated_val grids to list of lines when possible
                                try:
                                    if isinstance(generated_val, list) and generated_val and isinstance(generated_val[0], list):
                                        generated_lines = [" ".join(str(x) for x in row) for row in generated_val]
                                    else:
                                        generated_lines = generated_val
                                except Exception:
                                    generated_lines = generated_val
                                try:
                                    score_val = train_scores[j] if j < len(train_scores) else None
                                except Exception:
                                    score_val = None
                                comps.append({
                                    "example_index": j,
                                    "expected": expected_lines,
                                    "generated": generated_lines,
                                    "score": score_val,
                                })
                            gen_results.setdefault(i, {})["train_scores"] = train_scores
                            gen_results.setdefault(i, {})["training_comparison"] = comps
                        except Exception:
                            pass

                        # Now run apply on the test inputs as before
                            # attempt to build apply_prompt and report status
                            try:
                                apply_prompt = prompts.build_apply_prompt(parsed, challenges[tid], tid)
                                print(f"\033[92m[{tid} gen {i}] apply_prompt built\033[0m")
                            except Exception as e:
                                print(f"\033[91m[{tid} gen {i}] failed to build apply_prompt: {e}\033[0m")
                                # skip apply if we can't build the prompt
                                continue

                            t_apply0 = time.perf_counter()
                            apply_resp_text = await asyncio.to_thread(query_gemini, apply_prompt)
                            apply_text = apply_resp_text
                            if apply_text:
                                print(f"\033[92m[{tid} gen {i}] apply_prompt executed successfully\033[0m")
                            else:
                                print(f"\033[91m[{tid} gen {i}] apply_prompt execution failed\033[0m")
                        # metrics for apply
                        task_model_calls += 1
                        _incr_in = _estimate_tokens(apply_prompt) or 0
                        _incr_out = _estimate_tokens(apply_text) or 0
                        task_input_tokens += _incr_in
                        task_output_tokens += _incr_out
                        task_total_cost += _estimate_cost(_incr_in, selected_model_name, kind="input") + _estimate_cost(_incr_out, selected_model_name, kind="output")
                        print(f"(apply call time: {time.perf_counter() - t_apply0:.3f}s)")
                    except Exception as e:
                        print(f"Warning: failed to run apply prompt for {tid} generation {i}: {e}")
                        apply_text = None

                    extracted = None
                    if apply_text:
                        import re

                        m = re.search(r"<output>([\s\S]*?)</output>", apply_text, flags=re.IGNORECASE)
                        if m:
                            out_blob = m.group(1).strip()
                            try:
                                expected = solutions.get(tid)
                                parsed_out = None
                                try:
                                    parsed_out = __import__("json").loads(out_blob)
                                except Exception:
                                    lines = [ln.strip() for ln in out_blob.splitlines() if ln.strip()]
                                    if lines:
                                        grid = []
                                        ok = True
                                        for ln in lines:
                                            nums = re.findall(r"-?\d+", ln)
                                            if not nums:
                                                ok = False
                                                break
                                            grid.append([int(x) for x in nums])
                                        if ok:
                                            parsed_out = grid
                            except Exception:
                                parsed_out = None

                            match = False
                            score = 0.0
                            if parsed_out is not None and tid in solutions:
                                try:
                                    expected_grid = solutions[tid][0]
                                    er = len(expected_grid)
                                    ec = len(expected_grid[0]) if er > 0 else 0
                                    total = er * ec if er and ec else 0
                                    if total > 0:
                                        matches = 0
                                        for r in range(er):
                                            for c in range(ec):
                                                try:
                                                    if parsed_out[r][c] == expected_grid[r][c]:
                                                        matches += 1
                                                except Exception:
                                                    pass
                                        gr = len(parsed_out) if isinstance(parsed_out, list) else 0
                                        gc = len(parsed_out[0]) if gr > 0 and isinstance(parsed_out[0], list) else 0
                                        size_s = size_score(er, ec, gr, gc)
                                        color_s = color_score(expected_grid, parsed_out)
                                        score = size_s * color_s
                                        match = matches == total
                                    else:
                                        score = 0.0
                                        match = False
                                except Exception:
                                    match = False
                                    score = 0.0

                            try:
                                # find last inserted row id by selecting max id for this prompt_hash & tid & prompt_index
                                conn = __import__("sqlite3").connect(sqlite_store.DB_PATH)
                                cur = conn.cursor()
                                cur.execute(
                                    "SELECT id FROM responses WHERE prompt_hash = ? AND challenge_id = ? AND json_index = ? ORDER BY id DESC LIMIT 1",
                                    (prompt_hash, str(tid), i),
                                )
                                row = cur.fetchone()
                                if row:
                                    rid = row[0]
                                    # non-critical updates: enqueue
                                    try:
                                        db_enqueue("update_response_scores", rid, float(score))
                                    except Exception:
                                        pass
                                    try:
                                        if parsed_out is not None:
                                            import json as _json
                                            db_enqueue("update_response_apply_output", rid, _json.dumps(parsed_out))
                                    except Exception:
                                        pass
                                conn.close()
                                if apply_text and row:
                                    apply_tokens = _estimate_tokens(apply_text)
                                    try:
                                        db_enqueue("update_response_tokens", rid, apply_tokens)
                                    except Exception:
                                        pass
                                    try:
                                        # compute additional monetary cost for apply_text tokens and add to the response cost
                                        try:
                                            apply_cost = model_configs.estimate_cost(selected_model_name, input_tokens=0, output_tokens=apply_tokens)
                                        except Exception:
                                            apply_cost = 0.0
                                        db_enqueue("update_response_add_cost", rid, float(apply_cost))
                                    except Exception:
                                        pass
                                # collect testing comparison and per-generation values
                                try:
                                    comps_t = []
                                    try:
                                        expected_grid = solutions.get(tid)[0] if (solutions and tid in solutions) else None
                                    except Exception:
                                        expected_grid = None
                                    # convert expected and generated grids to list of space-separated lines
                                    try:
                                        if isinstance(expected_grid, list) and expected_grid and isinstance(expected_grid[0], list):
                                            expected_lines_t = [" ".join(str(x) for x in row) for row in expected_grid]
                                        else:
                                            expected_lines_t = expected_grid
                                    except Exception:
                                        expected_lines_t = expected_grid
                                    try:
                                        if isinstance(parsed_out, list) and parsed_out and isinstance(parsed_out[0], list):
                                            parsed_out_lines = [" ".join(str(x) for x in row) for row in parsed_out]
                                        else:
                                            parsed_out_lines = parsed_out
                                    except Exception:
                                        parsed_out_lines = parsed_out
                                    comps_t.append({
                                        "expected": expected_lines_t,
                                        "generated": parsed_out_lines,
                                        "score": score,
                                    })
                                    g = gen_results.setdefault(i, {})
                                    g["test_score"] = score
                                    g["testing_comparison"] = comps_t
                                    g["apply_cost"] = apply_cost if 'apply_cost' in locals() else None
                                    g["response_row_id"] = rid
                                except Exception:
                                    pass
                            except Exception as e:
                                print(f"Warning: failed to update scores for {tid} generation {i}: {e}")
                # generation record was created earlier; any per-generation metrics
                # are updated in-place on that record as they become available.

    # end for generations

    task_duration = time.perf_counter() - task_start_time

    # If we ran any apply/update flows, attempt to attach train/test scores and apply_cost
    # (They may have been enqueued asynchronously; we include what we captured above.)
    # Persist a per-task JSON file in the run output directory with collected data.
    try:
        import json as _json
        # merge any deferred per-generation results (train/test comparisons,
        # apply_cost, response_row_id) into the gen_records entries now that
        # the full task loop has completed.
        try:
            for rec in gen_records:
                idx = rec.get("generation_index")
                g = gen_results.get(idx, {})
                # ensure final_json is a parsed JSON object when possible
                try:
                    if rec.get("final_json"):
                        try:
                            parsed_obj = _json.loads(rec.get("final_json"))
                        except Exception:
                            parsed_obj = rec.get("final_json")
                        rec["final_json"] = parsed_obj
                except Exception:
                    pass
                # attach prompt details into the generation record
                try:
                    rec["prompt_hash"] = prompt_hash
                    rec["prompt_text"] = prompt.splitlines()
                except Exception:
                    pass
                # attach deferred comparison and scoring data
                try:
                    if g:
                        rec["train_scores"] = g.get("train_scores")
                        rec["training_comparison"] = g.get("training_comparison")
                        rec["test_score"] = g.get("test_score")
                        rec["testing_comparison"] = g.get("testing_comparison")
                        rec["apply_cost"] = g.get("apply_cost")
                        # prefer response_row_id from gen_results if present
                        rec["response_row_id"] = g.get("response_row_id", rec.get("response_row_id"))
                except Exception:
                    pass

        except Exception:
            pass

        # Reflection flow: iteratively attempt to refine any generation that
        # produced incorrect training outputs. Repeat up to max_reflections
        # times per generation or until all training examples score 1.0.
        try:
            for rec in list(gen_records):
                try:
                    idx = rec.get("generation_index")
                    # only consider generations that have training_comparison
                    tcomp = rec.get("training_comparison")
                    if not tcomp:
                        continue

                    # find any wrong cases (score < 1.0)
                    wrong_idxs = [e.get("example_index") for e in tcomp if (e.get("score") is not None and float(e.get("score")) < 1.0)]
                    if not wrong_idxs:
                        continue

                    # We'll iteratively reflect up to max_reflections times.
                    # Start with the original final_json (from the generation) as base.
                    base_final_json = rec.get("final_json")
                    try:
                        if isinstance(base_final_json, str):
                            base_final_json = _json.loads(base_final_json)
                    except Exception:
                        pass

                    current_final_json = base_final_json or {}
                    # For collecting per-iteration averages for summary
                    iteration = 0
                    while iteration < max_reflections:
                        iteration += 1

                        # Build wrong_cases and model_outputs from the latest training comparison
                        wrong_cases = []
                        model_outputs = []
                        train_examples = challenges.get(tid, {}).get("train", [])
                        # refresh tcomp from the most recent comparison if available
                        tcomp = rec.get("training_comparison") or []
                        for j in [e for e in range(len(train_examples)) if any(tc.get('example_index') == e and (tc.get('score') is not None and float(tc.get('score')) < 1.0) for tc in tcomp)]:
                            try:
                                ex = train_examples[j]
                            except Exception:
                                ex = None
                            if ex:
                                inp = ex.get("input")
                                expected = ex.get("output")
                            else:
                                inp = None
                                expected = None
                            wrong_cases.append((inp, expected))
                            # try to recover model output from the training_comparison entry
                            gen_entry = next((e for e in tcomp if e.get("example_index") == j), None)
                            gen_val = gen_entry.get("generated") if gen_entry is not None else None
                            # convert list-of-lines back to grid if needed
                            def _lines_to_grid(v):
                                try:
                                    if isinstance(v, list) and v and isinstance(v[0], str):
                                        grid = []
                                        for ln in v:
                                            parts = [p for p in ln.split() if p]
                                            grid.append([int(x) for x in parts])
                                        return grid
                                except Exception:
                                    pass
                                return v
                            model_outputs.append(_lines_to_grid(gen_val))

                        # build reflection prompt using the current_final_json
                        try:
                            reflection_prompt = prompts.build_reflection_prompt(current_final_json or {}, wrong_cases, model_outputs, task=challenges.get(tid), task_id=tid)
                        except Exception:
                            reflection_prompt = None

                        if not reflection_prompt:
                            break

                        # query the model for reflection (the actual reflection prompt)
                        try:
                            refl_text = await asyncio.to_thread(query_gemini, reflection_prompt)
                            refl_text = refl_text.strip()
                        except Exception:
                            refl_text = None

                        # attempt to extract trailing JSON from the reflection response
                        refl_body, refl_final_json = (None, None)
                        try:
                            if refl_text:
                                refl_body, refl_final_json = _extract_trailing_json(refl_text)
                                if refl_final_json:
                                    try:
                                        refl_final_json = _json.loads(refl_final_json)
                                    except Exception:
                                        pass
                        except Exception:
                            pass

                        try:
                            if refl_final_json:
                                print(f"[{tid} reflection {idx} iter={iteration}] final_json extracted")
                            else:
                                print(f"[{tid} reflection {idx} iter={iteration}] no final_json found")
                        except Exception:
                            pass

                        # If we have a revised final_json, apply it to all training and test examples
                        refl_train_scores = None
                        refl_training_comparison = None
                        refl_test_score = None
                        refl_testing_comparison = None
                        refl_apply_cost = None
                        if refl_final_json and isinstance(refl_final_json, dict):
                            try:
                                # apply to each training example
                                refl_train_scores = []
                                refl_training_comparison = []
                                train_examples = challenges.get(tid, {}).get("train", [])
                                for j, ex in enumerate(train_examples):
                                    try:
                                        apply_prompt_train = prompts.build_apply_prompt(refl_final_json, {"train": [ex], "test": []}, tid, include_examples=False)
                                        apply_resp_train = await asyncio.to_thread(query_gemini, apply_prompt_train)
                                        apply_text_train = apply_resp_train
                                    except Exception:
                                        apply_text_train = None

                                    parsed_out_t = None
                                    score_train_example = 0.0
                                    if apply_text_train:
                                        import re as _re
                                        m_t = _re.search(r"<output>([\s\S]*?)</output>", apply_text_train, flags=_re.IGNORECASE)
                                        if m_t:
                                            out_blob_t = m_t.group(1).strip()
                                            try:
                                                parsed_out_t = None
                                                try:
                                                    parsed_out_t = __import__("json").loads(out_blob_t)
                                                except Exception:
                                                    lines = [ln.strip() for ln in out_blob_t.splitlines() if ln.strip()]
                                                    if lines:
                                                        grid = []
                                                        ok = True
                                                        for ln in lines:
                                                            nums = _re.findall(r"-?\d+", ln)
                                                            if not nums:
                                                                ok = False
                                                                break
                                                            grid.append([int(x) for x in nums])
                                                        if ok:
                                                            parsed_out_t = grid
                                                if parsed_out_t is not None and tid in challenges:
                                                    try:
                                                        expected_grid = train_examples[j]["output"]
                                                        er = len(expected_grid)
                                                        ec = len(expected_grid[0]) if er > 0 else 0
                                                        gr = len(parsed_out_t) if isinstance(parsed_out_t, list) else 0
                                                        gc = len(parsed_out_t[0]) if gr > 0 and isinstance(parsed_out_t[0], list) else 0
                                                        total = er * ec if er and ec else 0
                                                        if total > 0:
                                                            matches = 0
                                                            for r in range(er):
                                                                for c in range(ec):
                                                                    try:
                                                                        if parsed_out_t[r][c] == expected_grid[r][c]:
                                                                            matches += 1
                                                                    except Exception:
                                                                        pass
                                                            gr = len(parsed_out_t) if isinstance(parsed_out_t, list) else 0
                                                            gc = len(parsed_out_t[0]) if gr > 0 and isinstance(parsed_out_t[0], list) else 0
                                                            size_s = size_score(er, ec, gr, gc)
                                                            color_s = color_score(expected_grid, parsed_out_t)
                                                            score_train_example = size_s * color_s
                                                    except Exception:
                                                        score_train_example = 0.0
                                            except Exception:
                                                score_train_example = 0.0
                                    refl_train_scores.append(float(score_train_example))
                                    # format expected/generated as list of space-joined lines
                                    try:
                                        expected_grid = ex.get("output")
                                        if isinstance(expected_grid, list) and expected_grid and isinstance(expected_grid[0], list):
                                            expected_lines = [" ".join(str(x) for x in row) for row in expected_grid]
                                        else:
                                            expected_lines = expected_grid
                                    except Exception:
                                        expected_lines = None
                                    try:
                                        gen_lines = None
                                        if isinstance(parsed_out_t, list) and parsed_out_t and isinstance(parsed_out_t[0], list):
                                            gen_lines = [" ".join(str(x) for x in row) for row in parsed_out_t]
                                        else:
                                            gen_lines = parsed_out_t
                                    except Exception:
                                        gen_lines = parsed_out_t
                                    refl_training_comparison.append({
                                        "example_index": j,
                                        "expected": expected_lines,
                                        "generated": gen_lines,
                                        "score": float(score_train_example),
                                    })

                                # apply to test inputs (single apply across tests)
                                try:
                                    apply_prompt = prompts.build_apply_prompt(refl_final_json, challenges[tid], tid)
                                    apply_resp_text = await asyncio.to_thread(query_gemini, apply_prompt)
                                    apply_text = apply_resp_text
                                except Exception:
                                    apply_text = None
                                if apply_text:
                                    import re as _re
                                    m = _re.search(r"<output>([\s\S]*?)</output>", apply_text, flags=_re.IGNORECASE)
                                    if m:
                                        out_blob = m.group(1).strip()
                                        try:
                                            parsed_out = None
                                            try:
                                                parsed_out = __import__("json").loads(out_blob)
                                            except Exception:
                                                lines = [ln.strip() for ln in out_blob.splitlines() if ln.strip()]
                                                if lines:
                                                    grid = []
                                                    ok = True
                                                    for ln in lines:
                                                        nums = _re.findall(r"-?\d+", ln)
                                                        if not nums:
                                                            ok = False
                                                            break
                                                        grid.append([int(x) for x in nums])
                                                    if ok:
                                                        parsed_out = grid
                                        except Exception:
                                            parsed_out = None
                                        # compare to solutions if available
                                        try:
                                            if solutions_path:
                                                with open(solutions_path, "r") as sf:
                                                    solutions = __import__("json").load(sf)
                                            else:
                                                solutions = {}
                                        except Exception:
                                            solutions = {}
                                        try:
                                            if parsed_out is not None and tid in solutions:
                                                expected_grid = solutions[tid][0]
                                                er = len(expected_grid)
                                                ec = len(expected_grid[0]) if er > 0 else 0
                                                gr = len(parsed_out) if isinstance(parsed_out, list) else 0
                                                gc = len(parsed_out[0]) if gr > 0 and isinstance(parsed_out[0], list) else 0
                                                total = er * ec if er and ec else 0
                                                if total > 0:
                                                    matches = 0
                                                    for r in range(er):
                                                        for c in range(ec):
                                                            try:
                                                                if parsed_out[r][c] == expected_grid[r][c]:
                                                                    matches += 1
                                                            except Exception:
                                                                pass
                                                    size_s = size_score(er, ec, gr, gc)
                                                    color_s = color_score(expected_grid, parsed_out)
                                                    refl_test_score = size_s * color_s
                                                else:
                                                    refl_test_score = 0.0
                                            else:
                                                refl_test_score = None
                                        except Exception:
                                            refl_test_score = None

                            except Exception:
                                pass

                        # build reflection generation record and append
                        try:
                            refl_rec = {
                                "generation_index": f"{idx}_reflection_{iteration}",
                                "timestamp": __import__('datetime').datetime.utcnow().isoformat() + "Z",
                                "prompt_response": refl_text.splitlines() if refl_text else None,
                                "prompt_text": reflection_prompt.splitlines() if reflection_prompt else None,
                                "final_json": refl_final_json,
                                "input_tokens": None,
                                "output_tokens": None,
                                "cost_estimate": None,
                                "apply_cost": refl_apply_cost,
                                "train_scores": refl_train_scores,
                                "training_comparison": refl_training_comparison,
                                "test_score": refl_test_score,
                                "testing_comparison": refl_testing_comparison,
                                "apply_cost": refl_apply_cost,
                                "response_row_id": None,
                            }
                            gen_records.append(refl_rec)
                        except Exception:
                            pass

                        # If the reflection produced full-accuracy training scores, stop
                        try:
                            if refl_train_scores and all(float(s) == 1.0 for s in refl_train_scores):
                                break
                        except Exception:
                            pass

                        # prepare for next iteration by updating rec's training_comparison
                        # so the next loop uses latest errors
                        try:
                            if refl_training_comparison:
                                rec["training_comparison"] = refl_training_comparison
                                rec["train_scores"] = refl_train_scores
                                # also update current_final_json for next reflection
                                if refl_final_json and isinstance(refl_final_json, dict):
                                    current_final_json = refl_final_json
                        except Exception:
                            pass

                    # end while iterations
                except Exception:
                    pass
        except Exception:
            pass

        record = {
            "task_id": tid,
            "run_name": run_name,
            "run_timestamp": run_timestamp,
            "model_name": selected_model_name,
            "generations": gen_records,
            "metrics": {
                "duration": task_duration,
                "model_calls": int(task_model_calls),
                "input_tokens": int(task_input_tokens),
                "output_tokens": int(task_output_tokens),
                "total_cost_usd": float(task_total_cost),
            },
        }
        if run_output_dir:
            out_path = os.path.join(run_output_dir, f"{tid}.json")
            try:
                with open(out_path, "w") as of:
                    _json.dump(record, of, indent=2, ensure_ascii=False)
            except Exception:
                # non-fatal: don't crash the worker if writing fails
                pass
    except Exception:
        pass

    # Per-task summary: print concise counts and any problematic generations
    try:
        total_gens = len(gen_records)
        final_json_count = sum(1 for r in gen_records if r.get("final_json"))
        apply_count = sum(1 for r in gen_records if (r.get("test_score") is not None or r.get("testing_comparison")))
        reflections_count = sum(1 for r in gen_records if isinstance(r.get("generation_index"), str) and "_reflection" in str(r.get("generation_index")))

        # find generations with imperfect training scores
        gens_with_issues = []
        for r in gen_records:
            try:
                ts = r.get("train_scores")
                if ts and any((s is None or float(s) < 1.0) for s in ts):
                    gens_with_issues.append(r.get("generation_index"))
            except Exception:
                pass

        print(f"\nSummary for task {tid}: gens={total_gens}, final_jsons={final_json_count}, applies={apply_count}, reflections={reflections_count}")
        print(f"Metrics: model_calls={int(task_model_calls)}, input_tokens={int(task_input_tokens)}, output_tokens={int(task_output_tokens)}, duration={task_duration:.3f}s")

        # Compute average train/test scores for initial generations
        try:
            import statistics as _stats
            initial_train_means = []
            initial_test_scores = []
            for r in gen_records:
                try:
                    gi = r.get("generation_index")
                    # initial gens use integer indices
                    if isinstance(gi, int):
                        ts = r.get("train_scores")
                        if ts:
                            initial_train_means.append(sum(ts) / len(ts))
                        if r.get("test_score") is not None:
                            initial_test_scores.append(float(r.get("test_score")))
                except Exception:
                    pass

            if initial_train_means:
                print(f"Initial average train score (mean over gens) = {_stats.mean(initial_train_means):.3f} (n={len(initial_train_means)})")
            else:
                print("Initial average train score: N/A")

            if initial_test_scores:
                print(f"Initial average test score (mean over gens) = {_stats.mean(initial_test_scores):.3f} (n={len(initial_test_scores)})")
            else:
                print("Initial average test score: N/A")

            # For each reflection iteration compute averages across generations
            for k in range(1, max_reflections + 1):
                try:
                    refl_train_means = []
                    refl_test_scores = []
                    for r in gen_records:
                        try:
                            gi = r.get("generation_index")
                            if isinstance(gi, str) and f"_reflection_{k}" in gi:
                                ts = r.get("train_scores")
                                if ts:
                                    refl_train_means.append(sum(ts) / len(ts))
                                if r.get("test_score") is not None:
                                    refl_test_scores.append(float(r.get("test_score")))
                        except Exception:
                            pass
                    if refl_train_means or refl_test_scores:
                        tavg = _stats.mean(refl_train_means) if refl_train_means else None
                        tavgs = _stats.mean(refl_test_scores) if refl_test_scores else None
                        print(f"Reflection {k} averages: train={tavg:.3f} (n={len(refl_train_means)})" if tavg is not None else f"Reflection {k} averages: train=N/A", end="")
                        if tavgs is not None:
                            print(f", test={tavgs:.3f} (n={len(refl_test_scores)})")
                        else:
                            print("")
                    else:
                        # no reflections of this iteration were produced
                        pass
                except Exception:
                    pass
        except Exception:
            pass
        if gens_with_issues:
            try:
                print("\033[91mGenerations with imperfect training scores:\033[0m ", ", ".join(map(str, gens_with_issues)))
            except Exception:
                pass
    except Exception:
        pass

    # return simple metrics for aggregation by the caller
    return {
        "task_id": tid,
        "duration": task_duration,
        "model_calls": int(task_model_calls),
        "input_tokens": int(task_input_tokens),
        "output_tokens": int(task_output_tokens),
    }


def _gather_task_ids(challenges: dict, limit: int | None):
    task_ids = list(challenges.keys())
    if limit is not None and limit > 0:
        task_ids = task_ids[:limit]
    return task_ids


def _remove_db_if_requested(clear: bool):
    if not clear:
        return
    try:
        if os.path.exists(sqlite_store.DB_PATH):
            print(f"Removing existing DB at {sqlite_store.DB_PATH} due to --clear_responses flag")
            os.remove(sqlite_store.DB_PATH)
    except Exception as e:
        print(f"Warning: failed to remove DB file: {e}")


def _run_async_main(args):
    # synchronous wrapper to run the async worker
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a Gemini prompt from prompts.py using a selected model"
    )
    parser.add_argument(
        "--challenges",
        "-c",
        dest="challenges",
        help="Path to a JSON file containing ARC challenges",
        required=True,
    )
    parser.add_argument(
        "--solutions",
        "-s",
        dest="solutions",
        help="Path to a JSON file containing ARC solutions",
        required=False,
    )
    parser.add_argument(
        "--limit",
        "-l",
        dest="limit",
        type=int,
        help="Maximum number of tasks to process from the challenges file",
        default=None,
    )
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"Model name to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--clear-responses",
        dest="clear_responses",
        action="store_true",
        help="Remove existing responses DB file before running",
    )
    parser.add_argument(
        "--run-name",
        dest="run_name",
        help="Optional name for this run (defaults to current datetime.isoformat())",
        required=False,
    )
    parser.add_argument(
        "--num-initial-generations",
        "-n",
        dest="num_initial_generations",
        type=int,
        default=1,
        help="Number of initial generations to perform",
    )
    parser.add_argument(
        "--max-reflections",
        dest="max_reflections",
        type=int,
        default=3,
        help="Maximum number of reflection iterations per generation (default: 3)",
    )
    args = parser.parse_args()

    selected_model_name = args.model
    challenges_path = args.challenges
    solutions_path = args.solutions
    limit = args.limit
    num_initial_generations = args.num_initial_generations
    max_reflections = args.max_reflections
    # compute a single run-level timestamp and default run_name
    from datetime import datetime, timezone
    run_timestamp = datetime.now(timezone.utc).isoformat() + "Z"
    run_name = args.run_name if args.run_name else run_timestamp

    # create per-run output directory and set DB path
    output_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
    run_output_dir = os.path.join(output_base, run_timestamp)
    os.makedirs(run_output_dir, exist_ok=True)
    sqlite_store.DB_PATH = os.path.join(run_output_dir, "responses.db")

    # optionally remove existing DB
    if args.clear_responses:
        _remove_db_if_requested(True)

    # build the model instance using the selected model name
    model = build_model(selected_model_name)
    # ensure DB is initialized
    try:
        sqlite_store.init_db()
    except Exception as e:
        print(f"Warning: failed to initialize DB: {e}")
    # start background DB writer so enqueue() calls have a running worker
    try:
        get_global_writer()
    except Exception:
        pass

    # ensure graceful shutdown flushes queued DB writes
    def _graceful_shutdown(*_args):
        try:
            w = get_global_writer()
            w.stop_and_flush()
        except Exception:
            pass

    atexit.register(_graceful_shutdown)
    try:
        signal.signal(signal.SIGINT, lambda *_: _graceful_shutdown())
        signal.signal(signal.SIGTERM, lambda *_: _graceful_shutdown())
    except Exception:
        # signal may not be available on some platforms (or when running in certain environments)
        pass

    # Load challenges json
    try:
        with open(challenges_path, "r") as f:
            challenges = __import__("json").load(f)
    except FileNotFoundError:
        print(f"Error: challenges file not found: {challenges_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading challenges file: {e}")
        sys.exit(1)

    # create list of task coroutines and run with concurrency limit
    async def main_async(args):
        # ensure selected_model_name is available in closure
        selected_model_name_local = selected_model_name

        # load challenges json
        try:
            with open(challenges_path, "r") as f:
                challenges_local = __import__("json").load(f)
        except FileNotFoundError:
            print(f"Error: challenges file not found: {challenges_path}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading challenges file: {e}")
            sys.exit(1)

        task_ids = _gather_task_ids(challenges_local, limit)

        max_concurrency = DEFAULT_MAX_CONCURRENCY
        # allow overriding via .env value or environment variable
        try:
            env_val = os.getenv("MAX_CONCURRENCY")
            if env_val:
                max_concurrency = int(env_val)
            else:
                max_concurrency = int(_env.get("MAX_CONCURRENCY", max_concurrency) or max_concurrency)
        except Exception:
            pass

        sem = asyncio.Semaphore(max_concurrency)
        num_initial_generations_local = num_initial_generations
        # schedule tasks
        # create per-run output directory so tasks can write per-id JSON files
        output_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
        run_output_dir = os.path.join(output_base, run_timestamp)
        try:
            os.makedirs(run_output_dir, exist_ok=True)
        except Exception:
            pass

        coros = [
            _process_task(tid, challenges_local, challenges_path, selected_model_name_local, solutions_path, sem, run_name, run_timestamp, run_output_dir, num_initial_generations_local, max_reflections)
            for tid in task_ids
        ]
        # run them concurrently
        await asyncio.gather(*coros)

    # run the async main
    try:
        _run_async_main(args)
    except Exception as e:
        print(f"Fatal error during run: {e}")