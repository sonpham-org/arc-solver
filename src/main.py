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
from dotenv import dotenv_values

# local DB helper
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from db import sqlite_store

# Ensure the package's src/ directory is importable when running this file as
# a script (python src/main.py). This inserts the file's directory onto
# sys.path so we can import local modules like model_configs reliably.
sys.path.insert(0, os.path.dirname(__file__))
import model_configs
DEFAULT_MODEL = model_configs.DEFAULT_MODEL
is_known_model = model_configs.is_known_model

# Load .env from src/
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
# load simple .env values for non-sensitive runtime config
_env = dotenv_values(os.path.join(os.path.dirname(__file__), ".env"))
# default concurrency if not provided in .env
DEFAULT_MAX_CONCURRENCY = int(_env.get("MAX_CONCURRENCY", 4) or 4)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Missing GEMINI_API_KEY in .env")
os.environ.setdefault("GRPC_VERBOSITY", "NONE")
os.environ.setdefault("GRPC_LOG_SEVERITY_LEVEL", "ERROR")

genai.configure(api_key=api_key)


def build_model(model_name: str):
    """Create and return a GenerativeModel for the given model_name.

    We keep this factory to make testing and future extension easier.
    """
    # Optionally warn if model is unknown (but still attempt to create it)
    if not is_known_model(model_name):
        print(f"Warning: model '{model_name}' is not in model_configs registry")
    return genai.GenerativeModel(model_name)

def query_gemini(prompt: str) -> str:
    """Send prompt to Gemini and return the response text.

    Note: the underlying client is synchronous, so callers should run this
    inside a thread via asyncio.to_thread when used from async code.
    """
    response = model.generate_content(prompt)
    return response.text.strip()


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
    import re

    if not text:
        return 0
    # count words as proxy for tokens
    words = re.findall(r"\w+", text)
    return max(1, len(words))

async def _process_task(tid: str, challenges: dict, challenges_path: str, selected_model_name: str, solutions_path: str, sem: asyncio.Semaphore, run_name: str = None, run_timestamp: str = None, num_initial_generations: int = 1):
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

        print(f"\n🧩 Built prompt for task {tid}:\n")
        print(prompt)

        # Send to Gemini n times
        for i in range(num_initial_generations):
            try:
                text = await asyncio.to_thread(query_gemini, prompt)
                text = text.strip()
            except Exception as e:
                print(f"Error querying Gemini for {tid} generation {i}: {e}")
                continue

            print(f"=== Gemini Response {i} ===\n")
            print(text)

            # store response in local sqlite DB (blocking -> thread)
            try:
                body, final_json = _extract_trailing_json(text)
                reasoning = text
                output_tokens = _estimate_tokens(text)
                # estimate input tokens from the prompt
                input_tokens = _estimate_tokens(prompt)
                import hashlib

                prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                try:
                    # insert prompt and response in thread to avoid blocking the event loop
                    await asyncio.to_thread(sqlite_store.insert_prompt, prompt_hash, prompt)
                except Exception:
                    pass

                # estimate monetary cost for this response using model pricing
                try:
                    cost_est = model_configs.estimate_cost(selected_model_name, input_tokens=input_tokens, output_tokens=output_tokens)
                except Exception:
                    cost_est = None

                # insert response and capture row id
                try:
                    rid = await asyncio.to_thread(
                        sqlite_store.insert_response,
                        selected_model_name,
                        prompt_hash,
                        reasoning,
                        final_json if final_json else None,
                        challenges_path,
                        solutions_path,
                        str(tid),
                        None,
                        None,
                        cost_est,
                        None,
                        input_tokens,
                        output_tokens,
                        run_name,
                        run_timestamp,
                        i,  # prompt_index
                    )
                except Exception:
                    rid = None
            except Exception as e:
                print(f"Warning: failed to save response {i} to DB: {e}")
                body = None
                final_json = None
                rid = None

            # If we have a parsed final_json and a solutions file, build APPLY_PROMPT
            if final_json and solutions_path:
                try:
                    parsed = __import__("json").loads(final_json)
                except Exception:
                    parsed = None

                if isinstance(parsed, dict) and parsed.get("step_by_step_rule"):
                    try:
                        with open(solutions_path, "r") as sf:
                            solutions = __import__("json").load(sf)
                    except Exception as e:
                        print(f"Warning: failed to load solutions file: {e}")
                        solutions = {}

                    try:
                        # First: apply the extracted rule to each training example (no examples included in prompt)
                        train_scores = []
                        train_examples = challenges[tid].get("train", [])
                        for j, ex in enumerate(train_examples):
                            try:
                                # build an apply prompt that applies the rule to this single training input
                                apply_prompt_train = prompts.build_apply_prompt(parsed, {"train": [ex], "test": []}, tid, include_examples=False)
                                apply_resp_train = await asyncio.to_thread(query_gemini, apply_prompt_train)
                                apply_text_train = apply_resp_train
                            except Exception as e:
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
                            if rid:
                                await asyncio.to_thread(sqlite_store.update_response_train_scores, rid, train_scores_json)
                                # update aggregate mean
                                mean_train = sum(train_scores) / len(train_scores) if train_scores else 0.0
                                await asyncio.to_thread(sqlite_store.update_response_train_score, rid, float(mean_train))
                        except Exception:
                            pass

                        # Now run apply on the test inputs as before
                        apply_prompt = prompts.build_apply_prompt(parsed, challenges[tid], tid)
                        apply_resp_text = await asyncio.to_thread(query_gemini, apply_prompt)
                        apply_text = apply_resp_text
                        print(f"=== Apply Response {i} ===\n")
                        print(apply_text)
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
                                        score = matches / total
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
                                    await asyncio.to_thread(sqlite_store.update_response_scores, rid, float(score))
                                    try:
                                        if parsed_out is not None:
                                            import json as _json

                                            await asyncio.to_thread(sqlite_store.update_response_apply_output, rid, _json.dumps(parsed_out))
                                    except Exception:
                                        pass
                                conn.close()
                                if apply_text and row:
                                    apply_tokens = _estimate_tokens(apply_text)
                                    try:
                                        await asyncio.to_thread(sqlite_store.update_response_tokens, rid, apply_tokens)
                                    except Exception:
                                        pass
                                    try:
                                        # compute additional monetary cost for apply_text tokens and add to the response cost
                                        try:
                                            apply_cost = model_configs.estimate_cost(selected_model_name, input_tokens=0, output_tokens=apply_tokens)
                                        except Exception:
                                            apply_cost = 0.0
                                        await asyncio.to_thread(sqlite_store.update_response_add_cost, rid, float(apply_cost))
                                    except Exception:
                                        pass
                            except Exception as e:
                                print(f"Warning: failed to update scores for {tid} generation {i}: {e}")


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
    args = parser.parse_args()

    selected_model_name = args.model
    challenges_path = args.challenges
    solutions_path = args.solutions
    limit = args.limit
    num_initial_generations = args.num_initial_generations
    # compute a single run-level timestamp and default run_name
    from datetime import datetime, timezone
    run_timestamp = datetime.now(timezone.utc).isoformat() + "Z"
    run_name = args.run_name if args.run_name else run_timestamp

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
        coros = [
            _process_task(tid, challenges_local, challenges_path, selected_model_name_local, solutions_path, sem, run_name, run_timestamp, num_initial_generations_local)
            for tid in task_ids
        ]
        # run them concurrently
        await asyncio.gather(*coros)

    # run the async main
    try:
        _run_async_main(args)
    except Exception as e:
        print(f"Fatal error during run: {e}")