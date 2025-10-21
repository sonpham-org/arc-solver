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
    """Send prompt to Gemini and return the response text."""
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
    args = parser.parse_args()

    selected_model_name = args.model
    challenges_path = args.challenges
    solutions_path = args.solutions
    limit = args.limit

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

    # Iterate over task ids up to limit
    task_ids = list(challenges.keys())
    if limit is not None and limit > 0:
        task_ids = task_ids[:limit]

    for tid in task_ids:
        print(f"\n--- Task {tid} ---")
        try:
            prompt = prompts.build_arc_prompt(challenges, tid)
        except Exception as e:
            print(f"Error building prompt for {tid}: {e}")
            continue

        print(f"\n🧩 Built prompt for task {tid}:\n")
        # Print the prompt being sent (optional large)
        print(prompt)

        # Send to Gemini using the model instance
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
        except Exception as e:
            print(f"Error querying Gemini for {tid}: {e}")
            continue

        print("=== Gemini Response ===\n")
        print(text)

        # store response in local sqlite DB
        try:
            body, final_json = _extract_trailing_json(text)
            # store the full generated text as reasoning per request
            reasoning = text
            tokens = _estimate_tokens(text)
            # compute prompt hash and register prompt to avoid storing duplicates
            import hashlib

            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            try:
                sqlite_store.insert_prompt(prompt_hash, prompt)
            except Exception:
                # non-fatal: continue even if prompt insert fails
                pass
            # cost_estimate left as None for now
            sqlite_store.insert_response(
                model_name=selected_model_name,
                prompt_hash=prompt_hash,
                reasoning=reasoning,
                final_json=final_json if final_json else None,
                challenges=challenges_path,
                solutions=solutions_path,
                challenge_id=str(tid),
                score_train=None,
                score_test=None,
                tokens=tokens,
                cost_estimate=None,
            )
        except Exception as e:
            print(f"Warning: failed to save response to DB: {e}")

        # If we have a parsed final_json and a solutions file, build APPLY_PROMPT
        if final_json and solutions_path:
            try:
                parsed = __import__("json").loads(final_json)
            except Exception:
                parsed = None

            if isinstance(parsed, dict) and parsed.get("step_by_step_rule"):
                try:
                    # load solutions file and get expected outputs for this task id
                    with open(solutions_path, "r") as sf:
                        solutions = __import__("json").load(sf)
                except Exception as e:
                    print(f"Warning: failed to load solutions file: {e}")
                    solutions = {}

                # build apply prompt and query Gemini to apply the rule
                try:
                    apply_prompt = prompts.build_apply_prompt(parsed, challenges[tid], tid)
                    apply_resp = model.generate_content(apply_prompt)
                    apply_text = apply_resp.text
                    # print the raw apply response so we can inspect reasoning and the <output> block
                    print("=== Apply Response ===\n")
                    print(apply_text)
                except Exception as e:
                    print(f"Warning: failed to run apply prompt for {tid}: {e}")
                    apply_text = None

                # extract text between <output>...</output>
                extracted = None
                if apply_text:
                    import re

                    m = re.search(r"<output>([\s\S]*?)</output>", apply_text, flags=re.IGNORECASE)
                    if m:
                        out_blob = m.group(1).strip()
                        # try to parse as JSON (grid) or simple representation
                        try:
                            expected = solutions.get(tid)
                            parsed_out = None
                            try:
                                parsed_out = __import__("json").loads(out_blob)
                            except Exception:
                                # try to parse as simple whitespace-separated grid
                                import re

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

                        # simple comparison: check equality with solutions[tid][0]
                        match = False
                        score = 0.0
                        if parsed_out is not None:
                            # print parsed output for inspection
                            print("=== Parsed <output> ===\n")
                            try:
                                import json as _json

                                print(_json.dumps(parsed_out, indent=2))
                            except Exception:
                                print(parsed_out)

                        if parsed_out is not None and tid in solutions:
                            try:
                                expected_grid = solutions[tid][0]
                                # If parsed_out is a list of rows (2D) but solutions are nested (list of outputs), handle accordingly
                                # Ensure both are 2D lists
                                if isinstance(parsed_out, list) and parsed_out and isinstance(parsed_out[0], list):
                                    out_grid = parsed_out
                                else:
                                    out_grid = parsed_out

                                # compute overlap: number of matching cells / total expected cells
                                er = len(expected_grid)
                                ec = len(expected_grid[0]) if er > 0 else 0
                                total = er * ec if er and ec else 0
                                if total > 0:
                                    matches = 0
                                    for r in range(er):
                                        for c in range(ec):
                                            try:
                                                if out_grid[r][c] == expected_grid[r][c]:
                                                    matches += 1
                                            except Exception:
                                                # out of bounds or invalid, treat as mismatch
                                                pass
                                    score = matches / total
                                    match = matches == total
                                else:
                                    score = 0.0
                                    match = False
                            except Exception:
                                match = False
                                score = 0.0

                        # update DB with test score (fraction between 0.0 and 1.0)
                        try:
                            # find last inserted row id by selecting max id for this prompt_hash & tid
                            conn = __import__("sqlite3").connect(sqlite_store.DB_PATH)
                            cur = conn.cursor()
                            cur.execute(
                                "SELECT id FROM responses WHERE prompt_hash = ? AND challenge_id = ? ORDER BY id DESC LIMIT 1",
                                (prompt_hash, str(tid)),
                            )
                            row = cur.fetchone()
                            if row:
                                rid = row[0]
                                sqlite_store.update_response_scores(rid, None, float(score))
                                # also store the parsed apply output JSON if available
                                try:
                                    if parsed_out is not None:
                                        import json as _json

                                        sqlite_store.update_response_apply_output(rid, _json.dumps(parsed_out))
                                except Exception:
                                    pass
                            conn.close()
                            # also add token estimate for apply_text
                            if apply_text and row:
                                apply_tokens = _estimate_tokens(apply_text)
                                try:
                                    sqlite_store.update_response_tokens(rid, apply_tokens)
                                except Exception:
                                    pass
                        except Exception as e:
                            print(f"Warning: failed to update scores for {tid}: {e}")