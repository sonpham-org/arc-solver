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
    limit = args.limit

    # build the model instance using the selected model name
    model = build_model(selected_model_name)

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