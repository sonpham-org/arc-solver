#!/usr/bin/env python3
"""Utility to count train and test examples for each challenge in a JSON file.

Usage: python count_challenges.py /path/to/challenges.json

Prints a table with challenge ID, number of train examples, and number of test examples.
"""
import sys
import json
from pathlib import Path


def print_challenge_counts(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object (dict) with challenge IDs as keys")

    print("Challenge ID       | Num Train | Num Test")
    print("-" * 45)
    for tid, challenge in data.items():
        if not isinstance(challenge, dict):
            continue
        num_train = len(challenge.get('train', []))
        num_test = len(challenge.get('test', []))
        print(f"{tid:<18} | {num_train:<10} | {num_test:<9}")


def main(argv):
    if len(argv) != 2:
        print("Usage: count_challenges.py /path/to/challenges.json", file=sys.stderr)
        return 2
    path = Path(argv[1])
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 3
    try:
        print_challenge_counts(path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
