#!/usr/bin/env python3
"""Simple utility: count top-level keys in a JSON file.

Usage: python count_challenges.py /path/to/file.json

Prints a single integer (the number of top-level keys) to stdout and exits 0.
Exits with non-zero status on error.
"""
import sys
import json
from pathlib import Path


def count_top_level_keys(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return len(data)
    # If the file contains a list of challenges, treat the length as count
    if isinstance(data, list):
        return len(data)
    # Otherwise, not a recognizable challenges file
    raise ValueError("JSON root must be an object (dict) or an array (list)")


def main(argv):
    if len(argv) != 2:
        print("Usage: count_challenges.py /path/to/challenges.json", file=sys.stderr)
        return 2
    path = Path(argv[1])
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 3
    try:
        count = count_top_level_keys(path)
    except Exception as e:
        print(f"Error reading JSON: {e}", file=sys.stderr)
        return 4
    print(count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
