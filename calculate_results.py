#!/usr/bin/env python3
"""
Interactive script to browse and analyze ARC test results.

This script allows users to:
1. Browse output directories from previous test runs
2. Select a run using an interactive command-line interface
3. Generate summary statistics if they don't exist
4. Display the summary results
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

from utils import calculate_results


def get_output_directories() -> List[Path]:
    """Find all directories in the output folder."""
    output_folder = Path("output")
    
    if not output_folder.exists():
        print("No output folder found. Please run test scripts first.")
        return []
    
    # Get all directories, sorted by modification time (newest first)
    directories = [d for d in output_folder.iterdir() if d.is_dir()]
    directories.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return directories


def display_directory_info(directory: Path) -> str:
    """Get display information for a directory."""
    # Try to read params.json for more info
    params_file = directory / "params.json"
    info_parts = [directory.name]
    
    if params_file.exists():
        try:
            with open(params_file, 'r') as f:
                params = json.load(f)
            
            script = params.get('script', 'unknown')
            model = params.get('model', 'unknown')
            num_tasks = params.get('number_of_tasks', 'unknown')
            
            info_parts.append(f"({script}, {model}, {num_tasks} tasks)")
        except Exception:
            pass
    
    # Count task files
    task_files = [f for f in directory.glob("*.json") 
                  if f.name not in ["summary.json", "params.json"]]
    info_parts.append(f"[{len(task_files)} results]")
    
    # Check if summary exists
    summary_file = directory / "summary.json"
    if summary_file.exists():
        info_parts.append("✓")
    else:
        info_parts.append("✗")
    
    return " ".join(info_parts)


def interactive_directory_selection(directories: List[Path]) -> Path:
    """Interactive CLI for selecting a directory."""
    if not directories:
        print("No output directories found.")
        sys.exit(1)
    
    current_index = 0
    
    while True:
        # Clear screen (works on most terminals)
        print("\033[2J\033[H", end="")
        
        print("📊 ARC Test Results Browser")
        print("=" * 50)
        print("Use ↑/↓ (or w/s) to navigate, Enter to select, q to quit")
        print("Format: [timestamp] (script, model, tasks) [results] [summary]")
        print()
        
        # Display directories with highlighting
        for i, directory in enumerate(directories):
            prefix = "➤ " if i == current_index else "  "
            info = display_directory_info(directory)
            
            if i == current_index:
                print(f"\033[92m{prefix}{info}\033[0m")  # Green highlight
            else:
                print(f"{prefix}{info}")
        
        print()
        print("Controls: w/↑=up, s/↓=down, Enter=select, q=quit")
        
        # Get user input
        try:
            # Try to read a single character
            import termios, tty
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                key = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except ImportError:
            # Fallback for Windows or systems without termios
            key = input().strip()
            if key == "":
                key = "\n"
            elif len(key) > 0:
                key = key[0].lower()
        
        # Handle input
        if key.lower() == 'q':
            print("Goodbye!")
            sys.exit(0)
        elif key in ['w', 'W'] or ord(key) == 27:  # Up arrow (starts with escape)
            if ord(key) == 27:
                # Read the rest of the arrow key sequence
                try:
                    next1 = sys.stdin.read(1)
                    next2 = sys.stdin.read(1)
                    if ord(next1) == 91 and ord(next2) == 65:  # Up arrow
                        current_index = max(0, current_index - 1)
                    elif ord(next1) == 91 and ord(next2) == 66:  # Down arrow
                        current_index = min(len(directories) - 1, current_index + 1)
                except:
                    pass
            else:
                current_index = max(0, current_index - 1)
        elif key in ['s', 'S']:  # Down
            current_index = min(len(directories) - 1, current_index + 1)
        elif key == '\n' or key == '\r':  # Enter
            return directories[current_index]


def load_and_display_summary(directory: Path) -> None:
    """Load and display the summary.json file."""
    # Instead of relying on a precomputed summary.json, scan task files
    # and count 'correct' fields inside the `test` (or similar) lists.
    task_files = [f for f in directory.glob("*.json")
                  if f.name not in ["summary.json", "params.json"]]

    total_tasks = 0
    summed_task_scores = 0.0  # sum of per-task averaged correctness in [0,1]
    total_test_items = 0
    raw_correct_items = 0
    total_tokens = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for f in task_files:
        try:
            with open(f, 'r') as fh:
                data = json.load(fh)

            # Prefer explicit 'test' or 'tests' lists for scoring
            tests_list = None
            if isinstance(data, dict):
                for preferred in ('test', 'tests'):
                    candidate = data.get(preferred)
                    if isinstance(candidate, list) and candidate and all(isinstance(x, dict) for x in candidate):
                        tests_list = candidate
                        break

                # Fallback: find any list-of-dicts that contains 'correct' keys
                if tests_list is None:
                    for key, val in data.items():
                        if isinstance(val, list) and val and all(isinstance(x, dict) for x in val) and any('correct' in x for x in val):
                            tests_list = val
                            break

            elif isinstance(data, list):
                if data and all(isinstance(x, dict) for x in data) and any('correct' in x for x in data):
                    tests_list = data

            if not tests_list:
                # nothing to score in this file
                continue

            # Compute per-task score: average correctness across test cases for this task
            num_cases = len(tests_list)
            num_correct_cases = sum(1 for item in tests_list if bool(item.get('correct')))
            task_score = (num_correct_cases / num_cases) if num_cases else 0.0

            total_tasks += 1
            summed_task_scores += task_score

            # keep raw counts as well
            total_test_items += num_cases
            raw_correct_items += num_correct_cases

            # Aggregate token info if available per-file
            if isinstance(data, dict):
                total_tokens += int(data.get('total_tokens', 0) or 0)
                total_input_tokens += int(data.get('input_tokens', 0) or 0)
                total_output_tokens += int(data.get('output_tokens', 0) or 0)

        except Exception:
            # Ignore malformed files and continue
            continue

    # Display computed summary where each file contributes at most 1.0 (averaged over its test cases)
    print(f"\n📊 Results Summary for {directory.name}")
    print("=" * 60)
    print(f"Total tasks processed (files with test lists): {total_tasks}")
    print(f"Raw test cases processed: {total_test_items}")
    print(f"Raw correct test cases: {raw_correct_items}")
    print()

    # summed_task_scores is in [0, total_tasks]
    average_score_per_task = (summed_task_scores / total_tasks) if total_tasks else 0.0
    correctness_percentage = average_score_per_task * 100.0
    print(f"Summed task scores (sum of per-task averages): {summed_task_scores:.3f}")
    print(f"Average score per task: {average_score_per_task:.3f} ({correctness_percentage:.2f}%)")
    print()

    # Token statistics (best-effort)
    if total_tokens or total_input_tokens or total_output_tokens:
        print(f"Total tokens (sum from files): {total_tokens:,}")
        print(f"  - Input tokens: {total_input_tokens:,}")
        print(f"  - Output tokens: {total_output_tokens:,}")
        avg_tokens_per_task = (total_tokens / total_tasks) if total_tasks else 0.0
        print(f"Average tokens per task (best-effort): {avg_tokens_per_task:.1f}")

    print("=" * 60)


def main():
    """Main function to run the interactive results browser."""
    print("🔍 Searching for ARC test results...")
    
    directories = get_output_directories()
    
    if not directories:
        print("No output directories found. Please run test scripts first.")
        sys.exit(1)
    
    print(f"Found {len(directories)} result directories")
    
    # Interactive selection
    selected_directory = interactive_directory_selection(directories)
    
    print(f"\n✅ Selected: {selected_directory.name}")
    
    # Load and display summary
    load_and_display_summary(selected_directory)


if __name__ == "__main__":
    main()