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
    summary_file = directory / "summary.json"
    
    if not summary_file.exists():
        print(f"\nSummary file not found. Generating summary for {directory.name}...")
        try:
            calculate_results(str(directory))
        except Exception as e:
            print(f"Error generating summary: {e}")
            return
    
    # Load and display the summary
    try:
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        print(f"\n📊 Results Summary for {directory.name}")
        print("=" * 60)
        
        # Basic statistics
        print(f"Total tasks processed: {summary['total_tasks']}")
        print(f"Completely correct tasks: {summary['completely_correct_tasks']}")
        print(f"Correctness percentage: {summary['correctness_percentage']:.2f}%")
        print()
        
        # Token and cost statistics
        print(f"Total tokens used: {summary['total_tokens']:,}")
        print(f"  - Input tokens: {summary['total_input_tokens']:,}")
        print(f"  - Output tokens: {summary['total_output_tokens']:,}")
        print(f"Average tokens per task: {summary['average_tokens_per_task']:.1f}")
        print()
        
        print(f"Total estimated cost: ${summary['total_estimated_cost']:.6f}")
        print(f"Average cost per task: ${summary['average_cost_per_task']:.6f}")
        print()
        
        # Performance breakdown
        if 'task_details' in summary:
            task_details = summary['task_details']
            correct_tasks = [t for t in task_details if t['completely_correct']]
            incorrect_tasks = [t for t in task_details if not t['completely_correct']]
            
            if correct_tasks:
                avg_tokens_correct = sum(t['tokens'] for t in correct_tasks) / len(correct_tasks)
                avg_cost_correct = sum(t['cost'] for t in correct_tasks) / len(correct_tasks)
                print(f"Correct tasks - Avg tokens: {avg_tokens_correct:.1f}, Avg cost: ${avg_cost_correct:.6f}")
            
            if incorrect_tasks:
                avg_tokens_incorrect = sum(t['tokens'] for t in incorrect_tasks) / len(incorrect_tasks)
                avg_cost_incorrect = sum(t['cost'] for t in incorrect_tasks) / len(incorrect_tasks)
                print(f"Incorrect tasks - Avg tokens: {avg_tokens_incorrect:.1f}, Avg cost: ${avg_cost_incorrect:.6f}")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"Error reading summary file: {e}")


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