#!/usr/bin/env python3
"""
Script to analyze performance by generations/loops.

This script processes ARC task results to show how many tasks are solved
at each generation/loop (0, 1, 2, 3, 4, 5).

Usage:
    python calculate_performance_by_generations.py
    
The script will interactively list all runs and allow you to select one.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime


def get_available_runs(base_output_dir: str = "output/output_agent") -> List[Tuple[str, Path, int, datetime]]:
    """
    Get all available runs sorted by timestamp (newest first).
    
    Args:
        base_output_dir: Base directory containing run folders
        
    Returns:
        List of tuples (run_name, run_path, num_tasks, timestamp)
    """
    base_path = Path(base_output_dir)
    if not base_path.exists():
        print(f"Error: Path {base_output_dir} does not exist")
        return []
    
    runs = []
    
    for run_dir in base_path.iterdir():
        if not run_dir.is_dir():
            continue
        
        # Count tasks with task_id.json
        num_tasks = 0
        for task_dir in run_dir.iterdir():
            if task_dir.is_dir():
                task_json = task_dir / f"{task_dir.name}.json"
                if task_json.exists():
                    num_tasks += 1
        
        # Try to parse timestamp from directory name
        # Expected format: YYYY-MM-DDTHH-MM-SS-microseconds
        try:
            # Extract timestamp part (before any additional suffix)
            name = run_dir.name
            # Handle formats like "2025-12-08T01-21-15-172665"
            timestamp_str = name.split('_')[0] if '_' in name else name
            timestamp_str = timestamp_str.replace('T', ' ').replace('-', ':', 2)
            # Parse: "2025-12-08 01:21:15-172665"
            parts = timestamp_str.split('-')
            if len(parts) >= 2:
                dt_part = parts[0]  # "2025-12-08 01:21:15"
                timestamp = datetime.strptime(dt_part, "%Y:%m:%d %H:%M:%S")
            else:
                # Fallback to modification time
                timestamp = datetime.fromtimestamp(run_dir.stat().st_mtime)
        except Exception:
            # Fallback to directory modification time
            timestamp = datetime.fromtimestamp(run_dir.stat().st_mtime)
        
        runs.append((run_dir.name, run_dir, num_tasks, timestamp))
    
    # Sort by timestamp, newest first
    runs.sort(key=lambda x: x[3], reverse=True)
    
    return runs


def select_run_interactive(runs: List[Tuple[str, Path, int, datetime]]) -> Path:
    """
    Display interactive menu to select a run using arrow keys.
    
    Args:
        runs: List of (run_name, run_path, num_tasks, timestamp)
        
    Returns:
        Selected run path
    """
    try:
        import curses
        
        def menu(stdscr, runs):
            curses.curs_set(0)  # Hide cursor
            current_idx = 0
            
            while True:
                stdscr.clear()
                h, w = stdscr.getmaxyx()
                
                # Title
                title = "Select a run to analyze (↑/↓ to navigate, Enter to select, q to quit):"
                stdscr.addstr(0, 0, title, curses.A_BOLD)
                stdscr.addstr(1, 0, "=" * min(len(title), w - 1))
                
                # Display runs
                start_row = 3
                for idx, (run_name, run_path, num_tasks, timestamp) in enumerate(runs):
                    if start_row + idx >= h - 1:
                        break
                    
                    # Format the line
                    time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    line = f"{idx + 1}. {run_name} ({num_tasks} tasks) - {time_str}"
                    
                    # Truncate if too long
                    if len(line) > w - 2:
                        line = line[:w - 5] + "..."
                    
                    # Highlight selected row
                    if idx == current_idx:
                        stdscr.addstr(start_row + idx, 0, "> " + line, curses.A_REVERSE)
                    else:
                        stdscr.addstr(start_row + idx, 0, "  " + line)
                
                stdscr.refresh()
                
                # Get input
                key = stdscr.getch()
                
                if key == curses.KEY_UP and current_idx > 0:
                    current_idx -= 1
                elif key == curses.KEY_DOWN and current_idx < len(runs) - 1:
                    current_idx += 1
                elif key == ord('\n') or key == curses.KEY_ENTER or key == 10:
                    return runs[current_idx][1]
                elif key == ord('q') or key == ord('Q'):
                    return None
        
        selected_path = curses.wrapper(menu, runs)
        return selected_path
        
    except ImportError:
        # Fallback to simple numbered selection if curses not available
        print("\nAvailable runs (sorted by timestamp, newest first):")
        print("=" * 80)
        for idx, (run_name, run_path, num_tasks, timestamp) in enumerate(runs):
            time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            print(f"{idx + 1}. {run_name} ({num_tasks} tasks) - {time_str}")
        
        print("\nEnter the number of the run to analyze (or 'q' to quit): ", end='')
        choice = input().strip()
        
        if choice.lower() == 'q':
            return None
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(runs):
                return runs[idx][1]
            else:
                print("Invalid selection.")
                return None
        except ValueError:
            print("Invalid input.")
            return None


def find_valid_tasks(base_path: str) -> List[Tuple[str, Path, Path]]:
    """
    Find all tasks that contain both [task_id].json and latest_state.json.
    
    Args:
        base_path: Path to the folder containing task directories
        
    Returns:
        List of tuples (task_id, task_json_path, state_json_path)
    """
    base_dir = Path(base_path)
    if not base_dir.exists():
        print(f"Error: Path {base_path} does not exist")
        return []
    
    valid_tasks = []
    
    # Iterate through subdirectories (task folders)
    for task_dir in base_dir.iterdir():
        if not task_dir.is_dir():
            continue
            
        task_id = task_dir.name
        task_json = task_dir / f"{task_id}.json"
        state_json = task_dir / "latest_state.json"
        
        # Check if both files exist
        if task_json.exists() and state_json.exists():
            valid_tasks.append((task_id, task_json, state_json))
    
    return valid_tasks


def extract_task_data(task_json_path: Path, state_json_path: Path) -> Dict:
    """
    Extract relevant data from task files.
    
    Args:
        task_json_path: Path to [task_id].json
        state_json_path: Path to latest_state.json
        
    Returns:
        Dictionary containing extracted data
    """
    data = {
        'task_id': task_json_path.stem,
        'highest_testing_solution_priority_score': None,
        'highest_training_solution_priority_score': None,
        'num_generations': None,
        'solved': False,
        'solved_at_loop': None,
        'overlap_scores_by_generation': []  # List of average_testing_overlap_score per generation
    }
    
    try:
        # Read task_id.json
        with open(task_json_path, 'r') as f:
            task_result = json.load(f)
            data['highest_testing_solution_priority_score'] = task_result.get('highest_testing_solution_priority_score')
            data['highest_training_solution_priority_score'] = task_result.get('highest_training_solution_priority_score')
        
        # Read latest_state.json to get generations
        with open(state_json_path, 'r') as f:
            state = json.load(f)
            generations = state.get('generations', [])
            data['num_generations'] = len(generations)
            
            # Extract average_training_success_rate from each generation
            for generation in generations:
                success_rate = generation.get('average_training_success_rate', 0.0)
                data['overlap_scores_by_generation'].append(success_rate)
        
        # Determine if task is solved (priority score > 0)
        if data['highest_testing_solution_priority_score'] is not None and \
           data['highest_testing_solution_priority_score'] > 0:
            data['solved'] = True
            
            # The generations list contains PAST generations that have completed.
            # If there are N generations, that means loops 0 through N-1 have completed.
            # The task was solved AFTER these loops, so it was solved at loop N.
            data['solved_at_loop'] = data['num_generations']
    
    except Exception as e:
        print(f"Error processing {task_json_path.stem}: {e}")
    
    return data


def calculate_cumulative_solves(tasks_data: List[Dict], max_loops: int = 6) -> Dict[int, int]:
    """
    Calculate cumulative number of tasks solved at each loop.
    If a task is solved at loop n, it's considered solved at all subsequent loops.
    
    Args:
        tasks_data: List of task data dictionaries
        max_loops: Maximum number of loops to track (0-5 = 6 loops)
        
    Returns:
        Dictionary mapping loop number to cumulative count of solved tasks
    """
    cumulative_solves = {i: 0 for i in range(max_loops)}
    
    for task in tasks_data:
        if task['solved'] and task['solved_at_loop'] is not None:
            solved_at = task['solved_at_loop']
            # Mark as solved for this loop and all subsequent loops
            for loop in range(solved_at, max_loops):
                cumulative_solves[loop] += 1
    
    return cumulative_solves


def calculate_average_overlap_scores(tasks_data: List[Dict], max_loops: int = 6) -> Dict[int, float]:
    """
    Calculate average training success rate at each loop across all tasks.
    
    For each loop:
    - If a task has a generation for that loop, use that generation's average_training_success_rate
    - If a task has fewer generations (i.e., solved earlier), use 100.0 for subsequent loops
    
    Args:
        tasks_data: List of task data dictionaries
        max_loops: Maximum number of loops to track (0-5 = 6 loops)
        
    Returns:
        Dictionary mapping loop number to average overlap score across all tasks
    """
    average_scores = {i: 0.0 for i in range(max_loops)}
    
    for loop in range(max_loops):
        scores_at_loop = []
        
        for task in tasks_data:
            num_gens = task.get('num_generations', 0)
            overlap_scores = task.get('overlap_scores_by_generation', [])
            
            if loop < num_gens:
                # This task has data for this loop (generation)
                if loop < len(overlap_scores):
                    scores_at_loop.append(overlap_scores[loop])
                else:
                    scores_at_loop.append(0.0)
            else:
                # Task was solved before this loop, so score is 100.0
                if task.get('solved', False):
                    scores_at_loop.append(100.0)
                else:
                    # Task not solved and no more generations, use last known score or 0
                    if overlap_scores:
                        scores_at_loop.append(overlap_scores[-1])
                    else:
                        scores_at_loop.append(0.0)
        
        # Calculate average for this loop
        if scores_at_loop:
            average_scores[loop] = sum(scores_at_loop) / len(scores_at_loop)
    
    return average_scores


def plot_overlap_scores(average_scores: Dict[int, float], total_tasks: int, output_path: str):
    """
    Create and save a line chart showing average training success rates at each loop.
    
    Args:
        average_scores: Dictionary of loop -> average training success rate
        total_tasks: Total number of tasks analyzed
        output_path: Path to save the plot
    """
    loops = sorted(average_scores.keys())
    scores = [average_scores[loop] for loop in loops]
    
    plt.figure(figsize=(12, 7))
    plt.plot(loops, scores, marker='o', linewidth=2, markersize=8, color='steelblue')
    
    # Add value labels on each point
    for loop, score in zip(loops, scores):
        plt.text(loop, score + 1.5, f'{score:.1f}%', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.xlabel('Loop / Generation', fontsize=12, fontweight='bold')
    plt.ylabel('Average Training Success Rate (%)', fontsize=12, fontweight='bold')
    plt.title(f'Average Training Success Rate by Loop (Total Tasks: {total_tasks})', 
             fontsize=14, fontweight='bold')
    plt.xticks(loops)
    plt.ylim(0, 105)  # Set y-axis from 0 to 105 to accommodate 100% and labels
    plt.grid(axis='both', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training success rate plot saved to: {output_path}")
    
    # Show the plot
    plt.show()


def calculate_cumulative_solves(tasks_data: List[Dict], max_loops: int = 6) -> Dict[int, int]:
    """
    Calculate cumulative number of tasks solved at each loop.
    If a task is solved at loop n, it's considered solved at all subsequent loops.
    
    Args:
        tasks_data: List of task data dictionaries
        max_loops: Maximum number of loops to track (0-5 = 6 loops)
        
    Returns:
        Dictionary mapping loop number to cumulative count of solved tasks
    """
    cumulative_solves = {i: 0 for i in range(max_loops)}
    
    for task in tasks_data:
        if task['solved'] and task['solved_at_loop'] is not None:
            solved_at = task['solved_at_loop']
            # Mark as solved for this loop and all subsequent loops
            for loop in range(solved_at, max_loops):
                cumulative_solves[loop] += 1
    
    return cumulative_solves


def plot_performance(cumulative_solves: Dict[int, int], total_tasks: int, output_path: str):
    """
    Create and save a bar chart showing tasks solved at each loop.
    
    Args:
        cumulative_solves: Dictionary of loop -> cumulative solve count
        total_tasks: Total number of tasks analyzed
        output_path: Path to save the plot
    """
    loops = sorted(cumulative_solves.keys())
    counts = [cumulative_solves[loop] for loop in loops]
    
    plt.figure(figsize=(12, 7))
    bars = plt.bar(loops, counts, color='steelblue', edgecolor='black', alpha=0.7)
    
    # Add value labels on top of bars
    for i, (loop, count) in enumerate(zip(loops, counts)):
        percentage = (count / total_tasks * 100) if total_tasks > 0 else 0
        plt.text(loop, count + 0.5, f'{count}\n({percentage:.1f}%)', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.xlabel('Loop / Generation', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Tasks Solved (Cumulative)', fontsize=12, fontweight='bold')
    plt.title(f'Tasks Solved by Generation/Loop (Total Tasks: {total_tasks})', 
             fontsize=14, fontweight='bold')
    plt.xticks(loops)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Also show the plot
    plt.show()


def main():
    """Main function to run the analysis."""
    print("ARC Performance by Generations Analyzer")
    print("=" * 80)
    
    # Get available runs
    runs = get_available_runs()
    
    if not runs:
        print("No runs found in output/output_agent/")
        sys.exit(1)
    
    print(f"\nFound {len(runs)} run(s)")
    
    # Let user select a run
    selected_path = select_run_interactive(runs)
    
    if selected_path is None:
        print("\nAnalysis cancelled.")
        sys.exit(0)
    
    folder_path = str(selected_path)
    
    print(f"\n{'=' * 80}")
    print(f"Analyzing tasks in: {folder_path}")
    print("=" * 80)
    
    # Find valid tasks
    valid_tasks = find_valid_tasks(folder_path)
    print(f"\nFound {len(valid_tasks)} valid tasks with both [task_id].json and latest_state.json")
    
    if not valid_tasks:
        print("No valid tasks found. Exiting.")
        sys.exit(1)
    
    # Extract data from each task
    print("\nProcessing tasks...")
    tasks_data = []
    for task_id, task_json, state_json in valid_tasks:
        data = extract_task_data(task_json, state_json)
        tasks_data.append(data)
    
    # Count solved tasks
    solved_tasks = [t for t in tasks_data if t['solved']]
    print(f"\nTotal tasks solved: {len(solved_tasks)} / {len(tasks_data)}")
    
    # Calculate cumulative solves
    cumulative_solves = calculate_cumulative_solves(tasks_data)
    
    # Calculate average overlap scores
    average_overlap_scores = calculate_average_overlap_scores(tasks_data)
    
    # Print results
    print("\n" + "=" * 80)
    print("CUMULATIVE TASKS SOLVED BY LOOP:")
    print("=" * 80)
    for loop in sorted(cumulative_solves.keys()):
        count = cumulative_solves[loop]
        percentage = (count / len(tasks_data) * 100) if len(tasks_data) > 0 else 0
        print(f"Loop {loop}: {count:3d} tasks ({percentage:5.1f}%)")
    
    # Show breakdown of when tasks were first solved
    print("\n" + "=" * 80)
    print("TASKS FIRST SOLVED AT EACH LOOP:")
    print("=" * 80)
    first_solved_at = {}
    for task in tasks_data:
        if task['solved'] and task['solved_at_loop'] is not None:
            loop = task['solved_at_loop']
            first_solved_at[loop] = first_solved_at.get(loop, 0) + 1
    
    for loop in sorted(first_solved_at.keys()):
        count = first_solved_at[loop]
        print(f"Loop {loop}: {count:3d} new tasks solved")
    
    # Show average training success rates
    print("\n" + "=" * 80)
    print("AVERAGE TRAINING SUCCESS RATE BY LOOP:")
    print("=" * 80)
    for loop in sorted(average_overlap_scores.keys()):
        score = average_overlap_scores[loop]
        print(f"Loop {loop}: {score:5.1f}%")
    
    # Count tasks where highest_training is 100% but highest_testing is not
    perfect_training_imperfect_testing = []
    for task in tasks_data:
        training_score = task.get('highest_training_solution_priority_score')
        testing_score = task.get('highest_testing_solution_priority_score')
        
        # Check if training is 100% (priority score of 1.0) but testing is not
        if training_score is not None and training_score >= 1.0:
            if testing_score is None or testing_score < 1.0:
                perfect_training_imperfect_testing.append(task['task_id'])
    
    print("\n" + "=" * 80)
    print("TASKS WITH 100% TRAINING BUT NOT 100% TESTING:")
    print("=" * 80)
    print(f"Count: {len(perfect_training_imperfect_testing)} / {len(tasks_data)}")
    if perfect_training_imperfect_testing:
        print("\nTask IDs:")
        for task_id in sorted(perfect_training_imperfect_testing):
            print(f"  - {task_id}")
    
    # Create output paths for plots
    output_dir = Path(folder_path)
    plot_path_solves = output_dir / "performance_by_generations.png"
    plot_path_overlap = output_dir / "training_success_rate_by_generations.png"
    
    # Generate plots
    print("\n" + "=" * 80)
    print("Generating plots...")
    plot_performance(cumulative_solves, len(tasks_data), str(plot_path_solves))
    plot_overlap_scores(average_overlap_scores, len(tasks_data), str(plot_path_overlap))
    
    print("\n" + "=" * 80)
    print("Analysis complete!")


if __name__ == "__main__":
    main()
