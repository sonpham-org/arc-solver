#!/usr/bin/env python3
"""
Interactive and CLI tool to compute average agent success rate for an output run.

Usage:
  - Interactive selection (curses): `python calculate_agent_results.py`
  - Non-interactive: `python calculate_agent_results.py --run output/2025-...`

It scans the chosen run folder for task JSON files and counts how many
have `testing_success_rate == 1.0`, then divides by total number of task files.
Skips `params.json` and `summary.json` when scanning.

This updated version also reads `llm_testing_success_rate` (if present) and
reports both the fraction of full LLM successes and the average LLM success
probability across tasks.
"""

import argparse
import curses
import json
import os
import sys
from typing import List, Tuple


def list_runs(output_dir: str) -> List[str]:
    if not os.path.isdir(output_dir):
        return []
    entries = [os.path.join(output_dir, p) for p in os.listdir(output_dir)]
    dirs = [p for p in entries if os.path.isdir(p)]
    # sort newest first if folders are timestamped
    dirs.sort(reverse=True)
    return dirs


def compute_run_stats(run_path: str) -> Tuple[int, int, float, int, float, List[Tuple[str, float, float]]]:
    """Return (success_count, total_count, success_rate, llm_success_count, llm_success_rate, list_of_task_rates).

    Only considers .json files in the run folder, excluding `params.json` and `summary.json`.
    For each task file, reads `testing_success_rate` and `llm_testing_success_rate` fields (expects numeric). Missing -> 0.0.
    The `rates` list contains tuples: (filename, testing_success_rate, llm_testing_success_rate).
    """
    files = [f for f in os.listdir(run_path) if f.endswith('.json')]
    files = [f for f in files if f not in ('params.json', 'summary.json')]

    total = 0
    success = 0
    llm_success = 0
    rates: List[Tuple[str, float, float]] = []

    for fname in sorted(files):
        fpath = os.path.join(run_path, fname)
        try:
            with open(fpath, 'r') as fh:
                data = json.load(fh)
        except Exception:
            # skip unreadable or non-json
            continue

        # Read standard testing_success_rate
        rate = data.get('testing_success_rate') if isinstance(data, dict) else None
        if rate is None:
            rate_val = 0.0
        else:
            try:
                rate_val = float(rate)
            except Exception:
                rate_val = 0.0

        # Read LLM testing success rate (may be absent)
        llm_rate = data.get('llm_testing_success_rate') if isinstance(data, dict) else None
        if llm_rate is None:
            llm_rate_val = 0.0
        else:
            try:
                llm_rate_val = float(llm_rate)
            except Exception:
                llm_rate_val = 0.0

        total += 1
        if rate_val == 1.0:
            success += 1
        if llm_rate_val == 1.0:
            llm_success += 1

        rates.append((fname, rate_val, llm_rate_val))

    success_rate = (success / total) if total > 0 else 0.0
    llm_success_rate = (llm_success / total) if total > 0 else 0.0
    return success, total, success_rate, llm_success, llm_success_rate, rates


def draw_menu(stdscr, items: List[tuple], selected_idx: int) -> None:
    stdscr.clear()
    h, w = stdscr.getmaxyx()
    title = 'Select a run folder and press Enter (q to quit)'
    stdscr.addstr(0, 0, title)
    for i, (it, count) in enumerate(items):
        # show just the basename for readability, append (num tasks)
        base = os.path.basename(it)
        display = f"{base} ({count} tasks)"
        y = i + 2
        if y >= h - 1:
            break
        if i == selected_idx:
            stdscr.attron(curses.A_REVERSE)
            stdscr.addstr(y, 0, display[: w - 1])
            stdscr.attroff(curses.A_REVERSE)
        else:
            stdscr.addstr(y, 0, display[: w - 1])
    stdscr.refresh()


def display_result(stdscr, run_path: str, success: int, total: int, rate: float, llm_success: int, llm_rate: float, rates: List[Tuple[str, float, float]]) -> None:
    stdscr.clear()
    h, w = stdscr.getmaxyx()
    stdscr.addstr(0, 0, f'Results for: {run_path}')
    stdscr.addstr(1, 0, f'Tasks counted: {total}')
    stdscr.addstr(2, 0, f'Full successes (testing_success_rate == 1.0): {success}')
    stdscr.addstr(3, 0, f'Success fraction: {rate:.4f}  ({rate*100:.2f}%)')
    stdscr.addstr(4, 0, f'LLM full successes (llm_testing_success_rate == 1.0): {llm_success}')
    stdscr.addstr(5, 0, f'LLM success fraction: {llm_rate:.4f}  ({llm_rate*100:.2f}%)')

    stdscr.addstr(7, 0, 'Per-task rates (filename : testing_rate , llm_testing_rate). Press any key to return.')
    line = 8
    for fname, r, lr in rates:
        if line >= h - 1:
            break
        stdscr.addstr(line, 0, f'{fname} : {r} , {lr}')
        line += 1

    stdscr.refresh()
    stdscr.getch()


def curses_selector(output_dir: str) -> int:
    raw_runs = list_runs(output_dir)
    if not raw_runs:
        print(f'No runs found in `{output_dir}`')
        return 1

    # Precompute task counts for each run to display in the menu
    runs_with_counts = []
    for r in raw_runs:
        try:
            files = [f for f in os.listdir(r) if f.endswith('.json') and f not in ('params.json', 'summary.json')]
            count = len(files)
        except Exception:
            count = 0
        runs_with_counts.append((r, count))

    # Sort runs by number of task files (descending) for easier selection of large runs
    runs_with_counts.sort(key=lambda x: x[1], reverse=True)

    def _main(stdscr):
        curses.curs_set(0)
        idx = 0
        while True:
            draw_menu(stdscr, runs_with_counts, idx)
            key = stdscr.getch()
            if key in (curses.KEY_UP, ord('k')):
                idx = max(0, idx - 1)
            elif key in (curses.KEY_DOWN, ord('j')):
                idx = min(len(runs_with_counts) - 1, idx + 1)
            elif key in (ord('\n'), curses.KEY_ENTER, 10, 13):
                # compute stats and show
                run_path = runs_with_counts[idx][0]
                success, total, rate, llm_success, llm_rate, rates = compute_run_stats(run_path)
                display_result(stdscr, run_path, success, total, rate, llm_success, llm_rate, rates)
            elif key in (ord('q'), 27):
                break

    curses.wrapper(_main)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Compute average testing success rate for agent runs')
    parser.add_argument('--output-dir', '-o', default=os.path.join(os.getcwd(), 'output'),
                        help='Path to the top-level output folder (default: ./output)')
    parser.add_argument('--run', '-r', default=None,
                        help='Path to a specific run folder (non-interactive). If given, prints results and exits.')
    args = parser.parse_args(argv)

    if args.run:
        run_path = args.run
        if not os.path.isabs(run_path):
            run_path = os.path.join(args.output_dir, run_path) if not os.path.exists(run_path) else run_path
        if not os.path.isdir(run_path):
            print(f'Run folder not found: {run_path}')
            return 2

        success, total, rate, llm_success, llm_rate, rates = compute_run_stats(run_path)
        print(f'Run: {run_path}')
        print(f'Tasks counted: {total}')
        print(f'Full successes (testing_success_rate == 1.0): {success}')
        print(f'Success fraction: {rate:.4f}  ({rate*100:.2f}%)')
        print(f'LLM full successes (llm_testing_success_rate == 1.0): {llm_success}')
        print(f'LLM success fraction: {llm_rate:.4f}  ({llm_rate*100:.2f}%)')
        # Also print average LLM success probability across tasks
        if total > 0:
            avg_llm_prob = sum(lr for _, _, lr in rates) / total
        else:
            avg_llm_prob = 0.0
        print(f'Average LLM success probability across tasks: {avg_llm_prob:.4f} ({avg_llm_prob*100:.2f}%)')
        return 0

    # Interactive selector
    return curses_selector(args.output_dir)


if __name__ == '__main__':
    sys.exit(main())
