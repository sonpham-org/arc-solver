#!/usr/bin/env python3
"""
Flask-based ARC Visualizer

Run with:
  pip install flask
  python3 arc_visualizer.py

Opens a local web server that lists runs under `output/`, shows run summary
and allows inspecting tasks and solutions with grid/difference rendering.
"""
from pathlib import Path
import json
from typing import Any, Dict, List, Optional
from flask import Flask, render_template, request, url_for, redirect

APP_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = APP_ROOT / "output"


def find_runs(output_root: Path) -> List[Path]:
    if not output_root.exists():
        return []
    return sorted([p for p in output_root.iterdir() if p.is_dir()])


def load_run(run_path: Path) -> Dict[str, Any]:
    res = {'path': run_path, 'params': {}, 'training_ids': [], 'evaluation_ids': [], 'tasks': {}}
    p_params = run_path / 'params.json'
    if p_params.exists():
        try:
            res['params'] = json.loads(p_params.read_text())
        except Exception:
            res['params'] = {}
    p_train = run_path / 'training_task_ids.txt'
    if p_train.exists():
        try:
            res['training_ids'] = [l.strip() for l in p_train.read_text().splitlines() if l.strip()]
        except Exception:
            res['training_ids'] = []
    p_eval = run_path / 'evaluation_task_ids.txt'
    if p_eval.exists():
        try:
            res['evaluation_ids'] = [l.strip() for l in p_eval.read_text().splitlines() if l.strip()]
        except Exception:
            res['evaluation_ids'] = []

    # New layout support: each task is stored in its own folder named by task_id
    # containing `task_id.json`. Prefer loading from these folders first.
    for p in sorted(run_path.iterdir()):
        if p.is_dir():
            candidate = p / f"{p.name}.json"
            if candidate.exists():
                try:
                    data = json.loads(candidate.read_text())
                    res['tasks'][p.name] = data
                except Exception:
                    # ignore corrupt/invalid files and continue
                    continue

    return res


def compute_run_accuracy(tasks: Dict[str, Any]) -> float:
    if not tasks:
        return 0.0
    vals = []
    for _, data in tasks.items():
        v = data.get('highest_testing_solution_priority_score')
        try:
            vals.append(float(v) if v is not None else 0.0)
        except Exception:
            vals.append(0.0)
    return sum(vals) / len(vals)


def build_diff_mask(expected: Optional[List[List[Any]]], predicted: Optional[List[List[Any]]]) -> Optional[List[List[bool]]]:
    if expected is None or predicted is None:
        return None
    rows = max(len(expected), len(predicted))
    cols = max(len(expected[0]) if expected else 0, len(predicted[0]) if predicted else 0)
    mask = [[False for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            e = expected[r][c] if r < len(expected) and c < len(expected[0]) else None
            p = predicted[r][c] if r < len(predicted) and c < len(predicted[0]) else None
            mask[r][c] = (e != p)
    return mask


def build_diff_status(expected: Optional[List[List[Any]]], predicted: Optional[List[List[Any]]]) -> Optional[List[List[str]]]:
    """Return a 2D status grid with values:
       'same' - expected==predicted
       'diff' - both exist but different
       'only_expected' - only expected has a cell at that position
       'only_predicted' - only predicted has a cell at that position
       'none' - neither grid has a cell at that position
    """
    exp = normalize_grid(expected) or []
    pred = normalize_grid(predicted) or []
    rows = max(len(exp), len(pred))
    cols = 0
    for r in range(rows):
        if r < len(exp):
            cols = max(cols, len(exp[r]))
        if r < len(pred):
            cols = max(cols, len(pred[r]))

    status = [["none" for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            has_e = (r < len(exp) and c < len(exp[r]))
            has_p = (r < len(pred) and c < len(pred[r]))
            if not has_e and not has_p:
                status[r][c] = 'none'
            elif has_e and not has_p:
                status[r][c] = 'only_expected'
            elif has_p and not has_e:
                status[r][c] = 'only_predicted'
            else:
                status[r][c] = 'same' if exp[r][c] == pred[r][c] else 'diff'
    return status


def normalize_grid(grid: Optional[List[Any]]) -> Optional[List[List[int]]]:
    """Convert various grid encodings into a canonical 2D list of ints.

    Supported input forms:
    - None -> None
    - list of strings (each string a row of digit characters) -> [[int,...], ...]
    - list of lists (cells may be ints or digit-strings) -> [[int,...], ...]
    """
    if grid is None:
        return None
    if not isinstance(grid, list):
        return None
    out: List[List[int]] = []
    for row in grid:
        if isinstance(row, str):
            # string row like "330900" -> each char
            new_row: List[int] = []
            for ch in row:
                try:
                    new_row.append(int(ch))
                except Exception:
                    new_row.append(0)
            out.append(new_row)
        elif isinstance(row, list):
            new_row = []
            for cell in row:
                if isinstance(cell, int):
                    new_row.append(cell)
                elif isinstance(cell, str):
                    try:
                        new_row.append(int(cell))
                    except Exception:
                        # maybe multi-char like '12' or non-digit; try best-effort
                        digits = ''.join([c for c in cell if c.isdigit()])
                        try:
                            new_row.append(int(digits) if digits else 0)
                        except Exception:
                            new_row.append(0)
                else:
                    try:
                        new_row.append(int(cell))
                    except Exception:
                        new_row.append(0)
            out.append(new_row)
        else:
            # unknown row type, skip or produce empty
            out.append([])
    return out


def prepare_task_for_view(task: Dict[str, Any]) -> Dict[str, Any]:
    # For each solution, compute diff masks for its examples and attach them
    sols = task.get('solutions_list') or []
    for sol in sols:
        for key in ('training_results', 'testing_results'):
            arr = sol.get(key) or []
            for ex in arr:
                # Normalize common grid fields so templates receive 2D int lists
                inp = ex.get('input') or ex.get('input_grid') or ex.get('input_raw')
                expected = ex.get('expected_output') or ex.get('output')
                predicted = ex.get('predicted_output') or ex.get('predicted') or ex.get('llm_predicted_output')
                ex['input'] = normalize_grid(inp)
                ex['expected_output'] = normalize_grid(expected)
                # store predicted output in a consistent key the templates expect
                ex['predicted_output'] = normalize_grid(predicted)

                ex['_diff_mask'] = build_diff_mask(ex['expected_output'], ex['predicted_output'])
                ex['_diff_status'] = build_diff_status(ex['expected_output'], ex['predicted_output'])
        # Normalize training/testing success rates to percentages:
        try:
            t = sol.get('training_success_rate')
            if t is None:
                sol['_training_pct'] = 0.0
            else:
                sol['_training_pct'] = (float(t) * 100.0) if float(t) <= 1.0 else float(t)
        except Exception:
            sol['_training_pct'] = 0.0
        try:
            tt = sol.get('testing_success_rate')
            if tt is None:
                sol['_testing_pct'] = 0.0
            else:
                sol['_testing_pct'] = (float(tt) * 100.0) if float(tt) <= 1.0 else float(tt)
        except Exception:
            sol['_testing_pct'] = 0.0
    return task


app = Flask(__name__)

# Color map for numeric cell values -> hex colors. Expose to templates via context processor.
COLOR_MAP = {
    0: '#ffffff',
    1: '#e6194b',
    2: '#3cb44b',
    3: '#ffe119',
    4: '#0082c8',
    5: '#f58231',
    6: '#911eb4',
    7: '#46f0f0',
    8: '#f032e6',
    9: '#d2f53c'
}


@app.context_processor
def inject_color_map():
    return dict(color_map=COLOR_MAP)


@app.route('/')
def index():
    runs = find_runs(OUTPUT_ROOT)
    run_name = request.args.get('run')
    run_info = None
    accuracy = None
    total_tasks = 0
    success_count = 0
    failure_count = 0
    if run_name:
        run_path = OUTPUT_ROOT / run_name
        if run_path.exists():
            run_info = load_run(run_path)
            accuracy = compute_run_accuracy(run_info['tasks'])
            # If a specific task is requested, prepare it for view (compute masks and percentages)
            task_id = request.args.get('task')
            if task_id and task_id in run_info['tasks']:
                prepare_task_for_view(run_info['tasks'][task_id])
            # compute task counts (use same threshold as template: score >= 1.0)
            try:
                # Determine per-task testing success percentage (max across solutions)
                total_tasks = len(run_info['tasks'])
                sc = 0
                for _, data in run_info['tasks'].items():
                    max_testing_pct = 0.0
                    max_training_pct = 0.0
                    overlaps: List[float] = []
                    sols = data.get('solutions_list') or []
                    for sol in sols:
                        # --- testing success ---
                        tt = None
                        if sol is not None:
                            tt = sol.get('testing_success_rate') or sol.get('testing_success')
                        try:
                            f = float(tt) if tt is not None else 0.0
                        except Exception:
                            f = 0.0
                        pct = (f * 100.0) if f <= 1.0 else f
                        if pct > max_testing_pct:
                            max_testing_pct = pct

                        # --- training success ---
                        tr = None
                        if sol is not None:
                            tr = sol.get('training_success_rate') or sol.get('training_success')
                        try:
                            tf = float(tr) if tr is not None else 0.0
                        except Exception:
                            tf = 0.0
                        tpct = (tf * 100.0) if tf <= 1.0 else tf
                        if tpct > max_training_pct:
                            max_training_pct = tpct

                        # --- gather overlap_percentage from example results ---
                        for key in ('training_results', 'testing_results'):
                            arr = sol.get(key) if sol is not None else []
                            if not arr:
                                continue
                            for ex in arr:
                                if not ex:
                                    continue
                                ov = ex.get('overlap_percentage') if isinstance(ex, dict) else None
                                if ov is None:
                                    ov = ex.get('overlap') if isinstance(ex, dict) else None
                                if ov is None:
                                    continue
                                try:
                                    ovf = float(ov)
                                except Exception:
                                    ovf = 0.0
                                ovpct = (ovf * 100.0) if ovf <= 1.0 else ovf
                                overlaps.append(ovpct)

                    # expose computed metrics back into the task dict for template consumption
                    data['_testing_pct_max'] = max_testing_pct
                    data['_training_pct_max'] = max_training_pct
                    if overlaps:
                        data['_avg_overlap_pct'] = sum(overlaps) / len(overlaps)
                    else:
                        data['_avg_overlap_pct'] = 0.0
                    # count as success only when testing reaches 100%
                    if max_testing_pct >= 100.0 - 1e-6:
                        sc += 1
                success_count = sc
                failure_count = total_tasks - success_count
            except Exception:
                total_tasks = 0
                success_count = 0
                failure_count = 0
    return render_template('index.html', runs=runs, selected_run=run_name, run_info=run_info, accuracy=accuracy,
                           total_tasks=total_tasks, success_count=success_count, failure_count=failure_count)


@app.route('/task')
def task_view():
    run_name = request.args.get('run')
    task_id = request.args.get('task')
    if not run_name or not task_id:
        return redirect(url_for('index'))
    run_path = OUTPUT_ROOT / run_name
    if not run_path.exists():
        return redirect(url_for('index'))
    run_info = load_run(run_path)
    task = run_info['tasks'].get(task_id)
    if not task:
        return redirect(url_for('index', run=run_name))
    task = prepare_task_for_view(task)
    return render_template('task.html', run_name=run_name, task_id=task_id, task=task)


if __name__ == '__main__':
    # For remote access, we bind to 0.0.0.0, but don't enable debug in production
    app.run(host='0.0.0.0', port=8501, debug=False)
#!/usr/bin/env python3
"""
ARC Visualizer

Streamlit app to inspect ARC LangGraph multi-solution run outputs.

Usage:
  pip install streamlit
  streamlit run arc_visualizer.py

Behavior summary:
- Lists runs under the `output/` folder (timestamped run folders)
- Loads `params.json`, `training_task_ids.txt` and `evaluation_task_ids.txt`
- Loads per-task JSON files (WorkflowOutput schema) and computes an accuracy
  metric (average of `highest_testing_solution_priority_score` across tasks)
- Left column: run selector, Show run button, run summary and clickable task list
- Right column: tabs for solutions (Highest solution + numbered solutions)
  and per-example cards showing input, expected, predicted and a difference map
"""

from pathlib import Path
import json
import os
from typing import Any, Dict, List, Optional, Tuple


