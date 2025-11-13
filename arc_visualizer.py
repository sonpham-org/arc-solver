#!/usr/bin/env python3
"""
ARC Solver Results Visualizer

A Gradio web application to visualize ARC (Abstraction and Reasoning Corpus) solver results.
Allows users to select different runs and view task results with input/predicted/actual output grids.
"""

import os
import json
from typing import Dict, List, Tuple, Optional

try:
    import gradio as gr
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install required packages:")
    print("pip install gradio matplotlib numpy")
    exit(1)


# ARC color scheme (0-9 color codes)
ARC_COLORS = [
    '#000000',  # 0: Black
    '#0074D9',  # 1: Blue
    '#FF4136',  # 2: Red
    '#2ECC40',  # 3: Green
    '#FFDC00',  # 4: Yellow
    '#AAAAAA',  # 5: Grey
    '#F012BE',  # 6: Magenta
    '#FF851B',  # 7: Orange
    '#7FDBFF',  # 8: Sky blue (background)
    '#870C25',  # 9: Dark red/brown
]

# Create custom colormap
arc_cmap = ListedColormap(ARC_COLORS)


class ARCVisualizer:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        self.runs_cache = {}
        self.current_run_data = {}
        
    def get_available_runs(self) -> List[str]:
        """Get list of available runs from the output directory."""
        if not os.path.exists(self.output_dir):
            return []
        
        runs = []
        for item in os.listdir(self.output_dir):
            run_path = os.path.join(self.output_dir, item)
            if os.path.isdir(run_path):
                # Check if it has the required files
                params_file = os.path.join(run_path, 'params.json')
                summary_file = os.path.join(run_path, 'summary.json')
                if os.path.exists(params_file) and os.path.exists(summary_file):
                    runs.append(item)
        
        # Sort runs by timestamp (newest first)
        runs.sort(reverse=True)
        return runs
    
    def get_available_runs_with_counts(self) -> List[Tuple[str, str]]:
        """Get list of available runs with task counts for display."""
        if not os.path.exists(self.output_dir):
            return []
        
        runs_with_counts = []
        for item in os.listdir(self.output_dir):
            run_path = os.path.join(self.output_dir, item)
            if os.path.isdir(run_path):
                # Check if it has the required files
                params_file = os.path.join(run_path, 'params.json')
                summary_file = os.path.join(run_path, 'summary.json')
                if os.path.exists(params_file) and os.path.exists(summary_file):
                    # Count tasks in this run
                    task_count = 0
                    try:
                        for file in os.listdir(run_path):
                            if file.endswith('.json') and file not in ['params.json', 'summary.json']:
                                # Quick check if it's a task file
                                try:
                                    with open(os.path.join(run_path, file), 'r') as f:
                                        data = json.load(f)
                                        if isinstance(data, dict) and ('tests' in data or 'trains' in data):
                                            task_count += 1
                                except:
                                    continue
                        
                        # Create display name with task count
                        display_name = f"{item} ({task_count} tasks)"
                        runs_with_counts.append((item, display_name))
                    except Exception as e:
                        # Fallback to original name if counting fails
                        runs_with_counts.append((item, item))
        
        # Sort by original name (newest first)
        runs_with_counts.sort(key=lambda x: x[0], reverse=True)
        return runs_with_counts
    
    def load_run_data(self, run_name: str) -> Dict:
        """Load all data for a specific run."""
        if run_name in self.runs_cache:
            return self.runs_cache[run_name]
        
        run_path = os.path.join(self.output_dir, run_name)
        
        try:
            # Load parameters
            with open(os.path.join(run_path, 'params.json'), 'r') as f:
                params = json.load(f)
            
            # Load summary
            with open(os.path.join(run_path, 'summary.json'), 'r') as f:
                summary = json.load(f)
            
            # Load task data
            tasks = {}
            for file in os.listdir(run_path):
                if file.endswith('.json') and file not in ['params.json', 'summary.json']:
                    # Skip any text files or other non-task JSON files
                    if file.endswith('_task_ids.json') or file.endswith('.txt'):
                        continue
                        
                    task_id = file[:-5]  # Remove .json extension
                    try:
                        with open(os.path.join(run_path, file), 'r') as f:
                            task_data = json.load(f)
                            # Verify this is actually a task file by checking for expected structure
                            if isinstance(task_data, dict) and ('tests' in task_data or 'trains' in task_data):
                                tasks[task_id] = task_data
                    except (json.JSONDecodeError, FileNotFoundError) as e:
                        print(f"Warning: Could not load task {task_id}: {e}")
                        continue
            
            run_data = {
                'params': params,
                'summary': summary,
                'tasks': tasks,
                'run_name': run_name
            }
            
            self.runs_cache[run_name] = run_data
            return run_data
            
        except Exception as e:
            print(f"Error loading run data for {run_name}: {e}")
            return {
                'params': {},
                'summary': {},
                'tasks': {},
                'run_name': run_name
            }
    
    def create_grid_plot(self, grid: List[List[int]], title: str = "") -> plt.Figure:
        """Create a matplotlib plot of an ARC grid."""
        if not grid or not grid[0]:
            # Create empty plot for invalid grids
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.text(0.5, 0.5, 'Invalid Grid', ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=12, fontweight='bold')
            return fig
        
        grid_array = np.array(grid)
        height, width = grid_array.shape
        
        # Create figure with appropriate size
        fig, ax = plt.subplots(1, 1, figsize=(max(4, width * 0.4), max(4, height * 0.4)))
        
        # Display the grid
        im = ax.imshow(grid_array, cmap=arc_cmap, vmin=0, vmax=9)
        
        # Add grid lines
        for i in range(height + 1):
            ax.axhline(y=i - 0.5, color='black', linewidth=1)
        for j in range(width + 1):
            ax.axvline(x=j - 0.5, color='black', linewidth=1)
        
        # Set ticks and labels
        ax.set_xticks(range(width))
        ax.set_yticks(range(height))
        ax.set_xticklabels(range(width), fontsize=8)
        ax.set_yticklabels(range(height), fontsize=8)
        
        # Set title
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        
        plt.tight_layout()
        return fig
    
    def parse_grid_string(self, grid_str: str) -> List[List[int]]:
        """Parse grid from string format to 2D list."""
        if not grid_str:
            return []
        
        try:
            # Remove quotes and split by lines
            lines = grid_str.strip().strip('"').split('\\n') if '\\n' in grid_str else grid_str.strip().split('\n')
            grid = []
            for line in lines:
                if line.strip():
                    row = [int(char) for char in line.strip()]
                    grid.append(row)
            return grid
        except (ValueError, TypeError):
            return []
    
    def get_task_list(self, run_data: Dict) -> List[str]:
        """Get list of tasks for the selected run."""
        if not run_data or 'tasks' not in run_data:
            print(f"Debug: No run data or no tasks in run data. Keys: {run_data.keys() if run_data else 'None'}")
            return []
        
        tasks = list(run_data['tasks'].keys())
        print(f"Debug: Found {len(tasks)} tasks: {tasks}")
        tasks.sort()
        return tasks
    
    def create_summary_info(self, run_data: Dict) -> str:
        """Create summary information for the selected run."""
        if not run_data:
            return "No run selected"
        
        summary = run_data.get('summary', {})
        params = run_data.get('params', {})
        
        info = f"""
        ## Run Summary: {run_data['run_name']}
        
        **Model:** {params.get('model', 'Unknown')}  
        **Total Tasks:** {summary.get('total_tasks', 0)}  
        **Correctly Solved:** {summary.get('completely_correct_tasks', 0)}  
        **Success Rate:** {summary.get('correctness_percentage', 0):.1f}%  
        **Total Tokens:** {summary.get('total_tokens', 0):,}  
        **Estimated Cost:** ${summary.get('total_estimated_cost', 0):.4f}  
        **Temperature:** {params.get('temperature', 'N/A')}  
        **Max Reflections:** {params.get('max_reflections', 'N/A')}  
        """
        
        return info
    
    def create_task_cards_html(self, run_data: Dict) -> str:
        """Create HTML cards for all tasks in the run."""
        if not run_data or 'tasks' not in run_data:
            return "<p>No tasks found for this run.</p>"
        
        tasks = run_data['tasks']
        if not tasks:
            return "<p>No valid tasks found for this run.</p>"
        
        html = """
        <style>
            .task-card {
                border: 2px solid #ddd;
                border-radius: 12px;
                padding: 20px;
                background: white;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 30px;
                transition: all 0.3s ease;
            }
            .task-card.correct {
                border-color: #28a745;
                background: linear-gradient(145deg, #ffffff, #f8fff9);
            }
            .task-card.incorrect {
                border-color: #dc3545;
                background: linear-gradient(145deg, #ffffff, #fff8f8);
            }
            .task-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                padding-bottom: 15px;
                border-bottom: 2px solid #eee;
            }
            .task-title {
                font-size: 20px;
                font-weight: bold;
                color: #333;
            }
            .task-status {
                display: flex;
                gap: 10px;
                align-items: center;
            }
            .status-badge {
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 14px;
                font-weight: bold;
            }
            .status-badge.correct {
                background-color: #d4edda;
                color: #155724;
            }
            .status-badge.incorrect {
                background-color: #f8d7da;
                color: #721c24;
            }
            .sample-count {
                font-size: 12px;
                color: #666;
                background: #f8f9fa;
                padding: 4px 8px;
                border-radius: 12px;
            }
            .samples-section {
                margin-bottom: 25px;
            }
            .samples-section h4 {
                color: #495057;
                margin-bottom: 15px;
                font-size: 16px;
                border-left: 4px solid #007bff;
                padding-left: 10px;
            }
            .sample-row {
                border: 1px solid #e9ecef;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 15px;
                background: #fafbfc;
            }
            .sample-label {
                font-weight: bold;
                color: #495057;
                margin-bottom: 10px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .test-result {
                font-size: 12px;
                padding: 2px 8px;
                border-radius: 10px;
            }
            .test-result.correct {
                background: #d1ecf1;
                color: #0c5460;
            }
            .test-result.incorrect {
                background: #f5c6cb;
                color: #721c24;
            }
            .train-result {
                font-size: 12px;
                padding: 2px 8px;
                border-radius: 10px;
            }
            .train-result.correct {
                background: #d1ecf1;
                color: #0c5460;
            }
            .train-result.incorrect {
                background: #f5c6cb;
                color: #721c24;
            }
            .metric-score {
                font-weight: bold;
                padding: 1px 4px;
                border-radius: 4px;
                transition: all 0.2s ease;
            }
            .metric-score.perfect-score {
                background: #28a745;
                color: white;
                box-shadow: 0 0 4px rgba(40, 167, 69, 0.3);
            }
            .grids-container {
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 12px;
                margin-top: 15px;
                width: 100%;
                align-items: start;
            }
            .grid-column {
                text-align: center;
                min-width: 0;
                width: 100%;
                overflow: visible;
                padding: 10px;
                background: #fafbfc;
                border: 2px solid #e9ecef;
                border-radius: 8px;
                min-height: 120px;
                display: flex;
                flex-direction: column;
                justify-content: flex-start;
            }
            .grid-column h5 {
                font-size: 16px;
                font-weight: bold;
                margin-bottom: 12px;
                margin-top: 0;
                color: #343a40;
                padding: 8px;
                background: linear-gradient(145deg, #f8f9fa, #e9ecef);
                border-radius: 6px;
                border: 1px solid #dee2e6;
            }
            .arc-grid {
                margin: 0 auto;
                border-collapse: collapse;
                border: 2px solid #333;
                max-width: 100%;
            }
            .arc-grid td {
                width: 8px;
                height: 8px;
                border: 1px solid #666;
                padding: 0;
                margin: 0;
                line-height: 8px;
            }
            .empty-grid {
                color: #999;
                font-style: italic;
                padding: 20px;
                background: #f8f9fa;
                border: 2px dashed #dee2e6;
                border-radius: 6px;
            }
            .iou-score {
                font-size: 12px;
                color: #666;
                background: #e9ecef;
                padding: 4px 8px;
                border-radius: 12px;
            }
        </style>
        """
        
        # Sort tasks for consistent display
        sorted_tasks = sorted(tasks.items())
        
        for task_id, task_data in sorted_tasks:
            # Get training and test data
            trains = task_data.get('trains', [])
            tests = task_data.get('tests', [])
            
            if not trains and not tests:
                continue
            
            # Get overall task status from first test case
            is_correct = False
            iou = 0
            overlap = 0
            if tests:
                test_case = tests[0]
                is_correct = test_case.get('correct', False)
                iou = test_case.get('iou', 0)
                overlap = test_case.get('overlap', 0)
            
            # Card styling based on correctness
            card_class = "task-card correct" if is_correct else "task-card incorrect"
            status_class = "correct" if is_correct else "incorrect"
            status_text = "✅ Correct" if is_correct else "❌ Incorrect"
            
            html += f"""
            <div class="{card_class}" id="task-{task_id}">
                <div class="task-header">
                    <div class="task-title">{task_id}</div>
                    <div class="task-status">
                        <span class="status-badge {status_class}">{status_text}</span>
                        <span class="metric-score {'perfect-score' if iou == 100 else ''}">IoU: {iou:.1f}%</span>
                        <span class="metric-score {'perfect-score' if overlap == 100 else ''}">Overlap: {overlap:.1f}%</span>
                        <span class="sample-count">Training: {len(trains)} | Test: {len(tests)}</span>
                    </div>
                </div>
            """
            
            # Add training examples
            if trains:
                html += f"""
                <div class="samples-section">
                    <h4>Training Examples ({len(trains)})</h4>
                """
                for i, train_case in enumerate(trains):
                    input_grid = self.parse_grid_from_strings(train_case.get('input', []))
                    output_grid = self.parse_grid_from_strings(train_case.get('output', []))
                    predicted_grid = self.parse_grid_from_strings(train_case.get('predict', []))
                    
                    # Get training metrics
                    train_correct = train_case.get('correct', False)
                    train_iou = train_case.get('iou', 0)
                    train_overlap = train_case.get('overlap', 0)
                    
                    # Create style classes for metrics
                    iou_class = "perfect-score" if train_iou == 100 else ""
                    overlap_class = "perfect-score" if train_overlap == 100 else ""
                    
                    html += f"""
                    <div class="sample-row">
                        <div class="sample-label">
                            Training {i+1}
                            <span class="train-result {'correct' if train_correct else 'incorrect'}">
                                {'✅' if train_correct else '❌'} 
                                <span class="metric-score {iou_class}">IoU: {train_iou:.1f}%</span> | 
                                <span class="metric-score {overlap_class}">Overlap: {train_overlap:.1f}%</span>
                            </span>
                        </div>
                        <div class="grids-container">
                            <div class="grid-column">
                                <h5>Input</h5>
                                {self.grid_to_html_table(input_grid)}
                            </div>
                            <div class="grid-column">
                                <h5>Expected Output</h5>
                                {self.grid_to_html_table(output_grid)}
                            </div>
                            <div class="grid-column">
                                <h5>Predicted Output</h5>
                                {self.grid_to_html_table(predicted_grid)}
                            </div>
                        </div>
                    </div>
                    """
                html += "</div>"
            
            # Add test cases
            if tests:
                html += f"""
                <div class="samples-section">
                    <h4>Test Cases ({len(tests)})</h4>
                """
                for i, test_case in enumerate(tests):
                    input_grid = self.parse_grid_from_strings(test_case.get('input', []))
                    actual_grid = self.parse_grid_from_strings(test_case.get('output', []))
                    predicted_grid = self.parse_grid_from_strings(test_case.get('predict', []))
                    
                    test_correct = test_case.get('correct', False)
                    test_iou = test_case.get('iou', 0)
                    
                    test_overlap = test_case.get('overlap', 0)
                    
                    # Create style classes for metrics
                    iou_class = "perfect-score" if test_iou == 100 else ""
                    overlap_class = "perfect-score" if test_overlap == 100 else ""
                    
                    html += f"""
                    <div class="sample-row">
                        <div class="sample-label">
                            Test {i+1}
                            <span class="test-result {'correct' if test_correct else 'incorrect'}">
                                {'✅' if test_correct else '❌'} 
                                <span class="metric-score {iou_class}">IoU: {test_iou:.1f}%</span> | 
                                <span class="metric-score {overlap_class}">Overlap: {test_overlap:.1f}%</span>
                            </span>
                        </div>
                        <div class="grids-container">
                            <div class="grid-column">
                                <h5>Input</h5>
                                {self.grid_to_html_table(input_grid)}
                            </div>
                            <div class="grid-column">
                                <h5>Expected Output</h5>
                                {self.grid_to_html_table(actual_grid)}
                            </div>
                            <div class="grid-column">
                                <h5>Predicted Output</h5>
                                {self.grid_to_html_table(predicted_grid)}
                            </div>
                        </div>
                    </div>
                    """
                html += "</div>"
            
            html += "</div>"
        
        return html
    
    def grid_to_html_table(self, grid: List[List[int]]) -> str:
        """Convert a grid to HTML table representation with full width utilization."""
        if not grid or not grid[0]:
            return '<div style="color: #999; font-style: italic; padding: 20px; text-align: center; min-height: 60px;">Invalid Grid</div>'
        
        # Calculate grid dimensions
        rows = len(grid)
        cols = len(grid[0])
        
        # Calculate cell size to maximize usage of available width
        # Each column should use most of its allocated 1/3 space
        available_width = 190  # Generous width for each column
        
        # Base cell size calculation
        border_space = (cols + 1) * 1 + 4  # Borders and table border
        usable_width = available_width - border_space
        
        if usable_width > 0 and cols > 0:
            cell_size = usable_width // cols
        else:
            cell_size = 10  # Fallback
        
        # Set reasonable bounds with preference for larger grids
        if cell_size < 4:
            cell_size = 4   # Absolute minimum
        elif cell_size > 30:
            cell_size = 30  # Maximum for very small grids
        elif cols <= 3 and rows <= 3:
            cell_size = max(cell_size, 20)  # Ensure small grids are large
        elif cols <= 10 and rows <= 10:
            cell_size = max(cell_size, 12)  # Medium grids get decent size
        
        # Calculate actual table width
        table_width = cols * cell_size + border_space
        
        # Ensure we don't exceed container but try to use most of it
        if table_width > available_width:
            cell_size = max(4, (available_width - border_space) // cols)
            table_width = cols * cell_size + border_space
        
        # Generate the table with calculated cell sizes
        html = f'''
        <div style="width: 100%; display: flex; justify-content: center;">
            <table style="
                border-collapse: collapse; 
                border: 2px solid #333; 
                background: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                width: {table_width}px;
            ">'''
        
        for row in grid:
            html += '<tr>'
            for cell in row:
                color = ARC_COLORS[cell] if 0 <= cell <= 9 else '#FFFFFF'
                html += f'''<td style="
                    background-color: {color}; 
                    width: {cell_size}px; 
                    height: {cell_size}px; 
                    border: 1px solid #666; 
                    padding: 0; 
                    margin: 0;
                    box-sizing: border-box;
                "></td>'''
            html += '</tr>'
        html += '</table></div>'
        return html
    
    def parse_grid_from_strings(self, grid_data) -> List[List[int]]:
        """Parse grid from various formats."""
        if not grid_data:
            return []
        
        try:
            if isinstance(grid_data, list):
                if grid_data and isinstance(grid_data[0], str):
                    # List of strings
                    return [[int(c) for c in row] for row in grid_data]
                elif grid_data and isinstance(grid_data[0], list):
                    # Already a 2D list
                    return [[int(c) for c in row] for row in grid_data]
            
            return []
        except (ValueError, TypeError, IndexError):
            return []
    
    def create_task_menu_html(self, run_data: Dict) -> str:
        """Create HTML for the task menu as a dropdown-style interface."""
        if not run_data or 'tasks' not in run_data or not run_data['tasks']:
            return "<p>No tasks available</p>"
        
        tasks = sorted(run_data['tasks'].keys())
        
        # Calculate statistics
        total_tasks = len(tasks)
        correct_tasks = 0
        
        for task_id in tasks:
            task_data = run_data['tasks'][task_id]
            tests = task_data.get('tests', [])
            if tests and tests[0].get('correct', False):
                correct_tasks += 1
        
        html = f"""
        <div style="width: 100%; max-width: 100%; overflow: hidden; box-sizing: border-box;">
        <style>
            .task-stats {{
                background: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 12px;
                margin-bottom: 15px;
                text-align: center;
            }}
            .task-dropdown-container {{
                margin-bottom: 10px;
            }}
            .task-dropdown {{
                width: 100%;
                padding: 8px 12px;
                border: 2px solid #dee2e6;
                border-radius: 6px;
                background: white;
                font-size: 14px;
                cursor: pointer;
                transition: border-color 0.2s ease;
            }}
            .task-dropdown:hover {{
                border-color: #007bff;
            }}
            .task-dropdown:focus {{
                outline: none;
                border-color: #007bff;
                box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
            }}
            .task-grid-nav {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 5px;
                margin-top: 10px;
                max-width: 100%;
                width: 100%;
            }}
            .task-nav-btn {{
                background: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 6px 4px;
                font-size: 11px;
                cursor: pointer;
                text-align: center;
                transition: all 0.2s ease;
                text-decoration: none;
                color: #495057;
                display: block;
            }}
            .task-nav-btn:hover {{
                background: #e9ecef;
                border-color: #007bff;
                color: #007bff;
            }}
            .task-nav-btn.correct {{
                background: #d4edda;
                border-color: #c3e6cb;
            }}
            .task-nav-btn.incorrect {{
                background: #f8d7da;
                border-color: #f5c6cb;
            }}
        </style>
        
        <div class="task-stats">
            <strong>{correct_tasks}/{total_tasks}</strong> tasks solved<br>
            <span style="color: #666; font-size: 12px;">({(correct_tasks/total_tasks*100):.1f}% success rate)</span>
        </div>
        
        <div class="task-dropdown-container">
            <select class="task-dropdown" onchange="if(this.value) document.getElementById('task-' + this.value).scrollIntoView({{behavior: 'smooth'}})">
                <option value="">Jump to task...</option>
        """
        
        for task_id in tasks:
            task_data = run_data['tasks'][task_id]
            tests = task_data.get('tests', [])
            trains = task_data.get('trains', [])
            status_icon = ""
            if tests:
                correct = tests[0].get('correct', False)
                status_icon = "✅" if correct else "❌"
            
            # Add sample counts to task display
            sample_info = f" (T:{len(trains)} | Te:{len(tests)})"
            html += f'<option value="{task_id}">{status_icon} {task_id}{sample_info}</option>'
        
        html += """
            </select>
        </div>
        
        <div class="task-grid-nav">
        """
        
        for task_id in tasks:
            task_data = run_data['tasks'][task_id]
            tests = task_data.get('tests', [])
            status_class = ""
            status_icon = ""
            if tests:
                correct = tests[0].get('correct', False)
                status_class = "correct" if correct else "incorrect"
                status_icon = "✅" if correct else "❌"
            
            html += f"""
            <a href="#task-{task_id}" class="task-nav-btn {status_class}" 
               onclick="document.getElementById('task-{task_id}').scrollIntoView({{behavior: 'smooth'}})">
                {status_icon}<br>{task_id[:8]}...
            </a>
            """
        
        html += "</div>"
        html += "</div>"  # Close the width constraint wrapper
        return html
    
    def create_sidebar_html(self, run_name=None) -> str:
        """Create HTML for the sidebar with run selection and navigation."""
        runs_with_counts = self.get_available_runs_with_counts()
        
        html = f"""
        <div style="height: 100%; overflow-y: auto; padding: 10px;">
            <h3 style="margin-top: 0;">🔧 Run Selection</h3>
            
            <select id="run-selector" style="width: 100%; padding: 8px; margin-bottom: 10px; border: 2px solid #dee2e6; border-radius: 6px;" onchange="selectRun(this.value)">
                <option value="">Choose a run...</option>
        """
        
        for run_id, display_name in runs_with_counts:
            selected = 'selected' if run_id == run_name else ''
            html += f'<option value="{run_id}" {selected}>{display_name}</option>'
        
        html += """
            </select>
            
            <button onclick="refreshRuns()" style="width: 100%; padding: 8px; margin-bottom: 15px; background: #6c757d; color: white; border: none; border-radius: 4px; cursor: pointer;">🔄 Refresh Runs</button>
            
            <div id="run-summary" style="background: #f8f9fa; padding: 10px; border-radius: 6px; margin-bottom: 15px; font-size: 12px;">
                Select a run to see summary
            </div>
            
            <h3>📋 Task Navigation</h3>
            <div id="task-navigation" style="max-height: 400px; overflow-y: auto;">
                Select a run to see tasks
            </div>
            
            <script>
                function selectRun(runName) {
                    if (runName) {
                        // This will be handled by Gradio events
                        console.log('Selected run:', runName);
                    }
                }
                
                function refreshRuns() {
                    // This will be handled by Gradio events
                    console.log('Refreshing runs');
                }
            </script>
        </div>
        """
        
        return html


def create_app():
    """Create the Gradio application."""
    visualizer = ARCVisualizer()
    
    with gr.Blocks(title="ARC Solver Results Visualizer", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🧠 ARC Solver Results Visualizer")
        gr.Markdown("Visualize and analyze results from ARC (Abstraction and Reasoning Corpus) solver runs.")
        
        # Custom CSS for improved layout and scrolling
        gr.HTML("""
        <style>
            .gradio-container {
                max-height: 100vh !important;
            }
            .sidebar-scroll {
                max-height: 85vh;
                overflow-y: auto;
                padding-right: 10px;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                background: #f8f9fa;
                padding: 15px;
                margin-bottom: 10px;
            }
            .main-content-scroll {
                max-height: 90vh;
                overflow-y: auto;
                padding: 10px;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                background: white;
            }
            /* Force task navigation to respect column width */
            .sidebar-scroll * {
                max-width: 100% !important;
                box-sizing: border-box !important;
            }
            /* Ensure HTML components don't overflow */
            .sidebar-scroll .gr-html {
                overflow-wrap: break-word !important;
                word-wrap: break-word !important;
                width: 100% !important;
            }

            /* Custom scrollbar styling */
            .sidebar-scroll::-webkit-scrollbar,
            .main-content-scroll::-webkit-scrollbar {
                width: 8px;
            }
            .sidebar-scroll::-webkit-scrollbar-track,
            .main-content-scroll::-webkit-scrollbar-track {
                background: #f1f1f1;
                border-radius: 4px;
            }
            .sidebar-scroll::-webkit-scrollbar-thumb,
            .main-content-scroll::-webkit-scrollbar-thumb {
                background: #888;
                border-radius: 4px;
            }
            .sidebar-scroll::-webkit-scrollbar-thumb:hover,
            .main-content-scroll::-webkit-scrollbar-thumb:hover {
                background: #555;
            }
        </style>
        """)
        
        with gr.Row():
            with gr.Column(scale=1, min_width=300, elem_classes=["sidebar-scroll"]):
                # Run selection
                runs_with_counts = visualizer.get_available_runs_with_counts()
                run_choices = [(display_name, run_name) for run_name, display_name in runs_with_counts]
                run_dropdown = gr.Dropdown(
                    choices=run_choices,
                    label="🔧 Select Run",
                    interactive=True
                )
                
                # Refresh button
                refresh_btn = gr.Button("🔄 Refresh Runs", variant="secondary")
                
                # Run summary (no special styling needed)
                summary_md = gr.Markdown("Select a run to see summary")
                
                # Task navigation section
                gr.Markdown("### 📋 Task Navigation")
                task_menu_html = gr.HTML(value="<p>Select a run to see tasks</p>")
                
            with gr.Column(scale=4, min_width=600, elem_classes=["main-content-scroll"]):
                # Task cards container
                task_cards_container = gr.HTML("Select a run to view tasks")
        
        # Event handlers
        def refresh_runs():
            runs_with_counts = visualizer.get_available_runs_with_counts()
            run_choices = [(display_name, run_name) for run_name, display_name in runs_with_counts]
            return gr.Dropdown(choices=run_choices)
        
        def on_run_selected(run_name):
            if not run_name:
                return "Select a run to see summary", "<p>Select a run to see tasks</p>", "Select a run to view tasks"
            
            try:
                run_data = visualizer.load_run_data(run_name)
                visualizer.current_run_data = run_data
                
                summary = visualizer.create_summary_info(run_data)
                task_menu = visualizer.create_task_menu_html(run_data)
                task_cards = visualizer.create_task_cards_html(run_data)
                
                return summary, task_menu, task_cards
            except Exception as e:
                error_msg = f"Error loading run {run_name}: {str(e)}"
                print(error_msg)
                return error_msg, "<p>Error loading tasks</p>", "<p>Error loading task data.</p>"
        
        # Wire up events
        refresh_btn.click(
            refresh_runs,
            outputs=[run_dropdown]
        )
        
        run_dropdown.change(
            on_run_selected,
            inputs=[run_dropdown],
            outputs=[summary_md, task_menu_html, task_cards_container]
        )
    
    return app


if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        import gradio
        import matplotlib.pyplot
        import pandas
        import numpy
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install required packages:")
        print("pip install gradio matplotlib pandas numpy")
        exit(1)
    
    app = create_app()
    
    # Launch the app
    print("Starting ARC Solver Results Visualizer...")
    print("Navigate to the provided URL to view the interface")
    
    app.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7861,       # Use different port to avoid conflicts
        share=False,            # Set to True to create a public link
        debug=False
    )