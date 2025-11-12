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
    
    def load_run_data(self, run_name: str) -> Dict:
        """Load all data for a specific run."""
        if run_name in self.runs_cache:
            return self.runs_cache[run_name]
        
        run_path = os.path.join(self.output_dir, run_name)
        
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
                task_id = file[:-5]  # Remove .json extension
                with open(os.path.join(run_path, file), 'r') as f:
                    tasks[task_id] = json.load(f)
        
        run_data = {
            'params': params,
            'summary': summary,
            'tasks': tasks,
            'run_name': run_name
        }
        
        self.runs_cache[run_name] = run_data
        return run_data
    
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
            return []
        
        tasks = list(run_data['tasks'].keys())
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
    
    def get_task_grids(self, run_data: Dict, task_id: str) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
        """Get the three grids (input, predicted, actual) for a task."""
        if not run_data or 'tasks' not in run_data or task_id not in run_data['tasks']:
            empty_fig = self.create_grid_plot([], "No Data")
            return empty_fig, empty_fig, empty_fig
        
        task_data = run_data['tasks'][task_id]
        
        # Get test case (use first test case if multiple exist)
        tests = task_data.get('tests', [])
        if not tests:
            empty_fig = self.create_grid_plot([], "No Test Data")
            return empty_fig, empty_fig, empty_fig
        
        test_case = tests[0]  # Use first test case
        
        # Parse grids
        input_grid = test_case.get('input', [])
        actual_grid = test_case.get('output', [])
        predicted_grid = test_case.get('predict', [])
        
        # Convert string grids to lists if needed
        if isinstance(input_grid, list) and input_grid and isinstance(input_grid[0], str):
            input_grid = [list(row) for row in input_grid]
            input_grid = [[int(c) for c in row] for row in input_grid]
        
        if isinstance(actual_grid, list) and actual_grid and isinstance(actual_grid[0], str):
            actual_grid = [list(row) for row in actual_grid]
            actual_grid = [[int(c) for c in row] for row in actual_grid]
        
        if isinstance(predicted_grid, list) and predicted_grid and isinstance(predicted_grid[0], str):
            predicted_grid = [list(row) for row in predicted_grid]
            predicted_grid = [[int(c) for c in row] for row in predicted_grid]
        
        # Create plots
        input_fig = self.create_grid_plot(input_grid, "Input")
        predicted_fig = self.create_grid_plot(predicted_grid, "Predicted Output")
        actual_fig = self.create_grid_plot(actual_grid, "Actual Output")
        
        return input_fig, predicted_fig, actual_fig


def create_app():
    """Create the Gradio application."""
    visualizer = ARCVisualizer()
    
    with gr.Blocks(title="ARC Solver Results Visualizer", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🧠 ARC Solver Results Visualizer")
        gr.Markdown("Visualize and analyze results from ARC (Abstraction and Reasoning Corpus) solver runs.")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Run selection
                run_dropdown = gr.Dropdown(
                    choices=visualizer.get_available_runs(),
                    label="Select Run",
                    interactive=True
                )
                
                # Refresh button
                refresh_btn = gr.Button("🔄 Refresh Runs", variant="secondary")
                
                # Run summary
                summary_md = gr.Markdown("Select a run to see summary")
                
                # Task list
                task_dropdown = gr.Dropdown(
                    choices=[],
                    label="Select Task",
                    interactive=True
                )
                
            with gr.Column(scale=2):
                # Grid displays
                with gr.Row():
                    input_plot = gr.Plot(label="Input Grid")
                
                with gr.Row():
                    predicted_plot = gr.Plot(label="Predicted Output")
                
                with gr.Row():
                    actual_plot = gr.Plot(label="Actual Output")
        
        # Event handlers
        def refresh_runs():
            return gr.Dropdown(choices=visualizer.get_available_runs())
        
        def on_run_selected(run_name):
            if not run_name:
                return "Select a run to see summary", [], None, None, None
            
            run_data = visualizer.load_run_data(run_name)
            visualizer.current_run_data = run_data
            
            summary = visualizer.create_summary_info(run_data)
            tasks = visualizer.get_task_list(run_data)
            
            # Clear plots when run changes
            empty_fig = visualizer.create_grid_plot([], "Select a task")
            
            return summary, tasks, empty_fig, empty_fig, empty_fig
        
        def on_task_selected(task_id):
            if not task_id or not visualizer.current_run_data:
                empty_fig = visualizer.create_grid_plot([], "No task selected")
                return empty_fig, empty_fig, empty_fig
            
            return visualizer.get_task_grids(visualizer.current_run_data, task_id)
        
        # Wire up events
        refresh_btn.click(
            refresh_runs,
            outputs=[run_dropdown]
        )
        
        run_dropdown.change(
            on_run_selected,
            inputs=[run_dropdown],
            outputs=[summary_md, task_dropdown, input_plot, predicted_plot, actual_plot]
        )
        
        task_dropdown.change(
            on_task_selected,
            inputs=[task_dropdown],
            outputs=[input_plot, predicted_plot, actual_plot]
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
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True to create a public link
        debug=False
    )