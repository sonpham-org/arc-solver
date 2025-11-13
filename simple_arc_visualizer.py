#!/usr/bin/env python3
"""
Simple ARC Solver Results Visualizer

A simple web application to visualize ARC solver results using basic HTML generation.
"""

import os
import json
import webbrowser
import http.server
import socketserver
from urllib.parse import parse_qs, urlparse
from typing import Dict, List, Optional


class SimpleARCVisualizer:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        self.current_run = None
        self.current_task = None
        
        # ARC color scheme
        self.ARC_COLORS = [
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
        
    def get_available_runs(self) -> List[str]:
        """Get list of available runs."""
        if not os.path.exists(self.output_dir):
            return []
        
        runs = []
        for item in os.listdir(self.output_dir):
            run_path = os.path.join(self.output_dir, item)
            if os.path.isdir(run_path):
                params_file = os.path.join(run_path, 'params.json')
                summary_file = os.path.join(run_path, 'summary.json')
                if os.path.exists(params_file) and os.path.exists(summary_file):
                    runs.append(item)
        
        runs.sort(reverse=True)
        return runs
    
    def load_run_data(self, run_name: str) -> Optional[Dict]:
        """Load data for a specific run."""
        if not run_name:
            return None
            
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
                    task_id = file[:-5]
                    with open(os.path.join(run_path, file), 'r') as f:
                        tasks[task_id] = json.load(f)
            
            return {
                'params': params,
                'summary': summary,
                'tasks': tasks,
                'run_name': run_name
            }
        except Exception as e:
            print(f"Error loading run {run_name}: {e}")
            return None
    
    def grid_to_html(self, grid: List[List[int]], title: str = "") -> str:
        """Convert a grid to HTML representation."""
        if not grid or not grid[0]:
            return f'<div class="grid-container"><h4>{title}</h4><p>Invalid Grid</p></div>'
        
        html = f'<div class="grid-container"><h4>{title}</h4><table class="arc-grid">'
        
        for row in grid:
            html += '<tr>'
            for cell in row:
                color = self.ARC_COLORS[cell] if 0 <= cell <= 9 else '#FFFFFF'
                html += f'<td style="background-color: {color}; width: 20px; height: 20px; border: 1px solid #333;"></td>'
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
    
    def generate_html(self, run_name: str = None, task_id: str = None) -> str:
        """Generate the full HTML page."""
        runs = self.get_available_runs()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ARC Solver Results Visualizer</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    display: flex;
                    gap: 20px;
                }}
                .sidebar {{
                    min-width: 300px;
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    height: fit-content;
                }}
                .main-content {{
                    flex: 1;
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .grid-container {{
                    margin: 10px 0;
                    text-align: center;
                }}
                .arc-grid {{
                    margin: 10px auto;
                    border-collapse: collapse;
                }}
                .grids-row {{
                    display: flex;
                    justify-content: space-around;
                    align-items: flex-start;
                    margin: 20px 0;
                }}
                select, button {{
                    width: 100%;
                    padding: 8px;
                    margin: 5px 0;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }}
                button {{
                    background-color: #007bff;
                    color: white;
                    cursor: pointer;
                }}
                button:hover {{
                    background-color: #0056b3;
                }}
                .summary {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 4px;
                    margin: 10px 0;
                }}
                h1, h2, h3 {{
                    color: #333;
                }}
                .task-info {{
                    background-color: #e9ecef;
                    padding: 10px;
                    border-radius: 4px;
                    margin: 10px 0;
                }}
            </style>
        </head>
        <body>
            <h1>🧠 ARC Solver Results Visualizer</h1>
            <div class="container">
                <div class="sidebar">
                    <h3>Select Run</h3>
                    <form method="GET" action="/">
                        <select name="run" onchange="this.form.submit()">
                            <option value="">Choose a run...</option>"""
        
        for run in runs:
            selected = 'selected' if run == run_name else ''
            html += f'<option value="{run}" {selected}>{run}</option>'
        
        html += """
                        </select>
                    </form>
        """
        
        # If run is selected, show run info and task selection
        if run_name:
            run_data = self.load_run_data(run_name)
            if run_data:
                summary = run_data['summary']
                params = run_data['params']
                tasks = list(run_data['tasks'].keys())
                tasks.sort()
                
                html += f"""
                    <div class="summary">
                        <h4>Run Summary</h4>
                        <p><strong>Model:</strong> {params.get('model', 'Unknown')}</p>
                        <p><strong>Total Tasks:</strong> {summary.get('total_tasks', 0)}</p>
                        <p><strong>Correct:</strong> {summary.get('completely_correct_tasks', 0)}</p>
                        <p><strong>Success Rate:</strong> {summary.get('correctness_percentage', 0):.1f}%</p>
                        <p><strong>Total Tokens:</strong> {summary.get('total_tokens', 0):,}</p>
                        <p><strong>Cost:</strong> ${summary.get('total_estimated_cost', 0):.4f}</p>
                    </div>
                    
                    <h3>Select Task</h3>
                    <form method="GET" action="/">
                        <input type="hidden" name="run" value="{run_name}">
                        <select name="task" onchange="this.form.submit()">
                            <option value="">Choose a task...</option>"""
                
                for task in tasks:
                    selected = 'selected' if task == task_id else ''
                    html += f'<option value="{task}" {selected}>{task}</option>'
                
                html += """
                        </select>
                    </form>
                """
        
        html += """
                </div>
                <div class="main-content">
        """
        
        # Show task grids if both run and task are selected
        if run_name and task_id:
            run_data = self.load_run_data(run_name)
            if run_data and task_id in run_data['tasks']:
                task_data = run_data['tasks'][task_id]
                
                # Show task information
                html += f"""
                    <div class="task-info">
                        <h2>Task: {task_id}</h2>
                """
                
                if 'tests' in task_data and task_data['tests']:
                    test_case = task_data['tests'][0]
                    
                    input_grid = self.parse_grid_from_strings(test_case.get('input', []))
                    actual_grid = self.parse_grid_from_strings(test_case.get('output', []))
                    predicted_grid = self.parse_grid_from_strings(test_case.get('predict', []))
                    
                    correct = test_case.get('correct', False)
                    iou = test_case.get('iou', 0)
                    
                    html += f"""
                        <p><strong>Result:</strong> {'✅ Correct' if correct else '❌ Incorrect'}</p>
                        <p><strong>IoU Score:</strong> {iou:.1f}%</p>
                    </div>
                    
                    <div class="grids-row">
                        {self.grid_to_html(input_grid, "Input")}
                        {self.grid_to_html(predicted_grid, "Predicted Output")}
                        {self.grid_to_html(actual_grid, "Actual Output")}
                    </div>
                    """
                else:
                    html += "<p>No test data available for this task.</p></div>"
            else:
                html += f"<p>Task {task_id} not found.</p>"
        elif run_name:
            html += "<p>Select a task to view the grids.</p>"
        else:
            html += "<p>Select a run to begin.</p>"
        
        html += """
                </div>
            </div>
        </body>
        </html>
        """
        
        return html


class SimpleHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, visualizer, *args, **kwargs):
        self.visualizer = visualizer
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)
        
        run_name = query_params.get('run', [None])[0]
        task_id = query_params.get('task', [None])[0]
        
        if parsed_url.path == '/':
            html_content = self.visualizer.generate_html(run_name, task_id)
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html_content.encode('utf-8'))
        else:
            super().do_GET()


def main():
    """Main function to start the server."""
    visualizer = SimpleARCVisualizer()
    
    # Check if output directory exists
    if not os.path.exists(visualizer.output_dir):
        print(f"Error: Output directory '{visualizer.output_dir}' not found.")
        print("Make sure you're running this from the arc-solver directory.")
        return
    
    # Check if there are any runs
    runs = visualizer.get_available_runs()
    if not runs:
        print(f"No valid runs found in '{visualizer.output_dir}' directory.")
        return
    
    PORT = 8080
    
    def handler(*args, **kwargs):
        return SimpleHTTPRequestHandler(visualizer, *args, **kwargs)
    
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"🚀 ARC Solver Results Visualizer started!")
        print(f"📊 Found {len(runs)} runs to visualize")
        print(f"🌐 Open your browser and go to: http://localhost:{PORT}")
        print(f"⏹️  Press Ctrl+C to stop the server")
        
        # Try to open browser automatically
        try:
            webbrowser.open(f'http://localhost:{PORT}')
        except:
            pass
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped!")


if __name__ == "__main__":
    main()