#!/usr/bin/env python3
"""
Simple test to demonstrate the parallel output buffering functionality.
"""

import sys
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Copy the OutputCapture class from test_arc_prompt.py
_thread_local = threading.local()

class OutputCapture:
    """Context manager to capture stdout and stderr for parallel execution."""
    
    def __init__(self):
        self.buffer = io.StringIO()
        self.original_stdout = None
        self.original_stderr = None
    
    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self.buffer
        sys.stderr = self.buffer
        _thread_local.output_buffer = self.buffer
        return self.buffer
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        if hasattr(_thread_local, 'output_buffer'):
            delattr(_thread_local, 'output_buffer')

def simulate_task_processing(task_id):
    """Simulate processing a task with lots of output."""
    print(f"Starting task {task_id}")
    print(f"Processing step 1 for task {task_id}")
    time.sleep(1)  # Simulate work
    print(f"Processing step 2 for task {task_id}")
    time.sleep(1)  # Simulate more work
    print(f"Processing step 3 for task {task_id}")
    print(f"Task {task_id} completed successfully!")
    return f"result_for_task_{task_id}"

def task_wrapper(task_id):
    """Wrapper that captures output for a task."""
    with OutputCapture() as captured_output:
        result = simulate_task_processing(task_id)
    
    output_text = captured_output.getvalue()
    return result, output_text, task_id

def main():
    print("=== Testing Parallel Output Buffering ===")
    print("Running 3 tasks in parallel...")
    print("Without buffering, the output would be mixed.")
    print("With buffering, each task's output stays together.\n")
    
    tasks = ["TaskA", "TaskB", "TaskC"]
    print_lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(task_wrapper, task_id): task_id 
                         for task_id in tasks}
        
        # Collect results as they complete
        for i, future in enumerate(as_completed(future_to_task)):
            original_task_id = future_to_task[future]
            try:
                result, captured_output, task_id = future.result()
                
                # Print the captured output in a synchronized way
                with print_lock:
                    print(f"\n{'='*60}")
                    print(f"COMPLETED: {task_id} ({i+1}/{len(tasks)})")
                    print(f"{'='*60}")
                    print(captured_output)
                    print(f"Result: {result}")
                    print(f"{'='*60}")
                    
            except Exception as e:
                with print_lock:
                    print(f"Error in {original_task_id}: {e}")

if __name__ == "__main__":
    main()