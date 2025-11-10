#!/usr/bin/env python3
"""
Test script to demonstrate parallel buffering with artificial delays
This shows that output appears as each task completes, not all at once.
"""

import time
import threading
import sys
import io
from concurrent.futures import ThreadPoolExecutor

class OutputCapture:
    def __init__(self):
        self.stdout_buffer = io.StringIO()
        self.stderr_buffer = io.StringIO()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.lock = threading.Lock()
        
    def __enter__(self):
        sys.stdout = self.stdout_buffer
        sys.stderr = self.stderr_buffer
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
    
    def get_output(self):
        return self.stdout_buffer.getvalue() + self.stderr_buffer.getvalue()

def simulate_task(task_id, delay):
    """Simulate a task with artificial delay"""
    start_time = time.time()
    
    print(f"🚀 Starting task {task_id}")
    print(f"📋 Task {task_id} - Step 1: Initialize")
    time.sleep(delay * 0.3)
    
    print(f"🔄 Task {task_id} - Step 2: Processing data")
    time.sleep(delay * 0.4)
    
    print(f"✨ Task {task_id} - Step 3: Generating results")
    time.sleep(delay * 0.3)
    
    elapsed = time.time() - start_time
    print(f"✅ Task {task_id} completed in {elapsed:.1f}s")
    
    return f"Task {task_id} result", elapsed

def run_task_with_capture(args):
    task_id, delay = args
    start_time = time.time()
    
    with OutputCapture() as capture:
        result, task_elapsed = simulate_task(task_id, delay)
        
    captured_output = capture.get_output()
    total_elapsed = time.time() - start_time
    
    return task_id, result, captured_output, total_elapsed

def main():
    print("🧪 Testing Parallel Buffering with Staggered Completion Times")
    print("=" * 70)
    
    # Define tasks with different delays (in seconds)
    task_configs = [
        ("FAST", 2),      # 2 second task
        ("MEDIUM", 5),    # 5 second task
        ("SLOW", 8),      # 8 second task
        ("VERY_SLOW", 12) # 12 second task
    ]
    
    print(f"Running {len(task_configs)} tasks with staggered delays:")
    for task_id, delay in task_configs:
        print(f"  - Task {task_id}: {delay}s delay")
    print()
    print("Output for each task will be displayed as it completes")
    print("=" * 70)
    
    overall_start = time.time()
    print_lock = threading.Lock()
    
    # Run tasks in parallel
    with ThreadPoolExecutor(max_workers=len(task_configs)) as executor:
        futures = [executor.submit(run_task_with_capture, config) for config in task_configs]
        
        # Process results as they complete
        for i, future in enumerate(futures):
            task_id, result, captured_output, elapsed = future.result()
            overall_elapsed = time.time() - overall_start
            
            with print_lock:
                print(f"⏱️  Task completed after {overall_elapsed:.1f}s, processing result {i+1}/{len(task_configs)}...")
                print()
                print("=" * 50)
                print(f"TASK COMPLETED: {task_id} ({i+1}/{len(task_configs)}) ✓")
                print("=" * 50)
                print(captured_output)
                print(f"Task {task_id} finished in {elapsed:.1f}s")
                print("=" * 50)
                print()
                
                # Flush output immediately
                sys.stdout.flush()
    
    total_time = time.time() - overall_start
    print(f"🎉 All tasks completed in {total_time:.1f}s total")

if __name__ == "__main__":
    main()