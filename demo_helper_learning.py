#!/usr/bin/env python3
"""
Demo script to show the helper learning system in action
"""

from arc_prompt_with_helpers import HelperDatabase
from prompts import build_arc_with_helpers_prompt
import json

def demo_helper_learning():
    """Demonstrate the helper learning system."""
    print("🎯 HELPER LEARNING SYSTEM DEMO")
    print("="*50)
    
    # Initialize helper database
    print("\n1. Initializing helper database...")
    db = HelperDatabase("demo_helpers.db")
    
    # Add some sample helper functions
    print("\n2. Adding sample helper functions...")
    
    helper1 = """def find_bounding_box(grid, value):
    \"\"\"Find the bounding box of all cells with a specific value.\"\"\"
    min_row, max_row = len(grid), -1
    min_col, max_col = len(grid[0]) if grid else 0, -1
    
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell == value:
                min_row = min(min_row, i)
                max_row = max(max_row, i)
                min_col = min(min_col, j)
                max_col = max(max_col, j)
    
    return (min_row, min_col, max_row, max_col) if max_row >= 0 else None"""
    
    helper2 = """def rotate_grid_90(grid):
    \"\"\"Rotate a grid 90 degrees clockwise.\"\"\"
    if not grid:
        return []
    return [[grid[len(grid)-1-j][i] for j in range(len(grid))] for i in range(len(grid[0]))]"""
    
    helper3 = """def flood_fill(grid, start_row, start_col, new_value):
    \"\"\"Flood fill starting from a position.\"\"\"
    if not grid or start_row < 0 or start_row >= len(grid):
        return grid
    if start_col < 0 or start_col >= len(grid[0]):
        return grid
    
    original_value = grid[start_row][start_col]
    if original_value == new_value:
        return grid
    
    # Create a copy
    result = [row[:] for row in grid]
    
    def fill(r, c):
        if (r < 0 or r >= len(result) or c < 0 or c >= len(result[0]) or 
            result[r][c] != original_value):
            return
        result[r][c] = new_value
        fill(r+1, c)
        fill(r-1, c)
        fill(r, c+1)
        fill(r, c-1)
    
    fill(start_row, start_col)
    return result"""
    
    # Store helpers with different success rates to simulate learning
    db.store_helper_function(helper1, "task_001", True)  # Successful
    db.store_helper_function(helper1, "task_002", True)  # Used again successfully
    db.store_helper_function(helper2, "task_003", True)  # Successful
    db.store_helper_function(helper3, "task_004", False)  # Failed task
    db.store_helper_function(helper2, "task_005", True)  # Used again successfully
    
    print("   ✓ Added helper functions to database")
    
    # Show database stats
    print("\n3. Helper database statistics:")
    stats = db.get_database_stats()
    for key, value in stats.items():
        if key != 'top_performers':
            print(f"   {key}: {value}")
    print("   top_performers:")
    for name, success, usage in stats['top_performers']:
        print(f"      {name}: {success} successes, {usage} total uses")
    
    # Get helpers for prompting
    print("\n4. Getting helpers for prompt generation...")
    top_helpers, random_helpers = db.get_helpers_for_prompting(2, 1)
    
    print(f"   Top helpers: {len(top_helpers)}")
    for i, helper in enumerate(top_helpers):
        func_name = helper[0]
        success_count = helper[4]
        print(f"      {i+1}. {func_name} ({success_count} successes)")
    
    print(f"   Random helpers: {len(random_helpers)}")
    for i, helper in enumerate(random_helpers):
        func_name = helper[0]
        success_count = helper[4]
        print(f"      {i+1}. {func_name} ({success_count} successes)")
    
    # Demo prompt generation
    print("\n5. Generating helper-aware prompt...")
    
    # Sample task data
    sample_task_data = {
        "demo_task": {
            "train": [
                {
                    "input": [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
                    "output": [[2, 0, 2], [0, 2, 0], [2, 0, 2]]
                }
            ],
            "test": [
                {
                    "input": [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
                }
            ]
        }
    }
    
    # Generate prompt with helpers
    prompt = build_arc_with_helpers_prompt(sample_task_data, "demo_task", top_helpers, random_helpers)
    
    # Show a snippet of the generated prompt
    prompt_lines = prompt.split('\n')
    print("   Generated prompt includes:")
    print("   - Task description and examples")
    print("   - Available helper functions section")
    print("   - Guidelines that encourage using helpers")
    
    # Show helper section snippet
    helper_section_start = None
    for i, line in enumerate(prompt_lines):
        if "Available Helper Functions" in line:
            helper_section_start = i
            break
    
    if helper_section_start:
        print(f"\n   Helper section preview (lines {helper_section_start+1}-{helper_section_start+10}):")
        for i in range(helper_section_start, min(helper_section_start + 10, len(prompt_lines))):
            print(f"      {prompt_lines[i]}")
    
    print(f"\n   Total prompt length: {len(prompt)} characters")
    
    print("\n🎉 Demo completed! The helper learning system:")
    print("   ✓ Stores helper functions from successful tasks")
    print("   ✓ Tracks usage and success rates")  
    print("   ✓ Provides top performers + random selection for prompts")
    print("   ✓ Enables iterative improvement through REDO_TASKS")
    print("\n   Ready to learn and improve from task to task! 🚀")

if __name__ == "__main__":
    demo_helper_learning()