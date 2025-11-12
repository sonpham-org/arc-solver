#!/usr/bin/env python3
"""
Quick test script for the helper learning system
"""

from arc_prompt_with_helpers import HelperDatabase

def test_helper_database():
    """Test the helper database functionality."""
    print("Testing Helper Database...")
    
    # Initialize database
    db = HelperDatabase("test_helpers.db")
    
    # Test storing a helper function
    test_function = """def find_connected_components(grid):
    \"\"\"Find connected components in a grid.\"\"\"
    import numpy as np
    from collections import deque
    
    h, w = len(grid), len(grid[0])
    visited = [[False]*w for _ in range(h)]
    components = []
    
    for i in range(h):
        for j in range(w):
            if not visited[i][j] and grid[i][j] != 0:
                component = []
                queue = deque([(i, j)])
                while queue:
                    r, c = queue.popleft()
                    if (0 <= r < h and 0 <= c < w and 
                        not visited[r][c] and grid[r][c] == grid[i][j]):
                        visited[r][c] = True
                        component.append((r, c))
                        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                            queue.append((r+dr, c+dc))
                components.append(component)
    
    return components"""
    
    # Store the function
    result = db.store_helper_function(test_function, "test_task_001", True)
    print(f"Stored helper function: {result}")
    
    # Store another function for testing
    test_function2 = """def rotate_grid_90(grid):
    \"\"\"Rotate a grid 90 degrees clockwise.\"\"\"
    return [[grid[len(grid)-1-j][i] for j in range(len(grid))] for i in range(len(grid[0]))]"""
    
    db.store_helper_function(test_function2, "test_task_002", True)
    
    # Get database stats
    stats = db.get_database_stats()
    print(f"Database stats: {stats}")
    
    # Get top helpers
    top_helpers = db.get_top_helpers(5)
    print(f"Top helpers: {len(top_helpers)}")
    
    # Get random helpers
    random_helpers = db.get_random_helpers(5)
    print(f"Random helpers: {len(random_helpers)}")
    
    print("Helper database test completed!")

if __name__ == "__main__":
    test_helper_database()