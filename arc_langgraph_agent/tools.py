"""
Tool definitions for common ARC helper functions.

This module provides tool definitions that can be used by the LLM to generate
helper functions for solving ARC problems. These tools represent common patterns
and operations needed in ARC tasks.
"""

from typing import List, Dict, Any
import numpy as np

# Helper function implementations
def copy_grid(grid: List[List[int]]) -> List[List[int]]:
    """Create a deep copy of a grid."""
    return [row[:] for row in grid]


def get_grid_dimensions(grid: List[List[int]]) -> tuple[int, int]:
    """Get the height and width of a grid."""
    if not grid:
        return 0, 0
    return len(grid), len(grid[0]) if grid[0] else 0


def create_empty_grid(height: int, width: int, fill_value: int = 0) -> List[List[int]]:
    """Create an empty grid filled with a specific value."""
    return [[fill_value for _ in range(width)] for _ in range(height)]


# Common grid transformation functions
def rotate_grid(grid: List[List[int]], degrees: int) -> List[List[int]]:
    """Rotate a grid by 90, 180, or 270 degrees."""
    if degrees == 90:
        return [[grid[len(grid) - 1 - j][i] for j in range(len(grid))] for i in range(len(grid[0]))]
    elif degrees == 180:
        return [[grid[len(grid) - 1 - i][len(grid[0]) - 1 - j] for j in range(len(grid[0]))] for i in range(len(grid))]
    elif degrees == 270:
        return [[grid[j][len(grid[0]) - 1 - i] for j in range(len(grid))] for i in range(len(grid[0]))]
    else:
        return copy_grid(grid)


def flip_grid(grid: List[List[int]], direction: str) -> List[List[int]]:
    """Flip a grid horizontally or vertically."""
    if direction == "horizontal":
        return [row[::-1] for row in grid]
    elif direction == "vertical":
        return grid[::-1]
    else:
        return copy_grid(grid)


def get_unique_colors(grid: List[List[int]]) -> List[int]:
    """Get all unique colors present in a grid."""
    colors = set()
    for row in grid:
        colors.update(row)
    return sorted(list(colors))


def replace_color(grid: List[List[int]], old_color: int, new_color: int) -> List[List[int]]:
    """Replace all instances of one color with another."""
    return [[new_color if cell == old_color else cell for cell in row] for row in grid]


def count_color_pixels(grid: List[List[int]], color: int) -> int:
    """Count the number of pixels of a specific color."""
    count = 0
    for row in grid:
        count += row.count(color)
    return count


def find_rectangles(grid: List[List[int]], color: int) -> List[Dict[str, int]]:
    """Find all rectangular regions of a specific color."""
    rectangles = []
    height, width = get_grid_dimensions(grid)
    visited = [[False for _ in range(width)] for _ in range(height)]
    
    for i in range(height):
        for j in range(width):
            if grid[i][j] == color and not visited[i][j]:
                # Find rectangle bounds - simplified approach
                min_row, max_row = i, i
                min_col, max_col = j, j
                
                # Expand right
                while max_col + 1 < width and grid[i][max_col + 1] == color:
                    max_col += 1
                
                # Mark as visited
                for r in range(min_row, max_row + 1):
                    for c in range(min_col, max_col + 1):
                        if r < height and c < width:
                            visited[r][c] = True
                
                rectangles.append({
                    'top': min_row,
                    'left': min_col,
                    'bottom': max_row,
                    'right': max_col,
                    'width': max_col - min_col + 1,
                    'height': max_row - min_row + 1
                })
    
    return rectangles


def flood_fill(grid: List[List[int]], start_row: int, start_col: int, new_color: int) -> List[List[int]]:
    """Fill a connected region starting from a point with a new color."""
    result = copy_grid(grid)
    height, width = get_grid_dimensions(grid)
    
    if start_row < 0 or start_row >= height or start_col < 0 or start_col >= width:
        return result
    
    original_color = grid[start_row][start_col]
    if original_color == new_color:
        return result
    
    def fill(row: int, col: int):
        if (row < 0 or row >= height or col < 0 or col >= width or 
            result[row][col] != original_color):
            return
        
        result[row][col] = new_color
        fill(row + 1, col)
        fill(row - 1, col) 
        fill(row, col + 1)
        fill(row, col - 1)
    
    fill(start_row, start_col)
    return result


# Map of function names to implementations
FUNCTION_MAP = {
    'copy_grid': copy_grid,
    'get_grid_dimensions': get_grid_dimensions,
    'create_empty_grid': create_empty_grid,
    'rotate_grid': rotate_grid,
    'flip_grid': flip_grid,
    'get_unique_colors': get_unique_colors,
    'replace_color': replace_color,
    'count_color_pixels': count_color_pixels,
    'find_rectangles': find_rectangles,
    'flood_fill': flood_fill,
}