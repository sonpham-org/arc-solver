import copy

def transform(grid):
    """
    Transforms a grid by diffusing non-zero elements into adjacent zero cells.

    The transformation simulates an "energy field" that encourages
    dense forms to spread out into empty spaces. Non-zero cells
    will "leak" into adjacent zero cells, effectively expanding
    their presence.

    Args:
        grid: A nested list representing the input grid.

    Returns:
        A nested list representing the transformed grid.
    """

    if not grid or not grid[0]:
        return []

    rows = len(grid)
    cols = len(grid[0])
    new_grid = copy.deepcopy(grid)

    # Iterate through each cell in the grid
    for r in range(rows):
        for c in range(cols):
            # If the current cell is non-zero, it can potentially spread its value
            if grid[r][c] != 0:
                # Check all 8 adjacent neighbors (including diagonals)
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        # Skip the current cell itself
                        if dr == 0 and dc == 0:
                            continue

                        nr, nc = r + dr, c + dc

                        # Check if the neighbor is within grid boundaries
                        if 0 <= nr < rows and 0 <= nc < cols:
                            # If the neighbor is a zero, set it to the value of the current cell
                            # This simulates the spreading or "leaching" of the non-zero value.
                            if grid[nr][nc] == 0:
                                new_grid[nr][nc] = grid[r][c]
    return new_grid