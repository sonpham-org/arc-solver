import copy

def transform(grid):
    """
    Transforms a grid based on the narrative description.

    The transformation logic is as follows:
    - Any '1' that has an adjacent '1' (horizontally or vertically) becomes a '2'.
    - Any '0' that is adjacent to a '1' becomes a '2'.
    - Cells that are not affected by these rules remain '0'.

    Args:
        grid: A nested list representing the input grid (e.g., [[0, 1, 0], [1, 0, 1]]).

    Returns:
        A nested list representing the transformed grid.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    transformed_grid = [[0 for _ in range(cols)] for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                # A '1' can become a '2' if it has an adjacent '1' or if it's a solitary '1'
                # that is adjacent to a '0' which would then become a '2'.
                # The narrative implies that any '1' becomes a '2' if it's part of any interaction.
                # A solitary '1' will still be a '2' if it's next to a '0' that becomes a '2'.
                # The simplest interpretation is that any '1' becomes a '2' if it has *any* neighbor
                # that is a '1' or a '0' that will become a '2'.
                # This also covers the case where a '1' is next to another '1'.
                transformed_grid[r][c] = 2
            elif grid[r][c] == 0:
                # Check for adjacent '1's
                is_adjacent_to_one = False
                # Check up
                if r > 0 and grid[r - 1][c] == 1:
                    is_adjacent_to_one = True
                # Check down
                if r < rows - 1 and grid[r + 1][c] == 1:
                    is_adjacent_to_one = True
                # Check left
                if c > 0 and grid[r][c - 1] == 1:
                    is_adjacent_to_one = True
                # Check right
                if c < cols - 1 and grid[r][c + 1] == 1:
                    is_adjacent_to_one = True

                if is_adjacent_to_one:
                    transformed_grid[r][c] = 2

    # Second pass to ensure all '1's become '2's if they have an adjacent '1'
    # This handles cases like '110' -> '220' where the first '1' might not
    # have triggered the second '1' to become '2' in the first pass if the logic
    # was strictly about '0's becoming '2's.
    # The narrative implies that '1's adjacent to '1's become '2's.
    # And '0's adjacent to '1's become '2's.
    # Let's refine:
    # If a cell is '1', it becomes '2'.
    # If a cell is '0' and adjacent to a '1', it becomes '2'.

    # Re-initialize for clarity based on refined understanding
    transformed_grid = [[0 for _ in range(cols)] for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                transformed_grid[r][c] = 2
            elif grid[r][c] == 0:
                # Check for adjacent '1's
                is_adjacent_to_one = False
                # Check up
                if r > 0 and grid[r - 1][c] == 1:
                    is_adjacent_to_one = True
                # Check down
                if r < rows - 1 and grid[r + 1][c] == 1:
                    is_adjacent_to_one = True
                # Check left
                if c > 0 and grid[r][c - 1] == 1:
                    is_adjacent_to_one = True
                # Check right
                if c < cols - 1 and grid[r][c + 1] == 1:
                    is_adjacent_to_one = True

                if is_adjacent_to_one:
                    transformed_grid[r][c] = 2

    return transformed_grid