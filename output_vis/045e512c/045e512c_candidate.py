import copy

def transform(grid):
    """
    Transforms a grid based on the narrative story.

    The transformation rules are derived from the descriptions of how
    different numbers interact and propagate.

    Args:
        grid: A nested list representing the input grid.

    Returns:
        A nested list representing the transformed grid.
    """

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    output_grid = [[0 for _ in range(cols)] for _ in range(rows)]

    def is_valid(r, c):
        return 0 <= r < rows and 0 <= c < cols

    # Helper to apply diffusion from a source with a specific value
    def apply_diffusion(start_r, start_c, source_value, diffusion_value, max_distance):
        queue = [(start_r, start_c, 0)]
        visited = set()

        while queue:
            r, c, dist = queue.pop(0)

            if (r, c) in visited or dist > max_distance:
                continue
            visited.add((r, c))

            # Apply transformation based on the source value and diffusion
            if grid[r][c] == source_value:
                # If the cell is the source itself, it retains its value or becomes the diffusion value
                if dist == 0:
                    output_grid[r][c] = source_value if source_value != 0 else diffusion_value
                else:
                    output_grid[r][c] = diffusion_value
            elif grid[r][c] == 0 and dist > 0:
                # If it's a void and within diffusion range, fill with diffusion value
                output_grid[r][c] = diffusion_value

            # Explore neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if is_valid(nr, nc):
                    queue.append((nr, nc, dist + 1))

    # Helper to apply expansion from a source with a specific value
    def apply_expansion(start_r, start_c, source_value, expansion_value):
        if grid[start_r][start_c] == source_value:
            output_grid[start_r][start_c] = expansion_value # Source becomes the expansion value

        queue = [(start_r, start_c)]
        visited = set()

        while queue:
            r, c = queue.pop(0)

            if (r, c) in visited:
                continue
            visited.add((r, c))

            # Propagate expansion to neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if is_valid(nr, nc):
                    # If the neighbor is a void and within influence, it becomes expansion_value
                    if grid[nr][nc] == 0:
                        output_grid[nr][nc] = expansion_value
                        queue.append((nr, nc))
                    # If the neighbor is the source value, it also becomes expansion_value
                    elif grid[nr][nc] == source_value:
                        output_grid[nr][nc] = expansion_value
                        queue.append((nr, nc))


    # Helper for gravitational pull of specific values
    def apply_gravity(source_value, pull_value):
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == source_value:
                    output_grid[r][c] = pull_value
                    # Propagate pull value to adjacent voids
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if is_valid(nr, nc) and grid[nr][nc] == 0:
                            output_grid[nr][nc] = pull_value

    # Helper for influence of a shape
    def apply_shape_influence(shape_coords, influence_value):
        for r, c in shape_coords:
            if is_valid(r, c):
                output_grid[r][c] = influence_value
                # Influence voids around the shape
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if is_valid(nr, nc) and grid[nr][nc] == 0:
                        output_grid[nr][nc] = influence_value

    # Helper for central nucleus division with mimicry
    def apply_nucleus_division(nucleus_r, nucleus_c, nucleus_value, shape_value):
        if is_valid(nucleus_r, nucleus_c) and grid[nucleus_r][nucleus_c] == nucleus_value:
            output_grid[nucleus_r][nucleus_c] = nucleus_value # Nucleus itself
            # Find surrounding cells that match the shape
            shape_cells = []
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == shape_value:
                        shape_cells.append((r, c))

            # Send tendrils mimicking the shape
            for sr, sc in shape_cells:
                # Simple direction-based mimicry for demonstration
                dr, dc = sr - nucleus_r, sc - nucleus_c
                if dr == 0 and dc != 0: # Horizontal
                    for i in range(1, abs(dc) + 1):
                        target_c = nucleus_c + (dc // abs(dc)) * i
                        if is_valid(nucleus_r, target_c) and grid[nucleus_r][target_c] == 0:
                            output_grid[nucleus_r][target_c] = nucleus_value
                elif dc == 0 and dr != 0: # Vertical
                    for i in range(1, abs(dr) + 1):
                        target_r = nucleus_r + (dr // abs(dr)) * i
                        if is_valid(target_r, nucleus_c) and grid[target_r][nucleus_c] == 0:
                            output_grid[target_r][nucleus_c] = nucleus_value

    # --- Apply transformations based on the stories ---

    # Story 1: Echo of 8s and a 3
    # '3' acts as a source of '3's (energy dispersal)
    # '8's reverberate outwards, creating '3's in voids up to a certain distance
    # '2's receive a parallel echo of '3's
    eight_coords = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 8]
    three_coords = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 3]
    two_coords = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 2]

    for r, c in three_coords:
        apply_diffusion(r, c, 3, 3, 1) # 3 itself becomes source, spreads 3s

    for r, c in eight_coords:
        apply_diffusion(r, c, 0, 3, 1) # 8s create 3s in adjacent voids

    for r, c in two_coords:
        # '2's receive a faint, parallel echo. This implies if a '2' is next to
        # a cell that becomes a '3' due to diffusion, the '2' also gets a '3'.
        # For simplicity, we'll apply diffusion from '3' sources and check if '2' is nearby
        for tr, tc in three_coords:
            if abs(r - tr) + abs(c - tc) <= 1 and grid[r][c] == 2: # If '2' is near a '3'
                output_grid[r][c] = 3 # It receives the echo


    # Story 2: Solitary 4, cross of 1s, and 2s on periphery
    # '4' radiates light ('4's)
    # '1's (cross) have gravitational pull, bending light
    # '2's orbit, influencing boundaries
    four_coords = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 4]
    ones_coords = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]
    twos_coords = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 2]

    # Find the cross shape of 1s
    cross_cells = []
    if len(ones_coords) > 0:
        # Assume the first '1' found is part of the cross, and find neighbors
        center_r, center_c = ones_coords[0]
        cross_cells.append((center_r, center_c))
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = center_r + dr, center_c + dc
            if is_valid(nr, nc) and grid[nr][nc] == 1:
                cross_cells.append((nr, nc))
        # If more than one '1', try to find a central one and expand
        if len(cross_cells) > 1:
            # Heuristic: Find the '1' with the most '1' neighbors
            best_center = None
            max_neighbors = -1
            for r1, c1 in ones_coords:
                neighbor_count = 0
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r1 + dr, c1 + dc
                    if is_valid(nr, nc) and grid[nr][nc] == 1:
                        neighbor_count += 1
                if neighbor_count > max_neighbors:
                    max_neighbors = neighbor_count
                    best_center = (r1, c1)
            if best_center:
                cross_cells = [best_center]
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = best_center[0] + dr, best_center[1] + dc
                    if is_valid(nr, nc) and grid[nr][nc] == 1:
                        cross_cells.append((nr, nc))

    # Apply gravitational pull of the cross
    for r, c in cross_cells:
        apply_gravity(1, 1)

    # '4' radiates light ('4's)
    for r, c in four_coords:
        apply_expansion(r, c, 4, 4)

    # '2's influence outer boundaries and also get influenced by the '4' and '1's
    # This is complex: '2's on the periphery witness and subtly reshape.
    # Let's assume they create a '2' boundary if they are on the edge and a '4' or '1' is present.
    for r, c in twos_coords:
        if grid[r][c] == 2:
            is_periphery = False
            if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                is_periphery = True

            has_neighbor_4_or_1 = False
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nr, nc = r + dr, c + dc
                if is_valid(nr, nc) and (grid[nr][nc] == 4 or grid[nr][nc] == 1):
                    has_neighbor_4_or_1 = True
                    break

            if is_periphery and has_neighbor_4_or_1:
                output_grid[r][c] = 2 # Retain '2' if on periphery and influenced

            # Also, '2's can become '4' if adjacent to a '4' (light pushing back darkness)
            for fr, fc in four_coords:
                if abs(r - fr) + abs(c - fc) <= 1 and grid[r][c] == 2:
                    output_grid[r][c] = 4 # '2' becomes '4' near '4'

    # Ensure the '1' cross remains as '1's and influences voids as '1's
    for r, c in ones_coords:
        output_grid[r][c] = 1 # Ensure the '1's themselves are present

    # Story 3: 6s, 5-shaped silhouette, 1 nucleus
    # '6's coalesce into primordial gases
    # '5'-shaped figure acts as a gravitational well, influencing '1'
    # '1' nucleus divides, sending tendrils mimicking the '5'
    # '6's become raw material
    six_coords = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 6]
    five_coords = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 5]
    one_nucleus_coords = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]

    # Find the '5' shape
    five_shape_cells = []
    if len(five_coords) > 0:
        # A simple heuristic: assume the '5' is contiguous.
        # We'll start with the first '5' and do a BFS/DFS to find connected '5's.
        # For simplicity, we'll just collect all '5's if found.
        five_shape_cells = five_coords

    # Find the '1' nucleus
    nucleus_cell = None
    if len(one_nucleus_coords) > 0:
        # Assume the first '1' is the nucleus
        nucleus_cell = one_nucleus_coords[0]

    # '6's become raw material, potentially coalescing or filling voids
    for r, c in six_coords:
        # If a '6' is near the '5' shape or nucleus, it might be incorporated
        is_near_influence = False
        if nucleus_cell and abs(r - nucleus_cell[0]) + abs(c - nucleus_cell[1]) <= 2:
            is_near_influence = True
        for fr, fc in five_shape_cells:
            if abs(r - fr) + abs(c - fc) <= 2:
                is_near_influence = True
                break

        if is_near_influence:
            output_grid[r][c] = 6 # '6' remains as raw material near influence
        else:
            output_grid[r][c] = 0 # '6' dissipates if not near influence

    # '5' shape influences surrounding voids with '5'
    for r, c in five_shape_cells:
        output_grid[r][c] = 5 # The '5' shape itself
        apply_shape_influence([(r, c)], 5)

    # '1' nucleus divides and sends tendrils
    if nucleus_cell and five_shape_cells:
        apply_nucleus_division(nucleus_cell[0], nucleus_cell[1], 1, 5)
        output_grid[nucleus_cell[0]][nucleus_cell[1]] = 1 # Ensure nucleus is '1'

    # Fill any remaining voids with '0'
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and output_grid[r][c] == 0:
                # If the original cell had a value and it wasn't transformed by any rule, keep it.
                # This is a fallback for unhandled cases or cells not part of transformations.
                # However, the narrative implies transformations should fill or change.
                # For now, let's assume 0 means it's a void unless explicitly changed.
                pass
            elif grid[r][c] == 0 and output_grid[r][c] == 0:
                output_grid[r][c] = 0 # Explicitly set voids to 0

    # --- Final adjustments and consolidations based on narrative interpretations ---

    # Consolidate '2's from story 1 and story 2.
    # Story 1: '2's receive an echo of '3's.
    # Story 2: '2's on periphery witness and reshape, can become '4' near '4'.
    # If a cell was a '2' and received a '3' echo, it should be '3'.
    # If a cell was a '2' and became a '4', it should be '4'.
    # If a cell was a '2' and remained a '2' on the periphery, it should be '2'.
    # This implies a priority. '4' > '3' > '2' (if it remained a '2').

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                if output_grid[r][c] == 4:
                    pass # Already set to 4
                elif output_grid[r][c] == 3:
                    pass # Already set to 3 (from story 1 echo)
                elif r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                    # If it's on the periphery and not transformed to 4 or 3, it remains 2.
                    # This check is implicitly handled by the initialization to 0 and then filling.
                    # However, to be explicit:
                    output_grid[r][c] = 2
                else:
                    # If it was a '2' and not on the periphery, and not transformed, it becomes 0.
                    output_grid[r][c] = 0


    return output_grid

if __name__ == '__main__':
    # Example usage (for testing purposes)

    # Story 1 Example
    grid1 = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 8, 8, 8, 0, 0, 0, 0, 0],
        [0, 8, 3, 8, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 0, 0]
    ]
    transformed_grid1 = transform(grid1)
    print("--- Story 1 ---")
    for row in transformed_grid1:
        print(row)

    # Expected (interpretation):
    # 0 0 0 0 0 0 0 0 0
    # 0 8 3 3 0 0 0 0 0
    # 0 3 3 3 0 0 0 0 0
    # 0 3 3 3 0 0 0 0 0
    # 0 3 0 0 0 0 0 0 0
    # 0 3 0 0 0 0 0 0 0

    print("\n")

    # Story 2 Example
    grid2 = [
        [2, 2, 2, 2, 2],
        [2, 0, 0, 0, 2],
        [2, 0, 1, 0, 2],
        [2, 0, 1, 0, 2],
        [2, 0, 4, 0, 2],
        [2, 0, 1, 0, 2],
        [2, 0, 1, 0, 2],
        [2, 2, 2, 2, 2]
    ]
    transformed_grid2 = transform(grid2)
    print("--- Story 2 ---")
    for row in transformed_grid2:
        print(row)

    # Expected (interpretation):
    # 2 2 2 2 2
    # 2 4 4 4 2
    # 2 4 1 4 2
    # 2 4 1 4 2
    # 2 4 4 4 2
    # 2 4 1 4 2
    # 2 4 1 4 2
    # 2 2 2 2 2

    print("\n")

    # Story 3 Example
    grid3 = [
        [0, 0, 6, 0, 0],
        [0, 6, 5, 6, 0],
        [0, 6, 1, 6, 0],
        [0, 6, 5, 6, 0],
        [0, 0, 6, 0, 0]
    ]
    transformed_grid3 = transform(grid3)
    print("--- Story 3 ---")
    for row in transformed_grid3:
        print(row)

    # Expected (interpretation):
    # 0 0 6 0 0
    # 0 6 5 6 0
    # 0 6 1 6 0
    # 0 6 5 6 0
    # 0 0 6 0 0

    # More complex Story 3 example to test nucleus division
    grid3_complex = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 6, 0, 0, 0, 6, 0],
        [0, 0, 5, 1, 5, 0, 0],
        [0, 6, 0, 0, 0, 6, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ]
    transformed_grid3_complex = transform(grid3_complex)
    print("--- Story 3 Complex ---")
    for row in transformed_grid3_complex:
        print(row)

    # Expected (interpretation of complex 3):
    # The '1' nucleus should send tendrils to mimic the '5' shape.
    # If the '5' is above and below the '1', tendrils might go up/down.
    # If '5' is to the sides, tendrils might go left/right.
    # 0 0 0 0 0 0 0
    # 0 6 0 1 0 6 0
    # 0 1 5 1 5 1 0
    # 0 6 0 1 0 6 0
    # 0 0 0 0 0 0 0