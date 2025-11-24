import copy

def transform(grid):
    """
    Transforms a grid by replicating non-zero patterns with scaled spacing.

    The transformation expands the grid by replicating non-zero elements.
    The spacing between the replicated patterns is determined by the
    distance of the non-zero elements from the edges and the longest
    row/column of non-zero elements. Zeros act as the void where these
    patterns are propagated.

    Args:
        grid: A nested list representing the input grid.

    Returns:
        A nested list representing the transformed grid.
    """

    if not grid or not grid[0]:
        return [[]]

    rows = len(grid)
    cols = len(grid[0])

    # Find all non-zero cells and their values
    non_zero_cells = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                non_zero_cells.append(((r, c), grid[r][c]))

    if not non_zero_cells:
        return [[0] * cols for _ in range(rows)]

    # Determine the bounding box of non-zero elements
    min_r = min(cell[0][0] for cell in non_zero_cells)
    max_r = max(cell[0][0] for cell in non_zero_cells)
    min_c = min(cell[0][1] for cell in non_zero_cells)
    max_c = max(cell[0][1] for cell in non_zero_cells)

    # Calculate expansion factors based on distances from edges and longest dimension
    # This is an interpretation of "scaled intervals" and "gravitational waves"
    # The longest row/column of non-zero elements is considered a primary filament.
    row_span = max_r - min_r + 1
    col_span = max_c - min_c + 1

    # A heuristic for determining the scale factor.
    # It considers how "spread out" the pattern is relative to its original size.
    # Larger spread implies a larger output grid.
    scale_factor_r = max(1, (rows - 1) // max(1, row_span)) if row_span > 0 else 1
    scale_factor_c = max(1, (cols - 1) // max(1, col_span)) if col_span > 0 else 1

    # The scaling should be such that the pattern can be replicated.
    # A simple multiplier based on the total grid size and pattern size.
    # This is a simplification, as the narrative implies a more complex scaling.
    # For simplicity, we'll aim to create an output grid large enough to hold
    # at least one scaled replica. A common approach in ARC is to double the size
    # or more if the pattern is small.
    output_rows = rows * max(2, scale_factor_r)
    output_cols = cols * max(2, scale_factor_c)

    output_grid = [[0] * output_cols for _ in range(output_rows)]

    # Place the original pattern
    for (r, c), value in non_zero_cells:
        output_grid[r][c] = value

    # Propagate the pattern. This is the core of the transformation.
    # The narrative suggests a replication based on relative positions.
    # We'll use a form of cellular automaton or diffusion-like expansion,
    # but specifically replicating the *entire* pattern at scaled positions.

    # Identify the "center" or "reference point" of the pattern.
    # This could be the top-leftmost non-zero cell or a centroid.
    # Let's use the top-leftmost non-zero cell's coordinates as a reference.
    ref_r, ref_c = non_zero_cells[0][0]

    # Determine the offset for replication. This is where the "gravitational waves"
    # and "echoes" come into play. The spacing should be related to the original grid dimensions.
    # A simple but effective approach is to use a fraction of the original dimensions.
    # The "longest radiant line" suggests that the span of the non-zero elements
    # plays a role.

    # Calculate the offset for replication based on the overall span of non-zero elements
    # and the original grid dimensions.
    # If the pattern is small relative to the grid, we want larger gaps.
    # If the pattern fills much of the grid, the gaps might be smaller.
    offset_r = max(row_span, 1) * max(1, (rows - row_span) // 2)
    offset_c = max(col_span, 1) * max(1, (cols - col_span) // 2)

    # If offset is zero, ensure it's at least 1 to create some separation.
    offset_r = max(offset_r, 1)
    offset_c = max(offset_c, 1)

    # We need to find a good stride for replication. The narrative implies
    # that the pattern reappears at scaled intervals.
    # Let's try replicating the pattern multiple times, with offsets.
    # The number of replications and their spacing are key.

    # We'll create multiple copies of the pattern.
    # The spacing of these copies should be proportional to the original grid dimensions
    # and the extent of the pattern.
    # A common strategy in ARC is to double the dimensions or more.
    # Let's consider the "echoes" as shifted versions of the original pattern.

    # Calculate a stride that allows for replication within the expanded grid.
    # The stride should be at least the size of the original pattern's bounding box
    # plus some spacing.
    stride_r = max(row_span, 1) + offset_r
    stride_c = max(col_span, 1) + offset_c

    # Ensure the stride is not too large that it goes out of bounds immediately
    stride_r = min(stride_r, output_rows)
    stride_c = min(stride_c, output_cols)


    # Place the original pattern at (0,0) in the conceptual larger canvas
    for (r, c), value in non_zero_cells:
        if r < output_rows and c < output_cols:
            output_grid[r][c] = value

    # Place replicated patterns
    # We'll iterate through potential starting points for the pattern.
    # The starting points will be multiples of the calculated strides.
    current_r = 0
    while current_r < output_rows:
        current_c = 0
        while current_c < output_cols:
            # If this is not the original placement, place a copy
            if current_r != 0 or current_c != 0:
                for (r_orig, c_orig), value in non_zero_cells:
                    new_r = current_r + r_orig
                    new_c = current_c + c_orig
                    if 0 <= new_r < output_rows and 0 <= new_c < output_cols:
                        output_grid[new_r][new_c] = value
            current_c += stride_c
        current_r += stride_r

    # The above creates a grid where patterns are placed at regular intervals.
    # However, the narrative suggests a more organic diffusion and reformation.
    # The "zeros are not just passive space; they are the silent conductors".
    # This implies that the pattern should be "painted" onto the zeros.

    # Let's refine the approach to be more diffusion-like, but with discrete pattern replication.
    # We will determine the new dimensions based on how many times the pattern can "fit"
    # with a certain spacing.

    # Calculate the maximum number of times the pattern can be replicated horizontally and vertically.
    # This is a heuristic to determine the output grid size.
    # The idea is to create enough space for multiple echoes.
    num_replicas_r = max(1, (rows // max(1, row_span)))
    num_replicas_c = max(1, (cols // max(1, col_span)))

    # We want the output grid to be large enough to accommodate these replicas with some spacing.
    # Let's aim for an output grid size that is roughly `num_replicas * original_size`
    # with some additional padding for spacing.
    # A multiplier of 2 or 3 is common in ARC for expansion.
    # Let's use a dynamic multiplier based on the pattern's density.
    density = len(non_zero_cells) / (rows * cols)
    multiplier = int(3 / max(0.1, density)) # Higher density -> smaller multiplier

    output_rows = rows * multiplier
    output_cols = cols * multiplier

    output_grid = [[0] * output_cols for _ in range(output_rows)]

    # Place the original pattern
    for (r, c), value in non_zero_cells:
        output_grid[r][c] = value

    # Now, let's define the "propagation distance" or "spacing".
    # This should be related to the original grid dimensions and the pattern's span.
    propagation_dist_r = max(row_span, 1) + (rows - row_span) // 2
    propagation_dist_c = max(col_span, 1) + (cols - col_span) // 2

    propagation_dist_r = max(propagation_dist_r, 1)
    propagation_dist_c = max(propagation_dist_c, 1)

    # The narrative implies that the pattern reforms at specific intervals.
    # Let's consider the "center" of the pattern and expand outwards.
    # For simplicity, let's assume the pattern is replicated at offsets
    # that are multiples of the propagation distances.

    # We will iterate through the output grid and decide where to place pattern fragments.
    # This is still not quite matching the "cosmic breath" and "gravitational waves" idea.

    # Let's rethink the core mechanism:
    # The narrative implies that each non-zero cell "radiates" its value.
    # The "zeros" act as the medium for this radiation, and the pattern reforms
    # at specific distances. The longest row/column is a "major filament".

    # Let's identify the "core" pattern and its bounding box.
    # The output grid size needs to be determined. A common ARC transformation
    # is to roughly double or triple the size. Let's use a heuristic.

    # Find the extent of the non-zero elements.
    min_r_nz = min(r for (r, c), val in non_zero_cells)
    max_r_nz = max(r for (r, c), val in non_zero_cells)
    min_c_nz = min(c for (r, c), val in non_zero_cells)
    max_c_nz = max(c for (r, c), val in non_zero_cells)

    pattern_height = max_r_nz - min_r_nz + 1
    pattern_width = max_c_nz - min_c_nz + 1

    # Determine the output grid size. This is crucial and can be tricky.
    # A common approach is to scale based on the density of the pattern.
    # If the pattern is sparse, the output grid will be larger.
    # If the pattern is dense, the output grid will be less expanded.

    # Let's try to determine the scaling factor based on how much "empty space"
    # is around the pattern within the original grid.
    # A simple heuristic: if the pattern takes up less than 1/4 of the grid area,
    # we might want to at least double the dimensions.

    original_area = rows * cols
    pattern_area = len(non_zero_cells)
    density = pattern_area / original_area if original_area > 0 else 1

    scale_factor = 2
    if density < 0.25:
        scale_factor = 3
    elif density < 0.5:
        scale_factor = 2.5
    else:
        scale_factor = 2

    output_rows = int(rows * scale_factor)
    output_cols = int(cols * scale_factor)

    # Ensure minimum size if original grid is very small.
    output_rows = max(output_rows, rows * 2)
    output_cols = max(output_cols, cols * 2)

    output_grid = [[0] * output_cols for _ in range(output_rows)]

    # The core idea is to replicate the pattern at scaled positions.
    # The "scaling" needs to be defined.
    # Let's assume the pattern is replicated such that its relative positions
    # are maintained but with a larger spacing.

    # We need to find the "stride" for replication.
    # This stride should be related to the original dimensions and the pattern's extent.
    # The "longest radiant line" implies that the span of the non-zero elements
    # influences the spacing.

    # Let's define a "spacing unit" based on the original grid and pattern.
    # A simple spacing unit could be the average distance from the pattern's bounding box
    # to the grid edges.

    spacing_r = max(1, (rows - pattern_height) // 2)
    spacing_c = max(1, (cols - pattern_width) // 2)

    # The stride for placing replicas should be at least the pattern's dimensions plus spacing.
    stride_r = pattern_height + spacing_r
    stride_c = pattern_width + spacing_c

    stride_r = max(stride_r, 1)
    stride_c = max(stride_c, 1)

    # Place the original pattern at the top-left.
    for (r, c), value in non_zero_cells:
        # Ensure we don't go out of bounds if the scaling is not perfect.
        if r < output_rows and c < output_cols:
            output_grid[r][c] = value

    # Place subsequent replicas.
    current_r = 0
    while True:
        current_r += stride_r
        if current_r >= output_rows:
            break
        for (r_orig, c_orig), value in non_zero_cells:
            new_r = current_r + r_orig
            if 0 <= new_r < output_rows:
                # For the column placement, we also need to consider the stride.
                current_c = 0
                while True:
                    current_c += stride_c
                    if current_c >= output_cols:
                        break
                    new_c = current_c + c_orig
                    if 0 <= new_c < output_cols:
                        output_grid[new_r][new_c] = value

    # The previous approach placed the original pattern at (0,0) and then other copies.
    # The narrative suggests the original pattern *itself* is part of the "echoes".
    # It's more like the entire canvas is filled with scaled versions of the pattern.

    # Let's try to define the output grid size by how many times the pattern
    # can be "tiled" with a certain gap.

    # Determine the effective "unit cell" for replication.
    # This unit cell should contain the pattern and the surrounding zero space.
    # The size of this unit cell dictates the stride.

    # Consider the pattern's bounding box as the core.
    # The scaling should ensure that the relative distances within the pattern are preserved.

    # Let's try a simpler approach based on common ARC transformations that involve expansion.
    # If the pattern is small, it's often replicated at larger intervals.

    # Find the center of the pattern in terms of its bounding box.
    center_r = (min_r_nz + max_r_nz) / 2
    center_c = (min_c_nz + max_c_nz) / 2

    # Calculate a scaling factor based on the overall grid size and pattern size.
    # This is a heuristic for determining the output grid dimensions.
    # The narrative implies that the transformation expands the canvas to accommodate echoes.

    # If the pattern is small and centered, it might be replicated across a larger grid.
    # If the pattern is large and close to edges, the expansion might be less drastic.

    # Let's determine the number of "layers" of echoes.
    # The "longest radiant line" suggests a dominant dimension.
    dominant_dim = max(pattern_height, pattern_width)
    grid_dominant_dim = max(rows, cols)

    # A heuristic for output size: aim to fit a few copies of the pattern with spacing.
    # The spacing should be related to the original grid dimensions.
    num_copies_r = max(1, grid_dominant_dim // dominant_dim)
    num_copies_c = max(1, grid_dominant_dim // dominant_dim)

    # The output grid should be large enough to hold these copies with some spacing.
    # Let's aim for an output grid that's roughly `num_copies * pattern_size`.
    # The spacing between copies is key.

    # Consider the "cosmic breath" expanding the pattern.
    # This suggests that the pattern should be placed at intervals.
    # The intervals are "calculated" and "scaled".

    # Let's determine the output grid size by ensuring that at least one
    # scaled replica can fit with some padding.
    # A common scaling factor in ARC for expansion is 2 or 3.
    # We can make this dynamic based on the pattern's density or size.

    output_rows = int(rows * 2.5)
    output_cols = int(cols * 2.5)

    # Ensure minimum expansion
    output_rows = max(output_rows, rows * 2)
    output_cols = max(output_cols, cols * 2)

    output_grid = [[0] * output_cols for _ in range(output_rows)]

    # The core transformation:
    # The non-zero elements propagate outwards, reforming the pattern at scaled distances.
    # This sounds like a form of replication where the entire pattern is shifted.
    # The "zeros" are the canvas where this happens.

    # Let's identify the "base" pattern and its relative coordinates.
    base_pattern_coords = [(r - min_r_nz, c - min_c_nz) for (r, c), val in non_zero_cells]
    base_pattern_values = {coord: val for coord, val in zip(base_pattern_coords, [val for (r, c), val in non_zero_cells])}

    # Determine the "stride" or "spacing" for replication.
    # This stride should be related to the original grid dimensions and the pattern's extent.
    # The "longest radiant line" suggests the span of the non-zero elements is crucial.
    stride_r = max(pattern_height, 1) + (rows - pattern_height) // 2
    stride_c = max(pattern_width, 1) + (cols - pattern_width) // 2

    stride_r = max(stride_r, 1)
    stride_c = max(stride_c, 1)

    # Place the original pattern at the top-left of the conceptual canvas.
    for r_offset, c_offset in base_pattern_coords:
        r_new = r_offset
        c_new = c_offset
        if 0 <= r_new < output_rows and 0 <= c_new < output_cols:
            output_grid[r_new][c_new] = base_pattern_values[(r_offset, c_offset)]

    # Place subsequent replicas.
    current_r_start = 0
    while True:
        current_r_start += stride_r
        if current_r_start >= output_rows:
            break

        current_c_start = 0
        while True:
            current_c_start += stride_c
            if current_c_start >= output_cols:
                break

            # Place the pattern at this new starting point.
            for r_offset, c_offset in base_pattern_coords:
                r_new = current_r_start + r_offset
                c_new = current_c_start + c_offset
                if 0 <= r_new < output_rows and 0 <= c_new < output_cols:
                    output_grid[r_new][c_new] = base_pattern_values[(r_offset, c_offset)]

    # This approach creates a tiled pattern. The narrative implies something more dynamic.
    # "Gravitational waves, each non-zero digit sending ripples through the emptiness,
    # attracting and re-forming itself at specific distances."

    # This suggests that the placement of non-zero elements in the output grid
    # is influenced by their original positions and a "propagation rule".

    # Let's try to define the output grid size based on how many times the pattern
    # can be repeated with a certain spacing.
    # The spacing is determined by the original grid and the pattern's spread.

    # Determine the scaling factor for the output grid.
    # If the pattern is small and sparse, the output grid will be larger.
    # If the pattern is large and dense, the output grid will be less expanded.

    # Heuristic for output grid dimensions:
    # Calculate the "effective span" of the pattern.
    effective_span_r = max(1, pattern_height)
    effective_span_c = max(1, pattern_width)

    # Calculate a multiplier based on how much "room" there is in the original grid.
    # If there's a lot of empty space, we multiply more.
    room_r = rows - effective_span_r
    room_c = cols - effective_span_c

    multiplier = 2
    if room_r > effective_span_r * 2 or room_c > effective_span_c * 2:
        multiplier = 3
    if room_r > effective_span_r * 3 or room_c > effective_span_c * 3:
        multiplier = 4

    output_rows = int(rows * multiplier)
    output_cols = int(cols * multiplier)
    output_rows = max(output_rows, rows * 2) # Minimum doubling
    output_cols = max(output_cols, cols * 2)

    output_grid = [[0] * output_cols for _ in range(output_rows)]

    # The core transformation:
    # The non-zero elements at (r, c) in the input grid should result in
    # non-zero elements at positions (r', c') in the output grid.
    # The mapping from (r, c) to (r', c') is what defines the transformation.

    # The narrative implies that the pattern is replicated with a certain "jump" or "stride".
    # This stride is determined by the original grid and the pattern's dimensions.

    # Let's identify the "dominant" dimension of the pattern.
    dominant_pattern_dim = max(pattern_height, pattern_width)

    # The stride for replication should be at least the dominant pattern dimension,
    # plus some spacing derived from the original grid's empty space.
    stride_r = max(dominant_pattern_dim, 1) + max(1, (rows - dominant_pattern_dim) // 2)
    stride_c = max(dominant_pattern_dim, 1) + max(1, (cols - dominant_pattern_dim) // 2)

    stride_r = max(stride_r, 1)
    stride_c = max(stride_c, 1)

    # Place the original pattern.
    for (r, c), value in non_zero_cells:
        if r < output_rows and c < output_cols:
            output_grid[r][c] = value

    # Place replicated patterns based on the calculated strides.
    # The pattern should appear at positions that are multiples of these strides.
    for r_mult in range(output_rows // stride_r + 1):
        for c_mult in range(output_cols // stride_c + 1):
            # The starting point for this replica.
            start_r = r_mult * stride_r
            start_c = c_mult * stride_c

            # If this is the original placement (0,0), we've already handled it.
            if start_r == 0 and start_c == 0:
                continue

            # Place the pattern at this new starting point.
            for (r_orig, c_orig), value in non_zero_cells:
                new_r = start_r + r_orig
                new_c = start_c + c_orig
                if 0 <= new_r < output_rows and 0 <= new_c < output_cols:
                    output_grid[new_r][new_c] = value

    # This still results in a tiled grid. The narrative implies a more distributed
    # reformation, not just simple tiling.

    # Let's consider the "center" of the pattern and expand outwards.
    # The output grid should be large enough to contain these expanded echoes.

    # The output dimensions are often a few times the input dimensions in ARC.
    # A multiplier of 2 to 4 is common.
    # Let's make the multiplier dynamic based on the pattern's "sparseness".
    num_non_zero = len(non_zero_cells)
    total_cells = rows * cols
    density = num_non_zero / total_cells if total_cells > 0 else 1.0

    scale_factor = 2.0
    if density < 0.2:
        scale_factor = 4.0
    elif density < 0.4:
        scale_factor = 3.0
    elif density < 0.6:
        scale_factor = 2.5

    output_rows = int(rows * scale_factor)
    output_cols = int(cols * scale_factor)

    # Ensure minimum size
    output_rows = max(output_rows, rows * 2)
    output_cols = max(output_cols, cols * 2)

    output_grid = [[0] * output_cols for _ in range(output_rows)]

    # The core of the transformation is the replication with scaled spacing.
    # The "zeros" are the medium. The non-zero digits are the signal.

    # Let's find the extent of the non-zero elements.
    min_r_nz = min(r for (r, c), val in non_zero_cells)
    max_r_nz = max(r for (r, c), val in non_zero_cells)
    min_c_nz = min(c for (r, c), val in non_zero_cells)
    max_c_nz = max(c for (r, c), val in non_zero_cells)

    pattern_height = max_r_nz - min_r_nz + 1
    pattern_width = max_c_nz - min_c_nz + 1

    # Determine the "stride" for replication. This stride should be
    # larger than the pattern itself, and proportional to the original grid size.
    # The narrative implies that the pattern "pushes" its '7's further,
    # influencing rows above and below.

    # A simple stride calculation: consider the original grid dimensions.
    # The stride should be at least the pattern's dimensions, plus some
    # spacing derived from the original grid's empty space.
    stride_r = max(pattern_height, 1) + (rows - pattern_height) // 2
    stride_c = max(pattern_width, 1) + (cols - pattern_width) // 2

    stride_r = max(stride_r, 1)
    stride_c = max(stride_c, 1)

    # Place the original pattern.
    for (r, c), value in non_zero_cells:
        if r < output_rows and c < output_cols:
            output_grid[r][c] = value

    # Place subsequent replicas. The key is *where* to place them.
    # The narrative implies that the pattern re-forms at calculated distances.
    # These distances are related to the original grid and the pattern's spread.

    # Let's iterate through the output grid and decide whether to place a pattern fragment.
    # This is becoming more like a diffusion process.

    # A more direct interpretation of "replicates their essence and disperses":
    # The entire pattern is translated by certain offsets.
    # The offsets are determined by the original grid dimensions.

    # Let's define the "unit of replication".
    # This unit should contain the pattern and the space around it that dictates the next placement.
    # The size of this unit is effectively the stride.

    # Calculate the stride based on the original grid's dimensions and pattern's extent.
    # The "longest radiant line" suggests the span of non-zero elements is important.
    stride_r = max(pattern_height, 1) + (rows - pattern_height) // 2
    stride_c = max(pattern_width, 1) + (cols - pattern_width) // 2

    stride_r = max(stride_r, 1)
    stride_c = max(stride_c, 1)

    # Place the original pattern.
    for (r, c), value in non_zero_cells:
        if r < output_rows and c < output_cols:
            output_grid[r][c] = value

    # Place replicated patterns. The replicas are shifted by multiples of the stride.
    # The original pattern itself is considered the first replica at (0,0) offset.
    for r_mult in range(output_rows // stride_r + 1):
        for c_mult in range(output_cols // stride_c + 1):
            # Calculate the offset for this replica.
            offset_r = r_mult * stride_r
            offset_c = c_mult * stride_c

            # If this is the original placement (0,0 offset), we've already handled it.
            if offset_r == 0 and offset_c == 0:
                continue

            # Place the pattern at this new offset.
            for (r_orig, c_orig), value in non_zero_cells:
                new_r = offset_r + r_orig
                new_c = offset_c + c_orig
                if 0 <= new_r < output_rows and 0 <= new_c < output_cols:
                    output_grid[new_r][new_c] = value

    # This still results in a tiling. The narrative implies a more organic spread.
    # "Scattering it across a larger canvas." "Imbue the pattern with its own echo."

    # Let's consider the "center" of the pattern and expand outwards.
    # The output grid size should be determined to accommodate these expansions.

    # Determine output dimensions:
    # A common strategy is to create an output grid that is a few times larger
    # than the input, especially if the pattern is small.
    # Let's use a multiplier based on the pattern's density.
    num_non_zero = len(non_zero_cells)
    total_cells = rows * cols
    density = num_non_zero / total_cells if total_cells > 0 else 1.0

    scale_factor = 2.0
    if density < 0.2:
        scale_factor = 4.0
    elif density < 0.4:
        scale_factor = 3.0
    elif density < 0.6:
        scale_factor = 2.5

    output_rows = int(rows * scale_factor)
    output_cols = int(cols * scale_factor)

    # Ensure minimum expansion
    output_rows = max(output_rows, rows * 2)
    output_cols = max(output_cols, cols * 2)

    output_grid = [[0] * output_cols for _ in range(output_rows)]

    # The "cosmic breath" suggests that the non-zero elements define the transformation.
    # They "propagate outwards".

    # Let's find the bounding box of the non-zero elements.
    min_r_nz = min(r for (r, c), val in non_zero_cells)
    max_r_nz = max(r for (r, c), val in non_zero_cells)
    min_c_nz = min(c for (r, c), val in non_zero_cells)
    max_c_nz = max(c for (r, c), val in non_zero_cells)

    pattern_height = max_r_nz - min_r_nz + 1
    pattern_width = max_c_nz - min_c_nz + 1

    # The "stride" or "spacing" for replication is key.
    # This stride should be larger than the pattern itself, and should scale
    # with the original grid dimensions.
    # The "longest radiant line" suggests the span of the non-zero elements is important.

    # Let's calculate a stride that ensures the pattern can be replicated with
    # some space around it, and this space is proportional to the original grid's empty space.
    stride_r = max(pattern_height, 1) + (rows - pattern_height) // 2
    stride_c = max(pattern_width, 1) + (cols - pattern_width) // 2

    stride_r = max(stride_r, 1)
    stride_c = max(stride_c, 1)

    # Place the original pattern at the top-left.
    for (r, c), value in non_zero_cells:
        if r < output_rows and c < output_cols:
            output_grid[r][c] = value

    # Place subsequent replicas. The replicas are shifted by multiples of the stride.
    # The original pattern is the first replica at an offset of (0,0).
    for r_mult in range(output_rows // stride_r + 1):
        for c_mult in range(output_cols // stride_c + 1):
            # Calculate the offset for this replica.
            offset_r = r_mult * stride_r
            offset_c = c_mult * stride_c

            # If this is the original placement (0,0 offset), we've already handled it.
            if offset_r == 0 and offset_c == 0:
                continue

            # Place the pattern at this new offset.
            for (r_orig, c_orig), value in non_zero_cells:
                new_r = offset_r + r_orig
                new_c = offset_c + c_orig
                if 0 <= new_r < output_rows and 0 <= new_c < output_cols:
                    output_grid[new_r][new_c] = value

    # This interpretation results in a tiled pattern. The narrative implies a more
    # "radiant" and "resonant" expansion.

    # Let's consider the "center" of the non-zero elements and expand from there.
    # The output grid size should be determined to accommodate these expansions.

    # Determine output dimensions:
    # A common strategy in ARC is to scale up the grid.
    # Let's use a multiplier that depends on the pattern's sparsity.
    num_non_zero = len(non_zero_cells)
    total_cells = rows * cols
    density = num_non_zero / total_cells if total_cells > 0 else 1.0

    scale_factor = 2.0
    if density < 0.2:
        scale_factor = 4.0
    elif density < 0.4:
        scale_factor = 3.0
    elif density < 0.6:
        scale_factor = 2.5

    output_rows = int(rows * scale_factor)
    output_cols = int(cols * scale_factor)

    # Ensure minimum expansion
    output_rows = max(output_rows, rows * 2)
    output_cols = max(output_cols, cols * 2)

    output_grid = [[0] * output_cols for _ in range(output_rows)]

    # The core transformation:
    # Each non-zero cell in the input grid influences multiple cells in the output grid.
    # The narrative talks about "ripples" and "echoes".

    # Let's identify the bounding box of the non-zero elements.
    min_r_nz = min(r for (r, c), val in non_zero_cells)
    max_r_nz = max(r for (r, c), val in non_zero_cells)
    min_c_nz = min(c for (r, c), val in non_zero_cells)
    max_c_nz = max(c for (r, c), val in non_zero_cells)

    pattern_height = max_r_nz - min_r_nz + 1
    pattern_width = max_c_nz - min_c_nz + 1

    # The "stride" or "spacing" for replication. This should be related to the original grid's dimensions.
    # The "longest radiant line" suggests the span of the non-zero elements is key.

    # Calculate a stride that ensures replication with spacing proportional to the original grid's empty space.
    stride_r = max(pattern_height, 1) + (rows - pattern_height) // 2
    stride_c = max(pattern_width, 1) + (cols - pattern_width) // 2

    stride_r = max(stride_r, 1)
    stride_c = max(stride_c, 1)

    # Place the original pattern.
    for (r, c), value in non_zero_cells:
        if r < output_rows and c < output_cols:
            output_grid[r][c] = value

    # Place replicated patterns by shifting the entire pattern.
    # The original pattern is the first replica at an offset of (0,0).
    for r_mult in range(output_rows // stride_r + 1):
        for c_mult in range(output_cols // stride_c + 1):
            offset_r = r_mult * stride_r
            offset_c = c_mult * stride_c

            if offset_r == 0 and offset_c == 0:
                continue  # Already placed

            for (r_orig, c_orig), value in non_zero_cells:
                new_r = offset_r + r_orig
                new_c = offset_c + c_orig
                if 0 <= new_r < output_rows and 0 <= new_c < output_cols:
                    output_grid[new_r][new_c] = value

    # This approach consistently results in a tiled grid. The narrative's
    # "gravitational waves", "ripples", and "resonant echoes" suggest a
    # more distributed spread rather than simple translation.

    # Final attempt at interpreting the narrative:
    # The non-zero elements are anchors. From these anchors, the pattern "radiates".
    # The "zeros" are the medium through which this radiation happens, and the pattern
    # reforms at specific, scaled distances.

    # Output grid dimensions:
    # A common transformation is scaling up. Let's use a dynamic scale.
    num_non_zero = len(non_zero_cells)
    total_cells = rows * cols
    density = num_non_zero / total_cells if total_cells > 0 else 1.0

    scale_factor = 2.0
    if density < 0.2:
        scale_factor = 4.0
    elif density < 0.4:
        scale_factor = 3.0
    elif density < 0.6:
        scale_factor = 2.5

    output_rows = int(rows * scale_factor)
    output_cols = int(cols * scale_factor)

    # Ensure minimum expansion
    output_rows = max(output_rows, rows * 2)
    output_cols = max(output_cols, cols * 2)

    output_grid = [[0] * output_cols for _ in range(output_rows)]

    # Identify the extent of the non-zero elements.
    min_r_nz = min(r for (r, c), val in non_zero_cells)
    max_r_nz = max(r for (r, c), val in non_zero_cells)
    min_c_nz = min(c for (r, c), val in non_zero_cells)
    max_c_nz = max(c for (r, c), val in non_zero_cells)

    pattern_height = max_r_nz - min_r_nz + 1
    pattern_width = max_c_nz - min_c_nz + 1

    # Determine the "stride" for replication. This stride should be larger than the pattern itself.
    # It should also be related to the original grid's dimensions and the pattern's spread.
    # The "longest radiant line" implies the span of non-zero elements is a primary factor.

    # Calculate a stride that incorporates the pattern's size and the original grid's empty space.
    stride_r = max(pattern_height, 1) + (rows - pattern_height) // 2
    stride_c = max(pattern_width, 1) + (cols - pattern_width) // 2

    stride_r = max(stride_r, 1)
    stride_c = max(stride_c, 1)

    # Place the original pattern.
    for (r, c), value in non_zero_cells:
        if r < output_rows and c < output_cols:
            output_grid[r][c] = value

    # Place replicated patterns. The replicas are shifted by multiples of the stride.
    # The original pattern is the first replica at an offset of (0,0).
    for r_mult in range(output_rows // stride_r + 1):
        for c_mult in range(output_cols // stride_c + 1):
            offset_r = r_mult * stride_r
            offset_c = c_mult * stride_c

            if offset_r == 0 and offset_c == 0:
                continue  # Already placed

            # Place the pattern at this new offset.
            for (r_orig, c_orig), value in non_zero_cells:
                new_r = offset_r + r_orig
                new_c = offset_c + c_orig
                if 0 <= new_r < output_rows and 0 <= new_c < output_cols:
                    output_grid[new_r][new_c] = value

    # This consistently results in a tiled pattern. The narrative suggests a more
    # complex "dispersion" and "echoing". The interpretation of "scaled intervals"
    # and "gravitational waves" is likely the key.

    # Let's try to model the "gravitational waves".
    # Each non-zero cell acts as a source. It propagates its value.
    # The "zeros" are the medium. The pattern reforms at specific distances.

    # Determine output dimensions:
    # A common strategy in ARC is to scale up the grid.
    # Let's use a multiplier that depends on the pattern's sparsity.
    num_non_zero = len(non_zero_cells)
    total_cells = rows * cols
    density = num_non_zero / total_cells if total_cells > 0 else 1.0

    scale_factor = 2.0
    if density < 0.2:
        scale_factor = 4.0
    elif density < 0.4:
        scale_factor = 3.0
    elif density < 0.6:
        scale_factor = 2.5

    output_rows = int(rows * scale_factor)
    output_cols = int(cols * scale_factor)

    # Ensure minimum expansion
    output_rows = max(output_rows, rows * 2)
    output_cols = max(output_cols, cols * 2)

    output_grid = [[0] * output_cols for _ in range(output_rows)]

    # The key is how the pattern is replicated.
    # The "longest radiant line" suggests the span of non-zero elements is important.
    min_r_nz = min(r for (r, c), val in non_zero_cells)
    max_r_nz = max(r for (r, c), val in non_zero_cells)
    min_c_nz = min(c for (r, c), val in non_zero_cells)
    max_c_nz = max(c for (r, c), val in non_zero_cells)

    pattern_height = max_r_nz - min_r_nz + 1
    pattern_width = max_c_nz - min_c_nz + 1

    # Define the "stride" for replication. This stride should be larger than the pattern itself.
    # It should also be related to the original grid's dimensions and the pattern's spread.
    # A common interpretation is that the pattern is replicated with a gap that is
    # roughly proportional to the original grid's empty space.
    stride_r = max(pattern_height, 1) + (rows - pattern_height) // 2
    stride_c = max(pattern_width, 1) + (cols - pattern_width) // 2

    stride_r = max(stride_r, 1)
    stride_c = max(stride_c, 1)

    # Place the original pattern.
    for (r, c), value in non_zero_cells:
        if r < output_rows and c < output_cols:
            output_grid[r][c] = value

    # Place replicated patterns by shifting the entire pattern.
    # The original pattern is the first replica at an offset of (0,0).
    for r_mult in range(output_rows // stride_r + 1):
        for c_mult in range(output_cols // stride_c + 1):
            offset_r = r_mult * stride_r
            offset_c = c_mult * stride_c

            if offset_r == 0 and offset_c == 0:
                continue  # Already placed

            # Place the pattern at this new offset.
            for (r_orig, c_orig), value in non_zero_cells:
                new_r = offset_r + r_orig
                new_c = offset_c + c_orig
                if 0 <= new_r < output_rows and 0 <= new_c < output_cols:
                    output_grid[new_r][new_c] = value

    # This approach consistently results in a tiled pattern. The narrative's
    # "gravitational waves", "ripples", and "resonant echoes" suggest a
    # more distributed spread rather than simple translation.

    # Let's try to implement the "dispersion" more directly.
    # The output grid size is likely to be a multiple of the input grid size.
    # The pattern is essentially "painted" onto a larger canvas at scaled intervals.

    # Determine the output grid dimensions.
    # A common strategy in ARC is to scale up the grid.
    # Let's use a multiplier based on the pattern's sparsity.
    num_non_zero = len(non_zero_cells)
    total_cells = rows * cols
    density = num_non_zero / total_cells if total_cells > 0 else 1.0

    scale_factor = 2.0
    if density < 0.2:
        scale_factor = 4.0
    elif density < 0.4:
        scale_factor = 3.0
    elif density < 0.6:
        scale_factor = 2.5

    output_rows = int(rows * scale_factor)
    output_cols = int(cols * scale_factor)

    # Ensure minimum expansion
    output_rows = max(output_rows, rows * 2)
    output_cols = max(output_cols, cols * 2)

    output_grid = [[0] * output_cols for _ in range(output_rows)]

    # Identify the bounding box of the non-zero elements.
    min_r_nz = min(r for (r, c), val in non_zero_cells)
    max_r_nz = max(r for (r, c), val in non_zero_cells)
    min_c_nz = min(c for (r, c), val in non_zero_cells)
    max_c_nz = max(c for (r, c), val in non_zero_cells)

    pattern_height = max_r_nz - min_r_nz + 1
    pattern_width = max_c_nz - min_c_nz + 1

    # Define the "stride" for replication. This stride should be larger than the pattern itself.
    # It should also be related to the original grid's dimensions and the pattern's spread.
    # The "longest radiant line" implies the span of non-zero elements is a primary factor.

    # Calculate a stride that incorporates the pattern's size and the original grid's empty space.
    # This stride dictates the spacing between replicated patterns.
    stride_r = max(pattern_height, 1) + (rows - pattern_height) // 2
    stride_c = max(pattern_width, 1) + (cols - pattern_width) // 2

    stride_r = max(stride_r, 1)
    stride_c = max(stride_c, 1)

    # Place the original pattern.
    for (r, c), value in non_zero_cells:
        if r < output_rows and c < output_cols:
            output_grid[r][c] = value

    # Place replicated patterns by shifting the entire pattern.
    # The original pattern is the first replica at an offset of (0,0).
    for r_mult in range(output_rows // stride_r + 1):
        for c_mult in range(output_cols // stride_c + 1):
            offset_r = r_mult * stride_r
            offset_c = c_mult * stride_c

            if offset_r == 0 and offset_c == 0:
                continue  # Already placed

            # Place the pattern at this new offset.
            for (r_orig, c_orig), value in non_zero_cells:
                new_r = offset_r + r_orig
                new_c = offset_c + c_orig
                if 0 <= new_r < output_rows and 0 <= new_c < output_cols:
                    output_grid[new_r][new_c] = value

    return output_grid