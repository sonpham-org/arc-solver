"""Grid manipulation and validation utilities."""


def grid_to_string_lines(grid):
    """Convert grid to array of line-separated strings."""
    if not grid:
        return []
    return [''.join(map(str, row)) for row in grid]


def is_valid_prediction(predicted):
    """Check if prediction is a valid 2D grid."""
    if not predicted:
        return False
    if not isinstance(predicted, list):
        return False
    if not all(isinstance(row, list) for row in predicted):
        return False
    if not predicted:
        return False
    row_length = len(predicted[0])
    return all(len(row) == row_length for row in predicted)


def calculate_grid_iou(predicted, expected):
    """Calculate intersection over union of grid dimensions."""
    if not predicted or not expected:
        return 0.0
    
    pred_h, pred_w = len(predicted), len(predicted[0]) if predicted else 0
    exp_h, exp_w = len(expected), len(expected[0]) if expected else 0
    
    if pred_h == 0 or pred_w == 0 or exp_h == 0 or exp_w == 0:
        return 0.0
    
    # Calculate intersection and union of dimensions
    intersection_h = min(pred_h, exp_h)
    intersection_w = min(pred_w, exp_w)
    union_h = max(pred_h, exp_h)
    union_w = max(pred_w, exp_w)
    
    intersection_area = intersection_h * intersection_w
    union_area = union_h * union_w
    
    return (intersection_area / union_area) * 100.0 if union_area > 0 else 0.0


def calculate_grid_overlap(predicted, expected):
    """Calculate percentage overlap between two grids."""
    if not predicted or not expected:
        return 0.0
    
    if len(predicted) != len(expected):
        return 0.0
    
    total_cells = 0
    matching_cells = 0
    
    for i, (pred_row, exp_row) in enumerate(zip(predicted, expected)):
        if len(pred_row) != len(exp_row):
            return 0.0
        for j, (pred_cell, exp_cell) in enumerate(zip(pred_row, exp_row)):
            total_cells += 1
            if pred_cell == exp_cell:
                matching_cells += 1
    
    return (matching_cells / total_cells) * 100.0 if total_cells > 0 else 0.0