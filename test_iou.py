#!/usr/bin/env python3
"""
Test IOU calculation to verify it's working correctly.
"""

def test_iou_calculation():
    """Test IOU calculation with known values."""
    
    # Test case: Expected 3x4 grid vs Predicted 4x3 grid
    expected = [
        [1, 2, 3, 4],
        [5, 6, 7, 8], 
        [9, 10, 11, 12]
    ]  # 3x4
    
    predicted = [
        [1, 2, 3],
        [5, 6, 7],
        [9, 10, 11],
        [13, 14, 15]
    ]  # 4x3
    
    # Calculate sizes
    pred_h, pred_w = len(predicted), len(predicted[0]) if predicted else 0
    exp_h, exp_w = len(expected), len(expected[0]) if expected else 0
    
    print(f"Expected size: {exp_h}x{exp_w} = {exp_h * exp_w} cells")
    print(f"Predicted size: {pred_h}x{pred_w} = {pred_h * pred_w} cells")
    
    # Calculate intersection (overlapping area)
    intersection_h = min(pred_h, exp_h)  # min(4, 3) = 3
    intersection_w = min(pred_w, exp_w)  # min(3, 4) = 3
    intersection_area = intersection_h * intersection_w  # 3 * 3 = 9
    
    print(f"Intersection area: {intersection_h}x{intersection_w} = {intersection_area} cells")
    
    # Calculate matching cells in intersection area
    matching_cells = 0
    for i in range(intersection_h):
        for j in range(intersection_w):
            if predicted[i][j] == expected[i][j]:
                matching_cells += 1
                print(f"Match at ({i},{j}): {predicted[i][j]} == {expected[i][j]}")
    
    print(f"Matching cells: {matching_cells}")
    
    # Calculate union (total area covered by both grids)
    union_area = pred_h * pred_w + exp_h * exp_w - intersection_area
    # union = 12 + 12 - 9 = 15
    
    print(f"Union area: {pred_h * pred_w} + {exp_h * exp_w} - {intersection_area} = {union_area}")
    
    # Calculate metrics
    overlap = (matching_cells / intersection_area * 100.0) if intersection_area > 0 else 0.0
    iou = (matching_cells / union_area * 100.0) if union_area > 0 else 0.0
    
    print(f"Overlap: {matching_cells}/{intersection_area} = {overlap:.1f}%")
    print(f"IOU: {matching_cells}/{union_area} = {iou:.1f}%")
    
    # Expected: 9 matching cells, union of 15 -> 9/15 = 60%
    print(f"Expected IOU: 9/15 = 60% -> Got: {iou:.1f}%")

if __name__ == "__main__":
    test_iou_calculation()