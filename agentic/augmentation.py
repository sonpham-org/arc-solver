"""
Augmentation utilities for ARC tasks.

Implements dihedral transformations and color permutations inspired by TRM.
"""

import numpy as np
import random
from typing import List, Dict, Any, Optional
import copy


def apply_dihedral_transform(grid: List[List[int]], trans_id: int) -> List[List[int]]:
    """
    Apply one of 8 dihedral group transformations to a grid.
    
    Args:
        grid: Input grid as list of lists
        trans_id: Transformation ID (0-7)
            0: Identity (no change)
            1: 90° rotation
            2: 180° rotation
            3: 270° rotation
            4: Horizontal flip
            5: Vertical flip
            6: Diagonal flip (transpose)
            7: Anti-diagonal flip
    
    Returns:
        Transformed grid
    """
    if not grid or not grid[0]:
        return grid
    
    arr = np.array(grid, dtype=np.uint8)
    
    if trans_id == 0:
        # Identity
        result = arr
    elif trans_id == 1:
        # 90° rotation (counter-clockwise)
        result = np.rot90(arr, k=1)
    elif trans_id == 2:
        # 180° rotation
        result = np.rot90(arr, k=2)
    elif trans_id == 3:
        # 270° rotation (or 90° clockwise)
        result = np.rot90(arr, k=3)
    elif trans_id == 4:
        # Horizontal flip (left-right)
        result = np.fliplr(arr)
    elif trans_id == 5:
        # Vertical flip (up-down)
        result = np.flipud(arr)
    elif trans_id == 6:
        # Diagonal flip (transpose)
        result = np.transpose(arr)
    elif trans_id == 7:
        # Anti-diagonal flip (transpose + 180° rotation)
        result = np.rot90(np.transpose(arr), k=2)
    else:
        result = arr
    
    return result.tolist()


def apply_color_permutation(grid: List[List[int]], mapping: Optional[np.ndarray] = None) -> tuple[List[List[int]], np.ndarray]:
    """
    Apply color permutation to a grid, keeping color 0 (black) fixed.
    
    Args:
        grid: Input grid as list of lists
        mapping: Optional color mapping array. If None, generates random permutation.
    
    Returns:
        Tuple of (transformed grid, color mapping used)
    """
    if not grid or not grid[0]:
        return grid, np.arange(10, dtype=np.uint8)
    
    # Generate random color permutation if not provided
    if mapping is None:
        # Keep color 0 fixed, permute colors 1-9
        mapping = np.concatenate([
            np.arange(0, 1, dtype=np.uint8),
            np.random.permutation(np.arange(1, 10, dtype=np.uint8))
        ])
    
    arr = np.array(grid, dtype=np.uint8)
    # Apply color mapping
    result = mapping[arr]
    
    return result.tolist(), mapping


def augment_example(example: Dict[str, List[List[int]]], 
                    trans_id: Optional[int] = None,
                    color_mapping: Optional[np.ndarray] = None) -> Dict[str, List[List[int]]]:
    """
    Augment a single ARC example (input-output pair) with dihedral transform and color permutation.
    
    Args:
        example: Dictionary with 'input' and 'output' grids
        trans_id: Optional transformation ID (0-7). If None, randomly selected.
        color_mapping: Optional color mapping. If None, randomly generated.
    
    Returns:
        Augmented example with same structure
    """
    # Select random transformation if not specified
    if trans_id is None:
        trans_id = np.random.randint(0, 8)
    
    # Apply dihedral transformation to both input and output
    aug_input = apply_dihedral_transform(example['input'], trans_id)
    aug_output = apply_dihedral_transform(example['output'], trans_id)
    
    # Apply color permutation to both input and output (use same mapping)
    aug_input, mapping = apply_color_permutation(aug_input, color_mapping)
    aug_output, _ = apply_color_permutation(aug_output, mapping)
    
    return {
        'input': aug_input,
        'output': aug_output
    }


def augment_task_data(task_data: Dict[str, Any], 
                      num_augmentations: int,
                      max_retries: int = 5) -> Dict[str, Any]:
    """
    Create augmented training examples from a task's training data.
    
    Args:
        task_data: Task data dictionary with 'train' and optionally 'test' keys
        num_augmentations: Number of augmented examples to generate
        max_retries: Maximum retries per augmentation to avoid duplicates
    
    Returns:
        Dictionary with 'train' key containing list of augmented examples
    """
    if not task_data or 'train' not in task_data or not task_data['train']:
        return {'train': []}
    
    training_examples = task_data['train']
    augmented_examples = []
    seen_augmentations = set()
    
    for _ in range(num_augmentations):
        for retry in range(max_retries):
            # Randomly select a training example to augment
            example = random.choice(training_examples)
            
            # Random transformation ID and color permutation
            trans_id = np.random.randint(0, 8)
            color_mapping = np.concatenate([
                np.arange(0, 1, dtype=np.uint8),
                np.random.permutation(np.arange(1, 10, dtype=np.uint8))
            ])
            
            # Apply augmentation
            aug_example = augment_example(example, trans_id, color_mapping)
            
            # Create hashable representation to check for duplicates
            aug_hash = (
                tuple(map(tuple, aug_example['input'])),
                tuple(map(tuple, aug_example['output']))
            )
            
            # If unique, add to results
            if aug_hash not in seen_augmentations:
                seen_augmentations.add(aug_hash)
                augmented_examples.append(aug_example)
                break
    
    return {'train': augmented_examples}
