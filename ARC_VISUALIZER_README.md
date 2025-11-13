# ARC Solver Results Visualizer

A comprehensive web application built with Gradio to visualize and analyze results from ARC (Abstraction and Reasoning Corpus) solver runs.

## Features

### 🎯 Enhanced Task Display
- **Task Cards**: Each task is displayed as a comprehensive card showing all training and test examples
- **Three-Column Layout**: Input, Expected Output, and Predicted Output side by side
- **Properly Scaled Grids**: Grids are automatically scaled to fit within the card layout
- **Training vs Test Separation**: Clear distinction between training examples and test cases

### 📋 Navigation Menu
- **Left Sidebar**: Contains run selection and task navigation menu
- **Task Menu**: Quick navigation to any task with status indicators (✅/❌)
- **Smooth Scrolling**: Click on any task in the menu to jump directly to that task card
- **Run Summary**: Overview of model performance, costs, and configuration

### 📊 Comprehensive Task Information
- **Training Examples**: Shows all training input/output pairs
- **Test Cases**: Displays test inputs, expected outputs, and AI predictions
- **Performance Metrics**: IoU scores and correctness indicators for each test case
- **Sample Counts**: Shows number of training examples and test cases per task

## Installation

1. Make sure you have the required dependencies:
```bash
pip install gradio matplotlib numpy
```

2. If using conda, activate the ARC environment:
```bash
conda activate ARC
```

## Usage

1. **Start the Visualizer**:
```bash
cd /path/to/arc-solver
python arc_visualizer.py
```

2. **Access the Interface**:
   - Open your browser and go to `http://localhost:7860`
   - The interface will automatically open in your default browser

3. **Navigate the Interface**:
   - **Select a Run**: Choose from available runs in the dropdown
   - **View Summary**: See run statistics and model configuration
   - **Browse Tasks**: Use the task menu on the left to jump to specific tasks
   - **Analyze Results**: Compare inputs, expected outputs, and predictions

## File Structure

The visualizer expects the following structure in your `output/` directory:

```
output/
├── 2025-11-11T21-52-52-015744/    # Run timestamp folder
│   ├── params.json                # Run parameters
│   ├── summary.json              # Run summary statistics
│   ├── evaluation_task_ids.txt   # List of evaluation tasks
│   ├── training_task_ids.txt     # List of training tasks
│   ├── fbf15a0b.json            # Individual task results
│   └── ...                      # More task files
└── ...                          # More runs
```

## Task Data Format

Each task JSON file contains:
- **trains**: Array of training examples with input/output pairs
- **tests**: Array of test cases with input/output/predicted data
- **Performance metrics**: IoU scores, correctness flags, etc.

## Features Explained

### Grid Visualization
- **Color Coding**: Uses the standard ARC color palette (0-9)
- **Proper Scaling**: Grids automatically resize to fit the three-column layout
- **Border Styling**: Clear borders distinguish individual cells

### Task Status Indicators
- **✅ Green**: Task solved correctly
- **❌ Red**: Task failed or incorrect prediction
- **IoU Scores**: Intersection over Union percentages for similarity measurement

### Training vs Testing
- **Training Examples**: Show the pattern without predictions (learning phase)
- **Test Cases**: Show AI predictions compared to ground truth
- **Clear Labeling**: Distinguishes between example types

## Troubleshooting

### Common Issues

1. **Module Import Errors**:
   ```bash
   pip install gradio matplotlib numpy
   ```

2. **No Tasks Found**:
   - Ensure you're running from the correct directory
   - Check that `output/` folder contains valid run directories
   - Verify JSON files are not corrupted

3. **Gradio Connection Issues**:
   - Try a different port by modifying the `server_port` parameter
   - Check firewall settings if accessing from remote machine

### Environment Setup

If using conda environments:
```bash
# Activate the correct environment
conda activate ARC

# Install missing packages
conda install -c conda-forge gradio matplotlib numpy
```

## Advanced Usage

### Customizing the Interface
- Modify `ARC_COLORS` to change the color scheme
- Adjust grid cell sizes in the CSS styles
- Customize the layout by modifying the Gradio column scales

### Adding New Features
The visualizer is modular and easy to extend:
- Add new analysis functions to the `ARCVisualizer` class
- Create additional HTML components for enhanced visualization
- Integrate with other analysis tools or databases

## Performance Notes

- **Caching**: Run data is cached after first load for faster navigation
- **Memory Usage**: Large runs with many tasks may use significant memory
- **Load Times**: Initial load depends on number of tasks and file sizes

## Support

For issues or questions:
1. Check the console output for error messages
2. Verify file formats match the expected structure
3. Ensure all dependencies are correctly installed
4. Check that the output directory contains valid run data