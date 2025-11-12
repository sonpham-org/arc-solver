# ARC Solver with Helper Learning System

This enhanced ARC solver implements a helper function learning system that allows tasks to learn from each other through shared helper functions.

## 🚀 Key Features

### Helper Learning System
- **Function Database**: Automatically stores helper functions from successful task solutions
- **Success Tracking**: Tracks usage count and success rate for each helper function
- **Intelligent Selection**: Provides top-performing helpers + random selection for diversity
- **Iterative Improvement**: Redoes failed tasks with accumulated helper knowledge

### Enhanced Prompting
- **Helper-Aware Prompts**: Includes relevant helper functions in task prompts
- **Learning Guidance**: Encourages models to use and adapt successful patterns
- **Progressive Knowledge**: Each task benefits from all previous successful tasks

### Iterative Task Execution
- **REDO_TASKS**: Configurable number of improvement iterations
- **Failure Analysis**: Automatically identifies tasks that need improvement
- **Knowledge Accumulation**: Each iteration has access to more learned helpers

## 🛠️ Usage

### Basic Usage with Helper Learning

```bash
# Run with helper learning system enabled (default)
python arc_prompt_with_helpers.py --number-of-tasks 5 --model gemini-2.5-flash-lite-preview-06-17

# Run with 3 redo iterations for failed tasks
python arc_prompt_with_helpers.py --number-of-tasks 10 --redo-tasks 3

# Use custom helper database filename
python arc_prompt_with_helpers.py --helper-db-path my_helpers.db
```

### Disable Helper Learning

```bash
# Run without helper learning (like original system)
python arc_prompt_with_helpers.py --number-of-tasks 5 --use-helpers=False --redo-tasks 0
```

### Advanced Options

```bash
# Full configuration example
python arc_prompt_with_helpers.py \
    --number-of-tasks 20 \
    --model gemini-2.5-flash-lite-preview-06-17 \
    --use-helpers \
    --redo-tasks 5 \
    --helper-db-path my_arc_helpers.db \
    --parallel \
    --use-smart-router \
    --max-reflections 3
```

## 📋 New Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use-helpers` | `True` | Enable helper function learning system |
| `--redo-tasks` | `3` | Number of redo iterations for failed tasks |
| `--helper-db-path` | `helpers.db` | Database filename (placed in run output directory for self-contained runs) |

## � Self-Contained Runs

Each run is completely self-contained with its own helper database:

- **Database Location**: `{output_directory}/helpers.db`
- **Run Isolation**: Each run maintains its own independent helper function database
- **No Conflicts**: Different runs or experiments don't interfere with each other
- **Easy Archival**: Complete run data (results + learned helpers) in one directory
- **Reproducibility**: All helper knowledge used in a run is preserved with that run

**Example**: For output directory `output/2025-11-09T22-45-32-534228/`, the helper database will be at `output/2025-11-09T22-45-32-534228/helpers.db`.

## �🔄 How the Learning System Works

### Phase 1: Initial Execution
1. Run all selected tasks with current helper knowledge
2. Extract helper functions from successful JSON responses
3. Store functions in database with validation and success tracking

### Phase 2: Iterative Improvement (if `redo-tasks > 0`)
1. **Identify Failed Tasks**: Find tasks with valid JSON but no correct test outputs
2. **Knowledge Query**: Get top 20 successful helpers + 20 random helpers
3. **Enhanced Prompting**: Include helper functions in prompts for failed tasks
4. **Execute Improvements**: Run failed tasks with accumulated knowledge
5. **Repeat**: Continue for specified number of redo iterations

### Helper Function Management
- **Extraction**: Parse `helper_python_functions` from successful JSON
- **Validation**: Syntax check and compilation validation
- **Deduplication**: Hash-based duplicate prevention
- **Ranking**: Success rate and usage count tracking
- **Selection**: Top performers for reliability + random for diversity

## 📊 Database Schema

### Helpers Table
- `function_hash`: Unique hash for deduplication
- `function_name`: Extracted function name
- `function_code`: Complete function code
- `function_signature`: Function signature
- `description`: Extracted docstring or description
- `success_count`: Number of times used in successful tasks
- `usage_count`: Total number of times referenced
- `is_validated`: Boolean for syntax validation
- `created_at`, `last_used_at`: Timestamps

### Helper Usage Table
- `helper_id`: Foreign key to helpers table
- `task_id`: Task that used this helper
- `was_successful`: Whether the task succeeded
- `used_at`: Usage timestamp

## 🎯 Benefits of Helper Learning

### Knowledge Accumulation
- Each successful task contributes to a shared knowledge base
- Helper functions become reusable building blocks
- Common patterns emerge and get reinforced

### Progressive Improvement
- Early tasks may struggle, but later tasks benefit from accumulated knowledge
- Failed tasks get multiple chances with growing helper database
- System learns which patterns work well for ARC problems

### Pattern Recognition
- Successful geometric transformations get preserved
- Useful utility functions become available to all tasks
- Complex operations get broken down into reusable components

## 📈 Expected Performance Improvements

1. **Cold Start**: Initial tasks perform like baseline system
2. **Knowledge Growth**: As helper database grows, tasks have more tools available
3. **Pattern Reuse**: Common ARC patterns get identified and reused
4. **Iterative Recovery**: Failed tasks get second chances with better knowledge

## 🔍 Monitoring and Debugging

### Helper Database Stats
The system prints helper database statistics showing:
- Total helpers learned
- Validated helpers (syntax-correct)
- Total successful applications
- Top performing functions

### Task Performance Tracking
Each task shows:
- Whether helper system was used
- Number of helpers included in prompt
- Success/failure status
- Token usage and costs

## 🧪 Testing and Validation

Run the demo script to see the system in action:

```bash
python demo_helper_learning.py
```

This demonstrates:
- Helper function storage and retrieval
- Database statistics
- Prompt generation with helpers
- Success tracking and ranking

## 🚧 Implementation Notes

### Thread Safety
- Database operations are thread-safe
- Parallel execution supported with helper learning
- Output capturing maintains task isolation

### Error Handling
- Invalid helper functions are filtered out
- Database failures don't crash task execution
- Graceful degradation when helpers unavailable

### Performance Considerations
- Helper selection limited to top 20 + random 20 for prompt size
- Database indexed for fast retrieval
- Function validation prevents infinite loops

## 🔮 Future Enhancements

Potential improvements to the helper learning system:

1. **Semantic Similarity**: Group similar helper functions
2. **Adaptive Selection**: Choose helpers based on task characteristics  
3. **Version Control**: Track helper function evolution
4. **Cross-Model Learning**: Share helpers between different AI models
5. **Pattern Templates**: Extract higher-level transformation patterns

## 📝 Example Output

```
🛠️  INITIALIZING HELPER LEARNING SYSTEM
📊 Helper database initialized:
  Database path: helpers.db
  Total helpers: 15
  Validated helpers: 14
  Total successes: 23
  Top performers: find_connected_components(5), rotate_grid_90(4), flood_fill(3)

🚀 INITIAL TASK EXECUTION ROUND
📋 INITIAL: Processing 10 tasks
📚 Including 20 top helpers and 20 random helpers

🔄 REDO ROUND 1/3 - LEARNING FROM FAILURES
🎯 Found 4 tasks with test failures
📚 Helper database now contains learned knowledge from successful tasks
📈 ROUND 1 RESULTS: 2/4 tasks now successful

🎓 FINAL HELPER LEARNING SUMMARY:
  Total helpers learned: 18
  Validated helpers: 17  
  Total successful applications: 31
  Database saved to: helpers.db
```

The helper learning system transforms the ARC solver from isolated task execution into a continuously learning system that gets smarter with each task! 🚀