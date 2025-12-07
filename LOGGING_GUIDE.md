# Logging Guide

## Quick Start

The logging system is configured globally - no need to pass debug flags around!

### In `run_langgraph_agent.py`

Set the debug level:
```python
DEBUG = False  # Default - only shows INFO level
# or
DEBUG = True  # Shows DEBUG level (LLM prompts, responses, etc.)
```

Or use CLI:
```bash
python run_langgraph_agent.py --debug  # Enable debug logging
```

### In Any Python Module

Just import logging and use it:

```python
import logging

# Get logger for your module
logger = logging.getLogger(__name__)

# Use it anywhere - no need to pass debug flags!
logger.debug("Detailed debug info - only shows with --debug")
logger.info("Important info - always shows")
logger.warning("Warning message")
logger.error("Error message")
```

### Existing Functions

The `print_prompt_and_response()` and `print_python_code()` functions in `agent/debug.py` already use this system:

```python
from agent.debug import print_prompt_and_response, print_python_code

# These automatically respect the --debug flag
print_prompt_and_response(prompt, response)  # Only shows if --debug enabled
print_python_code(code)  # Only shows if --debug enabled
```

## Benefits

1. **No parameter passing** - Configure once at startup, use everywhere
2. **Automatic filtering** - DEBUG messages only show when --debug is enabled
3. **Standard Python** - Uses built-in logging module
4. **Clean code** - No more `if debug:` checks everywhere
5. **Flexible** - Easy to add log levels (INFO, WARNING, ERROR)

## Examples

### Normal run (no debug):
```bash
python run_langgraph_agent.py
# Shows: INFO, WARNING, ERROR messages only
```

### Debug run:
```bash
python run_langgraph_agent.py --debug
# Shows: DEBUG, INFO, WARNING, ERROR messages
# Includes LLM prompts and responses
```

### In your code:
```python
import logging

logger = logging.getLogger(__name__)

def my_function():
    logger.info("Starting processing")  # Always shows
    logger.debug("Detailed state: %s", state)  # Only with --debug
    
    try:
        result = process()
        logger.debug("Result: %s", result)  # Only with --debug
        return result
    except Exception as e:
        logger.error("Failed: %s", e)  # Always shows
        raise
```
