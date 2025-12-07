"""Debug utilities using Python's logging system.

Use logging.getLogger(__name__).debug() for debug output.
The logging level is configured in run_langgraph_agent.py via the --debug flag.
"""
import logging
from typing import Any

# Get logger for this module
logger = logging.getLogger(__name__)

# ANSI color codes for terminal output (used for debugging LLM prompts/responses)
BLUE = "\033[34m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"

def _unescape_escapes(text: str) -> str:
    """Convert literal escape sequences like '\\n' into real newlines/tabs.

    If `text` is not a string, return it unchanged. Use a best-effort
    conversion: try `unicode_escape` decoding first, fall back to simple
    replacements on failure.
    """
    if not isinstance(text, str):
        return text
    try:
        return bytes(text, "utf-8").decode("unicode_escape")
    except Exception:
        return text.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")

def print_prompt_and_response(prompt: str, response: str) -> None:
    """Log the LLM prompt and response at DEBUG level.
    
    This will only show output when --debug flag is enabled.
    Uses ANSI colors for easier reading in terminals.
    """
    # Only proceed if debug logging is enabled
    if not logger.isEnabledFor(logging.DEBUG):
        return

    # Check if either strings are None or empty
    if not prompt:
        prompt = "<EMPTY PROMPT>"
    if not response:
        response = "<EMPTY RESPONSE>"
        
    # Convert any literal escape sequences into actual characters (e.g. "\\n" -> newline)
    prompt_to_print = _unescape_escapes(prompt)
    response_to_print = _unescape_escapes(response)

    # Log prompt and response with ANSI colors
    logger.debug(f"{BLUE}--- LLM Prompt ---{RESET}")
    logger.debug(f"{BLUE}{prompt_to_print}{RESET}")
    logger.debug(f"{GREEN}--- LLM Response ---{RESET}")
    logger.debug(f"{GREEN}{response_to_print}{RESET}")

def print_python_code(code: str) -> None:
    """Log generated Python code at DEBUG level.
    
    This will only show output when --debug flag is enabled.
    Uses ANSI colors for easier reading in terminals.
    """
    # Only proceed if debug logging is enabled
    if not logger.isEnabledFor(logging.DEBUG):
        return
    
    # Check if code string is None or empty
    if not code:
        code = "<EMPTY CODE>"
        
    code_to_print = _unescape_escapes(code)
    logger.debug(f"{YELLOW}--- Generated Python Code ---{RESET}")
    logger.debug(f"{YELLOW}{code_to_print}{RESET}")