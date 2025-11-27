from typing import Any

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

def print_prompt_and_response(prompt: str, response: Any) -> None:
    """Print the LLM prompt and response with ANSI colors for easier reading in terminals."""
    response_text = response.content if hasattr(response, 'content') else str(response)

    # Convert any literal escape sequences into actual characters (e.g. "\\n" -> newline)
    prompt_to_print = _unescape_escapes(prompt)
    response_to_print = _unescape_escapes(response_text)

    # Print prompt and response with ANSI colors for easier reading in terminals
    print(f"{BLUE}--- LLM Prompt ---{RESET}")
    print(f"{BLUE}{prompt_to_print}{RESET}")
    print(f"{GREEN}--- LLM Response ---{RESET}")
    print(f"{GREEN}{response_to_print}{RESET}")

def print_python_code(code: str) -> None:
    """Print Python code with ANSI colors for easier reading in terminals."""
    code_to_print = _unescape_escapes(code)
    print(f"{YELLOW}--- Generated Python Code ---{RESET}")
    print(f"{YELLOW}{code_to_print}{RESET}")