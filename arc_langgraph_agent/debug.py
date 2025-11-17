from typing import Any

# ANSI color codes for terminal output (used for debugging LLM prompts/responses)
BLUE = "\033[34m"
RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"

def print_prompt_and_response(prompt: str, response: Any) -> None:
    """Print the LLM prompt and response with ANSI colors for easier reading in terminals."""
    response_text = response.content if hasattr(response, 'content') else str(response)

    # Print prompt and response with ANSI colors for easier reading in terminals
    print(f"{BLUE}--- LLM Prompt ---{RESET}")
    print(f"{BLUE}{prompt}{RESET}")
    print(f"{GREEN}--- LLM Response ---{RESET}")
    print(f"{GREEN}{response_text}{RESET}")

def print_python_code(code: str) -> None:
    """Print Python code with ANSI colors for easier reading in terminals."""
    print(f"{RED}--- Generated Python Code ---{RESET}")
    print(f"{RED}{code}{RESET}")