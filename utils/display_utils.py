"""Display and terminal output utilities."""

# ANSI color codes
BLUE = '\033[94m'
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'


def print_colored(text, color=RESET, bold=False):
    """Print text with color formatting."""
    formatting = f"{BOLD if bold else ''}{color}"
    print(f"{formatting}{text}{RESET}")