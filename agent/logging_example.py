"""
Example showing how to use logging in your agent modules.

This demonstrates the key benefit: no need to pass debug flags around!
"""
import logging

# Get a logger for this module - each module gets its own logger
logger = logging.getLogger(__name__)

def analyze_pattern(grid):
    """Example function showing various log levels."""
    
    # INFO: Important events that always show
    logger.info(f"Analyzing grid of size {len(grid)}x{len(grid[0])}")
    
    # DEBUG: Detailed info that only shows with --debug flag
    logger.debug(f"Grid contents: {grid}")
    
    # Simulate some processing
    patterns_found = ["symmetry", "repetition"]
    
    # INFO: Key results
    logger.info(f"Found {len(patterns_found)} patterns")
    
    # DEBUG: Detailed results
    logger.debug(f"Patterns: {patterns_found}")
    
    return patterns_found

def generate_transformation(pattern):
    """Another example function."""
    
    logger.debug(f"Generating transformation for pattern: {pattern}")
    
    transformation = f"transform_{pattern}"
    
    logger.info(f"Generated transformation: {transformation}")
    
    return transformation

def call_llm(prompt):
    """Example showing LLM interaction logging."""
    
    # Import the debug helper that respects log level
    from agent.debug import print_prompt_and_response
    
    # Simulate LLM call
    response = "Sample LLM response"
    
    # This automatically only shows if --debug is enabled
    print_prompt_and_response(prompt, response)
    
    logger.info("LLM call completed")
    
    return response

# Example usage
if __name__ == "__main__":
    import logging
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,  # or logging.DEBUG for debug mode
        format='%(name)s - %(levelname)s - %(message)s'
    )
    
    print("\\n=== Running with INFO level (normal mode) ===")
    grid = [[1, 2], [3, 4]]
    patterns = analyze_pattern(grid)
    transform = generate_transformation(patterns[0])
    
    print("\\n=== Now with DEBUG level (debug mode) ===")
    logging.getLogger().setLevel(logging.DEBUG)
    
    patterns = analyze_pattern(grid)
    transform = generate_transformation(patterns[0])
    call_llm("Analyze this pattern")
