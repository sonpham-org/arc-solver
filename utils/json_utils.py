"""JSON parsing and validation utilities."""

import json
import re


def extract_json_from_response(response_text):
    """Extract JSON from response text, looking for ```json ``` blocks."""
    if not response_text:
        return None
    
    # Look for ```json ... ``` blocks
    json_pattern = r'```json\s*\n(.*?)\n```'
    matches = re.findall(json_pattern, response_text, re.DOTALL)
    
    if not matches:
        return None
    
    # Try to parse the first JSON block found
    try:
        return json.loads(matches[0])
    except json.JSONDecodeError as e:
        from .display_utils import print_colored, RED
        print_colored(f"JSON parsing error: {e}", RED)
        return None


def validate_json_structure(data):
    """Validate that the JSON has the expected ARC structure."""
    if not isinstance(data, dict):
        return False, "JSON is not a dictionary"
    
    if "python_code" not in data:
        return False, "Missing 'python_code' field"
    
    if "step_by_step_transformations" not in data:
        return False, "Missing 'step_by_step_transformations' field"
    
    steps = data["step_by_step_transformations"]
    if not isinstance(steps, list):
        return False, "'step_by_step_transformations' is not a list"
    
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            return False, f"Step {i+1} is not a dictionary"
        if "step_number" not in step or "description" not in step:
            return False, f"Step {i+1} missing required fields"
    
    return True, "Valid structure"