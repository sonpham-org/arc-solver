"""
Code repair prompt builder for when JSON is valid but Python code fails to execute.

This function creates a detailed prompt for fixing Python execution errors
while maintaining the JSON structure.
"""

import json


def build_code_repair_prompt(task_data: dict, task_id: str, previous_json: dict, execution_errors: list) -> str:
    """
    Build a code repair prompt for when JSON is valid but Python code fails to execute.
    
    Args:
        task_data: Dictionary containing all ARC tasks
        task_id: The unique identifier of the task
        previous_json: The JSON response with failing code
        execution_errors: List of execution errors for each training example
    
    Returns:
        A formatted prompt that includes the previous JSON, execution errors,
        and asks for corrected Python code.
    """
    task = task_data[task_id]
    train = task.get("train", [])
    
    # We don't need to show training examples again since errors already include input/expected output
    
    # Build execution errors block with line-specific information
    error_details = []
    for i, error_info in enumerate(execution_errors, 1):
        error_text = f"EXAMPLE {i} FAILURE:\n"
        error_text += f"Input: {error_info['input']}\n"
        error_text += f"Expected: {error_info['expected']}\n"
        
        # Add specific error information
        error_text += f"\nERROR: {error_info.get('error', 'Unknown error')}"
        
        # Add line-specific information if available
        if error_info.get('error_line'):
            error_text += f"\nFailed at: {error_info['error_line']}"
        
        if error_info.get('code_context'):
            error_text += f"\n{error_info['code_context']}"
        
        if 'predicted' in error_info and error_info['predicted'] is not None:
            error_text += f"\nActual output: {error_info['predicted']}"
        
        error_details.append(error_text)
    
    errors_block = "\n\n".join(error_details)
    
    # Format the previous JSON
    previous_json_str = json.dumps(previous_json, indent=2) if previous_json else "No previous JSON available"
    
    # Extract and format Python code with line numbers for easier reference
    python_code_with_lines = ""
    if previous_json and 'python_code' in previous_json:
        python_lines = previous_json['python_code']
        python_code_with_lines = "\n".join(f"{i+1:2d}: {line}" for i, line in enumerate(python_lines))
    
    return f"""
You are an expert Python programmer specializing in ARC puzzle transformations.
Your previous JSON response had valid structure but the Python code failed to execute on the training examples.

====================
Task {task_id} - CODE REPAIR NEEDED
====================

Your goal:
Fix the Python code in your JSON response so that it executes successfully on ALL training examples.
The JSON structure is correct, but the transformation code has execution errors that need to be resolved.

====================
Your Previous JSON (with failing code)
====================
```json
{previous_json_str}
```

====================
Current Python Code (with line numbers)
====================
{python_code_with_lines}

====================
Detailed Execution Errors
====================
{errors_block}

====================
Instructions for Code Repair
====================
Analyze the execution errors above and provide a CORRECTED JSON response with fixed Python code.

Focus on these common issues:
1. **Index out of range errors**: Check grid dimensions and array bounds (refer to line numbers above)
2. **Variable/function not defined**: Ensure all variables are properly initialized
3. **Type errors**: Make sure data types are compatible (lists, integers, etc.)
4. **Logic errors**: Verify the transformation logic matches the expected pattern
5. **Edge cases**: Handle special cases like empty grids, single values, etc.
6. **Line-specific fixes**: Pay attention to the specific lines mentioned in the error details

Provide the corrected JSON in the same format:

```json
{{
  "helper_python_functions": [
    "# Add any helper functions needed",
  ],
  "step_by_step_transformations": [{{
      "step_number": 1,
      "description": [
        "Describe what the transformation does",
      ],
      "pseudo_code": [
        "Outline the logic steps"
      ],
  }}],
  "python_code": [
    "def transform(grid):",
    "    # CORRECTED implementation that handles all the errors above",
    "    # Make sure this code executes successfully on all training examples",
    "    return processed_grid"
  ]
}}
```

CRITICAL: The Python code must execute without errors on ALL training examples. Test your logic carefully!
""".strip()


if __name__ == "__main__":
    import os
    import json
    from typing import Optional
    
    # Sample data for code repair testing
    sample_task_data = {
        "test_repair_001": {
            "train": [
                {
                    "input": [[1, 0], [0, 1]],
                    "output": [[0, 1], [1, 0]]
                }
            ]
        }
    }
    
    sample_previous_json = {
        "helper_python_functions": [],
        "step_by_step_transformations": [{
            "step_number": 1,
            "description": ["Flip the grid"],
            "pseudo_code": ["Reverse each row"]
        }],
        "python_code": [
            "def transform(grid):",
            "    result = []",
            "    for row in grid:",
            "        result.append(row.reverse())  # BUG: reverse() returns None",
            "    return result"
        ]
    }
    
    sample_execution_errors = [
        {
            "input": [[1, 0], [0, 1]],
            "expected": [[0, 1], [1, 0]],
            "error": "AttributeError: 'NoneType' object has no attribute 'append'",
            "error_line": "result.append(row.reverse())",
            "code_context": "Line 4: row.reverse() returns None instead of the reversed list",
            "predicted": None
        }
    ]
    
    def test_with_gemini_api(prompt_text: str, api_key: Optional[str] = None):
        """Test the prompt using Gemini API directly."""
        try:
            import google.generativeai as genai
            
            api_key = api_key or os.getenv('GEMINI_API_KEY')
            if not api_key:
                print("❌ Gemini API key not found. Set GEMINI_API_KEY environment variable.")
                return
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            print("🔄 Testing with Gemini API...")
            response = model.generate_content(prompt_text)
            print("✅ Gemini API Response:")
            print(response.text)
            
        except ImportError:
            print("❌ google-generativeai not installed. Run: pip install google-generativeai")
        except Exception as e:
            print(f"❌ Gemini API error: {e}")
    
    def test_with_ollama_api(prompt_text: str, model_name: str = "llama3.2"):
        """Test the prompt using Ollama API directly."""
        try:
            import requests
            
            print(f"🔄 Testing with Ollama API ({model_name})...")
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt_text,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Ollama API Response:")
                print(result.get('response', 'No response'))
            else:
                print(f"❌ Ollama API error: {response.status_code}")
                
        except ImportError:
            print("❌ requests library not installed. Run: pip install requests")
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to Ollama. Make sure Ollama is running on localhost:11434")
        except Exception as e:
            print(f"❌ Ollama API error: {e}")
    
    def test_with_langchain(prompt_text: str, api_key: Optional[str] = None):
        """Test the prompt using LangChain."""
        try:
            from langchain_google_vertexai import ChatVertexAI
            from langchain_core.messages import HumanMessage
            
            print("🔄 Testing with LangChain (VertexAI)...")
            llm = ChatVertexAI(
                model_name="gemini-2.5-flash",
                project=os.getenv('GOOGLE_CLOUD_PROJECT'),
                location=os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
            )
            
            message = HumanMessage(content=prompt_text)
            response = llm.invoke([message])
            print("✅ LangChain Response:")
            print(response.content)
            
        except ImportError as e:
            print(f"❌ LangChain dependencies not installed: {e}")
            print("Run: pip install langchain-google-vertexai")
        except Exception as e:
            print(f"❌ LangChain error: {e}")
    
    # Generate test prompt
    print("=" * 60)
    print("TESTING CODE REPAIR PROMPT")
    print("=" * 60)
    
    test_prompt = build_code_repair_prompt(
        task_data=sample_task_data,
        task_id="test_repair_001",
        previous_json=sample_previous_json,
        execution_errors=sample_execution_errors
    )
    
    print("Generated Prompt:")
    print("-" * 40)
    print(test_prompt)
    print("-" * 40)
    print()
    
    # Test with different APIs
    print("\n1. Testing with Gemini API:")
    test_with_gemini_api(test_prompt)
    
    print("\n2. Testing with Ollama API:")
    test_with_ollama_api(test_prompt)
    
    print("\n3. Testing with LangChain:")
    test_with_langchain(test_prompt)