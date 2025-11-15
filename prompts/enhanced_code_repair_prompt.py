import json


def create_code_repair_prompt(valid_json, execution_error, failed_example=None, all_failures=None):
    """Create a prompt to fix code execution errors in valid JSON with comprehensive failure details."""
    
    # Build detailed error context
    error_context = ""
    if failed_example:
        error_context = f"""
PRIMARY FAILED EXAMPLE (Example {failed_example.get('example_idx', 0) + 1}):
Input Grid: {failed_example.get('input', 'N/A')}
Expected Output: {failed_example.get('expected_output', 'N/A')}
Error: {failed_example.get('error', 'N/A')}
"""
    
    # Add additional failure context if available
    if all_failures and len(all_failures) > 1:
        error_context += f"\nADDITIONAL FAILURES ({len(all_failures) - 1} more examples):\n"
        for i, failure in enumerate(all_failures[1:6], 1):  # Show up to 5 additional failures
            error_context += f"Example {failure.get('example_idx', 0) + 1}: {failure.get('error', 'Unknown error')}\n"
            error_context += f"  Input: {failure.get('input', 'N/A')}\n"
            error_context += f"  Expected: {failure.get('expected_output', 'N/A')}\n"
        
        if len(all_failures) > 6:
            error_context += f"... and {len(all_failures) - 6} more failures\n"

    return f"""The JSON response is valid but the Python code fails to execute on multiple training examples.

CURRENT VALID JSON:
```json
{json.dumps(valid_json, indent=2)}
```

CODE EXECUTION ANALYSIS:
Total Training Examples: {len(all_failures) if all_failures else 1}
Failed Examples: {len(all_failures) if all_failures else 1}
Success Rate: 0%

DETAILED FAILURE INFORMATION:
{error_context}

CRITICAL ISSUE DETECTED:
The transformation logic has execution errors that need to be fixed.
The python_code should implement a complete transformation following the conceptual steps.

REQUIREMENTS FOR REPAIR:
1. There should be ONE complete `def transform(grid):` function in python_code
2. The function should implement all transformation logic from step_by_step_transformations
3. The step_by_step_transformations are conceptual guides - the actual implementation goes in python_code
4. Fix the specific execution errors mentioned above
5. Ensure the transform() function is properly defined and callable
6. Handle edge cases that cause failures across different input grids
7. Use proper error handling for robust execution
8. If a step has no actual transformation logic, return the input unchanged: `return grid`

Please provide a CORRECTED JSON with properly structured, complete transform functions:

```json
{{
  # Python compatible code that describes
  # any helper functions needed to implement the rule.
  # Each rule will run this code before applying the transformation code.
  "helper_python_functions": [
    "...",
  ],
  "step_by_step_transformations": [{{
      "step_number": 1,
      "description": [
        "...",
      ], # Describe the transformation conceptually
      "pseudo_code": [
      ],
  }},
  "python_code": [
    "def transform(grid):",
    "    # Complete transformation implementation",
    "    # This function must be fully executable on its own",
    "    # and return a complete transformed grid",
    "    return processed_grid"
  ]
}}
```

REMEMBER: Each transform function must be complete and executable independently!"""


if __name__ == "__main__":
    import os
    import json
    from typing import Optional
    
    # Sample data for enhanced code repair testing
    sample_valid_json = {
        "helper_python_functions": [],
        "step_by_step_transformations": [{
            "step_number": 1,
            "description": ["Rotate grid 90 degrees clockwise"],
            "pseudo_code": ["Transpose and reverse each row"]
        }],
        "python_code": [
            "def transform(grid):",
            "    # Missing implementation",
            "    pass"
        ]
    }
    
    sample_execution_error = "NameError: name 'transpose' is not defined"
    
    sample_failed_example = {
        "example_idx": 0,
        "input": [[1, 2], [3, 4]],
        "expected_output": [[3, 1], [4, 2]],
        "error": "Function returned None instead of transformed grid"
    }
    
    sample_all_failures = [
        {
            "example_idx": 0,
            "input": [[1, 2], [3, 4]],
            "expected_output": [[3, 1], [4, 2]],
            "error": "Function returned None"
        },
        {
            "example_idx": 1,
            "input": [[1, 0, 1], [0, 1, 0]],
            "expected_output": [[0, 1], [1, 0], [1, 1]],
            "error": "IndexError: list index out of range"
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
    print("TESTING ENHANCED CODE REPAIR PROMPT")
    print("=" * 60)
    
    test_prompt = create_code_repair_prompt(
        valid_json=sample_valid_json,
        execution_error=sample_execution_error,
        failed_example=sample_failed_example,
        all_failures=sample_all_failures
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