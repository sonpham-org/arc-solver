"""
ARC reflection prompt builder for failed tasks.

This function builds a reflection prompt that asks the model to analyze 
their previous attempt and generate an improved JSON solution.
"""

import json


def build_arc_reflection_prompt(task_data: dict, task_id: str, previous_json: dict, train_results: list) -> str:
    """
    Build a reflection prompt for failed tasks that asks the model to analyze 
    their previous attempt and generate an improved JSON solution.
    
    Args:
        task_data: Dictionary containing all ARC tasks
        task_id: The unique identifier of the task
        previous_json: The JSON response from the previous attempt
        train_results: List of training results showing predicted vs expected outputs
    
    Returns:
        A formatted prompt that includes the previous JSON, failure analysis,
        and asks for a corrected JSON response.
    """
    task = task_data[task_id]
    train = task.get("train", [])
    
    # Build training examples block
    examples_block = "\n\n".join(
        f"Training Example {i}\n--\nInput:\n" +
        "\n".join(" ".join(map(str, r)) for r in ex["input"]) +
        "\n\nOutput:\n" +
        "\n".join(" ".join(map(str, r)) for r in ex["output"])
        for i, ex in enumerate(train, 1)
    )
    
    # Build failure analysis block
    failure_analysis = []
    for i, result in enumerate(train_results, 1):
        if not result.get('correct', False):
            predicted = result.get('predicted', 'No output generated')
            expected = result.get('expected', 'Unknown')
            error = result.get('error', 'No error reported')
            
            failure_text = f"Training Example {i} - FAILED\n"
            failure_text += f"Expected Output:\n"
            if isinstance(expected, list):
                failure_text += "\n".join(" ".join(map(str, r)) for r in expected)
            else:
                failure_text += str(expected)
            
            failure_text += f"\n\nYour Predicted Output:\n"
            if isinstance(predicted, list):
                failure_text += "\n".join(" ".join(map(str, r)) for r in predicted)
            else:
                failure_text += str(predicted)
            
            if error and error != 'No error reported':
                failure_text += f"\n\nError Encountered:\n{error}"
            
            failure_analysis.append(failure_text)
    
    failures_block = "\n\n".join(failure_analysis) if failure_analysis else "No specific failure details available"
    
    # Format the previous JSON
    previous_json_str = json.dumps(previous_json, indent=2) if previous_json else "No previous JSON available"
    
    return f"""
You are an expert in reasoning about Abstract Reasoning Corpus (ARC) puzzles. 
You previously attempted to solve this task but your solution was incorrect on some training examples.

====================
Task {task_id} - REFLECTION AND CORRECTION
====================

Your goal:
Analyze your previous attempt, understand why it failed, and provide a CORRECTED transformation rule that:
1. Correctly maps every training input to its output.
2. Is general and intuitive (no memorization or hard-coded values).
3. Is logical, reproducible, and object-level.

====================
Original Training Examples
====================
{examples_block}

====================
Your Previous JSON Response
====================
```json
{previous_json_str}
```

====================
Analysis of Failures
====================
{failures_block}

====================
Instructions for Reflection
====================
First, analyze what went wrong in your previous attempt inside ```reasoning ``` block:
- Identify which part of your logic was incorrect
- Understand why the predicted outputs didn't match the expected outputs
- Determine what pattern or rule you missed or misinterpreted
- Think about how to fix the transformation logic

Then provide a CORRECTED JSON object inside ```json ``` block with the same format:
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
  }}],
  "python_code": [
    "def transform(grid):",
    "    # Complete transformation implementation",
    "    # This function must be fully executable on its own",
    "    # and return a complete transformed grid",
    "    return processed_grid"
  ]
}}

Make sure your corrected solution addresses all the failures identified above and works correctly on ALL training examples.
""".strip()


if __name__ == "__main__":
    import os
    import json
    from typing import Optional
    
    # Sample data for reflection prompt testing
    sample_task_data = {
        "test_reflection_001": {
            "train": [
                {
                    "input": [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                    "output": [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
                }
            ]
        }
    }
    
    sample_previous_json = {
        "helper_python_functions": [],
        "step_by_step_transformations": [{
            "step_number": 1,
            "description": ["Flip all 0s and 1s"],
            "pseudo_code": ["Replace 0 with 1, replace 1 with 0"]
        }],
        "python_code": [
            "def transform(grid):",
            "    # This was wrong - just returned input",
            "    return grid"
        ]
    }
    
    sample_train_results = [
        {
            "correct": False,
            "predicted": [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            "expected": [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
            "error": "Output did not match expected - values were not flipped"
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
    print("TESTING ARC REFLECTION PROMPT")
    print("=" * 60)
    
    test_prompt = build_arc_reflection_prompt(
        task_data=sample_task_data,
        task_id="test_reflection_001",
        previous_json=sample_previous_json,
        train_results=sample_train_results
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