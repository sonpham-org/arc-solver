"""
Reflection prompt builder for refining transformation rules based on failures.

This function creates a detailed prompt that instructs an AI model to analyze
incorrect outputs, identify flaws in the current rule, and propose improvements.
"""


def build_reflection_prompt(final_json: dict, wrong_cases: list, model_outputs: list,
                            task: dict = None, task_id: str = "") -> str:
    """
    Builds a prompt for reflecting on and refining a transformation rule based on failures.

    This function creates a detailed prompt that instructs an AI model to analyze
    incorrect outputs, identify flaws in the current rule, and propose improvements.
    It's used in an iterative refinement process for ARC task solving.

    Args:
        final_json (dict): A dictionary containing the current rule with keys like
            'rule_summary', 'step_by_step_rule', and optionally 'pseudocode'.
        wrong_cases (list): A list of tuples, where each tuple contains (input_grid, expected_output_grid)
            for cases where the current rule produced incorrect results.
        model_outputs (list): A list of the model's actual outputs for the wrong cases,
            corresponding to the wrong_cases list. Each output can be a 2D list or string.
        task (dict, optional): The full ARC task dictionary containing 'train' examples.
            Used to provide additional context. Defaults to None.
        task_id (str, optional): The unique identifier of the task. Defaults to "".

    Returns:
        str: A formatted string containing the complete reflection prompt, including
            the current rule, failure cases with inputs/expected/model outputs,
            training examples (if provided), and instructions for analysis and improvement.

    The prompt guides the model to:
    - Identify specific failures and their causes
    - Propose minimal, general fixes
    - Output an improved rule in JSON format within <json> tags
    - Show expected corrected outputs for each failure case
    - Ensure the refined rule is deterministic with tie-breakers
    """
    rule_summary = final_json.get("rule_summary", "")
    step_by_step = final_json.get("step_by_step_rule", [])
    pseudocode = final_json.get("pseudocode", "")

    wrong_block = "\n\n".join(
        f"Case {i}\n--\nInput:\n" +
        "\n".join(" ".join(map(str, r)) for r in inp) +
        "\n\nExpected Output:\n" +
        "\n".join(" ".join(map(str, r)) for r in exp) +
        "\n\nModel Output:\n" +
        ("\n".join(" ".join(map(str, r)) for r in model_outputs[i-1])
         if isinstance(model_outputs[i-1], list) else str(model_outputs[i-1]))
        for i, (inp, exp) in enumerate(wrong_cases, 1)
    )

    examples_block = ""
    if task and task.get("train"):
        examples_block = "\n\n".join(
            f"Training Example {i}\n--\nInput:\n" +
            "\n".join(" ".join(map(str, r)) for r in ex["input"]) +
            "\n\nOutput:\n" +
            "\n".join(" ".join(map(str, r)) for r in ex["output"])
            for i, ex in enumerate(task["train"], 1)
        )

    return f"""
You are a meticulous ARC reasoning assistant tasked with refining your transformation rule after observing failures.

====================
Task {task_id}
====================

Current Instruction (JSON Rule)
--------------------
Rule Summary:
{rule_summary}

Step-by-Step Rule:
{chr(10).join(step_by_step)}

Pseudocode:
{pseudocode}

====================
Observed Failures
====================
{wrong_block}

====================
Instructions for Reflection and Revision
====================
1) For each failure case, identify which assumption or omission caused the incorrect output.
2) Propose one minimal, general fix for that case.
3) Merge all fixes into a single improved JSON instruction of the same format:
  {{
    "rule_summary": "Describe the transformation conceptually.",
    "step_by_step_rule": [
      "1) ...", 
      "2) ..."]
  }}
4) Indicate which fixes correspond to which failure cases (via comments or inline notes).
5) Ensure the new rule is general, deterministic, and includes tie-breakers.
6) After the new JSON, conceptually apply it to each failed case and provide expected corrected outputs.

====================
Return Format
====================
- Step-by-step reflection for each case will be inside the <reasoning>...</reasoning> block.
- Final improved JSON inside <json>...</json>.
- For each wrong case, show expected fixed output inside <output_case_i>...</output_case_i>.
""".strip()


if __name__ == "__main__":
    import os
    import json
    from typing import Optional
    
    # Sample data for reflection prompt testing
    sample_final_json = {
        "rule_summary": "Copy input and change all 0s to 1s",
        "step_by_step_rule": [
            "1) Copy the input grid",
            "2) Replace all 0 values with 1"
        ],
        "pseudocode": "for each cell: if cell == 0 then cell = 1"
    }
    
    sample_wrong_cases = [
        ([[0, 1], [1, 0]], [[2, 1], [1, 2]]),  # Expected different transformation
        ([[0, 0, 1]], [[3, 3, 1]])  # Another failure case
    ]
    
    sample_model_outputs = [
        [[1, 1], [1, 1]],  # Wrong - just replaced 0s with 1s
        [[1, 1, 1]]       # Wrong - same issue
    ]
    
    sample_task = {
        "train": [
            {
                "input": [[0, 1], [1, 0]],
                "output": [[2, 1], [1, 2]]
            }
        ]
    }
    
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
    print("TESTING REFLECTION PROMPT")
    print("=" * 60)
    
    test_prompt = build_reflection_prompt(
        final_json=sample_final_json,
        wrong_cases=sample_wrong_cases,
        model_outputs=sample_model_outputs,
        task=sample_task,
        task_id="test_reflection_001"
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