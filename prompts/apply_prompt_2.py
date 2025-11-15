"""
Alternative apply prompt builder that suggests recommended transformation rulesets.

Unlike build_apply_prompt(), this version does not enforce exact step-by-step
execution. Instead, it presents the ruleset as recommended reasoning patterns
or transformation principles.
"""


def build_apply_prompt_2(final_json: dict, task: dict, task_id: str, include_examples=True) -> str:
    """
    Builds a prompt that suggests a recommended transformation ruleset to guide
    the model when applying it to test inputs.

    Unlike build_apply_prompt(), this version does not enforce exact step-by-step
    execution. Instead, it presents the ruleset as recommended reasoning patterns
    or transformation principles that should inform how the test inputs are handled.

    Args:
        final_json (dict): Contains inferred rules, with optional keys such as
            'rule_summary', 'step_by_step_rule', and 'pseudocode'.
        task (dict): The ARC task data (train/test examples).
        task_id (str): Unique identifier of the task.
        include_examples (bool, optional): Whether to include training examples
            for additional context. Defaults to True.

    Returns:
        str: A natural-language prompt emphasizing interpretation and reasoning,
             while still including reasoning and output blocks.
    """
    rule_summary = final_json.get("rule_summary", "")
    step_by_step = final_json.get("step_by_step_rule", [])
    pseudocode = final_json.get("pseudocode", "")
    examples_block = ""

    if include_examples:
        examples_block = "\n\n".join(
            f"Training Example {i}\n--\nInput:\n" +
            "\n".join(" ".join(map(str, r)) for r in ex["input"]) +
            "\n\nOutput:\n" +
            "\n".join(" ".join(map(str, r)) for r in ex["output"])
            for i, ex in enumerate(task.get("train", []), 1)
        )

    tests_block = "\n\n".join(
        f"Test Input {i}\n--\nInput:\n" +
        "\n".join(" ".join(map(str, r)) for r in t["input"])
        for i, t in enumerate(task.get("test", []), 1)
    )

    return f"""
You are an expert in visual reasoning and transformation synthesis for ARC tasks.

====================
Task {task_id}
====================

Recommended Rule Summary:
{rule_summary}

Guided Transformation Steps (for reference):
{chr(10).join(step_by_step)}

Reference Pseudocode (optional):
{pseudocode}

====================
Training Examples
====================
{examples_block}

====================
Test Inputs
====================
{tests_block}

====================
Instructions
====================
1) Use the above rules and examples as RECOMMENDED GUIDANCE — you may interpret or generalize them as needed.
2) For each test input, describe your reasoning and transformation process inside a <reasoning></reasoning> block.
3) Within the reasoning block, show intermediate reasoning and key decisions:
   <step>
   {{
      "thought": "Explain what you observed or inferred.",
      "action": "Describe the transformation applied (if any).",
      "grid_snapshot": [[...]]  // optional intermediate state
   }}
   </step>
4) Provide the final transformed grid for each test input inside <output></output>.
5) Stay consistent with the overall transformation logic demonstrated in the training examples.
6) Keep valid ARC grid formatting (2D arrays of integers).
""".strip()


if __name__ == "__main__":
    import os
    import json
    from typing import Optional
    
    # Test data for apply prompt 2
    sample_final_json = {
        "rule_summary": "Mirror the input horizontally and fill empty spaces with blue (1)",
        "step_by_step_rule": [
            "1) Create a copy of the input grid",
            "2) Mirror/flip the grid horizontally",
            "3) Fill any empty or zero cells with blue (1)"
        ],
        "pseudocode": "mirrored_grid = flip_horizontal(input); fill_zeros(mirrored_grid, 1)"
    }
    
    sample_task = {
        "train": [
            {
                "input": [[2, 0, 0], [0, 3, 0], [0, 0, 2]],
                "output": [[1, 1, 2], [1, 3, 1], [2, 1, 1]]
            }
        ],
        "test": [
            {"input": [[3, 0, 2], [0, 0, 0], [2, 3, 0]]}
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
    print("TESTING APPLY PROMPT 2 (FLEXIBLE INTERPRETATION)")
    print("=" * 60)
    
    test_prompt = build_apply_prompt_2(
        final_json=sample_final_json,
        task=sample_task,
        task_id="test_002",
        include_examples=True
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