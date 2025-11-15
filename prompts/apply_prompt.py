"""
Apply prompt builder for executing inferred transformation rules on test inputs.

This function creates a detailed prompt that instructs an AI model to execute
a previously inferred rule on test inputs for an ARC task.
"""


def build_apply_prompt(final_json: dict, task: dict, task_id: str, include_examples=True) -> str:
    """
    Builds a prompt to apply an inferred transformation rule to test inputs.

    This function creates a detailed prompt that instructs an AI model to execute
    a previously inferred rule on test inputs for an ARC task. The prompt includes
    the rule summary, step-by-step instructions, and optionally the training examples
    for context.

    Args:
        final_json (dict): A dictionary containing the inferred rule with keys like
            'rule_summary', 'step_by_step_rule', and optionally 'pseudocode'.
        task (dict): A dictionary representing the ARC task, containing 'train' and
            'test' keys with their respective example lists.
        task_id (str): The unique identifier of the task.
        include_examples (bool, optional): Whether to include training examples in the
            prompt for additional context. Defaults to True.

    Returns:
        str: A formatted string containing the complete application prompt, including
            the rule details, training examples (if included), test inputs, and
            instructions for step-by-step execution with intermediate reasoning.

    The prompt requires the model to:
    - Apply the rule exactly to each test input
    - Show intermediate reasoning in <reasoning> blocks with step details
    - Provide final outputs in <output> blocks
    - Maintain valid ARC grid format (2D arrays of integers)
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
You are an expert ARC executor.

====================
Task {task_id}
====================

Rule Summary:
{rule_summary}

Step-by-Step Rule:
{chr(10).join(step_by_step)}

Pseudocode:
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
1) Apply the rule EXACTLY, step-by-step, to each test input.
2) Show your intermediate reasoning inside a <reasoning></reasoning> block.
3) Within the reasoning block, after each step, shows the intermediate grid state in the following format:
  <step>
  {{
    "step_number": i,
    "description": "Describe what is being done in this step.",
    "before": [[...]],   // grid state before applying this step
    "after":  [[...]]    // grid state after applying this step
  }}
  </step>
4) Provide ONLY the final output grid for each test input inside <output></output>.
5) Ensure valid ARC format (2D array of integers).
6) No text outside of <reasoning> and <output> blocks.
""".strip()


if __name__ == "__main__":
    import os
    import json
    from pathlib import Path
    from typing import Optional
    
    # ANSI color codes
    class Colors:
        BLUE = '\033[94m'      # Blue for prompts
        WHITE = '\033[97m'     # White for responses
        GREEN = '\033[92m'     # Green for success
        RED = '\033[91m'       # Red for errors
        RESET = '\033[0m'      # Reset to default
        BOLD = '\033[1m'       # Bold text
    
    def print_colored(text: str, color: str = Colors.WHITE, bold: bool = False):
        """Print text with specified color."""
        formatting = f"{Colors.BOLD if bold else ''}{color}"
        print(f"{formatting}{text}{Colors.RESET}")
    
    # Load real ARC task data
    def load_real_task():
        sample_task_path = Path(__file__).parent / "sample_task.json"
        if not sample_task_path.exists():
            print_colored(f"❌ Sample task file not found: {sample_task_path}", Colors.RED)
            return None, None
        
        try:
            with open(sample_task_path, 'r') as f:
                data = json.load(f)
            task_id = "009d5c81"
            task = data[task_id]
            return task, task_id
        except Exception as e:
            print_colored(f"❌ Error loading sample task: {e}", Colors.RED)
            return None, None
    
    # Real ARC task analysis for apply prompt
    sample_final_json = {
        "rule_summary": "Replace all 8s with a color based on the 1-pattern below. Pattern analysis shows: 1s form a small cross/plus shape, and this determines the replacement color for 8s. Cross pointing up/down = color 2, L-shaped = color 3, etc.",
        "step_by_step_rule": [
            "1) Locate all 1s in the grid to identify the pattern shape",
            "2) Determine the pattern type (cross, L-shape, T-shape, etc.)",
            "3) Based on pattern type, choose replacement color (2, 3, 7, etc.)",
            "4) Replace all 8s with the determined color",
            "5) Remove all 1s (set to 0)"
        ],
        "pseudocode": "pattern = analyze_ones_pattern(grid); color = get_color_for_pattern(pattern); replace_all(grid, 8, color); replace_all(grid, 1, 0)"
    }
    
    sample_task, task_id = load_real_task()
    if not sample_task:
        exit(1)
    
    def test_with_gemini_api(prompt_text: str, api_key: Optional[str] = None):
        """Test the prompt using Gemini API directly."""
        try:
            import google.generativeai as genai
            
            api_key = api_key or os.getenv('GEMINI_API_KEY')
            if not api_key:
                print_colored("❌ Gemini API key not found. Set GEMINI_API_KEY environment variable.", Colors.RED)
                return
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            print_colored("🔄 Testing with Gemini API...", Colors.GREEN)
            print_colored("\n" + "="*80, Colors.BLUE)
            print_colored("PROMPT SENT TO LLM:", Colors.BLUE, bold=True)
            print_colored("="*80, Colors.BLUE)
            print_colored(prompt_text, Colors.BLUE)
            print_colored("="*80, Colors.BLUE)
            
            response = model.generate_content(prompt_text)
            
            print_colored("\n" + "="*80, Colors.WHITE)
            print_colored("GEMINI API RESPONSE:", Colors.GREEN, bold=True)
            print_colored("="*80, Colors.WHITE)
            print_colored(response.text, Colors.WHITE)
            print_colored("="*80, Colors.WHITE)
            
        except ImportError:
            print_colored("❌ google-generativeai not installed. Run: pip install google-generativeai", Colors.RED)
        except Exception as e:
            print_colored(f"❌ Gemini API error: {e}", Colors.RED)
    
    def test_with_ollama_api(prompt_text: str, model_name: str = "llama3.2"):
        """Test the prompt using Ollama API directly."""
        try:
            import requests
            
            print_colored(f"🔄 Testing with Ollama API ({model_name})...", Colors.GREEN)
            print_colored("\n" + "="*80, Colors.BLUE)
            print_colored("PROMPT SENT TO LLM:", Colors.BLUE, bold=True)
            print_colored("="*80, Colors.BLUE)
            print_colored(prompt_text, Colors.BLUE)
            print_colored("="*80, Colors.BLUE)
            
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
                print_colored("\n" + "="*80, Colors.WHITE)
                print_colored("OLLAMA API RESPONSE:", Colors.GREEN, bold=True)
                print_colored("="*80, Colors.WHITE)
                print_colored(result.get('response', 'No response'), Colors.WHITE)
                print_colored("="*80, Colors.WHITE)
            else:
                print_colored(f"❌ Ollama API error: {response.status_code}", Colors.RED)
                
        except ImportError:
            print_colored("❌ requests library not installed. Run: pip install requests", Colors.RED)
        except requests.exceptions.ConnectionError:
            print_colored("❌ Cannot connect to Ollama. Make sure Ollama is running on localhost:11434", Colors.RED)
        except Exception as e:
            print_colored(f"❌ Ollama API error: {e}", Colors.RED)
    
    def test_with_langchain(prompt_text: str, api_key: Optional[str] = None):
        """Test the prompt using LangChain."""
        try:
            from langchain_google_vertexai import ChatVertexAI
            from langchain_core.messages import HumanMessage
            
            print_colored("🔄 Testing with LangChain (VertexAI)...", Colors.GREEN)
            print_colored("\n" + "="*80, Colors.BLUE)
            print_colored("PROMPT SENT TO LLM:", Colors.BLUE, bold=True)
            print_colored("="*80, Colors.BLUE)
            print_colored(prompt_text, Colors.BLUE)
            print_colored("="*80, Colors.BLUE)
            
            llm = ChatVertexAI(
                model_name="gemini-2.5-flash",
                project=os.getenv('GOOGLE_CLOUD_PROJECT'),
                location=os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
            )
            
            message = HumanMessage(content=prompt_text)
            response = llm.invoke([message])
            
            print_colored("\n" + "="*80, Colors.WHITE)
            print_colored("LANGCHAIN VERTEXAI RESPONSE:", Colors.GREEN, bold=True)
            print_colored("="*80, Colors.WHITE)
            print_colored(response.content, Colors.WHITE)
            print_colored("="*80, Colors.WHITE)
            
        except ImportError as e:
            print_colored(f"❌ LangChain dependencies not installed: {e}", Colors.RED)
            print_colored("Run: pip install langchain-google-vertexai", Colors.RED)
        except Exception as e:
            print_colored(f"❌ LangChain VertexAI error: {e}", Colors.RED)
    
    # Generate test prompt
    print_colored("=" * 60, Colors.GREEN)
    print_colored(f"TESTING APPLY PROMPT WITH REAL ARC TASK {task_id}", Colors.GREEN, bold=True)
    print_colored("=" * 60, Colors.GREEN)
    
    test_prompt = build_apply_prompt(
        final_json=sample_final_json,
        task=sample_task,
        task_id=task_id,
        include_examples=True
    )
    
    # Test with different APIs
    print_colored("\n1. Testing with Gemini API:", Colors.GREEN, bold=True)
    test_with_gemini_api(test_prompt)
    
    print_colored("\n2. Testing with Ollama API:", Colors.GREEN, bold=True)
    test_with_ollama_api(test_prompt)
    
    print_colored("\n3. Testing with LangChain:", Colors.GREEN, bold=True)
    test_with_langchain(test_prompt)