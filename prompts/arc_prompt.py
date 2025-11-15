"""
Main ARC prompt builder for comprehensive ARC task solving.

This function constructs a detailed prompt that instructs an AI model to reason about
and solve an ARC puzzle. The prompt includes the task's training examples, test inputs,
guidelines for rule inference, and a specified output format.
"""


def build_arc_prompt(task_data: dict, task_id: str) -> str:
    """
    Builds a comprehensive prompt for an ARC (Abstraction and Reasoning Corpus) task.

    This function constructs a detailed prompt that instructs an AI model to reason about
    and solve an ARC puzzle. The prompt includes the task's training examples, test inputs,
    guidelines for rule inference, and a specified output format.

    Args:
        task_data (dict): A dictionary containing all ARC tasks, where keys are task IDs
            and values are task dictionaries with 'train' and 'test' keys.
        task_id (str): The unique identifier of the specific task to build a prompt for.

    Returns:
        str: A formatted string containing the complete prompt for the ARC task, including
            task description, training examples, test inputs, guidelines, and output format
            instructions.

    The prompt encourages the model to:
    - Infer a single, general transformation rule from training pairs
    - Use object-level reasoning rather than memorization
    - Output the rule in a structured JSON format within <json> tags
    """
    task = task_data[task_id]
    train, test = task.get("train", []), task.get("test", [])

    examples_block = "\n\n".join(
        f"Training Example {i}\n--\nInput:\n" +
        "\n".join("".join(map(str, r)) for r in ex["input"]) +
        "\n\nOutput:\n" +
        "\n".join("".join(map(str, r)) for r in ex["output"])
        for i, ex in enumerate(train, 1)
    )
    tests_block = "\n\n".join(
        f"Test Input {i}\n--\nInput:\n" +
        "\n".join(" ".join(map(str, r)) for r in t["input"])
        for i, t in enumerate(test, 1)
    )

    return f"""
You are an expert in reasoning about Abstract Reasoning Corpus (ARC) puzzles.

====================
Task {task_id}
====================

Your goal:
Given the training pairs and test inputs, infer a general transformation rule that:
1. Correctly maps every training input to its output.
2. Is general and intuitive (no memorization or hard-coded values).
3. Is logical, reproducible, and object-level.

====================
Guidelines
====================
- The SAME rule must successfully transform all training pairs.
- Treat all grid values (numbers/characters) as categorical labels, not magnitudes. Do not use arithmetic operations.
- Avoid rules that depend on specific values or characters. 
- Make rules in a general manner using object-level reasoning (movements, shapes, fills, mirrors, rotations, bounding boxes, duplicates, etc.).
- Take as many rules as you need to achieve your goals.

====================
Training Examples
====================
{examples_block}

====================
Output Format
====================
First, show your reasoning process inside ```reasoning ``` block:
- Analyze each training example carefully
- Look for patterns, transformations, and commonalities
- Identify the core transformation rule(s)

Be concise in your reasoning. Keep it short but informative.

Then you MUST return a JSON object inside ```json ``` block:
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
```""".strip()


if __name__ == "__main__":
    import os
    import json
    from pathlib import Path
    from typing import Optional
    
    # Load environment variables from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("Warning: python-dotenv not installed. Run: pip install python-dotenv")
        print("Continuing with system environment variables...")
    
    # ANSI color codes
    class Colors:
        BLUE = '\033[94m'
        WHITE = '\033[97m'
        GREEN = '\033[92m'
        RED = '\033[91m'
        RESET = '\033[0m'
        BOLD = '\033[1m'
    
    def print_colored(text: str, color: str = Colors.WHITE, bold: bool = False):
        formatting = f"{Colors.BOLD if bold else ''}{color}"
        print(f"{formatting}{text}{Colors.RESET}")
    
    # Load real ARC task data
    def load_real_task():
        # Try multiple possible paths for sample_task.json
        possible_paths = [
            Path(__file__).parent / "sample_task.json",  # Same directory as script
            Path.cwd() / "prompts" / "sample_task.json",  # Called from parent directory
            Path.cwd() / "sample_task.json",  # Called from same directory
        ]
        
        for sample_task_path in possible_paths:
            if sample_task_path.exists():
                try:
                    with open(sample_task_path, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    print_colored(f"❌ Error loading sample task from {sample_task_path}: {e}", Colors.RED)
                    continue
        
        print_colored("❌ Sample task file not found in any of the expected locations:", Colors.RED)
        for path in possible_paths:
            print_colored(f"  - {path}", Colors.RED)
        return {}
    
    # Load real task data
    sample_task_data = load_real_task()
    if not sample_task_data:
        exit(1)
    
    def test_with_gemini_api(prompt_text: str, api_key: Optional[str] = None):
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
    
    def test_with_ollama_api(prompt_text: str, model_name: str = "llama3.1:latest"):
        try:
            import requests
            print_colored(f"🔄 Testing with Ollama API ({model_name})...", Colors.GREEN)
            print_colored("\\n" + "="*80, Colors.BLUE)
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
                print_colored("\\n" + "="*80, Colors.WHITE)
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
            print_colored("\\n" + "="*80, Colors.BLUE)
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
            print_colored("\\n" + "="*80, Colors.WHITE)
            print_colored("LANGCHAIN VERTEXAI RESPONSE:", Colors.GREEN, bold=True)
            print_colored("="*80, Colors.WHITE)
            print_colored(response.content, Colors.WHITE)
            print_colored("="*80, Colors.WHITE)
            
        except ImportError as e:
            print_colored(f"❌ LangChain dependencies not installed: {e}", Colors.RED)
            print_colored("Run: pip install langchain-google-vertexai", Colors.RED)
        except Exception as e:
            print_colored(f"❌ LangChain VertexAI error: {e}", Colors.RED)
    
    # Generate test prompt with real ARC task
    task_id = "009d5c81"
    print_colored("=" * 60, Colors.GREEN)
    print_colored(f"TESTING MAIN ARC PROMPT WITH REAL TASK {task_id}", Colors.GREEN, bold=True)
    print_colored("=" * 60, Colors.GREEN)
    
    test_prompt = build_arc_prompt(
        task_data=sample_task_data,
        task_id=task_id
    )
    
    # Test with different APIs
    print_colored("\n1. Testing with Gemini API:", Colors.GREEN, bold=True)
    test_with_gemini_api(test_prompt)
    
    print_colored("\n2. Testing with Ollama API:", Colors.GREEN, bold=True)
    test_with_ollama_api(test_prompt)
    
    print_colored("\n3. Testing with LangChain:", Colors.GREEN, bold=True)
    test_with_langchain(test_prompt)