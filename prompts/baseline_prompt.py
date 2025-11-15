"""
Baseline prompt builder for minimal ARC task solving.

This function builds a minimal prompt that asks the model to output 
the test outputs directly without detailed reasoning or JSON structure.
"""


def build_arc_baseline_prompt(task_data: dict, task_id: str) -> str:
    """Build a minimal prompt that asks the model to output the test outputs.

    The prompt includes the training examples for context but then asks the model
    explicitly to return ONLY the final test output arrays in a ```json ``` block
    as a plain 2D array or list of 2D arrays (if multiple test inputs).
    """
    task = task_data[task_id]
    train, test = task.get("train", []), task.get("test", [])

    examples_block = "\n\n".join(
        f"Training Example {i}\nInput:\n" + "\n".join(" ".join(map(str, r)) for r in ex["input"]) +
        "\n\nOutput:\n" + "\n".join(" ".join(map(str, r)) for r in ex["output"]) 
        for i, ex in enumerate(train, 1)
    )

    tests_block = "\n\n".join(
        f"Test Input {i}\nInput:\n" + "\n".join(" ".join(map(str, r)) for r in t["input"]) 
        for i, t in enumerate(test, 1)
    )

    return f"""
You are an assistant that given ARC training examples should provide the
final transformed grid(s) for the TEST input(s) only.

Task: {task_id}

Training examples (for context):
{examples_block}

Test inputs:
{tests_block}

Instruction:
Return ONLY a JSON block (delimited with ```json ... ```). The JSON MUST be
either a single 2D array (if there is one test input) or a list of 2D arrays
(if there are multiple test inputs). Each 2D array should be composed of integers
matching ARC grid format. No extra text, reasoning, or explanation should be
included outside the ```json ``` block.
""".strip()


if __name__ == "__main__":
    import os
    import json
    from pathlib import Path
    from typing import Optional
    
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
        sample_task_path = Path(__file__).parent / "sample_task.json"
        if not sample_task_path.exists():
            print_colored(f"❌ Sample task file not found: {sample_task_path}", Colors.RED)
            return {}
        try:
            with open(sample_task_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print_colored(f"❌ Error loading sample task: {e}", Colors.RED)
            return {}
    
    sample_task_data = load_real_task()
    if not sample_task_data:
        exit(1)
    
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
    
    # Generate test prompt with real task
    task_id = "009d5c81"
    print_colored("=" * 60, Colors.GREEN)
    print_colored(f"TESTING BASELINE PROMPT WITH REAL TASK {task_id}", Colors.GREEN, bold=True)
    print_colored("=" * 60, Colors.GREEN)
    
    test_prompt = build_arc_baseline_prompt(
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