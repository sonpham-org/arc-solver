#!/usr/bin/env python3
"""
Test runner for prompt files using the real ARC task from sample_task.json.

This script loads the real ARC task and tests each prompt with it, providing
colored output for better visibility.

Usage:
    python test_with_sample_task.py [prompt_name]

Examples:
    python test_with_sample_task.py                    # Test all prompts
    python test_with_sample_task.py arc_prompt         # Test only arc_prompt.py
    python test_with_sample_task.py apply_prompt       # Test only apply_prompt.py
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any


# ANSI color codes
class Colors:
    BLUE = '\033[94m'      # Blue for prompts
    WHITE = '\033[97m'     # White for responses
    GREEN = '\033[92m'     # Green for success
    RED = '\033[91m'       # Red for errors
    YELLOW = '\033[93m'    # Yellow for warnings
    CYAN = '\033[96m'      # Cyan for info
    RESET = '\033[0m'      # Reset to default
    BOLD = '\033[1m'       # Bold text


def print_colored(text: str, color: str = Colors.WHITE, bold: bool = False):
    """Print text with specified color."""
    formatting = f"{Colors.BOLD if bold else ''}{color}"
    print(f"{formatting}{text}{Colors.RESET}")


def load_sample_task() -> Dict[str, Any]:
    """Load the sample task from sample_task.json."""
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


def test_with_gemini_api(prompt_text: str, api_key: Optional[str] = None):
    """Test the prompt using Gemini API directly with colored output."""
    try:
        import google.generativeai as genai
        
        api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            print_colored("❌ Gemini API key not found. Set GEMINI_API_KEY environment variable.", Colors.RED)
            return
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        print_colored("🔄 Testing with Gemini API...", Colors.CYAN)
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
    """Test the prompt using Ollama API directly with colored output."""
    try:
        import requests
        
        print_colored(f"🔄 Testing with Ollama API ({model_name})...", Colors.CYAN)
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
    """Test the prompt using LangChain with colored output."""
    try:
        from langchain_google_vertexai import ChatVertexAI
        from langchain_core.messages import HumanMessage
        
        
        print_colored("🔄 Testing with LangChain (VertexAI)...", Colors.CYAN)
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
        print_colored("LANGCHAIN RESPONSE:", Colors.GREEN, bold=True)
        print_colored("="*80, Colors.WHITE)
        print_colored(response.content, Colors.WHITE)
        print_colored("="*80, Colors.WHITE)
        
    except ImportError as e:
        print_colored(f"❌ LangChain dependencies not installed: {e}", Colors.RED)
        print_colored("Run: pip install langchain-google-vertexai", Colors.YELLOW)
    except Exception as e:
        print_colored(f"❌ LangChain error: {e}", Colors.RED)


def test_arc_prompt():
    """Test the main ARC prompt with sample task."""
    try:
        from arc_prompt import build_arc_prompt
        
        sample_data = load_sample_task()
        if not sample_data:
            return False
        
        task_id = "009d5c81"  # The task ID from sample_task.json
        
        print_colored(f"\n{'='*60}", Colors.CYAN)
        print_colored("TESTING ARC PROMPT WITH REAL TASK", Colors.CYAN, bold=True)
        print_colored(f"Task ID: {task_id}", Colors.CYAN)
        print_colored(f"{'='*60}", Colors.CYAN)
        
        test_prompt = build_arc_prompt(
            task_data=sample_data,
            task_id=task_id
        )
        
        print_colored("\n1. Testing with Gemini API:", Colors.YELLOW, bold=True)
        test_with_gemini_api(test_prompt)
        
        print_colored("\n2. Testing with Ollama API:", Colors.YELLOW, bold=True)
        test_with_ollama_api(test_prompt)
        
        print_colored("\n3. Testing with LangChain:", Colors.YELLOW, bold=True)
        test_with_langchain(test_prompt)
        
        return True
        
    except ImportError as e:
        print_colored(f"❌ Could not import arc_prompt: {e}", Colors.RED)
        return False
    except Exception as e:
        print_colored(f"❌ Error testing arc_prompt: {e}", Colors.RED)
        return False


def test_baseline_prompt():
    """Test the baseline prompt with sample task."""
    try:
        from baseline_prompt import build_arc_baseline_prompt
        
        sample_data = load_sample_task()
        if not sample_data:
            return False
        
        task_id = "009d5c81"
        
        print_colored(f"\n{'='*60}", Colors.CYAN)
        print_colored("TESTING BASELINE PROMPT WITH REAL TASK", Colors.CYAN, bold=True)
        print_colored(f"Task ID: {task_id}", Colors.CYAN)
        print_colored(f"{'='*60}", Colors.CYAN)
        
        test_prompt = build_arc_baseline_prompt(
            task_data=sample_data,
            task_id=task_id
        )
        
        print_colored("\n1. Testing with Gemini API:", Colors.YELLOW, bold=True)
        test_with_gemini_api(test_prompt)
        
        print_colored("\n2. Testing with Ollama API:", Colors.YELLOW, bold=True)
        test_with_ollama_api(test_prompt)
        
        print_colored("\n3. Testing with LangChain:", Colors.YELLOW, bold=True)
        test_with_langchain(test_prompt)
        
        return True
        
    except ImportError as e:
        print_colored(f"❌ Could not import baseline_prompt: {e}", Colors.RED)
        return False
    except Exception as e:
        print_colored(f"❌ Error testing baseline_prompt: {e}", Colors.RED)
        return False


def test_apply_prompts():
    """Test the apply prompts with sample task (requires a rule first)."""
    try:
        from apply_prompt import build_apply_prompt
        from apply_prompt_2 import build_apply_prompt_2
        
        sample_data = load_sample_task()
        if not sample_data:
            return False
        
        task_id = "009d5c81"
        task = sample_data[task_id]
        
        # Create a sample rule based on pattern analysis
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
        
        print_colored(f"\n{'='*60}", Colors.CYAN)
        print_colored("TESTING APPLY PROMPTS WITH REAL TASK", Colors.CYAN, bold=True)
        print_colored(f"Task ID: {task_id}", Colors.CYAN)
        print_colored(f"{'='*60}", Colors.CYAN)
        
        # Test apply_prompt.py
        print_colored("\n📋 TESTING EXACT RULE APPLICATION:", Colors.YELLOW, bold=True)
        test_prompt_1 = build_apply_prompt(
            final_json=sample_final_json,
            task=task,
            task_id=task_id,
            include_examples=True
        )
        
        print_colored("\n1. Testing with Gemini API:", Colors.YELLOW, bold=True)
        test_with_gemini_api(test_prompt_1)
        
        # Test apply_prompt_2.py  
        print_colored("\n📋 TESTING FLEXIBLE RULE INTERPRETATION:", Colors.YELLOW, bold=True)
        test_prompt_2 = build_apply_prompt_2(
            final_json=sample_final_json,
            task=task,
            task_id=task_id,
            include_examples=True
        )
        
        print_colored("\n1. Testing with Gemini API:", Colors.YELLOW, bold=True)
        test_with_gemini_api(test_prompt_2)
        
        return True
        
    except ImportError as e:
        print_colored(f"❌ Could not import apply prompts: {e}", Colors.RED)
        return False
    except Exception as e:
        print_colored(f"❌ Error testing apply prompts: {e}", Colors.RED)
        return False


def main():
    """Main function to run prompt tests with real ARC task."""
    # Check command line arguments
    specific_prompt = None
    if len(sys.argv) > 1:
        specific_prompt = sys.argv[1].lower()
    
    print_colored("🚀 ARC SOLVER PROMPT TESTING WITH REAL TASK", Colors.GREEN, bold=True)
    print_colored("=" * 60, Colors.GREEN)
    
    # Load and verify sample task
    sample_data = load_sample_task()
    if not sample_data:
        print_colored("❌ Cannot proceed without sample task data", Colors.RED)
        return
    
    task_id = "009d5c81"
    task = sample_data.get(task_id, {})
    train_examples = len(task.get('train', []))
    test_examples = len(task.get('test', []))
    
    print_colored(f"✅ Loaded task {task_id}:", Colors.GREEN)
    print_colored(f"   - Training examples: {train_examples}", Colors.WHITE)
    print_colored(f"   - Test examples: {test_examples}", Colors.WHITE)
    
    # Display setup status
    print_colored("\n📋 Environment Check:", Colors.CYAN, bold=True)
    gemini_key = os.getenv('GEMINI_API_KEY')
    print_colored(f"   GEMINI_API_KEY: {'✅ Set' if gemini_key else '❌ Not set'}", Colors.WHITE)
    
    try:
        import requests
        ollama_response = requests.get("http://localhost:11434/api/version", timeout=2)
        ollama_running = ollama_response.status_code == 200
    except:
        ollama_running = False
    print_colored(f"   Ollama Server: {'✅ Running' if ollama_running else '❌ Not running'}", Colors.WHITE)
    
    # Run tests based on selection
    if not specific_prompt or specific_prompt in ['arc', 'arc_prompt']:
        test_arc_prompt()
    
    if not specific_prompt or specific_prompt in ['baseline', 'baseline_prompt']:
        test_baseline_prompt()
    
    if not specific_prompt or specific_prompt in ['apply', 'apply_prompt', 'apply_prompts']:
        test_apply_prompts()
    
    if specific_prompt and specific_prompt not in ['arc', 'arc_prompt', 'baseline', 'baseline_prompt', 'apply', 'apply_prompt', 'apply_prompts']:
        print_colored(f"\n❌ Unknown prompt: {specific_prompt}", Colors.RED)
        print_colored("Available prompts: arc, baseline, apply", Colors.YELLOW)
    
    print_colored("\n🎉 Testing complete!", Colors.GREEN, bold=True)


if __name__ == "__main__":
    main()