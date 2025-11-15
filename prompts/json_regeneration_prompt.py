def create_json_regeneration_prompt(previous_response):
    """Create a prompt to regenerate JSON when reasoning is complete but JSON is invalid."""
    return f"""Your reasoning analysis is complete, but the JSON output needs to be regenerated in the correct format.

Your previous analysis:
{previous_response}

Based on your analysis above, please provide ONLY the final JSON response in the exact required format:

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

CRITICAL REQUIREMENTS:
- The `python_code` section contains ONE complete `def transform(grid):` function
- The `step_by_step_transformations` describe the logic conceptually (not executable code)
- The single `transform(grid)` function must implement ALL transformation steps
- The function must be fully executable and return a complete transformed grid
- DO NOT split the actual implementation across multiple functions"""


if __name__ == "__main__":
    import os
    import json
    from typing import Optional
    
    # Sample incomplete previous response
    sample_previous_response = """
Analyzing the training examples, I can see that:

1. In example 1, the input grid has objects in certain positions
2. The output shows these objects moved according to a pattern
3. It seems like objects are being moved to fill empty spaces

The transformation rule appears to be:
- Find all non-zero objects
- Move them towards the center
- Fill gaps systematically

Based on this analysis, the transformation should:
1. Identify all colored objects (non-zero values)
2. Calculate center of mass
3. Move objects towards center while maintaining relative positions

This needs to be implemented as a complete transform function.
    """
    
    def test_with_gemini_api(prompt_text: str, api_key: Optional[str] = None):
        """Test the prompt using Gemini API directly."""
        try:
            import google.generativeai as genai
            
            api_key = api_key or os.getenv('GEMINI_API_KEY')
            if not api_key:
                print("❌ Gemini API key not found. Set GEMINI_API_KEY environment variable.")
                return
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
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
                model_name="gemini-1.5-flash",
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
    print("TESTING JSON REGENERATION PROMPT")
    print("=" * 60)
    
    test_prompt = create_json_regeneration_prompt(sample_previous_response)
    
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