def create_continuation_prompt(previous_response):
    """Create a prompt to continue reasoning when it appears incomplete."""
    return f"""Continue your analysis from where you left off. Your previous response was:

{previous_response}

Please continue your reasoning process and complete the analysis. Make sure to:

1. Continue from exactly where you stopped
2. Complete your step-by-step analysis of the patterns
3. Identify the core transformation rule(s)
4. Provide the final JSON response in the required format

Continue your reasoning:"""


if __name__ == "__main__":
    import os
    import json
    from typing import Optional
    
    # Sample incomplete response for testing
    sample_previous_response = """
Looking at the training examples, I can see a clear pattern:

Example 1: The input has some shapes and the output shows...
Example 2: Similarly, there are objects that get transformed by...

I notice that the transformation involves:
1) Identifying certain colored regions
2) Moving them according to some rule
3) But I need to analyze this more carefully...
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
    print("TESTING CONTINUATION PROMPT")
    print("=" * 60)
    
    test_prompt = create_continuation_prompt(sample_previous_response)
    
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