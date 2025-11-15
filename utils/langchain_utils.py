"""LangChain integration utilities."""

import os


def run_with_langchain(prompt, model_name):
    """Run the prompt using LangChain with the specified model."""
    try:
        # Try different LangChain integrations
        if "gemini" in model_name.lower():
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=os.getenv('GEMINI_API_KEY')
                )
            except ImportError:
                try:
                    from langchain_google_vertexai import ChatVertexAI
                    llm = ChatVertexAI(
                        model_name=model_name,
                        project=os.getenv('GOOGLE_CLOUD_PROJECT'),
                        location=os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
                    )
                except ImportError:
                    raise ImportError("Neither langchain_google_genai nor langchain_google_vertexai is installed")
                    
        elif "gpt" in model_name.lower() or "openai" in model_name.lower():
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=model_name,
                api_key=os.getenv('OPENAI_API_KEY')
            )
            
        elif "claude" in model_name.lower():
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(
                model=model_name,
                api_key=os.getenv('ANTHROPIC_API_KEY')
            )
            
        else:
            # Default to Ollama for local models
            from langchain_ollama import ChatOllama
            llm = ChatOllama(model=model_name)
        
        from langchain_core.messages import HumanMessage
        
        # Generate response
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])
        
        return response.content
        
    except ImportError as e:
        from .display_utils import print_colored, RED, YELLOW
        print_colored(f"Missing LangChain dependency: {e}", RED)
        print_colored("Install required packages:", YELLOW)
        print_colored("  pip install langchain-google-genai  # For Gemini", YELLOW)
        print_colored("  pip install langchain-openai       # For OpenAI", YELLOW)
        print_colored("  pip install langchain-anthropic    # For Claude", YELLOW)
        print_colored("  pip install langchain-ollama       # For local models", YELLOW)
        return None
        
    except Exception as e:
        from .display_utils import print_colored, RED
        print_colored(f"LangChain error: {e}", RED)
        return None