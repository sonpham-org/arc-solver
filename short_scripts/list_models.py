#!/usr/bin/env python3
"""
List all available Gemini models accessible with your current GEMINI_API_KEY.
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

def main():
    # Load environment variables
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in .env")

    # Configure Gemini client
    genai.configure(api_key=api_key)

    # List available models
    print("Fetching available Gemini models...\n")
    models = genai.list_models()

    for m in models:
        print(f"🧠 {m.name}")
        if hasattr(m, "display_name"):
            print(f"   Display Name: {m.display_name}")
        if hasattr(m, "description"):
            print(f"   Description : {m.description}")
        if hasattr(m, "supported_generation_methods"):
            print(f"   Supported   : {m.supported_generation_methods}")
        print("-" * 60)

if __name__ == "__main__":
    main()