#!/usr/bin/env python3
"""
Test runner for all prompt files in the prompts folder.

This script demonstrates how to test each prompt file individually.
Each prompt file now has its own __main__ section with test functionality for:
1. Direct API calls (Gemini/Ollama)
2. LangChain integration

Usage:
    python test_all_prompts.py [prompt_name]
    
    If prompt_name is provided, only that prompt will be tested.
    Otherwise, all prompts will be tested.

Examples:
    python test_all_prompts.py                    # Test all prompts
    python test_all_prompts.py apply_prompt       # Test only apply_prompt.py
    python test_all_prompts.py arc_prompt         # Test only arc_prompt.py

Requirements:
    - Set GEMINI_API_KEY environment variable for Gemini API testing
    - Have Ollama running on localhost:11434 for Ollama API testing
    - Install dependencies: pip install google-generativeai langchain-google-genai requests
"""

import os
import sys
import subprocess
from pathlib import Path


def get_prompt_files():
    """Get all prompt files that have __main__ sections."""
    prompt_files = [
        "apply_prompt.py",
        "apply_prompt_2.py", 
        "arc_prompt.py",
        "arc_reflection_prompt.py",
        "baseline_prompt.py",
        "code_repair_prompt.py",
        "continuation_prompt.py",
        "enhanced_code_repair_prompt.py",
        "json_regeneration_prompt.py",
        "json_repair_prompt.py",
        "reflection_prompt.py"
    ]
    
    # Verify files exist
    prompts_dir = Path(__file__).parent
    existing_files = []
    for file in prompt_files:
        file_path = prompts_dir / file
        if file_path.exists():
            existing_files.append(file)
        else:
            print(f"Warning: {file} not found in {prompts_dir}")
    
    return existing_files


def run_prompt_test(prompt_file):
    """Run the test for a specific prompt file."""
    prompts_dir = Path(__file__).parent
    file_path = prompts_dir / prompt_file
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    print(f"\n{'='*80}")
    print(f"TESTING {prompt_file.upper()}")
    print(f"{'='*80}")
    
    try:
        # Change to the prompts directory to run the script
        original_cwd = os.getcwd()
        os.chdir(prompts_dir)
        
        # Run the prompt file directly
        result = subprocess.run([sys.executable, prompt_file], 
                              capture_output=False, 
                              text=True)
        
        os.chdir(original_cwd)
        
        if result.returncode == 0:
            print(f"\n✅ {prompt_file} test completed successfully")
            return True
        else:
            print(f"\n❌ {prompt_file} test failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error testing {prompt_file}: {e}")
        return False


def main():
    """Main function to run prompt tests."""
    # Check if a specific prompt was requested
    specific_prompt = None
    if len(sys.argv) > 1:
        specific_prompt = sys.argv[1]
        if not specific_prompt.endswith('.py'):
            specific_prompt += '.py'
    
    # Get available prompt files
    prompt_files = get_prompt_files()
    
    if not prompt_files:
        print("❌ No prompt files found!")
        return
    
    # Filter to specific prompt if requested
    if specific_prompt:
        if specific_prompt in prompt_files:
            prompt_files = [specific_prompt]
        else:
            print(f"❌ Prompt file '{specific_prompt}' not found!")
            print(f"Available prompts: {', '.join(prompt_files)}")
            return
    
    print("🚀 ARC Solver Prompt Testing Suite")
    print(f"Found {len(prompt_files)} prompt files to test")
    
    # Display setup instructions
    print("\n📋 Setup Requirements:")
    print("1. Set GEMINI_API_KEY environment variable for Gemini API testing")
    print("2. Have Ollama running on localhost:11434 for Ollama API testing")
    print("3. Install dependencies:")
    print("   pip install google-generativeai langchain-google-genai requests")
    
    # Check environment setup
    print("\n🔍 Environment Check:")
    gemini_key = os.getenv('GEMINI_API_KEY')
    print(f"   GEMINI_API_KEY: {'✅ Set' if gemini_key else '❌ Not set'}")
    
    try:
        import requests
        ollama_response = requests.get("http://localhost:11434/api/version", timeout=2)
        ollama_running = ollama_response.status_code == 200
    except:
        ollama_running = False
    print(f"   Ollama Server: {'✅ Running' if ollama_running else '❌ Not running'}")
    
    # Test each prompt file
    successful_tests = 0
    total_tests = len(prompt_files)
    
    for prompt_file in prompt_files:
        success = run_prompt_test(prompt_file)
        if success:
            successful_tests += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    print(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")
    
    if successful_tests == total_tests:
        print("\n🎉 All prompt tests completed successfully!")
    else:
        print(f"\n⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()