#!/usr/bin/env python3

import os
import json
import sys
from typing import Dict, List, Tuple, Any

try:
    import termios
    import tty
    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False


def get_key():
    """Get a single key press from user input"""
    if not HAS_TERMIOS:
        # Fallback for Windows or systems without termios
        return input("Press Enter to select, 'u' for up, 'd' for down, 'q' to quit: ").strip()
    
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def display_folders(folders: List[str], selected_index: int) -> None:
    """Display the list of folders with the current selection highlighted"""
    os.system('clear' if os.name == 'posix' else 'cls')
    print("=" * 50)
    print("ARC Solver Output Folder Selection")
    print("=" * 50)
    print("Use UP/DOWN arrow keys to navigate, ENTER to select, 'q' to quit\n")
    
    for i, folder in enumerate(folders):
        if i == selected_index:
            print(f"-> {folder} <-")
        else:
            print(f"   {folder}")
    
    print(f"\nSelected: {selected_index + 1}/{len(folders)}")


def select_folder(output_dir: str) -> str:
    """Interactive folder selection with up/down navigation"""
    if not os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' does not exist!")
        sys.exit(1)
    
    folders = [f for f in os.listdir(output_dir) 
              if os.path.isdir(os.path.join(output_dir, f))]
    
    if not folders:
        print("No folders found in output directory!")
        sys.exit(1)
    
    # Sort folders by modification time, newest first
    folders.sort(key=lambda f: os.path.getmtime(os.path.join(output_dir, f)), reverse=True)
    
    # Start cursor on the last (newest) option
    selected_index = 0
    
    while True:
        display_folders(folders, selected_index)
        
        if not HAS_TERMIOS:
            # Fallback interface
            key = get_key().lower()
            if key == 'u':
                selected_index = (selected_index - 1) % len(folders)
            elif key == 'd':
                selected_index = (selected_index + 1) % len(folders)
            elif key == 'q':
                sys.exit(0)
            elif key == '':  # Enter
                return os.path.join(output_dir, folders[selected_index])
        else:
            key = get_key()
            
            if key == '\x1b':  # ESC sequence
                key2 = get_key()
                if key2 == '[':
                    key3 = get_key()
                    if key3 == 'A':  # Up arrow
                        selected_index = (selected_index - 1) % len(folders)
                    elif key3 == 'B':  # Down arrow
                        selected_index = (selected_index + 1) % len(folders)
            elif key == '\r' or key == '\n':  # Enter
                return os.path.join(output_dir, folders[selected_index])
            elif key == 'q' or key == 'Q':
                sys.exit(0)


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load and parse a JSON file with error handling"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data if data is not None else {}
    except (json.JSONDecodeError, FileNotFoundError, TypeError) as e:
        # Silently handle errors - we'll report "No valid test data found" later
        return {}
    except Exception:
        # Handle any other unexpected errors
        return {}


def compare_arrays(output: Any, produce: Any) -> bool:
    """Compare two arrays for exact match, handling NoneType and other edge cases"""
    try:
        # Handle NoneType cases
        if output is None or produce is None:
            return False
        
        # Handle non-list types
        if not isinstance(output, list) or not isinstance(produce, list):
            return output == produce
        
        # Compare list lengths
        if len(output) != len(produce):
            return False
        
        # Compare each element
        for i in range(len(output)):
            if isinstance(output[i], list) and isinstance(produce[i], list):
                # Handle 2D arrays
                if len(output[i]) != len(produce[i]):
                    return False
                for j in range(len(output[i])):
                    if output[i][j] != produce[i][j]:
                        return False
            else:
                # Handle 1D arrays or other types
                if output[i] != produce[i]:
                    return False
        
        return True
    except Exception:
        # If any error occurs during comparison, return False
        return False


def calculate_score_for_file(file_path: str) -> Tuple[int, int]:
    """Calculate score for a single JSON file with proper error handling"""
    try:
        data = load_json_file(file_path)
        if not data:
            return 0, 0
        
        total_tests = 0
        correct_tests = 0
        
        # Handle both possible JSON structures
        for key, value in data.items():
            try:
                if isinstance(value, dict) and 'test' in value:
                    # Structure: { test_id: {test: [{input:, output:, produce:}]} }
                    test_data = value['test']
                    if isinstance(test_data, list):
                        for test_case in test_data:
                            try:
                                if 'output' in test_case and 'produce' in test_case:
                                    total_tests += 1
                                    output = test_case['output']
                                    produce = test_case['produce']
                                    
                                    # Handle NoneType or missing data
                                    if output is None or produce is None:
                                        continue  # Score remains 0 for this test
                                    
                                    if compare_arrays(output, produce):
                                        correct_tests += 1
                            except Exception:
                                # If error occurs processing this test case, count it as 0
                                total_tests += 1
                                continue
                                
                elif isinstance(value, dict) and all(k in value for k in ['input', 'output', 'produce']):
                    # Structure: {input:, output:, produce:}
                    try:
                        total_tests += 1
                        output = value['output']
                        produce = value['produce']
                        
                        # Handle NoneType or missing data
                        if output is None or produce is None:
                            continue  # Score remains 0 for this test
                        
                        if compare_arrays(output, produce):
                            correct_tests += 1
                    except Exception:
                        # If error occurs processing this test case, count it as 0
                        continue
            except Exception:
                # If error occurs processing this key-value pair, skip it
                continue
        
        # Check if the data itself has the direct structure
        try:
            if 'input' in data and 'output' in data and 'produce' in data:
                total_tests += 1
                output = data['output']
                produce = data['produce']
                
                # Handle NoneType or missing data
                if output is not None and produce is not None:
                    if compare_arrays(output, produce):
                        correct_tests += 1
        except Exception:
            # If error occurs processing root level data, skip it
            pass
        
        return correct_tests, total_tests
        
    except Exception:
        # If any error occurs with the entire file, return 0, 0
        return 0, 0


def calculate_scores_for_folder(folder_path: str) -> None:
    """Calculate and display scores for all JSON files in the selected folder"""
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist!")
        return
    
    json_files = [f for f in os.listdir(folder_path) 
                 if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in '{folder_path}'")
        return
    
    total_correct = 0
    total_tests = 0
    
    print(f"\nCalculating scores for folder: {os.path.basename(folder_path)}")
    print("=" * 60)
    
    for json_file in sorted(json_files):
        file_path = os.path.join(folder_path, json_file)
        correct, total = calculate_score_for_file(file_path)
        total_correct += correct
        total_tests += total
        
        if total > 0:
            percentage = (correct / total) * 100
            print(f"{json_file:<30} {correct:>3}/{total:<3} ({percentage:>5.1f}%)")
        else:
            print(f"{json_file:<30} No valid test data found")
    
    print("=" * 60)
    if total_tests > 0:
        overall_percentage = (total_correct / total_tests) * 100
        print(f"{'TOTAL':<30} {total_correct:>3}/{total_tests:<3} ({overall_percentage:>5.1f}%)")
    else:
        print("No valid test data found in any files")
    
    print(f"\nOverall Score: {total_correct} / {total_tests}")


def main():
    """Main function"""
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    
    print("Welcome to ARC Solver Score Calculator!")
    print("This tool will help you calculate scores for JSON test files.")
    
    selected_folder = select_folder(output_dir)
    calculate_scores_for_folder(selected_folder)


if __name__ == "__main__":
    main()