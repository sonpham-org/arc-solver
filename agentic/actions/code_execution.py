"""
Code execution and evaluation functions for ARC agent.
"""

import ast
import traceback
import subprocess
import sys
import re
from typing import List, Dict, Optional, Any, Tuple

from .code_generation import extract_python_solutions, ensure_imports_in_code
from .utilities import format_grid_for_prompt, generate_llm_predicted_output
from ..schema import ExampleResult
from agentic.debug import print_prompt_and_response

# Whitelist of safe scientific/data libraries that can be auto-installed
SAFE_LIBRARIES = {
    'scipy', 'scikit-learn', 'sklearn', 'networkx', 'sympy', 'statsmodels',
    'seaborn', 'plotly', 'opencv-python', 'cv2', 'imageio', 'skimage',
    'scikit-image', 'nltk', 'spacy', 'gensim', 'xgboost', 'lightgbm',
    'catboost', 'pulp', 'cvxpy', 'ortools', 'numba', 'dask', 'joblib'
}

# Map common import names to their pip package names
IMPORT_TO_PACKAGE = {
    'cv2': 'opencv-python',
    'sklearn': 'scikit-learn',
    'skimage': 'scikit-image',
}

# Track which libraries we've already asked about this session
_ASKED_LIBRARIES = set()


def is_safe_library(module_name: str) -> bool:
    """Check if a module is in the safe library whitelist."""
    # Extract base module name (e.g., 'scipy.stats' -> 'scipy')
    base_module = module_name.split('.')[0]
    return base_module.lower() in SAFE_LIBRARIES


def get_package_name(module_name: str) -> str:
    """Get the pip package name for a given module import name."""
    base_module = module_name.split('.')[0]
    return IMPORT_TO_PACKAGE.get(base_module, base_module)


def attempt_library_installation(module_name: str, auto_install: bool = False) -> bool:
    """Attempt to install a missing library with user confirmation.
    
    Args:
        module_name: The module that failed to import
        auto_install: If True, install without asking (use with caution)
    
    Returns:
        True if installation succeeded, False otherwise
    """
    # Skip if we've already asked about this library
    if module_name in _ASKED_LIBRARIES:
        return False
    
    _ASKED_LIBRARIES.add(module_name)
    
    # Check if it's a safe library
    if not is_safe_library(module_name):
        print(f"\n⚠️  Missing library '{module_name}' is not in the safe library list.")
        print(f"   For security reasons, automatic installation is not available.")
        print(f"   Please install manually if needed: pip install {module_name}")
        return False
    
    package_name = get_package_name(module_name)
    
    if not auto_install:
        print(f"\n📦 Missing library detected: '{module_name}'")
        print(f"   This is a recognized scientific library and safe to install.")
        response = input(f"   Install '{package_name}' now? [y/N]: ").strip().lower()
        if response not in ('y', 'yes'):
            print(f"   Skipping installation. Code requiring '{module_name}' will fail.")
            return False
    
    print(f"\n📥 Installing {package_name}...")
    try:
        # Use subprocess to install the package
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            print(f"✅ Successfully installed {package_name}")
            return True
        else:
            print(f"❌ Failed to install {package_name}")
            print(f"   Error: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"❌ Installation of {package_name} timed out")
        return False
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False


def test_code_on_examples(code: str, examples: List[Dict], timeout_seconds: int = 5) -> List[ExampleResult]:
    """Test code on a list of examples and return results.
    
    Args:
        code: Python code containing transform function
        examples: List of {'input': grid, 'output': grid} dicts
        timeout_seconds: Timeout for each example execution
    
    Returns:
        List of ExampleResult dictionaries
    """
    results = []
    for i, example in enumerate(examples):
        predicted_output, error_message = execute_transformation_code(code, example['input'])
        
        # Calculate overlap and matching size
        matching_size = False
        overlap_percentage = 0.0
        code_success = False
        
        if predicted_output is not None and error_message is None:
            expected_output = example['output']
            matching_size, overlap_percentage = calculate_grid_results(predicted_output, expected_output)
            code_success = matching_size and overlap_percentage >= 99.9
        
        result = ExampleResult(
            example_index=i,
            input=example['input'],
            expected_output=example['output'],
            predicted_output=predicted_output,
            matching_size=matching_size,
            overlap_percentage=overlap_percentage,
            error_message=error_message,
            code_success=code_success,
            llm_predicted_output=None,
            llm_matching_size=None,
            llm_overlap_percentage=None,
            llm_error_message=None,
            llm_success=False,
        )
        results.append(result)
    
    return results


def execute_transformation_code(main_code: str,
                                input_grid: List[List[int]]) -> Tuple[Optional[List[List[int]]], Optional[str]]:
    """Execute the transformation code on an input grid.

    Returns:
        (result_grid, error_message)

    - `result_grid` is the transformed grid when execution succeeds, otherwise `None`.
    - `error_message` is `None` on success, otherwise contains the exception traceback or
      a short error description useful for refinement.
    """
    try:
        # Create execution namespace
        namespace = {"__builtins__": __builtins__}

        # Normalize code strings that contain escaped newlines (e.g. "\\n") so
        # they become properly formatted Python source before printing/execution.
        if isinstance(main_code, str):
            try:
                # If the string appears to contain literal backslash-n sequences
                # but no real newlines, attempt to un-escape it.
                if "\\n" in main_code and "\n" not in main_code:
                    stripped = main_code.strip()
                    if (stripped.startswith(('"', "'")) and stripped.endswith(('"', "'"))):
                        try:
                            main_code = ast.literal_eval(main_code)
                        except Exception:
                            main_code = main_code.encode('utf-8').decode('unicode_escape')
                    else:
                        main_code = main_code.encode('utf-8').decode('unicode_escape')
                else:
                    # Replace any remaining escaped newlines/tabs with real ones
                    main_code = main_code.replace('\\r\\n', '\n').replace('\\n', '\n').replace('\\t', '\t')

                # Trim excessive leading/trailing blank lines
                main_code = main_code.strip('\n') + '\n'
            except Exception:
                # Best-effort fallback
                main_code = main_code.replace('\\n', '\n').replace('\\t', '\t')

        # Execute the main code
        exec(main_code, namespace)

    except (ImportError, ModuleNotFoundError) as import_err:
        # Handle missing library imports
        error_msg = str(import_err)
        # Extract module name from error message (e.g., "No module named 'scipy'")
        match = re.search(r"No module named ['\"]([^'\"]+)['\"]|cannot import name .+ from ['\"]([^'\"]+)['\"]|No module named: ([^\s]+)", error_msg)
        if match:
            module_name = match.group(1) or match.group(2) or match.group(3)
            print(f"\n⚠️  Code execution failed: Missing module '{module_name}'")
            
            # Attempt to install the library
            if attempt_library_installation(module_name):
                # Retry execution after successful installation
                print(f"\n🔄 Retrying code execution after installing {module_name}...")
                try:
                    namespace = {"__builtins__": __builtins__}
                    exec(main_code, namespace)
                    
                    # Call transform function if execution succeeded
                    if "transform" in namespace:
                        try:
                            result = namespace["transform"](input_grid)
                            return result, None
                        except Exception as inner_e:
                            tb = traceback.format_exc()
                            print(f"Error while running transform(): {inner_e}\n{tb}")
                            return None, str(inner_e) + "\n" + tb
                    else:
                        err = "transform function not found in executed code"
                        print(err)
                        return None, err
                except Exception as retry_e:
                    tb = traceback.format_exc()
                    print(f"Error executing code after library installation: {retry_e}\n{tb}")
                    return None, str(retry_e) + "\n" + tb
            else:
                # Installation failed or was declined
                tb = traceback.format_exc()
                return None, f"Missing required library: {module_name}\n" + tb
        else:
            # Couldn't parse module name from error
            tb = traceback.format_exc()
            print(f"Import error (could not determine module): {import_err}\n{tb}")
            return None, str(import_err) + "\n" + tb

    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error executing transformation code: {e}\n{tb}")
        return None, str(e) + "\n" + tb

    # Call the transform function (executes when exec succeeded without exceptions)
    if "transform" in namespace:
        try:
            result = namespace["transform"](input_grid)
            return result, None
        except Exception as inner_e:
            tb = traceback.format_exc()
            print(f"Error while running transform(): {inner_e}\n{tb}")
            return None, str(inner_e) + "\n" + tb
    else:
        err = "transform function not found in executed code"
        print(err)
        return None, err


def calculate_grid_results(predicted: List[List[int]], expected: List[List[int]]) -> Tuple[bool, float]:
    """Compare two 2D grids and return (size_match, value_match_percent).

    - size_match: True iff the predicted grid has the same dimensions as the
      expected grid (dimensions only; values are not considered).
    - value_match_percent: Percentage (0.0-100.0) of cells that match between
      the predicted and expected grids. The percentage is calculated relative
      to the expected grid's total cells. If the predicted grid is smaller or
      larger, non-overlapping cells count as mismatches.

    Args:
        predicted: 2D list representing the predicted output grid.
        expected: 2D list representing the expected output grid.

    Returns:
        (size_match, value_match_percent)
    """
    # Compute dimensions
    pred_h = len(predicted) if predicted is not None else 0
    pred_w = len(predicted[0]) if pred_h > 0 and predicted[0] else 0
    exp_h = len(expected) if expected is not None else 0
    exp_w = len(expected[0]) if exp_h > 0 and expected[0] else 0

    # First return value: size match (dimensions only)
    size_match = (pred_h == exp_h and pred_w == exp_w)

    # Calculate value match percentage relative to expected grid area
    total_cells = exp_h * exp_w
    if total_cells == 0:
        return (size_match, 0.0)

    matching_cells = 0
    for i in range(exp_h):
        for j in range(exp_w):
            if i < pred_h and j < pred_w:
                try:
                    if predicted[i][j] == expected[i][j]:
                        matching_cells += 1
                except Exception:
                    # Treat any comparison error as mismatch
                    pass
            else:
                # Out-of-range predicted cell counts as mismatch
                pass

    value_match_percent = (matching_cells / total_cells) * 100.0

    return (size_match, value_match_percent)


def evaluate_example(llm,
                     main_code: str,
                     transformation_steps: List[str],
                     input_grid: List[List[int]],
                     expected_output: List[List[int]],
                     enable_code_predict: bool = True,
                     enable_llm_predict: bool = True) -> Dict[str, Any]:
    """Evaluate a single training/test example.

    Runs the provided `main_code` on `input_grid`, computes grid comparison
    metrics against `expected_output`, and asks the LLM to apply the
    `transformation_steps` to the `input_grid` for a comparison baseline.

    Returns a result dict compatible with `nodes.test_code_node` usage.
    """
    
    # Execute the code only if enabled
    exec_predicted_output = None
    exec_error = None
    matching_size = False
    overlap_percentage = 0.0
    error_message = None
    code_success = False

    if enable_code_predict:
        try:
            exec_predicted_output, exec_error = execute_transformation_code(main_code, input_grid)
        except Exception as e:
            exec_predicted_output = None
            exec_error = str(e)

        # If there is no expected output available, report comparison-related
        # metrics as None rather than attempting to compute them.
        if expected_output is None:
            matching_size, overlap_percentage = None, None
            error_message = None
            code_success = None
        else:
            # Compute code metrics (if execution produced an output)
            if exec_predicted_output is not None and exec_error is None:
                matching_size, overlap_percentage = calculate_grid_results(exec_predicted_output, expected_output)
            else:
                matching_size, overlap_percentage = False, 0.0
            error_message = exec_error or None
            code_success = bool(matching_size) and (overlap_percentage == 100.0)

    else:
        # Not executing code — leave defaults (no prediction)
        exec_predicted_output = None
        exec_error = None
        if expected_output is None:
            matching_size = None
            overlap_percentage = None
            error_message = None
            code_success = None
        else:
            matching_size = False
            overlap_percentage = 0.0
            error_message = None
            code_success = False

    # Ask the LLM to apply the step-by-step transformation to the input only if enabled
    llm_predicted_output = None
    llm_error = None
    llm_matching_size = False
    llm_overlap_percentage = 0.0
    llm_error_message = None
    llm_success = False

    if enable_llm_predict:
        try:
            # transformation_steps is expected to be a dict with key 'transformation_steps' in our flow
            steps_for_llm = transformation_steps["transformation_steps"] if isinstance(transformation_steps, dict) and "transformation_steps" in transformation_steps else transformation_steps
            llm_predicted_output, llm_error = generate_llm_predicted_output(llm, steps_for_llm, input_grid)
        except Exception as e:
            llm_predicted_output = None
            llm_error = str(e)

        # If there is no expected output available, report comparison-related
        # metrics as None rather than attempting to compute them.
        if expected_output is None:
            llm_matching_size, llm_overlap_percentage = None, None
            llm_error_message = None
            llm_success = None
        else:
            # Compute LLM-specific metrics (if LLM produced an output)
            if llm_predicted_output is not None and llm_error is None:
                llm_matching_size, llm_overlap_percentage = calculate_grid_results(llm_predicted_output, expected_output)
            else:
                llm_matching_size, llm_overlap_percentage = False, 0.0
            llm_error_message = llm_error or None
            llm_success = bool(llm_matching_size) and (llm_overlap_percentage == 100.0)
    else:
        llm_predicted_output = None
        llm_error = None
        if expected_output is None:
            llm_matching_size = None
            llm_overlap_percentage = None
            llm_error_message = None
            llm_success = None
        else:
            llm_matching_size = False
            llm_overlap_percentage = 0.0
            llm_error_message = None
            llm_success = False

    result = {
        "input": input_grid,
        "expected_output": expected_output,
        "predicted_output": exec_predicted_output,
        "matching_size": matching_size,
        "overlap_percentage": overlap_percentage,
        "error_message": error_message,
        "code_success": code_success,
        "llm_predicted_output": llm_predicted_output,
        "llm_matching_size": llm_matching_size,
        "llm_overlap_percentage": llm_overlap_percentage,
        "llm_error_message": llm_error_message,
        "llm_success": llm_success,
    }

    return result


def test_and_fix_code_from_trial_run(code_llm, python_codes_list: List[str], training_examples: List[Dict], probe_index: int = 0) -> Tuple[List[str], List[Dict]]:
    """Run candidate codes on a training example, collect diagnostics,
    and, if failures exist, ask the LLM to produce fixed implementations.

    Returns a tuple: (possibly_updated_python_codes_list, trial_run_results)
    """
    python_codes_list = python_codes_list[:]  # Make a copy
    trial_run_results: List[Dict] = []

    # Quick exit if nothing to test
    if not python_codes_list or not training_examples:
        return python_codes_list, trial_run_results

    example_index = 0
    example_input = training_examples[example_index].get('input', [])

    # Execute each candidate and record results/errors
    for idx, code in enumerate(python_codes_list, start=1):
        try:
            src = ensure_imports_in_code(code)
        except Exception:
            src = code
        result, error = execute_transformation_code(src, example_input)
        trial_run_results.append({
            'index': idx,
            'code': src,
            'predicted': result,
            'error': error
        })

    # Collect failing candidates
    errors = [r for r in trial_run_results if r.get('error')]
    if not errors or not code_llm:
        return python_codes_list, trial_run_results

    # Build prompt for fixes
    def build_fix_prompt():
        parts = [
            "You are an expert Python programmer tasked with fixing implementations of a function `transform(input_grid)` for the ARC task.",
            "However, several candidate implementations have failed when tested against a training example.",
            "Your job is to analyze each failing candidate, understand the error and produce a corrected implementation that runs successfully and produces output",
        
            "----------------",
            "TRAINING EXAMPLE",
            "----------------",
            format_grid_for_prompt(example_input),
            "",
        ]

        # Include per-candidate code and error info
        for r in trial_run_results:
            if r.get('error'):
                parts.extend([
                    f"CANDIDATE {r.get('index')} DETAILS",
                    "Code:",
                    r.get('code', '') or '',
                    "",
                ])
                err_text = r.get('error') or ''
                parts.extend([
                    "Execution failed with error:",
                    err_text if len(err_text) < 4000 else err_text[:4000]
                ])

        parts += [
            "------------"
            "INSTRUCTIONS",
            "------------",
            "- Only return fixed solutions for the candidates that failed. If a candidate already succeeded, you may paste the solution as is.",
            "- Each solution must be a standalone Python code block that defines `transform(input_grid)` and any helpers it needs.",
            "- Return solutions using the XML-like tags exactly as: <count>n</count> followed by <solution>...code...</solution>.",
            "- Make sure you respect the original candidate numbering.",
            "- Ensure that each solution includes any necessary imports at the top.",
            "- Do NOT add any explanations or comments outside the code blocks.",
            "",
            "Example structure:",
            "<count>1</count>",
            "<solution>",
            "from typing import List",
            "import ... # Import ANY necessary standard libraries to run the code here",
            "def helper_function_1(...):",
            "    # Add helper functions if neeeded",
            "def helper_function_2(...):",
            "    # Add helper functions if needed",
            "def transform(input_grid):",
            "    [implementations of transformation steps]",
            "    return transformed_grid",
            "</solution>",
            "<count>2</count>",
            "<solution>...</solution>",
            "...",
            "",
            "Generate the Python code now:",
            "",
        ]
        return "\n".join(parts)

    prompt = build_fix_prompt()
    try:
        response = code_llm.invoke(prompt, temperature=0.2)
        response_text = response.content if hasattr(response, 'content') else str(response)
        print_prompt_and_response(prompt, response_text)
    except Exception as e:
        print(f"test_and_fix_code_from_trial_run: LLM invocation failed: {e}")
        return python_codes_list, trial_run_results

    fixed_solutions = extract_python_solutions(response_text)
    if not fixed_solutions:
        return python_codes_list, trial_run_results

    # Normalize by ensuring imports are present and return
    try:
        fixed_solutions = [ensure_imports_in_code(s) for s in fixed_solutions]
    except Exception as e:
        print(f"test_and_fix_code_from_trial_run: Failed to normalize fixed solutions: {e}")
        return python_codes_list, trial_run_results
    
    fixed_idx = 0
    for r in trial_run_results:
        idx = r.get('index', 0)
        if r.get('error'):
            if fixed_idx < len(fixed_solutions):
                python_codes_list[idx - 1] = fixed_solutions[fixed_idx]
                fixed_idx += 1
    return python_codes_list, trial_run_results
