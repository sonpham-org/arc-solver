"""
Action helpers for the ARC LangGraph Agent workflow.

This module contains helper functions, prompt builders, LLM-driven
generation utilities, execution helpers, and refinement logic.
"""

import json
import copy
from typing import List, Dict, Optional, Any, TypedDict, Tuple, Union
import traceback
import ast
import sys
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import math
import uuid
import os
import textwrap
import traceback
import re

# Import schema and tools
from .schema import AgentState, CodeSolution, ExampleResult, ReasoningTraceRecord
from .tools import FUNCTION_MAP
from .debug import print_prompt_and_response, print_python_code


def generate_llm_predicted_output(llm,
                                  transformation_steps: Dict[str, Any],
                                  input_grid: List[List[int]]) -> Tuple[Optional[List[List[int]]], Optional[str]]:
    """Use the LLM to apply the step-by-step transformation to a single input grid.

    The LLM is instructed to return the transformed grid as a JSON array
    (list of lists of integers). Returns (grid, None) on success, or
    (None, error_message) on failure.
    """
    try:
        steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(transformation_steps)) if transformation_steps else "(no steps provided)"

        prompt_parts = [
            "You are an expert that can execute step-by-step grid transformations by following instructions",
            "Given the following input grid and transformation steps, you are tasked with applying the steps and return the resulting grid.",
            
            "Do NOT return any other text after that block.",
            "",
            "INPUT GRID:",
            format_grid_for_prompt(input_grid),
            "",
            "TRANSFORMATION STEPS:",
            steps_text,
            "",
            "Follow the transformation steps carefully, show your detailed step-by-step transformation"
            "After finishing all the steps, show the 2D grid inside a fenced block labeled ```llm_predicted_output``` containing only the grid rows as lines of numbers (space-separated or contiguous digits)."
        ]

        prompt = "\n".join(prompt_parts)
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)

        # print_prompt_and_response(prompt, response_text)

        # Prefer a fenced block labelled ```llm_predicted_output``` containing
        # the grid as lines of numbers (space-separated or run-together digits).
        import re, json
        block_match = re.search(r'```llm_predicted_output\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
        if block_match:
            block = block_match.group(1).strip()
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            parsed_grid = []
            for line in lines:
                # Split on whitespace; if no whitespace, split into single chars
                if re.search(r'\s+', line):
                    parts = re.split(r'\s+', line.strip())
                else:
                    parts = list(line.strip())
                row = []
                for p in parts:
                    try:
                        row.append(int(p))
                    except Exception:
                        # If conversion fails, try to strip non-digits then int
                        digits = re.findall(r'-?\d+', p)
                        if digits:
                            row.append(int(digits[0]))
                        else:
                            # Give up and return error with raw block
                            return None, f"Non-numeric token in llm_predicted_output block: '{p}'"
                parsed_grid.append(row)
            return parsed_grid, None

        # Fallback: try to find a JSON array in the response
        json_match = re.search(r'(\[\s*\[.*?\]\s*\])', response_text, re.DOTALL)
        if json_match:
            candidate = json_match.group(1)
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list) and all(isinstance(row, list) for row in parsed):
                    norm = []
                    for row in parsed:
                        new_row = []
                        for cell in row:
                            try:
                                new_row.append(int(cell))
                            except Exception:
                                new_row.append(cell)
                        norm.append(new_row)
                    return norm, None
            except Exception:
                pass

        # Fallback: try to parse any Python-style list literal
        try:
            parsed2 = eval(response_text, {"__builtins__": {}}, {})
            if isinstance(parsed2, list) and all(isinstance(r, list) for r in parsed2):
                return parsed2, None
        except Exception:
            pass

        # If response contains an explicit error line, return it as error
        if isinstance(response_text, str) and ("error" in response_text.lower() or "cannot" in response_text.lower() or "failed" in response_text.lower()):
            return None, response_text.strip()

        return None, f"Could not parse LLM response as grid. Raw response: {response_text[:1000]}"

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return None, f"Exception calling LLM for predicted output: {e}\n{tb}"


def analyze_training_examples(training_examples: List[Dict]) -> str:
    """Analyze training examples to understand the pattern."""
    if not training_examples:
        return "No training examples provided."
    
    analysis = []
    analysis.append(f"Found {len(training_examples)} training examples.")
    
    # Analyze input/output dimensions
    input_sizes = [(len(ex["input"]), len(ex["input"][0]) if ex["input"] else 0) 
                   for ex in training_examples]
    output_sizes = [(len(ex["output"]), len(ex["output"][0]) if ex["output"] else 0) 
                    for ex in training_examples]
    
    analysis.append(f"Input sizes: {input_sizes}")
    analysis.append(f"Output sizes: {output_sizes}")
    
    # Check if sizes are consistent
    if len(set(input_sizes)) == 1:
        analysis.append("All inputs have the same size.")
    else:
        analysis.append("Input sizes vary.")
    
    if len(set(output_sizes)) == 1:
        analysis.append("All outputs have the same size.")
    else:
        analysis.append("Output sizes vary.")
    
    return "\n".join(analysis)


def _grid_to_image_bytes(grid: List[List[int]], cell_size: int = 24, padding: int = 8) -> bytes:
    """Render a grid (list of lists of ints) to a PNG bytes buffer.

    Returns PNG bytes. If Pillow is not available, raises ImportError.
    """
    try:
        from PIL import Image, ImageDraw
    except Exception:
        raise ImportError("Pillow is required to generate visual cues (pip install pillow)")

    # Simple color map for values 0..9
    DEFAULT_COLORS = {
        0: (255, 255, 255),
        1: (230, 25, 75),
        2: (60, 180, 75),
        3: (255, 225, 25),
        4: (0, 130, 200),
        5: (245, 130, 48),
        6: (145, 30, 180),
        7: (70, 240, 240),
        8: (240, 50, 230),
        9: (210, 245, 60),
    }

    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    img_w = w * cell_size + padding * 2
    img_h = h * cell_size + padding * 2
    img = Image.new('RGB', (img_w, img_h), (240, 240, 240))
    draw = ImageDraw.Draw(img)

    for r in range(h):
        for c in range(w):
            val = grid[r][c]
            color = DEFAULT_COLORS.get(val, (200, 200, 200))
            x0 = padding + c * cell_size
            y0 = padding + r * cell_size
            x1 = x0 + cell_size - 1
            y1 = y0 + cell_size - 1
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=(100, 100, 100))

    import io
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def generate_distilled_reasoning(llm, reasoning_trace, transformation_steps, python_codes):
    """Distill a detailed `reasoning_trace` into a JSON-like structure.

    Returns a dict with keys:
      - 'strategy': concise single-paragraph summary (<=150 words)
      - 'concepts': list of short concept strings

    The LLM is instructed to return ONLY valid JSON. This function will try
    to robustly parse common response shapes (```json``` fenced block, bare
    JSON, or a JSON-like substring). On failure it will produce a best-effort
    dict with empty `concepts`.
    """

    def build_distill_reasoning_prompt() -> str:
        prompt_parts = [
            "---------------",
            "REASONING TRACE",
            "---------------",
            reasoning_trace,
            "",
            "------------",
            "INSTRUCTIONS",
            "------------",
            "Read ONLY the reasoning trace above and produce a JSON object with exactly two fields:\n",
            "1) \"strategy\": a concise single-paragraph summary (<=150 words) describing the high-level strategy used to solve the task;\n",
            "2) \"concepts\": an array of short strings naming the key operations or concepts used (e.g., \"connected components\", \"symmetry\", \"color mapping\").\n",
            "Return ONLY valid JSON within a ```json```. Do NOT include any additional text, commentary, or markdown.\n",
            "Example output:",
            '```json',
            '{"strategy": "Brief summary...", "concepts": ["symmetry", "fill", "mirror"]}',
            '```',
            "Perform the distillation now:"
        ]
        return "\n".join(prompt_parts)

    prompt = build_distill_reasoning_prompt()

    try:
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        return {"strategy": f"(LLM error during distillation: {e})", "concepts": []}

    # Extract JSON candidate
    json_candidate = None
    m = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
    if m:
        json_candidate = m.group(1).strip()
    else:
        m2 = re.search(r'\{.*?\}', response_text, re.DOTALL)
        if m2 and 'strategy' in m2.group(0):
            json_candidate = m2.group(0)

    if json_candidate:
        try:
            parsed = json.loads(json_candidate)
            strategy = str(parsed.get('strategy', '')).strip()
            concepts = parsed.get('concepts') or parsed.get('concept') or []
            if isinstance(concepts, str):
                concepts = [c.strip() for c in re.split(r'[;,\n]', concepts) if c.strip()]
            elif not isinstance(concepts, (list, tuple)):
                concepts = []

            words = strategy.split()
            if len(words) > 150:
                strategy = " ".join(words[:150]) + "..."

            return {"strategy": strategy, "concepts": concepts}
        except Exception:
            pass

    # Forgiving fallback
    text = textwrap.dedent(str(response_text)).strip()
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    strategy = paragraphs[0] if paragraphs else text
    words = strategy.split()
    if len(words) > 150:
        strategy = " ".join(words[:150]) + "..."

    concepts = []
    for line in paragraphs[1:4]:
        if len(line) < 200 and (',' in line or ';' in line or line.lower().startswith('concepts') or len(line.split()) <= 6):
            cand = re.sub(r'^(concepts?:\s*)', '', line, flags=re.IGNORECASE)
            parts = [c.strip() for c in re.split(r'[;,\n]', cand) if c.strip()]
            for p in parts:
                if 2 <= len(p) <= 60:
                    concepts.append(p)
        if concepts:
            break

    return {"strategy": strategy, "concepts": concepts}


def generate_embedding_from_distilled_reasoning(distilled_text) -> List[float]:
    """Generate an embedding vector from the distilled reasoning text.

    This is a placeholder function. In a real implementation, this would
    call an embedding model (e.g., OpenAI's text-embedding-ada-002)
    to generate a vector representation of the text.
    """
    # Try Google Generative AI (GenAI) embeddings first, if available.
    # This keeps the dependency optional and falls back to a deterministic
    # SHA-256-based vector when the GenAI client or credentials are absent.
    if not distilled_text:
        return []

    # Local import to avoid hard dependency at module import time
    try:
        import google.generativeai as genai  # type: ignore
        # genai uses environment or configure() for credentials.
        # Model name example: 'textembedding-gecko@001' — change as needed.
        model = getattr(genai, 'DEFAULT_EMBEDDING_MODEL', None) or 'textembedding-gecko@001'
        resp = genai.embeddings.create(model=model, input=distilled_text)
        # Response shape: resp.data[0].embedding (list[float])
        emb = None
        try:
            emb = resp.data[0].embedding
        except Exception:
            # Some client versions may use resp['data'][0]['embedding']
            try:
                emb = resp['data'][0]['embedding']
            except Exception:
                emb = None
        if emb is not None:
            try:
                vec = [float(x) for x in emb]
                # Optionally normalize to unit length
                norm = math.sqrt(sum(x * x for x in vec))
                if norm > 0:
                    vec = [x / norm for x in vec]
                return vec
            except Exception:
                # If provider returned an unexpected shape, fall back
                pass
    except Exception:
        # Fall through to deterministic fallback
        pass

    # Fallback: deterministic SHA-256 based vector (length = 1536 to match Qdrant collection)
    dim = 1536
    seed = (distilled_text or "").encode('utf-8')
    vec = []
    for i in range(dim):
        h = hashlib.sha256()
        h.update(seed)
        h.update(b'||')
        h.update(str(i).encode('utf-8'))
        digest = h.digest()
        u64 = int.from_bytes(digest[:8], 'big')
        f = (u64 / float(2**64 - 1)) * 2.0 - 1.0
        vec.append(float(f))

    # Normalize
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]

    return vec


def store_record(record, qdrant_client=None, collection_name=None) -> bool:
    """Store a ReasoningTraceRecord into a Qdrant collection if available.

    - `record` may be a dataclass-like object (ReasoningTraceRecord) or a dict.
    - If `qdrant_client` is not provided, we will try to import the global
      QDRANT_CLIENT from run_langgraph_agent (which uses embedded Qdrant).
    - If `collection_name` is not provided we will try to use the global
      QDRANT_COLLECTION_NAME or read from environment/collection_info.json.

    Returns True on success (or when the store was skipped because qdrant is
    unavailable), False only on explicit failure to upsert when qdrant is
    available but the upsert fails.
    """
    try:
        if not record:
            return False

        # Extract fields from the record (object or dict)
        def _get(obj, key):
            return getattr(obj, key, None) if not isinstance(obj, dict) else obj.get(key)

        payload = {
            "reasoning_text": _get(record, 'reasoning_text'),
            "reasoning_summary": _get(record, 'reasoning_summary'),
            "concepts": _get(record, 'concepts') or _get(record, 'concept'),
            "helpers": _get(record, 'helpers'),
        }
        vector = _get(record, 'vector')
        point_id = _get(record, 'id') or str(uuid.uuid4())

        # Try to get global client/collection from run_langgraph_agent module
        if not qdrant_client:
            try:
                import run_langgraph_agent
                qdrant_client = run_langgraph_agent.QDRANT_CLIENT
                if not collection_name:
                    collection_name = run_langgraph_agent.QDRANT_COLLECTION_NAME
            except Exception:
                pass

        # Determine collection name from environment if still not set
        if not collection_name:
            collection_name = os.environ.get('QDRANT_COLLECTION_NAME')
        
        # Try to locate collection_info.json if env var not set
        if not collection_name:
            try:
                for root, dirs, files in os.walk(os.getcwd()):
                    if 'collection_info.json' in files:
                        try:
                            with open(os.path.join(root, 'collection_info.json'), 'r') as f:
                                info = json.load(f)
                            collection_name = info.get('collection_name') or info.get('name')
                            if collection_name:
                                break
                        except Exception:
                            pass
            except Exception:
                # If os.walk fails for any reason, ignore and proceed
                pass

        # If no qdrant info available, silently skip storing (not an error)
        if not qdrant_client or not collection_name:
            # Provide debug info only on first call (avoid spam)
            if not hasattr(store_record, '_warned'):
                store_record._warned = True
                print(f"Debug: store_record skipped - qdrant_client={'available' if qdrant_client else 'None'}, collection_name={collection_name or 'None'}")
            return False

        # Prepare point and upsert
        try:
            # Ensure point_id is a string (Qdrant requires string IDs)
            point_id = str(point_id)
            
            # Try different import paths for PointStruct
            PointStruct = None
            try:
                from qdrant_client.http.models import PointStruct as PS
                PointStruct = PS
            except ImportError:
                try:
                    from qdrant_client.models import PointStruct as PS
                    PointStruct = PS
                except ImportError:
                    pass
            
            if PointStruct is not None:
                point = PointStruct(id=point_id, vector=vector, payload=payload)
                qdrant_client.upsert(collection_name=collection_name, points=[point])
            else:
                # If PointStruct unavailable, use the client's upsert method directly with kwargs
                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=[
                        {
                            "id": point_id,
                            "vector": vector,
                            "payload": payload
                        }
                    ]
                )
            return True
        except Exception as e:
            print(f"Warning: failed to upsert record to Qdrant collection '{collection_name}': {e}")
            import traceback
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"Warning: unexpected error in store_record: {e}")
        return False


def retrieve_similar_distillations(vector: List[float], top_k: int = 5, qdrant_client=None, collection_name=None) -> List[Dict[str, Any]]:
    """Retrieve top_k most similar distilled reasoning records from Qdrant.

    This function requires an explicit embedding `vector` (list of floats).
    Returns a list of dicts with keys: `id`, `score`, `payload`, `vector`.
    If Qdrant is unavailable or an error occurs, returns an empty list.
    """
    try:
        if not vector:
            return []

        # Try to get global client/collection from run_langgraph_agent module
        if not qdrant_client:
            try:
                import run_langgraph_agent
                qdrant_client = run_langgraph_agent.QDRANT_CLIENT
                if not collection_name:
                    collection_name = run_langgraph_agent.QDRANT_COLLECTION_NAME
            except Exception:
                pass

        # Determine collection name if not provided
        if not collection_name:
            collection_name = os.environ.get('QDRANT_COLLECTION_NAME')

        # Try to locate collection_info.json if env var not set
        if not collection_name:
            try:
                for root, dirs, files in os.walk(os.getcwd()):
                    if 'collection_info.json' in files:
                        try:
                            with open(os.path.join(root, 'collection_info.json'), 'r') as f:
                                info = json.load(f)
                            collection_name = info.get('collection_name') or info.get('name')
                            if collection_name:
                                break
                        except Exception:
                            pass
            except Exception:
                pass

        # If no qdrant client or collection available, return empty
        if not qdrant_client or not collection_name:
            return []

        # Perform search - embedded Qdrant uses different API
        hits = []
        try:
            # Try the query method with proper parameters
            results = qdrant_client.query(
                collection_name=collection_name,
                query_vector=vector,
                limit=top_k
            )
            # results should be a list of ScoredPoint objects
            hits = results
        except (AttributeError, TypeError) as e:
            # If query doesn't work, try using scroll + manual similarity
            try:
                # Get all points and manually compute similarities
                import numpy as np
                
                # Scroll through collection
                all_points, _ = qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=100,  # Get more points to search through
                    with_payload=True,
                    with_vectors=True
                )
                
                # Compute cosine similarities
                query_vec = np.array(vector)
                query_norm = np.linalg.norm(query_vec)
                
                scored_points = []
                for point in all_points:
                    try:
                        point_vec = point.vector if hasattr(point, 'vector') else None
                        if point_vec:
                            point_vec_arr = np.array(point_vec)
                            point_norm = np.linalg.norm(point_vec_arr)
                            if query_norm > 0 and point_norm > 0:
                                similarity = float(np.dot(query_vec, point_vec_arr) / (query_norm * point_norm))
                            else:
                                similarity = 0.0
                            
                            scored_points.append({
                                'id': point.id if hasattr(point, 'id') else None,
                                'score': similarity,
                                'payload': point.payload if hasattr(point, 'payload') else {},
                                'vector': point_vec
                            })
                    except Exception:
                        continue
                
                # Sort by similarity and take top_k
                scored_points.sort(key=lambda x: x['score'], reverse=True)
                hits = scored_points[:top_k]
                
            except Exception as e2:
                print(f"Warning: Qdrant search/scroll failed: {e}, {e2}")
                return []

        results = []
        for h in hits or []:
            try:
                # hit may be a typed object or dict-like depending on client
                if isinstance(h, dict):
                    item_id = h.get('id')
                    score = h.get('score') or h.get('payload', {}).get('score')
                    payload = h.get('payload')
                    vec = h.get('vector')
                else:
                    item_id = getattr(h, 'id', None)
                    score = getattr(h, 'score', None) or (getattr(h, 'payload', {}) or {}).get('score')
                    payload = getattr(h, 'payload', None)
                    vec = getattr(h, 'vector', None)

                results.append({"id": item_id, "score": score, "payload": payload, "vector": vec})
            except Exception:
                continue

        return results

    except Exception as e:
        print(f"Warning: unexpected error in retrieve_similar_distillations: {e}")
        return []


def create_solutions_with_reasoning(llm, transformation_llm, code_llm, 
                                      training_examples: List[Dict], num_solutions: int,
                                      enable_visual_cue: bool = False,
                                      enable_rag_hint: bool = False) -> Tuple[List[str], str, List[Dict]]:
    """Generate Python transformation code using reasoning-first approach.

    When `enable_visual_cue` is True, this function will render training
    input/output pairs to PNG images, encode them as base64 and include them
    in the LLM invocation (if the LLM driver supports image messages).

    Returns:
        Tuple of (python_code_list, reasoning_trace, transformation_steps)
    """

    visual_cues = []
    if enable_visual_cue:
        # Build visual cues: for each training example, create a small image
        # that shows the input and expected output stacked vertically.
        import base64
        for i, ex in enumerate(training_examples):
            inp = ex.get('input') or []
            out = ex.get('output') or []
            inp_bytes = _grid_to_image_bytes(inp)
            out_bytes = _grid_to_image_bytes(out)
            b64_in = base64.b64encode(inp_bytes).decode('utf-8')
            b64_out = base64.b64encode(out_bytes).decode('utf-8')
            visual_cues.append({
                'example_index': i,
                'input_b64': b64_in,
                'output_b64': b64_out,
            })

    # Step 1: Generate reasoning trace
    reasoning_trace, reasoning_retries = (generate_reasoning_trace(llm, training_examples) if not enable_visual_cue 
                                          else generate_reasoning_trace(llm, training_examples, visual_cues=visual_cues))

    # Step 2: Extract step-by-step transformation from reasoning
    transformation_solutions_list, transformation_retries = generate_transformation_steps(transformation_llm, reasoning_trace, training_examples, num_solutions)

    # Step 3: Generate Python code(s) based on reasoning and steps
    # Note: `generate_code_from_reasoning` may return multiple candidate code strings.
    if not transformation_solutions_list:
        python_codes_list = generate_code_from_reasoning(code_llm, reasoning_trace, training_examples)
    else:
        python_codes_list = generate_code_from_reasoning_and_transformations(code_llm, reasoning_trace, transformation_solutions_list,
                                                                             training_examples)
    # Trial-run + automatic fix: run candidates on a probe example and
    # request fixes from the code LLM if needed. This logic is encapsulated
    # in `test_and_fix_code_from_trial_run` which returns possibly-updated
    # candidates and the trial run diagnostics.
    try:
        python_codes_list, trial_run_results = test_and_fix_code_from_trial_run(code_llm, python_codes_list, training_examples)
    except Exception as e:
        print(f"Warning: test_and_fix_code_from_trial_run failed: {e}")
    
    # Step 4: Create rag entry if enabled
    if enable_rag_hint:
        distilled_reasoning = generate_distilled_reasoning(llm, reasoning_trace, transformation_solutions_list, python_codes_list)
        distilled_text = f"Strategy: {distilled_reasoning.get('strategy', '')}\nConcepts: {', '.join(distilled_reasoning.get('concepts', []))}"
        embedding = generate_embedding_from_distilled_reasoning(distilled_text)
        helpers = extract_helpers_from_python_codes(python_codes_list)
        rag_entry = ReasoningTraceRecord(
            id=str(uuid.uuid4()),
            reasoning_text=reasoning_trace,
            reasoning_summary=distilled_reasoning.get('strategy', ''),
            concepts=distilled_reasoning.get('concepts', []),
            helpers=helpers,
            vector=embedding,
        )

        # Best-effort: store the distilled reasoning into the Qdrant vector store
        # If qdrant is not available this will be a no-op and will not raise.
        try:
            stored = store_record(rag_entry)
            if not stored:
                # Quietly continue if storing was skipped or unavailable
                pass
        except Exception as e:
            print(f"Warning: store_record raised an exception: {e}")
    else:
        rag_entry = None

    # Attach visual cue data onto each transformation dict (best-effort)
    if enable_visual_cue and visual_cues:
        # Try to attach example-level cues to the first solution dict to be saved later
        for sol in transformation_solutions_list:
            sol['_visual_cues'] = visual_cues

    # Return the list of candidate codes, plus reasoning and steps.
    return python_codes_list, reasoning_trace, transformation_solutions_list, rag_entry, reasoning_retries, transformation_retries

def generate_reasoning_trace(llm, training_examples: List[Dict], visual_cues: Optional[List[Dict]] = None, max_retries: int = 3) -> Tuple[str, int]:
    """Generate detailed reasoning trace analyzing ARC patterns.
    
    Returns:
        Tuple of (reasoning_trace, num_retries_used)
    """
    
    def _flatten_content(c):
        """Flatten various response content types to a single string."""
        try:
            if isinstance(c, str):
                return c
            if isinstance(c, dict):
                # Common forms: {'content': '...'} or {'choices': [...]}
                if 'content' in c and isinstance(c['content'], (str, list)):
                    return _flatten_content(c['content'])
                if 'choices' in c and isinstance(c['choices'], list):
                    return '\n'.join(_flatten_content(ch) for ch in c['choices'])
                return str(c)
            if isinstance(c, (list, tuple)):
                parts = []
                for item in c:
                    if isinstance(item, dict):
                        # Message-like item with 'content' or 'type' fields
                        if 'content' in item:
                            parts.append(_flatten_content(item['content']))
                            continue
                        # Structured content items: {'type':'text','text':...} or {'type':'image',...}
                        if item.get('type') == 'text' and 'text' in item:
                            parts.append(str(item['text']))
                            continue
                        if item.get('type') == 'image':
                            parts.append('[IMAGE]')
                            continue
                        parts.append(str(item))
                    else:
                        parts.append(str(item))
                return '\n'.join([p for p in parts if p])
            return str(c)
        except Exception:
            return str(c)

    def build_initial_reasoning_prompt(training_examples: List[Dict]) -> str:
        """Build prompt for generating detailed reasoning about ARC patterns."""
        
        prompt_parts = [
            "You are an expert mathematician, logistician and pattern recognizier who is solving the"
            "Abstract Reasoning Corpus (ARC) problems.",
            "Your task is to deeply analyze the input-output examples and understand the underlying pattern.",
            "Focus on identifying the core transformation rule that maps inputs to outputs.",

            "YOUR GOAL:",
            "Given the training pairs and test inputs, infer a general transformation rule that:",
            "- Correctly maps every training input to its output.",
            "- Is general and intuitive (no memorization or hard-coded values).",
            "- Is logical, reproducible, and object-level.",

            "GUIDELINES:",
            "- The SAME rule must successfully transform all training pairs.",
            "- Treat all grid values (numbers/characters) as categorical labels, not magnitudes. Do not use arithmetic operations.",
            "- Avoid rules that depend on specific values or characters.",
            "- Make rules in a general manner using object-level reasoning (movements, shapes, fills, mirrors, rotations, bounding boxes, duplicates, etc.).",
            "- Take as many rules as you need to achieve your goals.",
            "",
            "TRAINING EXAMPLES:"
        ]
        
        # Add training examples with detailed formatting
        for i, example in enumerate(training_examples):
            prompt_parts.append(f"\nExample {i+1}:")
            prompt_parts.append(f"Input Grid ({len(example['input'])}x{len(example['input'][0]) if example['input'] else 0}):")
            prompt_parts.append(format_grid_for_analysis(example['input']))
            prompt_parts.append(f"Output Grid ({len(example['output'])}x{len(example['output'][0]) if example['output'] else 0}):")
            prompt_parts.append(format_grid_for_analysis(example['output']))
        
        prompt_parts.extend([
            "",
            "ANALYSIS INSTRUCTIONS:",
            "Provide a ```reasoning``` block that contains your detailed analysis.",
        ])
        
        return "\n".join(prompt_parts)
    

    prompt = build_initial_reasoning_prompt(training_examples)
    # If visual cues are provided and the llm driver supports image messages,
    # send a structured message containing the images (base64 data URLs).
    if visual_cues:
        pass

    # Retry up to max_retries times if extraction fails
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            # Flatten content to handle list/dict/string responses
            if hasattr(response, 'content'):
                resp_content = response.content
            else:
                resp_content = response
            
            response_text = _flatten_content(resp_content)
            # print_prompt_and_response(prompt, response_text)

            # Extract reasoning from response
            reasoning = extract_reasoning_content(response_text)
            if reasoning and reasoning != "Unable to generate reasoning trace":
                return reasoning, attempt
            
            # If this isn't the last attempt, log and retry
            if attempt < max_retries - 1:
                print(f"Warning: Failed to extract reasoning content (attempt {attempt + 1}/{max_retries}). Retrying...")
        
        except Exception as e:
            print(f"Warning: Error in generate_reasoning_trace (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
            else:
                traceback.print_exc()
    
    # After all retries failed
    return "Unable to generate reasoning trace", max_retries
    

def build_steps_text_from_transformation_steps(transformation_steps: List[str]) -> str:
    """Build a numbered steps text block from a list of transformation steps."""
    if not transformation_steps:
        return "(no transformation steps provided)"
    return "\n".join(f"{i+1}. {s}" for i, s in enumerate(transformation_steps))


def generate_reflection_reasoning_trace(llm,
                                        current_solution: CodeSolution,
                                        training_results: List[ExampleResult],
                                        training_examples: List[Dict],
                                        enable_rag_hint: bool,
                                        max_retries: int = 3) -> Tuple[str, int]:
    """Generate a reflection-focused reasoning trace using the ARC-style reflection prompt.

    This is intended for refinement: it asks the model to analyze failures, explain
    what went wrong, and produce a reasoning trace focused on correcting the logic.
    
    Returns:
        Tuple of (reasoning_trace, num_retries_used)
    """
    def build_refinement_reasoning_prompt() -> str:
        """Build reflection prompt based on ARC reflection prompt style for deep analysis."""
        
        # Format previous solution
        previous_code = current_solution["main_code"]
        transformation_steps = current_solution["step_by_step_transformation"]
        reasoning_trace = current_solution["reasoning_trace"]

        # Retrieve the relevant concepts based on RAG hints if enabled
        rag_concepts = set()
        rag_hints_parts = []
        if enable_rag_hint:
            vector = current_solution.get('vector')
            entries = retrieve_similar_distillations(vector=vector, top_k=5)

            rag_concepts = set()
            for entry in entries:
                payload = entry.get('payload', {})
                concepts = payload.get('concepts') or []
                if isinstance(concepts, str):
                    concepts = [c.strip() for c in re.split(r'[;,\n]', concepts) if c.strip()]
                elif not isinstance(concepts, (list, tuple)):
                    concepts = []
                for c in concepts:
                    rag_concepts.add(c)
        
        if rag_concepts:
            rag_hints_parts = [
                "---------------------",
                "RELATED CONCEPT HINTS",
                "---------------------",
                "The following concepts were found in similar prior solutions. Feel free to consider them in your analysis:",
                "\n".join(f"- {c}" for c in rag_concepts),
                "",
            ]
        # Format transformation steps
        steps_text = build_steps_text_from_transformation_steps(transformation_steps)
        
        # Build detailed failure analysis
        failure_analysis = []
        for test in training_results:
            example_idx = test.get("example_index", 0)
            if example_idx < len(training_examples):
                example = training_examples[example_idx]
                
                analysis = f"Training Example {example_idx + 1} - FAILED\\n"
                analysis += "--\\n"
                analysis += f"Input:\\n{format_grid_for_prompt(example['input'])}\\n\\n"
                analysis += f"Expected Output:\\n{format_grid_for_prompt(example['output'])}\\n\\n"
                
                predicted = test.get("predicted_output")
                if predicted:
                    analysis += f"Your Predicted Output:\\n{format_grid_for_prompt(predicted)}\\n\\n"
                    # Calculate sizes
                    pred_h, pred_w = len(predicted), len(predicted[0]) if predicted else 0
                    exp_h, exp_w = len(example['output']), len(example['output'][0]) if example['output'] else 0
                    analysis += f"Expected size: {exp_h}x{exp_w}, Predicted size: {pred_h}x{pred_w}\n"
                    # Add a visual difference map ('.' match, 'X' mismatch; non-overlap = X)
                    try:
                        diff_map = format_difference_map(predicted, example['output'])
                        analysis += f"Difference:\n{diff_map}\n\n"
                    except Exception:
                        analysis += "Difference: (could not compute difference map)\n\n"
                else:
                    analysis += "Your Predicted Output: No output generated\\n\\n"
                    exp_h, exp_w = len(example['output']), len(example['output'][0]) if example['output'] else 0
                    analysis += f"Expected size: {exp_h}x{exp_w}, Predicted size: 0x0\\n"
                    # When there's no predicted output, mark the entire expected area as mismatches
                    try:
                        diff_map = format_difference_map(None, example['output'])
                        analysis += f"Difference:\n{diff_map}\n\n"
                    except Exception:
                        analysis += "Difference: (could not compute difference map)\n\n"
                
                analysis += f"Overlap: {test.get('overlap_percentage', 0):.1f}%\\n"
                analysis += f"IOU (Intersection over Union): {test.get('iou_percentage', 0):.1f}%\\n"
                
                error_msg = test.get("error_message")
                if error_msg:
                    analysis += f"Error: {error_msg}\\n"
                
                failure_analysis.append(analysis)
        
        failures_block = "\\n".join(failure_analysis)
        
        # Build training examples block
        examples_block = ""
        for i, example in enumerate(training_examples, 1):
            examples_block += f"Training Example {i}\\n--\\n"
            examples_block += f"Input:\\n{format_grid_for_prompt(example['input'])}\\n\\n"
            examples_block += f"Output:\\n{format_grid_for_prompt(example['output'])}\\n\\n"
        
        prompt_parts = [
            "You are an expert mathematician, logistician and pattern recognizier who is solving the"
            "Abstract Reasoning Corpus (ARC) problems.",
            "You previously attempted to solve this task but your solution was incorrect on some training examples."
            "",
            "---------"
            "YOUR GOAL",
            "---------"
            "Analyze your previous attempt deeply, understand why it failed"
            "- Was there an issue with the logic of the code that led to the failure?",
            "- If the code succeeds but the output doesn't match up, what are the difference between the intended output and your predicted output?",
            "- What is missing from your reasoning and solution that leads to these differences?",
            "- How to modify your reasoning and code to correct for these errors and ensure it solves the task fully?",
            "",
            "------------------"
            "YOUR PREVIOUS CODE",
            "------------------",
            f"{previous_code}",
            ""
            "-----------------------"
            "YOUR PREVIOUS REASONING",
            "-----------------------",
            f"{reasoning_trace}",
            "",
            "----------------------------------"
            "YOUR PREVIOUS TRANSFORMATION RULES",
            "----------------------------------",
            f"{steps_text}",
            "",
            "-------------------------",
            "DETAILED FAILURE ANALYSIS",
            "-------------------------",
            f"{failures_block}",
            ""] + rag_hints_parts if rag_hints_parts else [] + [
            "---------------------",
            "ANALYSIS INSTRUCTIONS",
            "---------------------",
            "Provide a ```reasoning``` block that contains your detailed analysis.",
        ]

        prompt = "\n".join(prompt_parts)
        return prompt

    prompt = build_refinement_reasoning_prompt()
    
    def _flatten_content(c):
        # Return a string representation of various response content shapes
        try:
            if isinstance(c, str):
                return c
            if isinstance(c, dict):
                # common forms: {'content': '...'} or {'choices': [...]}
                if 'content' in c and isinstance(c['content'], (str, list)):
                    return _flatten_content(c['content'])
                if 'choices' in c and isinstance(c['choices'], list):
                    return '\n'.join(_flatten_content(ch) for ch in c['choices'])
                # fallback to string
                return str(c)
            if isinstance(c, (list, tuple)):
                parts = []
                for item in c:
                    if isinstance(item, dict):
                        # message-like item with 'content' or 'type' fields
                        if 'content' in item:
                            parts.append(_flatten_content(item['content']))
                            continue
                        # structured content items: {'type':'text','text':...} or {'type':'image',...}
                        if item.get('type') == 'text' and 'text' in item:
                            parts.append(str(item['text']))
                            continue
                        if item.get('type') == 'image' and item.get('image_url'):
                            parts.append('[IMAGE]')
                            continue
                        # generic dict
                        parts.append(str(item))
                    else:
                        parts.append(str(item))
                return '\n'.join([p for p in parts if p])
            # fallback
            return str(c)
        except Exception:
            return str(c)
    
    # Retry up to max_retries times if extraction fails
    for attempt in range(max_retries):
        response = llm.invoke(prompt)
        if hasattr(response, 'content'):
            resp_content = response.content
        else:
            resp_content = response

        response_text = _flatten_content(resp_content)

        # Prefer structured reflection extraction first
        reasoning = extract_reasoning_content(response_text)
        if reasoning and reasoning != "Unable to generate reasoning trace":
            return reasoning, attempt
        
        # If this isn't the last attempt, log and retry
        if attempt < max_retries - 1:
            print(f"Warning: Failed to extract reflection reasoning content (attempt {attempt + 1}/{max_retries}). Retrying...")
    
    # After all retries failed
    return "Unable to generate reasoning trace", max_retries

def format_grid_for_analysis(grid: List[List[int]]) -> str: 
    """Format grid for detailed analysis in reasoning prompts."""
    if not grid:
        return "(empty grid)"
    
    formatted_rows = []
    for row in grid:
        formatted_rows.append("".join(str(cell) for cell in row))
    
    return "\n".join(formatted_rows)


def format_difference_map(predicted: Optional[List[List[int]]], expected: Optional[List[List[int]]], indent: int = 0) -> str:
    """Return a visual difference map where '.' indicates a matching cell and 'X' a mismatch.

    If the predicted and expected grids have different sizes, the non-overlapping
    area is marked with 'X'. The returned string contains one line per row.
    """
    if not expected:
        return "(no expected output)"

    pred_h = len(predicted) if predicted else 0
    pred_w = len(predicted[0]) if pred_h > 0 and predicted[0] else 0
    exp_h = len(expected) if expected else 0
    exp_w = len(expected[0]) if exp_h > 0 and expected[0] else 0

    h = max(exp_h, pred_h)
    w = max(exp_w, pred_w)

    rows = []
    for i in range(h):
        cols = []
        for j in range(w):
            if i < pred_h and j < pred_w and i < exp_h and j < exp_w:
                try:
                    cols.append('.' if predicted[i][j] == expected[i][j] else 'X')
                except Exception:
                    cols.append('X')
            else:
                # Out-of-range or missing cell => mismatch
                cols.append('X')
        rows.append(''.join(cols))

    # Apply indentation to each row
    indentation = " " * indent
    rows = [indentation + row for row in rows]

    return "\n".join(rows)


def extract_reasoning_content(response_text: str) -> str:
    """Extract reasoning content from LLM response."""
    import re
    
    # Ensure we have a string to work with
    if not isinstance(response_text, str):
        try:
            # Try to convert to string if it's not already
            if isinstance(response_text, (list, dict)):
                response_text = str(response_text)
            else:
                response_text = str(response_text)
        except Exception:
            return "Unable to generate reasoning trace"
    
    if not response_text or not response_text.strip():
        return "Unable to generate reasoning trace"
    
    # Look for reasoning block
    try:
        reasoning_match = re.search(r'```reasoning\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            return reasoning_match.group(1).strip()
    except (TypeError, AttributeError) as e:
        print(f"Warning: regex search failed in extract_reasoning_content: {e}")
        # Continue to fallback extraction methods
    
    # Fallback: look for structured content
    patterns = [
        r'PATTERN OBSERVATION:(.*?)(?=TRANSFORMATION HYPOTHESIS:|$)',
        r'TRANSFORMATION HYPOTHESIS:(.*?)(?=VERIFICATION:|$)',
        r'VERIFICATION:(.*?)(?=CORE INSIGHT:|$)',
        r'CORE INSIGHT:(.*?)(?=$)'
    ]
    
    reasoning_parts = []
    try:
        for pattern in patterns:
            match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if match:
                reasoning_parts.append(match.group(1).strip())
        
        if reasoning_parts:
            return '\n\n'.join(reasoning_parts)
    except (TypeError, AttributeError) as e:
        print(f"Warning: pattern matching failed in extract_reasoning_content: {e}")
    
    # Ultimate fallback: return first substantial paragraph
    try:
        lines = response_text.split('\n')
        substantial_lines = [line.strip() for line in lines if len(line.strip()) > 20]
        
        return '\n'.join(substantial_lines[:10]) if substantial_lines else response_text[:500]
    except Exception:
        return response_text[:500] if len(response_text) > 500 else response_text


def generate_transformation_steps(llm, reasoning_trace: str, training_examples: List[Dict], num_solutions: int, max_retries: int = 3) -> Tuple[List[Dict], int]:
    """Extract clear step-by-step transformation from reasoning trace.

    Returns:
        Tuple of (solution_objects_list, num_retries_used)
        Where solution_objects_list is [{"solution_number": int, "transformation_steps": [str, ...]}, ...]
    """

    def build_transformation_steps_prompt() -> str:
        """Build prompt for extracting clear transformation steps."""
        # Build training examples block
        examples_block = ""
        for i, example in enumerate(training_examples, 1):
            examples_block += f"Training Example {i}\\n--\\n"
            examples_block += f"Input:\\n{format_grid_for_prompt(example['input'])}\\n\\n"
            examples_block += f"Output:\\n{format_grid_for_prompt(example['output'])}\\n\\n"
        prompt_parts = [
            "You are an expert mathematician, logistician and pattern recognizier who is solving the Abstract Reasoning Corpus (ARC) problems.",
            "Based on the following reasoning analysis, extract clear step-by-step transformation instructions.",
            "",
            "TRAINING EXAMPLES",
            f"{examples_block}",
            "",
            "REASONING ANALYSIS",
            f"{reasoning_trace}",
            "",
            "INSTRUCTIONS",
            f"Produce {num_solutions} different candidate solutions. Each solution should be a numbered sequence of clear, actionable transformation steps.",
            "Be creative: try different, plausible interpretations of the reasoning so the set of solutions explores diverse approaches (use different object-level operations, orders, or heuristics).",
            "Each solution should be concise and concrete so it can be executed programmatically.",
            "",
            "RESPONSE FORMAT (JSON):",
            f"Return a JSON array in a json block containing {num_solutions} solution objects. Each object should have two keys:",
            f"- \"solution_number\": an integer (1..{num_solutions})",
            "- \"transformation_steps\": a JSON array of strings, each string being a single transformation step (in order)",
            "",
            "Example response structure:",
            "```json",
            "[",
            "  {",
            "    \"solution_number\": 1,",
            "    \"transformation_steps\": [\"Step 1 text\", \"Step 2 text\"]",
            "  },",
            "  {",
            "    \"solution_number\": 2,",
            "    \"transformation_steps\": [\"Step 1 text\", \"Step 2 text\"]",
            "  },",
            "  ...",
            "]",
            "```",
            f"Do NOT output any additional text outside the ```json``` block. Generate the {num_solutions} solutions now."
        ]

        prompt = "\n".join(prompt_parts)
        return prompt
        
    prompt = build_transformation_steps_prompt()
    
    # Retry up to max_retries times if parsing fails
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt, temperature=0.7)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # print_prompt_and_response(prompt, response_text)

            solutions = parse_transformation_steps(response_text)
            if solutions:
                return solutions, attempt
            
            # If parsing failed and this isn't the last attempt, log and retry
            if attempt < max_retries - 1:
                print(f"Warning: Failed to parse transformation steps (attempt {attempt + 1}/{max_retries}). Retrying...")
                
        except Exception as e:
            print(f"Error extracting transformation steps (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
    
    # After all retries failed, return empty list
    return [], max_retries


def generate_refined_transformation_steps(llm, reasoning_trace: str, sol: Dict, training_examples: List[Dict], num_solutions: int, max_retries: int = 3) -> Tuple[List[Dict], int]:
    """Extract refined step-by-step transformation from reasoning trace and previous solution failures.

    Returns:
        Tuple of (solution_objects_list, num_retries_used)
        Where solution_objects_list is [{"solution_number": int, "transformation_steps": [str, ...]}, ...]
    """

    def build_refined_transformation_steps_prompt() -> str:
        """Build a prompt that includes reasoning, the previous solution (and its failures), and training examples.

        The `sol` argument is expected to be a solution dict that may contain keys like
        'solution_number', 'transformation_steps', 'training_results', 'main_code', and any
        failure analysis produced during evaluation. The prompt encourages the LLM to
        produce refined candidate transformation-step sequences that take the failures
        into account.
        """
        # Get relevant information out
        training_results = sol["training_results"]

        # Build detailed failure analysis
        failure_analysis = []
        for test in training_results:
            example_idx = test.get("example_index", 0)
            if example_idx < len(training_examples):
                example = training_examples[example_idx]
                
                analysis = f"Training Example {example_idx + 1} - FAILED\\n"
                analysis += "--\\n"
                analysis += f"Input:\\n{format_grid_for_prompt(example['input'])}\\n\\n"
                analysis += f"Expected Output:\\n{format_grid_for_prompt(example['output'])}\\n\\n"
                
                predicted = test.get("predicted_output")
                if predicted:
                    analysis += f"Your Predicted Output:\\n{format_grid_for_prompt(predicted)}\\n\\n"
                    # Calculate sizes
                    pred_h, pred_w = len(predicted), len(predicted[0]) if predicted else 0
                    exp_h, exp_w = len(example['output']), len(example['output'][0]) if example['output'] else 0
                    analysis += f"Expected size: {exp_h}x{exp_w}, Predicted size: {pred_h}x{pred_w}\n"
                    # Add a visual difference map ('.' match, 'X' mismatch; non-overlap = X)
                    try:
                        diff_map = format_difference_map(predicted, example['output'])
                        analysis += f"Difference:\n{diff_map}\n\n"
                    except Exception:
                        analysis += "Difference: (could not compute difference map)\n\n"
                else:
                    analysis += "Your Predicted Output: No output generated\\n\\n"
                    exp_h, exp_w = len(example['output']), len(example['output'][0]) if example['output'] else 0
                    analysis += f"Expected size: {exp_h}x{exp_w}, Predicted size: 0x0\\n"
                    # When there's no predicted output, mark the entire expected area as mismatches
                    try:
                        diff_map = format_difference_map(None, example['output'])
                        analysis += f"Difference:\n{diff_map}\n\n"
                    except Exception:
                        analysis += "Difference: (could not compute difference map)\n\n"
                
                analysis += f"Overlap: {test.get('overlap_percentage', 0):.1f}%\\n"
                analysis += f"IOU (Intersection over Union): {test.get('iou_percentage', 0):.1f}%\\n"
                
                error_msg = test.get("error_message")
                if error_msg:
                    analysis += f"Error: {error_msg}\\n"
                
                failure_analysis.append(analysis)
            
        failures_block = "\\n".join(failure_analysis) 

        prompt_parts = [
            "You are an expert mathematician, logistician and pattern recognizier who is solving the Abstract Reasoning Corpus (ARC) problems.",
            "You previously attempted a solution which failed to solve the task. You are asked to use the failure information to generate refined candidate transformation steps.",
            "",
            "PREVIOUS SOLUTION & FAILURES:",
            f"{failures_block}",
            ""
            "REFLECTION ON PAST FAILURE:",
            f"{reasoning_trace}",
            "",
            "INSTRUCTIONS:",
            f"Produce {num_solutions} different candidate solutions. Each solution should be a numbered sequence of clear, actionable transformation steps.",
            "Give special attention to correcting the failure modes shown in the previous solution summary.",
            "Each solution should be concise and concrete so it can be executed programmatically.",
            "",
            "RESPONSE FORMAT (JSON):",
            f"Return a JSON array in a json block containing {num_solutions} solution objects. Each object should have two keys:",
            f"- \"solution_number\": an integer (1..{num_solutions})",
            "- \"transformation_steps\": a JSON array of strings, each string being a single transformation step (in order)",
            "",
            "Example response structure:",
            "```json",
            "[",
            "  {",
            "    \"solution_number\": 1,",
            "    \"transformation_steps\": [\"Step 1 text\", \"Step 2 text\"]",
            "  },",
            "  {",
            "    \"solution_number\": 2,",
            "    \"transformation_steps\": [\"Step 1 text\", \"Step 2 text\"]",
            "  },",
            "  ...",
            "]",
            "```",
            f"Do NOT output any additional text outside the ```json``` block. Generate the {num_solutions} solutions now."
        ]

        prompt = "\n".join(prompt_parts)
        return prompt

    prompt = build_refined_transformation_steps_prompt()
    
    # Retry up to max_retries times if parsing fails
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt, temperature=0.7)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # print_prompt_and_response(prompt, response_text)

            solutions = parse_transformation_steps(response_text)
            if solutions:
                return solutions, attempt
            
            # If parsing failed and this isn't the last attempt, log and retry
            if attempt < max_retries - 1:
                print(f"Warning: Failed to parse refined transformation steps (attempt {attempt + 1}/{max_retries}). Retrying...")
                
        except Exception as e:
            print(f"Error extracting refined transformation steps (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
    
    # After all retries failed, return empty list
    return [], max_retries

def parse_transformation_steps(response_text: str) -> List[str]:
    """Parse transformation steps from LLM response."""
    import re, json

    # Attempt 1: prefer a fenced ```json``` block containing the JSON array
    try:
        json_block_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
        candidate = None
        if json_block_match:
            candidate = json_block_match.group(1).strip()
        else:
            # Fallback: try to find a bare JSON array anywhere in the text
            start = response_text.find('[')
            end = response_text.rfind(']')
            if start != -1 and end != -1 and end > start:
                candidate = response_text[start:end+1]

        if candidate:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                solutions = []
                for item in parsed:
                    if isinstance(item, dict):
                        sol_num = item.get('solution_number') or item.get('solution') or item.get('solution_number')
                        steps = item.get('transformation_steps') or item.get('steps') or []
                        # Normalize steps to list of strings
                        if isinstance(steps, str):
                            step_lines = [ln.strip() for ln in steps.splitlines() if ln.strip()]
                            steps = [re.sub(r'^\d+\.\s*', '', ln).strip() for ln in step_lines]
                        elif isinstance(steps, list):
                            steps = [str(s).strip() for s in steps if str(s).strip()]
                        else:
                            steps = []

                        solutions.append({
                            "solution_number": int(sol_num) if (sol_num is not None and str(sol_num).isdigit()) else sol_num,
                            "transformation_steps": steps
                        })

                if solutions:
                    return solutions
    except Exception:
        pass

    # Fallback 1: look for fenced 'steps' block
    steps_match = re.search(r'```steps\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
    if steps_match:
        steps_content = steps_match.group(1).strip()
    else:
        steps_content = response_text

    # Try to split into multiple "Solution X" sections
    sol_splits = re.split(r'\bSolution\s*(\d+)\b', steps_content, flags=re.IGNORECASE)
    # re.split returns [before, num1, block1, num2, block2, ...] if matches
    if len(sol_splits) > 1:
        solutions = []
        # iterate pairs
        it = iter(sol_splits)
        pre = next(it)
        for token in it:
            try:
                num = token
                block = next(it)
            except StopIteration:
                break
            # extract numbered steps within block
            step_pattern = r'\d+\.\s*(.+)'
            matches = re.findall(step_pattern, block)
            steps = [m.strip() for m in matches]
            solutions.append({"solution_number": int(num) if num.isdigit() else num, "transformation_steps": steps})

        if solutions:
            return solutions

    # Fallback 2: extract any numbered steps across the text as a single solution
    step_pattern = r'(\d+\.\s*(.+))'
    matches = re.findall(step_pattern, steps_content, re.MULTILINE)
    if matches:
        steps = [match[1].strip() for match in matches]
        return [{"solution_number": 1, "transformation_steps": steps}]

    # Ultimate fallback: split long lines and return as single solution entries
    lines = [line.strip() for line in steps_content.split('\n') if line.strip()]
    steps = []
    for line in lines:
        cleaned_line = re.sub(r'^\d+\.\s*|^-\s*|^\*\s*', '', line).strip()
        if len(cleaned_line) > 5:
            steps.append(cleaned_line)

    return [{"solution_number": 1, "transformation_steps": steps[:50]}]


def generate_code_from_reasoning(code_llm, reasoning_trace: str, training_examples: List[Dict], num_solutions: int, max_retries: int = 3) -> Tuple[List[str], int]:
    """Generate Python code based on reasoning trace only (without transformation steps).

    Returns:
        Tuple of (python_codes_list, num_retries_used)
    """

    def build_code_from_reasoning_only_prompt(reasoning_trace: str, training_examples: List[Dict], num_solutions: int) -> str:
        """Build prompt for generating Python code from reasoning only."""
        prompt_parts = [
            "You are a Python expert implementing ARC transformations.",
            "",
            "Given the following reasoning analysis, implement Python functions that solve the task.",
            "",
            "------------------",
            "REASONING ANALYSIS",
            "------------------",
            f"{reasoning_trace}",
            "",
            "-----------------",
            "TRAINING EXAMPLES",
            "------------------",
            f"{len(training_examples)} input-output example pairs are provided for validation.",
            "",
        ]
        
        for i, ex in enumerate(training_examples, 1):
            prompt_parts.append(f"Example {i} Input:")
            prompt_parts.append(format_grid_for_prompt(ex.get('input', [])))
            prompt_parts.append(f"Example {i} Output:")
            prompt_parts.append(format_grid_for_prompt(ex.get('output', [])))
            prompt_parts.append("")
        
        prompt_parts.extend([
            "---------------------------",
            "IMPLEMENTATION REQUIREMENTS",
            "---------------------------",
            f"Produce {num_solutions} different candidate solutions based on the reasoning above.",
            "For each solution:",
            "1. Write a function called 'transform(input_grid)' that takes a 2D list of integers as input and returns a transformed 2D list of integers",
            "2. Implement the transformation clearly and precisely based on the reasoning",
            "3. Import any necessary standard libraries at the top for EACH solution",
            "4. Include helper functions where necessary",
            "5. DO NOT ADD ANY EXPLANATIONS OR COMMENTS IN THE CODE",
            "6. Address any error cases or edge conditions mentioned in the reasoning to ensure correctness and robustness",
            "7. Return ONLY executable Python code",
            "The <count> will help you keep track of what-th solution you are at. Make sure you have all solutions implemented.",
            "",
            "Example structure:",
            "<count>1</count>",
            "<solution>",
            "from typing import List",
            "import ... # Import ANY necessary standard libraries to run the code here",
            "def helper_function_1(...):",
            "    # Add helper functions if needed",
            "def helper_function_2(...):",
            "    # Add helper functions if needed",
            "def transform(input_grid):",
            "    [implementation based on reasoning]",
            "    return transformed_grid",
            "</solution>",
            "<count>2</count>",
            "<solution>...</solution>",
            "...",
            "",
            "Generate the Python code now:"
        ])

        prompt = "\n".join(prompt_parts)
        return prompt

    prompt = build_code_from_reasoning_only_prompt(reasoning_trace, training_examples, num_solutions)
    
    # Retry up to max_retries times if extraction fails
    for attempt in range(max_retries):
        try:
            response = code_llm.invoke(prompt, temperature=0.3)
            response_text = response.content if hasattr(response, 'content') else str(response)
            # print_prompt_and_response(prompt, response_text)
            
            # Extract candidate python solutions (may be multiple)
            candidate_codes = extract_python_solutions(response_text)
            # Ensure common imports are present in each candidate code block
            candidate_codes = [ensure_imports_in_code(c) for c in candidate_codes]
            
            if candidate_codes:
                print(f"{len(candidate_codes)} candidate code solutions generated.")
                return candidate_codes, attempt
                
        except Exception as e:
            print(f"Warning: Failed to generate code from reasoning (attempt {attempt + 1}/{max_retries}). Error: {e}")
        
        # If this isn't the last attempt, log and retry
        if attempt < max_retries - 1:
            print(f"Warning: Retrying code generation (attempt {attempt + 1}/{max_retries})...")
    
    # After all retries failed, return empty list
    print(f"Error: Failed to generate code from reasoning after {max_retries} attempts.")
    return [], max_retries


def generate_code_from_reasoning_and_transformations(code_llm, reasoning_trace: str, transformation_steps: List[str],
                                 training_examples: List[Dict]) -> str:
    """Generate Python code based on reasoning trace and transformation steps.

    This function will request code from the LLM, then immediately try to execute
    the generated `transform(input_grid)` on the first training example (if present).
    If execution fails and a `code_llm` is provided, it will invoke that
    LLM up to 3 times to refine the main `transform` function and retry execution.
    The function returns the (possibly refined) Python source for the transform
    function (or a fallback template on failure).
    """

    def build_code_from_reasoning_prompt(reasoning_trace: str, transformation_steps: List[Dict],
                                         training_examples: List[Dict]) -> str:
        """Build prompt for generating Python code from reasoning and steps."""
        # Only support the new structured format: a list of solution dicts
        if not (transformation_steps and isinstance(transformation_steps, list) and isinstance(transformation_steps[0], dict)):
            raise ValueError("transformation_steps must be a list of solution dicts of the form {'solution_number': int, 'transformation_steps': [str,...]}")

        parts = []
        for sol in transformation_steps:
            sol_num = sol.get('solution_number', '?')
            parts.append(f"Solution {sol_num}:")
            for i, s in enumerate(sol.get('transformation_steps', []) or [], 1):
                parts.append(f"{i}. {s}")
            parts.append("")
        steps_text = '\n'.join(parts).strip()

        prompt_parts = [
            "You are a Python expert implementing ARC transformations.",
            "",
            "Given the following reasoning analysis and step-by-step transformation, implement a Python function.",
            "",
            "------------------",
            "REASONING ANALYSIS",
            "------------------",
            f"{reasoning_trace}",
            "",
            "-----------------",
            "TRAINING EXAMPLES",
            "------------------",
            f"{len(training_examples)} input-output example pairs are provided for validation.",
            "",
            "-----------------------------",
            "TRANSFORMATION STEP SOLUTIONS",
            "-----------------------------",
            f"{steps_text}",
            "",
            "---------------------------",
            "IMPLEMENTATION REQUIREMENTS",
            "---------------------------",
            "For each solution",
            "1. Write a function called 'transform(input_grid)' that takes a 2D list of integers as input and returns a transformed 2D list of integers",
            "2. Implement each transformation step clearly and precisely",
            "3. Import any necessary standard libraries at the top for EACH solution",
            "4. Include helper functions where necessary.",
            "5. DO NOT ADD ANY EXPLANATIONS OR COMMENTS IN THE CODE",
            "6. Address any error cases or edge conditions mentioned in the reasoning to ensure correctness and robustness",
            "7. Return ONLY executable Python code",
            "The <count> will help you keep track of what-th solution are you at. Make sure you have all solutions implemented."
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
            "..."
            "",
            "Generate the Python code now:"
        ]

        prompt = "\n".join(prompt_parts)

        return prompt

    prompt = build_code_from_reasoning_prompt(reasoning_trace, transformation_steps, 
                                              training_examples)

    try:
        response = code_llm.invoke(prompt, temperature=0.3)
        response_text = response.content if hasattr(response, 'content') else str(response)

        # print_prompt_and_response(prompt, response_text)
        
        # Extract candidate python solutions (may be multiple)
        candidate_codes = extract_python_solutions(response_text)
        # Ensure common imports are present in each candidate code block so
        # the probe can execute without trivial missing-import errors.
        candidate_codes = [ensure_imports_in_code(c) for c in candidate_codes]
        print(len(candidate_codes), "candidate code solutions generated.")

        # No test input available — return all candidates
        return candidate_codes

    except Exception as e:
        print(f"Error generating code from reasoning: {e}")
        return [generate_fallback_code_from_steps(transformation_steps)]


def generate_fallback_code_from_steps(transformation_steps: List[Union[str, Dict]]) -> str:
    """Generate fallback Python code template from transformation steps."""
    
    code_lines = ["def transform(input_grid):"]
    code_lines.append("    # Copy input grid to work with")
    code_lines.append("    result = copy_grid(input_grid)")
    code_lines.append("    height, width = get_grid_dimensions(input_grid)")
    code_lines.append("")
    # If the new structured format (list of solution dicts) is provided,
    # use the first solution's steps for the fallback template.
    steps_list = []
    try:
        if transformation_steps and isinstance(transformation_steps[0], dict):
            steps_list = transformation_steps[0].get('transformation_steps', []) or []
        else:
            steps_list = transformation_steps or []
    except Exception:
        steps_list = transformation_steps or []

    for i, step in enumerate(steps_list, 1):
        s_text = str(step)
        code_lines.append(f"    # Step {i}: {s_text[:80]}{'...' if len(s_text) > 80 else ''}")
        code_lines.append(f"    # TODO: Implement step {i}")
        code_lines.append("")
    
    code_lines.append("    return result")
    
    return "\n".join(code_lines)


def ensure_imports_in_code(code: str) -> str:
    """Ensure common imports exist at top of a generated Python code string.

    This scans the provided `code` for usage of common modules and typing
    names and prepends import lines that are missing. The function performs
    a conservative, best-effort check and only adds imports for a small set
    of common utilities (typing, json, re, copy, itertools, collections,
    math, numpy).

    The function attempts to combine typing imports into a single
    `from typing import ...` line.
    """
    import re as _re

    if not code or not isinstance(code, str):
        return code

    # Find already-present import lines to avoid duplicates
    existing_imports = set()
    for m in _re.finditer(r'^\s*(?:from|import)\s+([^\n]+)', code, _re.MULTILINE):
        existing_imports.add(m.group(0).strip())

    # Map usage tokens -> import statements (conservative set)
    typing_tokens = {tok for tok in ("List", "Dict", "Any", "Tuple", "Optional", "Set") if _re.search(r'\b%s\b' % tok, code)}

    imports_needed = []
    if typing_tokens:
        typing_line = f"from typing import {', '.join(sorted(typing_tokens))}"
        if not any(l.startswith('from typing') for l in existing_imports):
            imports_needed.append(typing_line)

    token_map = [
        (r'\bjson\b', 'import json'),
        (r'\bre\b', 'import re'),
        (r'\bcopy\b', 'import copy'),
        (r'\bitertools\b', 'import itertools'),
        (r'\bdefaultdict\b|\bCounter\b', 'from collections import defaultdict, Counter'),
        (r'\bcollections\b', 'import collections'),
        (r'\bmath\b', 'import math'),
        (r'\bnp\.', 'import numpy as np'),
        (r'\bnumpy\b', 'import numpy as np'),
        (r'\bdataclass\b', 'from dataclasses import dataclass'),
    ]

    for pattern, imp in token_map:
        if _re.search(pattern, code) and not any(imp in s for s in existing_imports):
            imports_needed.append(imp)

    # If there are no imports to add, return original code
    if not imports_needed:
        return code

    # Prepend imports to the code, keeping a blank line separation
    header = "\n".join(imports_needed) + "\n\n"
    return header + code


def extract_python_solutions(response_text: str) -> List[str]:
    """Extract Python solution code blocks from LLM response and return a list of code strings.

    The LLM response is expected to contain multiple solutions in the following structure:

    <count>1</count>
    <solution>...</solution>
    <count>2</count>
    <solution>...</solution>
    ...

    This function returns a list of the extracted solution bodies (strings). If a
    `<solution>` block contains fenced python code, it will extract the inner
    python; otherwise the raw block text is returned (trimmed).
    """
    import re

    solutions: List[str] = []

    # 1) Prefer explicit <solution>...</solution> blocks
    sol_blocks = re.findall(r'<solution>(.*?)</solution>', response_text, re.DOTALL | re.IGNORECASE)
    if sol_blocks:
        for blk in sol_blocks:
            blk = blk.strip()
            # Use the raw content inside the <solution>...</solution> tags without further parsing.
            # This keeps the original block exactly as the LLM returned it.
            solutions.append(blk)

        return [s for s in solutions if s]


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
        # print_prompt_and_response(prompt, response_text)
    except Exception as e:
        print(f"test_and_fix_code_from_trial_run: LLM invocation failed: {e}")
        return python_codes_list, trial_run_results

    fixed_solutions = extract_python_solutions(response_text)
    if not fixed_solutions:
        return python_codes_list, trial_run_results

    # Normalize by ensuring imports are present and return
    fixed_idx = 0
    fixed_solutions = [ensure_imports_in_code(s) for s in fixed_solutions]
    for r in trial_run_results:
        idx = r.get('index', 0)
        if r.get('error'):
            if fixed_idx < len(fixed_solutions):
                python_codes_list[idx - 1] = fixed_solutions[fixed_idx]
                fixed_idx += 1
    return python_codes_list, trial_run_results


def extract_helpers_from_python_codes(python_codes: List[str]) -> List[Dict[str, str]]:
    """Extract deduplicated helper function signatures and short descriptions.

    Args:
        python_codes: list of Python source strings (each may contain multiple functions).

    Returns:
        A list of dictionaries in the form {"signature": "func(arg1, arg2)",
        "description": "short one-line description"} deduplicated by
        function name and argument names.

    Strategy:
    - Parse each source string using `ast`.
    - For every `FunctionDef`, build a signature using the argument names
        (positional args only for brevity).
    - Prefer the function docstring (first line) as description. If missing,
        fall back to the first source line of the function body.
    - Deduplicate by (name, arg-names) tuple.
    """

    results: List[Dict[str, str]] = []
    seen = set()

    for src in python_codes or []:
        if not isinstance(src, str) or not src.strip():
            continue
        try:
            tree = ast.parse(src)
        except Exception:
            # Skip code that doesn't parse
            continue

        for node in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            name = node.name
            # Collect positional/keyword-only arg names (skip varargs/kw)
            arg_names = []
            try:
                for a in node.args.args:
                    arg_names.append(a.arg)
                for a in getattr(node.args, 'kwonlyargs', []) or []:
                    arg_names.append(a.arg)
            except Exception:
                pass

            key = (name, tuple(arg_names))
            if key in seen:
                continue
            seen.add(key)

            signature = f"{name}({', '.join(arg_names)})"

            # Prefer docstring first
            desc = ast.get_docstring(node) or ""
            if desc:
                desc = desc.strip().splitlines()[0]
            else:
                # Fallback: try to get the first statement source inside the function
                desc = ""
                try:
                    if node.body:
                        first_stmt = node.body[0]
                        snippet = ast.get_source_segment(src, first_stmt) or ""
                        # Clean up snippet onto one line
                        desc = " ".join(snippet.strip().splitlines())[:200]
                except Exception:
                    desc = ""

            results.append({"signature": signature, "description": desc})

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

        # Call the transform function
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

    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error executing transformation code: {e}\n{tb}")
        return None, str(e) + "\n" + tb


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


def analyze_failures(failed_tests: List[ExampleResult], training_examples: List[Dict]) -> Dict[str, Any]:
    """Analyze the pattern of failures to understand what went wrong."""
    analysis = {
        "num_failures": len(failed_tests),
        "error_types": [],
        "size_mismatches": [],
        "color_issues": []
    }
    
    for test in failed_tests:
        if test["error_message"]:
            analysis["error_types"].append(test["error_message"])
        
        if test["predicted_output"] and test["expected_output"]:
            pred_shape = (len(test["predicted_output"]), len(test["predicted_output"][0]) if test["predicted_output"] else 0)
            exp_shape = (len(test["expected_output"]), len(test["expected_output"][0]) if test["expected_output"] else 0)
            
            if pred_shape != exp_shape:
                analysis["size_mismatches"].append({
                    "predicted": pred_shape,
                    "expected": exp_shape
                })
    
    return analysis


def refine_solutions_with_reasoning(llm,
                                    transformation_llm,
                                    code_llm,
                                    current_solution: CodeSolution,
                                    training_examples: List[Dict],
                                    num_refined_solutions: int,
                                    enable_visual_cue: bool = False,
                                    enable_rag_hint: bool = False) -> Tuple[List[str], str, List[Dict]]:
    """Refine a CodeSolution using LLM reflection, transformation extraction, and code regeneration.

    Steps:
    1. Identify failed/partial training examples from `current_solution['training_results']`.
    2. Ask the LLM to reflect on differences between expected vs predicted and produce a
       corrected `reasoning_trace` (via `generate_reflection_reasoning_trace`).
    3. Extract concrete `transformation_steps` from the reasoning.
    4. Generate candidate Python implementations from the reasoning + steps.
    5. Pick a candidate, evaluate on training examples, and return an updated CodeSolution
       with updated `main_code`, `reasoning_trace`, `step_by_step_transformation`, and metrics.

    Returns an updated CodeSolution dict (may be same as input if refinement fails).
    """
    # Defensive copy of solution to avoid mutating the caller's object
    sol = copy.deepcopy(current_solution)

    # Create visual cuesif needed
    visual_cues = []
    if enable_visual_cue:
        # Build visual cues: for each training example, create a small image
        # that shows the input and expected output stacked vertically.
        import base64
        for i, ex in enumerate(training_examples):
            inp = ex.get('input') or []
            out = ex.get('output') or []
            inp_bytes = _grid_to_image_bytes(inp)
            out_bytes = _grid_to_image_bytes(out)
            b64_in = base64.b64encode(inp_bytes).decode('utf-8')
            b64_out = base64.b64encode(out_bytes).decode('utf-8')
            visual_cues.append({
                'example_index': i,
                'input_b64': b64_in,
                'output_b64': b64_out,
            })

    training_results: List[ExampleResult] = sol.get('training_results', []) or []

    # Step 1: Generate reflection reasoning that focuses on what went wrong
    reasoning_trace, reasoning_retries = generate_reflection_reasoning_trace(llm, sol, training_results, training_examples, enable_rag_hint)
    sol['reasoning_trace'] = reasoning_trace

    # Step 2: Extract transformation steps from the reflection reasoning
    transformation_solutions_list, transformation_retries = generate_refined_transformation_steps(transformation_llm, reasoning_trace, sol, training_examples, num_refined_solutions)
    
    # Step 3: Generate candidate code implementations
    if not transformation_solutions_list:
        python_codes_list = generate_code_from_reasoning(code_llm, reasoning_trace, training_examples)
    else:
        python_codes_list = generate_code_from_reasoning_and_transformations(code_llm, reasoning_trace, transformation_solutions_list, training_examples)
    
    # Step 4: Create rag entry if enabled
    if enable_rag_hint:
        distilled_reasoning = generate_distilled_reasoning(llm, reasoning_trace, transformation_solutions_list, python_codes_list)
        distilled_text = f"Strategy: {distilled_reasoning.get('strategy', '')}\nConcepts: {', '.join(distilled_reasoning.get('concepts', []))}"
        embedding = generate_embedding_from_distilled_reasoning(distilled_text)
        helpers = extract_helpers_from_python_codes(python_codes_list)
        rag_entry = ReasoningTraceRecord(
            id=str(uuid.uuid4()),
            reasoning_text=reasoning_trace,
            reasoning_summary=distilled_reasoning.get('strategy', ''),
            concepts=distilled_reasoning.get('concepts', []),
            helpers=helpers,
            vector=embedding,
        )
        # Best-effort: store the distilled reasoning into the Qdrant vector store
        # If qdrant is not available this will be a no-op and will not raise.
        try:
            stored = store_record(rag_entry)
            if stored:
                print(f"✓ Stored refined RAG entry (concepts: {len(rag_entry.concepts)}, helpers: {len(rag_entry.helpers)})")
        except Exception as e:
            print(f"Warning: store_record raised an exception: {e}")
    else:
        rag_entry = None

    # Attach visual cue data onto each transformation dict (best-effort)
    if enable_visual_cue:
        # TODO: Think about a way to add the visual cues here
        pass

    # Return the list of candidate codes, plus reasoning and steps.
    return python_codes_list, reasoning_trace, transformation_solutions_list, rag_entry, reasoning_retries, transformation_retries

def extract_reasoning_from_reflection(response_content: str) -> str:
    """Extract reasoning section from ARC-style reflection response."""
    import re
    
    # Look for reasoning block
    reasoning_match = re.search(r'```reasoning\s*(.*?)\s*```', response_content, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        return reasoning_match.group(1).strip()
    
    # Fallback: look for analysis patterns
    patterns = [
        r'PATTERN MISINTERPRETATION:(.*?)(?=\d\.|\n\n|$)',
        r'LOGIC ERRORS:(.*?)(?=\d\.|\n\n|$)',
        r'EDGE CASES:(.*?)(?=\d\.|\n\n|$)',
        r'CORE INSIGHT:(.*?)(?=\d\.|\n\n|$)'
    ]
    
    reasoning_parts = []
    for pattern in patterns:
        match = re.search(pattern, response_content, re.DOTALL | re.IGNORECASE)
        if match:
            reasoning_parts.append(match.group(1).strip())
    
    if reasoning_parts:
        return '; '.join(reasoning_parts)
    
    # Ultimate fallback
    return "No structured reasoning found in response"


def extract_key_insight_from_reasoning(reasoning: str) -> str:
    """Extract the key insight from reasoning text."""
    # Look for core insight patterns
    import re
    
    patterns = [
        r'(?:CORE INSIGHT|key insight|main insight|crucial insight)[:\s]+(.*?)(?:\n|$)',
        r'(?:The pattern is|Pattern:|Main pattern)[:\s]+(.*?)(?:\n|$)',
        r'(?:I need to|Should|Must)[:\s]+(.*?)(?:\n|$)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, reasoning, re.IGNORECASE)
        if match:
            insight = match.group(1).strip()
            # Clean up and limit length
            insight = re.sub(r'\s+', ' ', insight)
            return insight[:200] + '...' if len(insight) > 200 else insight
    
    # Fallback: take first meaningful sentence
    sentences = re.split(r'[.!?]+', reasoning)
    for sentence in sentences:
        if len(sentence.strip()) > 20:  # Skip very short sentences
            cleaned = re.sub(r'\s+', ' ', sentence.strip())
            return cleaned[:200] + '...' if len(cleaned) > 200 else cleaned
    
    return "Pattern recognition issue identified"


def format_grid_for_prompt(grid: List[List[int]], indent: int = 0) -> str:
    """Format grid for display in prompts."""
    indentation = " " * indent
    return "\n".join(indentation + " ".join(map(str, row)) for row in grid)

def result_comparison_text(training_results1: List[ExampleResult],
                           training_results2: List[ExampleResult]) -> str:
    """Generate a comparison text of training results between two solutions."""

    training_results_comparison = []
    for i, (tr1, tr2) in enumerate(zip(training_results1, training_results2)):
        size_match_a = tr1.get('matching_size', False)
        overlap_a = tr1.get('overlap_percentage', 0.0)
        size_match_b = tr2.get('matching_size', False)
        overlap_b = tr2.get('overlap_percentage', 0.0)
        training_results_comparison.extend([
            f"Example {i+1}:",
            f"  Input:\n{format_grid_for_prompt(tr1.get('input', []), indent=4)}",
            f"  Expected Output:\n{format_grid_for_prompt(tr1.get('expected_output', []), indent=4)}",
            f"  Solution A - size match: {size_match_a}, overlap: {overlap_a:.1f}%",
            f"  Solution A - predicted Output:\n{format_grid_for_prompt(tr1.get('predicted_output', []), indent=4)}",
            f"  Solution A - difference:\n{format_difference_map(tr1.get('predicted_output', []), tr1.get('expected_output', []), indent=4)}",
            f"  Solution B - size match: {size_match_b}, overlap: {overlap_b:.1f}%",
            f"  Solution B - predicted Output:\n{format_grid_for_prompt(tr2.get('predicted_output', []), indent=4)}",
            f"  Solution B - difference:\n{format_difference_map(tr2.get('predicted_output', []), tr2.get('expected_output', []), indent=4)}",
            ""
        ])
    training_results_text = "\n".join(training_results_comparison).strip()
    return training_results_text


def generate_fused_reasoning_trace(llm,
                                   sola: Dict,
                                   solb: Dict,
                                   training_results1: List[ExampleResult],
                                   training_results2: List[ExampleResult],
                                   training_examples: List[Dict],
                                   enable_rag_hint: bool,
                                   max_retries: int = 3) -> Tuple[str, int]:
    """Generate a fused reasoning trace that reconciles two candidate solutions.

    The prompt includes both solutions' reasoning, transformation steps, code (if available),
    and the training results. The LLM is asked to produce a single, coherent reasoning
    trace that explains how to combine their strengths and address their failure modes.
    
    Returns:
        Tuple of (reasoning_trace, num_retries_used)
    """
    def build_fused_reasoning_trace_prompt():
        reasoning_a = sola.get('reasoning_trace') or "(no reasoning)"
        steps_text_a = build_steps_text_from_transformation_steps(sola.get('step_by_step_transformation') or [])
        code_a = sola.get('main_code') or "(no code)"

        reasoning_b = solb.get('reasoning_trace') or "(no reasoning)"
        steps_text_b = build_steps_text_from_transformation_steps(solb.get('step_by_step_transformation') or [])
        code_b = solb.get('main_code') or "(no code)"

        training_results_text = result_comparison_text(training_results1, training_results2)

        rag_concepts = set()
        rag_hints_parts = []

        if enable_rag_hint:
            vectora = sola.get('vector')
            vectorb = solb.get('vector')
            entries_a = retrieve_similar_distillations(vector=vectora, top_k=5)
            if entries_a:
                print("✓ Retrieved RAG entries for Solution A")
            entries_b = retrieve_similar_distillations(vector=vectorb, top_k=5)
            if entries_b:
                print("✓ Retrieved RAG entries for Solution B")

            rag_concepts = set()
            for entry in entries_a + entries_b:
                payload = entry.get('payload', {})
                concepts = payload.get('concepts') or []
                if isinstance(concepts, str):
                    concepts = [c.strip() for c in re.split(r'[;,\n]', concepts) if c.strip()]
                elif not isinstance(concepts, (list, tuple)):
                    concepts = []
                for c in concepts:
                    rag_concepts.add(c)
            if rag_concepts:
                print(f"✓ Found {len(rag_concepts)} RAG concepts for fused reasoning prompt")
        
        if rag_concepts:
            rag_hints_parts = [
                "---------------------",
                "RELATED CONCEPT HINTS",
                "---------------------",
                "The following concepts were found in similar prior solutions. Feel free to consider them in your analysis:",
                "\n".join(f"- {c}" for c in rag_concepts),
                ""
            ]

        parts = [
            "You are an expert ARC solver. Two candidate solutions were produced for the same task.",
            "Your job is to reconcile them into a single solution that combines their strengths and remedies their weaknesses.",
            "",
            "----------",
            "SOLUTION A",
            "----------",
            "",
            "REASONING TRACE A:",
            f"{reasoning_a}",
            "",
            "TRANSFORMATION STEPS A:",
            f"{steps_text_a}",
            "",
            "CODE A:",
            f"{code_a}",
            "",
            "----------",
            "SOLUTION B",
            "----------",
            "",
            "REASONING TRACE B:",
            f"{reasoning_b}",
            "",
            "TRANSFORMATION STEPS B:",
            f"{steps_text_b}",
            "",
            "CODE B:",
            f"{code_b}",
            "",
            "---------------------------",
            "TRAINING RESULTS COMPARISON",
            "---------------------------",
            "For each training example, show the performance of each solution in terms of size match and overlap percentage.",
            "",
            f"{training_results_text}",
            ""] + rag_hints_parts + [
            "---------------------",
            "ANALYSIS INSTRUCTIONS",
            "---------------------",
            "Produce a single ```reasoning``` block that: "
            "- Explains how the two solutions related to the final solution",
            "- The strengths and weaknesses of each solution, and how they complement each other",
            "- Proposes a fused general rule that combines the two"
        ]
        return "\n".join(parts)
    
    prompt = build_fused_reasoning_trace_prompt()
    # If visual cues are provided and the llm driver supports image messages,
    # send a structured message containing the images (base64 data URLs).
    
    # Retry up to max_retries times if extraction fails
    for attempt in range(max_retries):
        response = llm.invoke(prompt)

        # Extract reasoning from response
        response_text = response.content if hasattr(response, 'content') else str(response)
        print_prompt_and_response(prompt, response_text)

        reasoning = extract_reasoning_content(response_text)
        if reasoning and reasoning != "Unable to generate reasoning trace":
            return reasoning, attempt
        
        # If this isn't the last attempt, log and retry
        if attempt < max_retries - 1:
            print(f"Warning: Failed to extract fused reasoning content (attempt {attempt + 1}/{max_retries}). Retrying...")
    
    # After all retries failed
    return "Unable to generate reasoning trace", max_retries


def generate_fused_transformation_steps(llm,
                                        reasoning_trace: str,
                                        sola: Dict,
                                        solb: Dict,
                                        training_results_a: List[ExampleResult],
                                        training_results_b: List[ExampleResult],
                                        training_examples: List[Dict],
                                        num_solutions: int,
                                        max_retries: int = 3) -> Tuple[List[Dict], int]:
    """Generate candidate fused transformation step sequences from a fused reasoning prompt.

    Returns:
        Tuple of (solution_objects_list, num_retries_used)
        Where solution_objects_list is [{{"solution_number": int, "transformation_steps": [str, ...]}}, ...]
    """
    def build_fused_transformation_steps_prompt() -> str:
        steps_text_a = build_steps_text_from_transformation_steps(sola.get('step_by_step_transformation') or [])
        code_a = sola.get('main_code') or "(no code)"

        steps_text_b = build_steps_text_from_transformation_steps(solb.get('step_by_step_transformation') or [])
        code_b = solb.get('main_code') or "(no code)"

        training_results_text = result_comparison_text(training_results_a, training_results_b)

        parts = [
            "You are an expert ARC solver. Based on the reasoning that represents fusion of two solutions below, produce multiple candidate fused transformation rules.",
            "Each candidate should be a clear ordered list of transformation steps that can be implemented programmatically.",
            "",
            "",
            "----------",
            "SOLUTION A",
            "----------",
            "",
            "TRANSFORMATION STEPS A:",
            f"{steps_text_a}",
            "",
            "CODE A:",
            f"{code_a}",
            "",
            "----------",
            "SOLUTION B",
            "----------",
            "",
            "TRANSFORMATION STEPS B:",
            f"{steps_text_b}",
            "",
            "CODE B:",
            f"{code_b}",
            "",
            "---------------------",
            "RESULTS COMPARISON",
            "---------------------",
            "",
            f"{training_results_text}",
            "",
            "---------------------",
            "FUSED REASONING TRACE",
            "---------------------",
            "",
            f"{reasoning_trace}",
            "",
            "------------",
            "INSTRUCTIONS",
            "------------",
            f"Produce {num_solutions} different candidate solutions based on the above fused reasoning which attempts to combine the strengths of both solutions.",
            "Try to pick different aspects from each solution to create diverse candidates while still addressing the failures of individual solutions.",
            "Each solution should be concise and concrete so it can be executed programmatically.",
            "",
            "RESPONSE FORMAT (JSON):",
            f"Return a JSON array in a json block containing {num_solutions} solution objects. Each object should have two keys:",
            f"- \"solution_number\": an integer (1..{num_solutions})",
            "- \"transformation_steps\": a JSON array of strings, each string being a single transformation step (in order)",
            "",
            "Example response structure:",
            "```json",
            "[",
            "  {",
            "    \"solution_number\": 1,",
            "    \"transformation_steps\": [\"Step 1 text\", \"Step 2 text\"]",
            "  },",
            "  {",
            "    \"solution_number\": 2,",
            "    \"transformation_steps\": [\"Step 1 text\", \"Step 2 text\"]",
            "  },",
            "  ...",
            "]",
            "```",
            f"Do NOT output any additional text outside the ```json``` block. Generate the {num_solutions} solutions now."
        ]

        for i, ex in enumerate(training_examples, 1):
            parts.append(f"Example {i} Input:\n{format_grid_for_prompt(ex.get('input', []))}")
            parts.append(f"Example {i} Output:\n{format_grid_for_prompt(ex.get('output', []))}")
            parts.append("")

        parts.extend([
            "",
            "INSTRUCTIONS:",
            f"Produce {num_solutions} candidate fused solutions. Return them as a single JSON array inside a ```json``` fenced block. Each object should have keys: 'solution_number' (int) and 'transformation_steps' (array of strings).",
            "Do NOT include extra commentary. Generate the JSON array now."
        ])

        return "\n".join(parts)

    prompt = build_fused_transformation_steps_prompt()
    
    # Retry up to max_retries times if parsing fails
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt, temperature=0.7)
            response_text = response.content if hasattr(response, 'content') else str(response)
            # print_prompt_and_response(prompt, response_text)
            solutions = parse_transformation_steps(response_text)
            
            if solutions:
                return solutions, attempt
                
        except Exception as e:
            print(f"Warning: Failed to generate fused transformation steps (attempt {attempt + 1}/{max_retries}). Error: {e}")
        
        # If this isn't the last attempt, log and retry
        if attempt < max_retries - 1:
            print(f"Warning: Failed to parse fused transformation steps (attempt {attempt + 1}/{max_retries}). Retrying...")
    
    # After all retries failed, return empty list
    return [], max_retries


def fuse_solutions_with_reasoning(llm,
                                  transformation_llm,
                                  code_llm,
                                  sola: CodeSolution,
                                  solb: CodeSolution,
                                  training_examples: List[Dict],
                                  num_fused_solutions: int,
                                  enable_visual_cue: bool = False,
                                  enable_rag_hint: bool = False) -> Tuple[List[str], str, List[Dict]]:
    """Attempt to fuse two CodeSolution candidates into a stronger combined solution.

    Returns a tuple: (python_codes_list, fused_reasoning_trace, fused_transformation_solutions_list)
    """
    # Build visual cues if requested
    if enable_visual_cue:
        # TODO: Implement visual cue generation for fused solutions if needed
        pass

    # Merge training_results from both solutions (concatenate, allowing duplicates)
    tra = sola.get('training_results') or []
    trb = solb.get('training_results') or []

    # 1) Generate fused reasoning trace
    fused_reasoning, reasoning_retries = generate_fused_reasoning_trace(llm, sola, solb, tra, trb, training_examples, enable_rag_hint)

    # 2) Generate fused transformation steps
    fused_transformation_solutions, transformation_retries = generate_fused_transformation_steps(transformation_llm, fused_reasoning, sola, solb, tra, trb, training_examples, num_fused_solutions)

    # 3) Generate candidate Python implementations from fused reasoning and steps
    if not fused_transformation_solutions:
        python_codes_list = generate_code_from_reasoning(code_llm, fused_reasoning, training_examples)
    else:
        python_codes_list = generate_code_from_reasoning_and_transformations(code_llm, fused_reasoning, fused_transformation_solutions, training_examples)

    # Step 4: Create rag entry if enabled
    if enable_rag_hint:
        distilled_reasoning = generate_distilled_reasoning(llm, fused_reasoning, fused_transformation_solutions, python_codes_list)
        distilled_text = f"Strategy: {distilled_reasoning.get('strategy', '')}\nConcepts: {', '.join(distilled_reasoning.get('concepts', []))}"
        embedding = generate_embedding_from_distilled_reasoning(distilled_text)
        helpers = extract_helpers_from_python_codes(python_codes_list)
        rag_entry = ReasoningTraceRecord(
            id=str(uuid.uuid4()),
            reasoning_text=fused_reasoning,
            reasoning_summary=distilled_reasoning.get('strategy', ''),
            concepts=distilled_reasoning.get('concepts', []),
            helpers=helpers,
            vector=embedding,
        )
        # Best-effort: store the distilled reasoning into the Qdrant vector store
        # If qdrant is not available this will be a no-op and will not raise.
        try:
            stored = store_record(rag_entry)
            if stored:
                print(f"✓ Stored fused RAG entry (concepts: {len(rag_entry.concepts)}, helpers: {len(rag_entry.helpers)})")
        except Exception as e:
            print(f"Warning: store_record raised an exception: {e}")

    return python_codes_list, fused_reasoning, fused_transformation_solutions, rag_entry, reasoning_retries, transformation_retries