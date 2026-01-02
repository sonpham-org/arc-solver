"""
RAG (Retrieval-Augmented Generation) operations for ARC agent.
"""

import hashlib
import math
import uuid
import os
import json
import ast
from typing import List, Dict, Any, Optional


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
