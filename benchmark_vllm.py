#!/usr/bin/env python3
"""Benchmark vLLM vs non-vLLM model runtimes.

This script attempts to initialize an LLM using the project's
`initialize_llm_from_config` helper and run a small number of prompt
invocations to measure latency. It will try both vLLM-backed and
non-vLLM-backed initializations for comparison.

Usage examples:
  python benchmark_vllm.py --model "qwen2.5:32b" --iterations 5
  python benchmark_vllm.py --model "qwen2.5:32b" --iterations 20 --max-tokens 64

The script will save results to `benchmark_vllm_results.json` by default.
"""
import argparse
import time
import json
import statistics
import traceback
from typing import Any


def safe_invoke(llm: Any, prompt: str, max_tokens: int = 128):
    """Try several common LLM invocation patterns and return (text, raw_response).

    This helper is intentionally permissive because different wrappers expose
    different call signatures (e.g., `invoke`, `generate`, `__call__`, etc.).
    """
    try:
        # Common Chat/LLM wrappers sometimes expose an `invoke` method
        if hasattr(llm, 'invoke'):
            try:
                return llm.invoke(prompt=prompt, max_tokens=max_tokens), None
            except TypeError:
                return llm.invoke(prompt), None

        # LangChain-style LLMs sometimes expose `generate` which may accept a list
        if hasattr(llm, 'generate'):
            try:
                res = llm.generate([prompt])
                return res, None
            except Exception:
                try:
                    res = llm.generate(prompt)
                    return res, None
                except Exception:
                    pass

        # Some wrappers are callable
        if callable(llm):
            try:
                return llm(prompt), None
            except TypeError:
                return llm(prompt, max_tokens=max_tokens), None

        # Last resort: try a `complete` method
        if hasattr(llm, 'complete'):
            try:
                return llm.complete(prompt=prompt, max_tokens=max_tokens), None
            except Exception:
                return llm.complete(prompt), None

        return None, f"No known invocation method on LLM object of type {type(llm)}"

    except Exception as e:
        return None, f"Invocation failed: {e}\n{traceback.format_exc()}"


def extract_text(resp: Any) -> str:
    """Extract human-readable text from common response shapes."""
    if resp is None:
        return ""
    try:
        # LangChain-like objects sometimes have .generations or .generations[0][0].text
        if hasattr(resp, 'generations'):
            gens = getattr(resp, 'generations')
            try:
                # generations -> list[list[Generation]]
                if isinstance(gens, (list, tuple)) and gens:
                    first = gens[0]
                    if isinstance(first, (list, tuple)) and first:
                        g = first[0]
                        return getattr(g, 'text', str(g))
            except Exception:
                pass

        # simple objects with .content or .text
        if hasattr(resp, 'content'):
            return str(getattr(resp, 'content'))
        if hasattr(resp, 'text'):
            return str(getattr(resp, 'text'))

        # dict-like
        if isinstance(resp, dict):
            for k in ('text', 'content', 'response'):
                if k in resp:
                    return str(resp[k])
            # try first value
            if resp:
                return str(next(iter(resp.values())))

        # fall back to string conversion
        return str(resp)
    except Exception:
        return str(resp)


def benchmark_one(llm: Any, prompt: str, iterations: int, max_tokens: int):
    # Warm up
    _ = safe_invoke(llm, prompt, max_tokens)

    latencies = []
    outputs = []
    errors = []

    for i in range(iterations):
        start = time.time()
        resp, err = safe_invoke(llm, prompt, max_tokens)
        dur = time.time() - start
        latencies.append(dur)
        if err:
            errors.append(err)
            outputs.append(None)
        else:
            outputs.append(extract_text(resp))

    return {
        'iterations': iterations,
        'latencies': latencies,
        'avg_latency': statistics.mean(latencies) if latencies else None,
        'p50': statistics.median(latencies) if latencies else None,
        'p90': (sorted(latencies)[int(len(latencies)*0.9)-1] if latencies and len(latencies) > 1 else None),
        'examples': outputs,
        'errors': errors,
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark vLLM vs non-vLLM for a model')
    parser.add_argument('--model', type=str, default='qwen2.5:32b')
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--prompt', type=str, default='Write a short poem about a curious robot.')
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--results-file', type=str, default='benchmark_vllm_results.json')
    args = parser.parse_args()

    results = {'model': args.model, 'iterations': args.iterations, 'max_tokens': args.max_tokens, 'runs': {}}

    # Import the helper from the project's runner to reuse model config logic.
    try:
        from run_multi_solution_langgraph_agent import initialize_llm_from_config
    except Exception as e:
        print(f"Could not import initialize helper: {e}")
        initialize_llm_from_config = None

    # Try vLLM-backed initialization
    print('\n=== vLLM-backed run ===')
    vllm_llm = None
    try:
        if initialize_llm_from_config:
            vllm_llm = initialize_llm_from_config(args.model, use_vllm=True)
        else:
            try:
                from langchain_vllm import VLLM
                vllm_llm = VLLM(model=args.model)
            except Exception as e:
                print(f"Could not initialize langchain_vllm directly: {e}")
                vllm_llm = None
    except Exception as e:
        print(f"vLLM init error: {e}")
        vllm_llm = None

    if vllm_llm is not None:
        try:
            res_v = benchmark_one(vllm_llm, args.prompt, args.iterations, args.max_tokens)
            results['runs']['vllm'] = res_v
            print(f"vLLM avg latency: {res_v['avg_latency']:.3f}s")
        except Exception as e:
            print(f"vLLM benchmark failed: {e}\n{traceback.format_exc()}")
            results['runs']['vllm'] = {'error': str(e)}
    else:
        print('vLLM initialization unavailable; skipping vLLM run')
        results['runs']['vllm'] = {'error': 'initialization_failed'}

    # Try non-vLLM initialization (provider default)
    print('\n=== non-vLLM run ===')
    nonv_llm = None
    try:
        if initialize_llm_from_config:
            nonv_llm = initialize_llm_from_config(args.model, use_vllm=False)
        else:
            # try common langchain wrappers (best-effort)
            try:
                from langchain_openai import ChatOpenAI
                nonv_llm = ChatOpenAI(model_name=args.model)
            except Exception:
                nonv_llm = None
    except Exception as e:
        print(f"non-vLLM init error: {e}")
        nonv_llm = None

    if nonv_llm is not None:
        try:
            res_nv = benchmark_one(nonv_llm, args.prompt, args.iterations, args.max_tokens)
            results['runs']['non_vllm'] = res_nv
            print(f"non-vLLM avg latency: {res_nv['avg_latency']:.3f}s")
        except Exception as e:
            print(f"non-vLLM benchmark failed: {e}\n{traceback.format_exc()}")
            results['runs']['non_vllm'] = {'error': str(e)}
    else:
        print('non-vLLM initialization unavailable; skipping non-vLLM run')
        results['runs']['non_vllm'] = {'error': 'initialization_failed'}

    # Save results
    with open(args.results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {args.results_file}")


if __name__ == '__main__':
    main()
