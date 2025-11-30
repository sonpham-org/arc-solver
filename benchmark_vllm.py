#!/usr/bin/env python3
#"""Benchmark vLLM vs transformers (clean, single copy).
#...

"""Benchmark vLLM vs transformers (clean, single copy).

This script avoids running both backends concurrently on the same GPU
by offering options to run only one backend, run transformers first when
both use the GPU, and attempt best-effort GPU cleanup between runs.

Usage examples:
  python benchmark_vllm.py --model "qwen2.5:7b" --iterations 5 --transformers-device cuda
  python benchmark_vllm.py --model "qwen2.5:7b" --only-vllm
  python benchmark_vllm.py --model "qwen2.5:7b" --only-transformers --transformers-device cpu

Notes:
- If you run both backends and want the transformers pipeline on GPU,
  the script will initialize transformers first to avoid vLLM holding
  most of the GPU memory during pipeline init.
- When possible the script will attempt to free GPU memory between runs
  by deleting objects, running `gc.collect()` and `torch.cuda.empty_cache()`.
"""
import argparse
import time
import sys
import os
import subprocess
import json
import statistics
import traceback
import gc
from typing import Any, Optional


# ---------------------- Helpers ---------------------------------


def _resolve_model(model_name: str) -> str:
    try:
        from model_configs import resolve_hf_model
        return resolve_hf_model(model_name) or model_name
    except Exception:
        return model_name


def safe_invoke(llm: Any, prompt: str, max_tokens: int = 128):
    try:
        if hasattr(llm, 'invoke'):
            try:
                return llm.invoke(prompt=prompt, max_tokens=max_tokens), None
            except TypeError:
                return llm.invoke(prompt), None

        if hasattr(llm, 'generate'):
            try:
                return llm.generate([prompt]), None
            except Exception:
                try:
                    return llm.generate(prompt), None
                except Exception:
                    pass

        if callable(llm):
            try:
                return llm(prompt), None
            except TypeError:
                return llm(prompt, max_new_tokens=max_tokens), None

        if hasattr(llm, 'complete'):
            try:
                return llm.complete(prompt=prompt, max_tokens=max_tokens), None
            except Exception:
                return llm.complete(prompt), None

        return None, f"No known invocation method on LLM object of type {type(llm)}"

    except Exception as e:
        return None, f"Invocation failed: {e}\n{traceback.format_exc()}"


def extract_text(resp: Any) -> str:
    if resp is None:
        return ""
    try:
        if isinstance(resp, list) and resp:
            first = resp[0]
            if isinstance(first, dict):
                for k in ('generated_text', 'text', 'content'):
                    if k in first:
                        return str(first[k])
                return str(first)

        if hasattr(resp, 'generations'):
            gens = getattr(resp, 'generations')
            try:
                if isinstance(gens, (list, tuple)) and gens:
                    first = gens[0]
                    if isinstance(first, (list, tuple)) and first:
                        g = first[0]
                        return getattr(g, 'text', str(g))
            except Exception:
                pass

        if hasattr(resp, 'content'):
            return str(getattr(resp, 'content'))
        if hasattr(resp, 'text'):
            return str(getattr(resp, 'text'))

        if isinstance(resp, dict):
            for k in ('text', 'content', 'response', 'generated_text'):
                if k in resp:
                    return str(resp[k])
            if resp:
                return str(next(iter(resp.values())))

        return str(resp)
    except Exception:
        return str(resp)


def benchmark_one(llm: Any, prompt: str, iterations: int, max_tokens: int):
    _ = safe_invoke(llm, prompt, max_tokens)

    latencies = []
    outputs = []
    errors = []

    for _ in range(iterations):
        start = time.monotonic()
        resp, err = safe_invoke(llm, prompt, max_tokens)
        dur = time.monotonic() - start
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


# ---------------------- Backends init ---------------------------------


def init_vllm(model_id: str, dtype_hint: Optional[str] = None):
    try:
        import os
        os.environ.pop('HF_HUB_OFFLINE', None)
        os.environ.pop('HUGGINGFACE_HUB_OFFLINE', None)
        from langchain_community.llms import VLLM
        print(f"[vLLM] initializing with model id: {model_id} dtype_hint={dtype_hint}")

        # Map dtype hint to torch dtype if provided
        dtype_kw = {}
        if dtype_hint:
            try:
                import torch
                if dtype_hint == 'bf16':
                    dtype_kw['dtype'] = torch.bfloat16
                elif dtype_hint == 'fp16':
                    dtype_kw['dtype'] = torch.float16
                elif dtype_hint == 'float32':
                    dtype_kw['dtype'] = torch.float32
            except Exception:
                # If torch not available or mapping fails, continue without dtype kw
                pass

        return VLLM(model=model_id, local_files_only=False, **dtype_kw)
    except Exception as e:
        print(f"[vLLM] initialization failed: {e}")
        return None


def init_transformers_pipeline(model_id: str, device: int = -1):
    try:
        import os
        os.environ.pop('HF_HUB_OFFLINE', None)
        os.environ.pop('HUGGINGFACE_HUB_OFFLINE', None)

        from transformers import pipeline
        device_arg = device if device is not None else -1
        print(f"[transformers] initializing pipeline with model id: {model_id} device={device_arg}")
        gen = pipeline('text-generation', model=model_id, device=device_arg)
        return gen
    except Exception as e:
        print(f"[transformers] pipeline init failed: {e}")
        return None


def try_free_gpu():
    try:
        gc.collect()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    except Exception:
        pass


# ---------------------- Main ---------------------------------


def main():
    parser = argparse.ArgumentParser(description='Benchmark vLLM vs transformers')
    parser.add_argument('--model', type=str, default='qwen2.5:7b')
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--prompt', type=str, default='Write a short poem about a curious robot.')
    parser.add_argument('--max-tokens', type=int, default=10000)
    parser.add_argument('--results-file', type=str, default='benchmark_vllm_results.json')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--only-vllm', action='store_true', help='Run only the vLLM-backed backend')
    group.add_argument('--only-transformers', action='store_true', help='Run only the transformers pipeline')

    parser.add_argument('--transformers-device', type=str, choices=['cpu', 'cuda'], default='cuda',
                        help='Device for the transformers pipeline (default: cuda). If cuda and running both backends, transformers will be initialized first to avoid vLLM occupying GPU memory.')
    parser.add_argument('--vllm-dtype', type=str, choices=['bf16', 'fp16', 'float32'], default=None,
                        help='Hint for vLLM dtype. If not set, vLLM default is used (often bf16).')
    parser.add_argument('--order', type=str, choices=['vllm-first', 'transformers-first'], default=None,
                        help='Explicit backend ordering when running both backends. If not set, script chooses an order to minimize OOMs.')
    parser.add_argument('--isolate', action='store_true', help='Run each backend in a separate subprocess to guarantee GPU memory is released between runs')
    parser.add_argument('--isolate-wait-seconds', type=float, default=1.0, help='Seconds to wait after a subprocess exits before continuing (defaults to 1.0)')

    args = parser.parse_args()

    model_raw = args.model
    model_for_vllm = _resolve_model(model_raw)
    model_for_transformers = _resolve_model(model_raw)

    results = {'model_requested': model_raw, 'model_vllm': model_for_vllm, 'model_transformers': model_for_transformers, 'iterations': args.iterations, 'runs': {}}

    run_vllm = True
    run_transformers = True
    if args.only_vllm:
        run_transformers = False
    if args.only_transformers:
        run_vllm = False

    # Determine ordering preference
    order = args.order
    if order is None:
        # prefer transformers-first when transformers target GPU (to avoid vLLM occupying GPU)
        if run_transformers and run_vllm and args.transformers_device == 'cuda':
            order = 'transformers-first'
        else:
            order = 'vllm-first'

    # If isolation requested, run each backend in its own subprocess (ensures GPU memory freed)
    if args.isolate and run_vllm and run_transformers:
        def _child_results_file(base: str, suffix: str) -> str:
            # produce e.g. base.vllm.json or base.transformers.json
            if base.lower().endswith('.json'):
                return base[:-5] + f'.{suffix}.json'
            return base + f'.{suffix}.json'

        def _run_subprocess_for(flag_list, child_results: str):
            cmd = [sys.executable, os.path.abspath(__file__)] + flag_list + ['--model', args.model, '--iterations', str(args.iterations), '--max-tokens', str(args.max_tokens), '--results-file', child_results]
            # preserve transformers-device when running transformers subprocess
            try:
                print(f"Running subprocess: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Subprocess failed: {e}")

        base_results = args.results_file
        vllm_results = _child_results_file(base_results, 'vllm')
        trf_results = _child_results_file(base_results, 'transformers')

        if order == 'transformers-first':
            _run_subprocess_for(['--only-transformers', '--transformers-device', args.transformers_device], trf_results)
            time.sleep(args.isolate_wait_seconds)
            try_free_gpu()
            # pass vllm dtype to child if set
            child_flags = ['--only-vllm']
            if args.vllm_dtype:
                child_flags += ['--vllm-dtype', args.vllm_dtype]
            _run_subprocess_for(child_flags, vllm_results)
        else:
            child_flags = ['--only-vllm']
            if args.vllm_dtype:
                child_flags += ['--vllm-dtype', args.vllm_dtype]
            _run_subprocess_for(child_flags, vllm_results)
            time.sleep(args.isolate_wait_seconds)
            try_free_gpu()
            _run_subprocess_for(['--only-transformers', '--transformers-device', args.transformers_device], trf_results)

        print('\nSubprocess runs complete; individual results written to:', vllm_results, trf_results)
        return

    # If both backends requested and transformers will use GPU, initialize transformers first
    if run_transformers and run_vllm and order == 'transformers-first':
        print('\n=== transformers (GPU) run first to avoid vLLM GPU contention ===')
        device = 0
        trf = init_transformers_pipeline(model_for_transformers, device=device)
        if trf is not None:
            try:
                res_t = benchmark_one(trf, args.prompt, args.iterations, args.max_tokens)
                results['runs']['transformers'] = res_t
                print(f"transformers avg latency: {res_t['avg_latency']:.3f}s")
            except Exception as e:
                print(f"transformers benchmark failed: {e}\n{traceback.format_exc()}")
                results['runs']['transformers'] = {'error': str(e)}
        else:
            print('transformers pipeline unavailable; skipping non-vLLM run')
            results['runs']['transformers'] = {'error': 'initialization_failed'}

        # Attempt to free GPU before starting vLLM
        try_free_gpu()

        # Now init vLLM (likely uses GPU)
        if run_vllm:
            print('\n=== vLLM-backed run ===')
            vllm_llm = init_vllm(model_for_vllm)
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

    else:
        # Default ordering: run vLLM first (if requested), then transformers (on requested device)
        if run_vllm:
            print('\n=== vLLM-backed run ===')
            vllm_llm = init_vllm(model_for_vllm)
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

        # Attempt to free GPU before starting transformers
        try_free_gpu()

        if run_transformers:
            print('\n=== transformers run ===')
            device = 0 if args.transformers_device == 'cuda' else -1
            trf = init_transformers_pipeline(model_for_transformers, device=device)
            if trf is not None:
                try:
                    res_t = benchmark_one(trf, args.prompt, args.iterations, args.max_tokens)
                    results['runs']['transformers'] = res_t
                    print(f"transformers avg latency: {res_t['avg_latency']:.3f}s")
                except Exception as e:
                    print(f"transformers benchmark failed: {e}\n{traceback.format_exc()}")
                    results['runs']['transformers'] = {'error': str(e)}
            else:
                print('transformers pipeline unavailable; skipping non-vLLM run')
                results['runs']['transformers'] = {'error': 'initialization_failed'}

    # Final GPU cleanup attempt
    try_free_gpu()

    with open(args.results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {args.results_file}")


if __name__ == '__main__':
    main()

