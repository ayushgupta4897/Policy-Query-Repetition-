#!/usr/bin/env python3
"""
Multi-model concurrent runner for optimized PQR experiments.
"""

import json
import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import google.generativeai as genai
from openai import OpenAI

from config import (
    OPENAI_API_KEY, DEEPINFRA_API_KEY, DEEPINFRA_BASE_URL, GEMINI_API_KEY,
    MODELS, ACTIVE_MODELS, TEMPERATURE, MAX_TOKENS, RESULTS_DIR
)
from optimized_test_cases import ALL_OPTIMIZED_CASES

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
deepinfra_client = OpenAI(api_key=DEEPINFRA_API_KEY, base_url=DEEPINFRA_BASE_URL)
genai.configure(api_key=GEMINI_API_KEY)

MAX_WORKERS = 15


def build_baseline_prompt(policy: str, query: str) -> list[dict]:
    return [
        {"role": "system", "content": policy},
        {"role": "user", "content": query}
    ]


def build_pqr_prompt(policy: str, query: str) -> list[dict]:
    repeated_user_content = f"""{query}

─────────────────────────────────────────
IMPORTANT: Before responding, re-read these rules:

{policy}
─────────────────────────────────────────

Now answer the original query while following ALL rules above."""
    
    return [
        {"role": "system", "content": policy},
        {"role": "user", "content": repeated_user_content}
    ]


def call_openai(messages: list[dict], model_id: str) -> str:
    try:
        response = openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            timeout=90.0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR] {str(e)}"


def call_deepinfra(messages: list[dict], model_id: str) -> str:
    try:
        response = deepinfra_client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            timeout=90.0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR] {str(e)}"


def call_gemini(messages: list[dict], model_id: str) -> str:
    try:
        model = genai.GenerativeModel(model_id)
        system_content = ""
        user_content = ""
        
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user":
                user_content = msg["content"]
        
        full_prompt = f"""SYSTEM INSTRUCTIONS:
{system_content}

USER REQUEST:
{user_content}"""
        
        response = model.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(
                temperature=TEMPERATURE,
                max_output_tokens=MAX_TOKENS
            )
        )
        return response.text
    except Exception as e:
        return f"[ERROR] {str(e)}"


def call_model(messages: list[dict], model_name: str) -> str:
    if model_name not in MODELS:
        return f"[ERROR] Unknown model: {model_name}"
    
    config = MODELS[model_name]
    provider = config["provider"]
    model_id = config["model_id"]
    
    if provider == "openai":
        return call_openai(messages, model_id)
    elif provider == "deepinfra":
        return call_deepinfra(messages, model_id)
    elif provider == "gemini":
        return call_gemini(messages, model_id)
    return f"[ERROR] Unknown provider"


def run_single_test(tc, mode: str, sample: int, model_name: str) -> dict:
    if mode == "baseline":
        messages = build_baseline_prompt(tc.policy, tc.query)
    else:
        messages = build_pqr_prompt(tc.policy, tc.query)
    
    response = call_model(messages, model_name)
    is_violation, reason = tc.violation_check(response)
    
    return {
        "model": model_name,
        "test_id": tc.id,
        "name": tc.name,
        "category": tc.category,
        "mode": mode,
        "sample": sample,
        "is_violation": is_violation,
        "reason": reason,
        "response": response[:300] if response else ""
    }


def run_all_models(models: list[str] = None, samples_per_case: int = 5):
    if models is None:
        models = ACTIVE_MODELS
    
    test_cases = ALL_OPTIMIZED_CASES
    modes = ["baseline", "pqr"]
    
    # Create all tasks
    tasks = []
    for model in models:
        for tc in test_cases:
            for mode in modes:
                for s in range(samples_per_case):
                    tasks.append((tc, mode, s, model))
    
    total_tasks = len(tasks)
    
    print("=" * 80)
    print("MULTI-MODEL OPTIMIZED PQR EXPERIMENT")
    print("=" * 80)
    print(f"Models: {models}")
    print(f"Test cases: {len(test_cases)}")
    print(f"Samples per case: {samples_per_case}")
    print(f"Total API calls: {total_tasks}")
    print(f"Concurrent workers: {MAX_WORKERS}")
    print("=" * 80)
    
    all_results = []
    completed = 0
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(run_single_test, tc, mode, s, model): (tc, mode, s, model)
            for tc, mode, s, model in tasks
        }
        
        for future in as_completed(future_to_task):
            try:
                result = future.result()
                all_results.append(result)
                completed += 1
                
                if completed % 20 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (total_tasks - completed) / rate if rate > 0 else 0
                    print(f"Progress: {completed}/{total_tasks} ({100*completed/total_tasks:.1f}%) - ETA: {eta:.0f}s", flush=True)
            except Exception as e:
                print(f"Task error: {e}")
    
    elapsed = time.time() - start_time
    
    # Analyze results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total time: {elapsed:.1f}s\n")
    
    summary = {}
    
    for model in models:
        mdata = [r for r in all_results if r["model"] == model]
        baseline = [r for r in mdata if r["mode"] == "baseline"]
        pqr = [r for r in mdata if r["mode"] == "pqr"]
        
        b_viols = sum(1 for r in baseline if r["is_violation"])
        p_viols = sum(1 for r in pqr if r["is_violation"])
        
        b_rate = b_viols / len(baseline) * 100 if baseline else 0
        p_rate = p_viols / len(pqr) * 100 if pqr else 0
        improvement = ((b_viols - p_viols) / b_viols * 100) if b_viols > 0 else 0
        
        summary[model] = {
            "baseline_violations": b_viols,
            "baseline_total": len(baseline),
            "baseline_rate": b_rate,
            "pqr_violations": p_viols,
            "pqr_total": len(pqr),
            "pqr_rate": p_rate,
            "improvement_pct": improvement
        }
        
        arrow = "↑" if improvement > 0 else "↓" if improvement < 0 else "="
        status = "✓" if improvement >= 30 else "✗"
        print(f"{status} {model}:")
        print(f"    Baseline: {b_viols}/{len(baseline)} ({b_rate:.1f}%)")
        print(f"    PQR:      {p_viols}/{len(pqr)} ({p_rate:.1f}%)")
        print(f"    {arrow} Improvement: {improvement:.1f}%")
        print()
    
    # Overall
    total_b = sum(s["baseline_violations"] for s in summary.values())
    total_bt = sum(s["baseline_total"] for s in summary.values())
    total_p = sum(s["pqr_violations"] for s in summary.values())
    total_pt = sum(s["pqr_total"] for s in summary.values())
    overall_imp = ((total_b - total_p) / total_b * 100) if total_b > 0 else 0
    
    print("-" * 40)
    print(f"OVERALL: {total_b}/{total_bt} -> {total_p}/{total_pt} ({overall_imp:.1f}% improvement)")
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_file = os.path.join(RESULTS_DIR, f"all_models_optimized_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump({
            "results": all_results,
            "summary": summary,
            "overall_improvement": overall_imp,
            "timestamp": timestamp
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return all_results, summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()
    
    run_all_models(models=args.models, samples_per_case=args.samples)
