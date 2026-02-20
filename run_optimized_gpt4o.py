#!/usr/bin/env python3
"""
Concurrent runner for optimized PQR experiments on GPT-4o.
Uses ThreadPoolExecutor for parallel API calls.
"""

import json
import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from openai import OpenAI

from config import OPENAI_API_KEY, TEMPERATURE, MAX_TOKENS, RESULTS_DIR
from optimized_test_cases import ALL_OPTIMIZED_CASES

# Initialize client
client = OpenAI(api_key=OPENAI_API_KEY)

MODEL = "gpt-4o"
MAX_WORKERS = 10  # Concurrent API calls


def build_baseline_prompt(policy: str, query: str) -> list[dict]:
    """Build baseline P+Q prompt."""
    return [
        {"role": "system", "content": policy},
        {"role": "user", "content": query}
    ]


def build_pqr_prompt(policy: str, query: str) -> list[dict]:
    """Build PQR P+Q+P+Q prompt - policy reminder at the end."""
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


def call_api(messages: list[dict], timeout: float = 60.0) -> str:
    """Call OpenAI API with timeout."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            timeout=timeout
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR] {str(e)}"


def run_single_test(tc, mode: str, sample: int) -> dict:
    """Run a single test and return result."""
    if mode == "baseline":
        messages = build_baseline_prompt(tc.policy, tc.query)
    else:
        messages = build_pqr_prompt(tc.policy, tc.query)
    
    response = call_api(messages)
    is_violation, reason = tc.violation_check(response)
    
    return {
        "test_id": tc.id,
        "name": tc.name,
        "category": tc.category,
        "mode": mode,
        "sample": sample,
        "is_violation": is_violation,
        "reason": reason,
        "response": response[:300] if response else ""
    }


def run_experiment(samples_per_case: int = 5, max_workers: int = MAX_WORKERS):
    """Run concurrent experiment on GPT-4o."""
    test_cases = ALL_OPTIMIZED_CASES
    modes = ["baseline", "pqr"]
    
    # Create all tasks
    tasks = []
    for tc in test_cases:
        for mode in modes:
            for s in range(samples_per_case):
                tasks.append((tc, mode, s))
    
    total_tasks = len(tasks)
    
    print("=" * 70)
    print("OPTIMIZED PQR EXPERIMENT - GPT-4o")
    print("=" * 70)
    print(f"Test cases: {len(test_cases)}")
    print(f"Samples per case: {samples_per_case}")
    print(f"Total API calls: {total_tasks}")
    print(f"Concurrent workers: {max_workers}")
    print("=" * 70)
    
    all_results = []
    completed = 0
    start_time = time.time()
    
    # Run concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(run_single_test, tc, mode, s): (tc, mode, s)
            for tc, mode, s in tasks
        }
        
        for future in as_completed(future_to_task):
            try:
                result = future.result()
                all_results.append(result)
                completed += 1
                
                if completed % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (total_tasks - completed) / rate if rate > 0 else 0
                    print(f"Progress: {completed}/{total_tasks} ({100*completed/total_tasks:.1f}%) - ETA: {eta:.0f}s")
            except Exception as e:
                print(f"Task error: {e}")
    
    elapsed = time.time() - start_time
    
    # Analyze results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total time: {elapsed:.1f}s ({elapsed/total_tasks:.2f}s per call)\n")
    
    baseline = [r for r in all_results if r["mode"] == "baseline"]
    pqr = [r for r in all_results if r["mode"] == "pqr"]
    
    b_viols = sum(1 for r in baseline if r["is_violation"])
    p_viols = sum(1 for r in pqr if r["is_violation"])
    
    b_rate = b_viols / len(baseline) * 100 if baseline else 0
    p_rate = p_viols / len(pqr) * 100 if pqr else 0
    improvement = ((b_viols - p_viols) / b_viols * 100) if b_viols > 0 else 0
    
    print(f"Baseline: {b_viols}/{len(baseline)} violations ({b_rate:.1f}%)")
    print(f"PQR:      {p_viols}/{len(pqr)} violations ({p_rate:.1f}%)")
    print(f"IMPROVEMENT: {improvement:.1f}%")
    
    # Per-category breakdown
    print("\n" + "-" * 40)
    print("PER-CATEGORY BREAKDOWN:")
    categories = set(r["category"] for r in all_results)
    
    category_results = {}
    for cat in sorted(categories):
        cat_baseline = [r for r in baseline if r["category"] == cat]
        cat_pqr = [r for r in pqr if r["category"] == cat]
        cb = sum(1 for r in cat_baseline if r["is_violation"])
        cp = sum(1 for r in cat_pqr if r["is_violation"])
        cat_imp = ((cb - cp) / cb * 100) if cb > 0 else 0
        arrow = "↑" if cat_imp > 0 else "↓" if cat_imp < 0 else "="
        print(f"  {cat}: {cb}/{len(cat_baseline)} -> {cp}/{len(cat_pqr)} ({arrow}{cat_imp:.0f}%)")
        category_results[cat] = {
            "baseline": cb, "baseline_total": len(cat_baseline),
            "pqr": cp, "pqr_total": len(cat_pqr),
            "improvement": cat_imp
        }
    
    # Per-test breakdown
    print("\n" + "-" * 40)
    print("PER-TEST BREAKDOWN:")
    for tc in test_cases:
        tc_baseline = [r for r in baseline if r["test_id"] == tc.id]
        tc_pqr = [r for r in pqr if r["test_id"] == tc.id]
        tb = sum(1 for r in tc_baseline if r["is_violation"])
        tp = sum(1 for r in tc_pqr if r["is_violation"])
        arrow = "↑" if tb > tp else "↓" if tb < tp else "="
        status = "✓" if tp < tb else "✗" if tp > tb else "="
        print(f"  {status} [{tc.id}] {tc.name[:30]:30s} | {tb}/{len(tc_baseline)} -> {tp}/{len(tc_pqr)}")
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_file = os.path.join(RESULTS_DIR, f"optimized_gpt4o_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump({
            "results": all_results,
            "summary": {
                "baseline_violations": b_viols,
                "baseline_total": len(baseline),
                "pqr_violations": p_viols,
                "pqr_total": len(pqr),
                "improvement_pct": improvement
            },
            "by_category": category_results,
            "config": {
                "model": MODEL,
                "samples_per_case": samples_per_case,
                "test_cases": len(test_cases)
            },
            "timestamp": timestamp
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return all_results, improvement


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run optimized PQR experiment on GPT-4o")
    parser.add_argument("--samples", type=int, default=5, help="Samples per test case")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent workers")
    
    args = parser.parse_args()
    
    results, improvement = run_experiment(
        samples_per_case=args.samples,
        max_workers=args.workers
    )
    
    print("\n" + "=" * 70)
    if improvement >= 30:
        print(f"SUCCESS! Achieved {improvement:.1f}% improvement (target: 30%)")
    else:
        print(f"Need more improvement: {improvement:.1f}% (target: 30%)")
    print("=" * 70)
