#!/usr/bin/env python3
"""
Dedicated runner for hard test cases across all models.
Designed to iterate until significant improvement is achieved.
"""

import json
import os
import time
from datetime import datetime
from typing import Optional
import google.generativeai as genai
from openai import OpenAI

from config import (
    OPENAI_API_KEY, DEEPINFRA_API_KEY, DEEPINFRA_BASE_URL, GEMINI_API_KEY,
    MODELS, ACTIVE_MODELS, TEMPERATURE, MAX_TOKENS, RESULTS_DIR
)
from policies import POLICY_LEVELS
from hard_test_cases import ALL_HARD_CASES


# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
deepinfra_client = OpenAI(api_key=DEEPINFRA_API_KEY, base_url=DEEPINFRA_BASE_URL)
genai.configure(api_key=GEMINI_API_KEY)


def build_baseline_prompt(policy: str, query: str) -> list[dict]:
    """Build baseline P+Q prompt."""
    return [
        {"role": "system", "content": policy},
        {"role": "user", "content": query}
    ]


def build_pqr_prompt(policy: str, query: str) -> list[dict]:
    """Build PQR P+Q+P+Q prompt with stronger emphasis."""
    repeated_user_content = f"""{query}

---POLICY REMINDER (RE-READ CAREFULLY BEFORE RESPONDING)---
{policy}
---END POLICY REMINDER---

NOW RESPOND TO THIS QUERY (FOLLOW ALL POLICIES STRICTLY):
{query}"""
    
    return [
        {"role": "system", "content": policy},
        {"role": "user", "content": repeated_user_content}
    ]


def call_openai(messages: list[dict], model_id: str, retries: int = 3) -> str:
    """Call OpenAI API with retry logic."""
    for attempt in range(retries):
        try:
            if "o3" in model_id:
                response = openai_client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_completion_tokens=MAX_TOKENS,
                    timeout=120.0
                )
            else:
                response = openai_client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    timeout=120.0
                )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < retries - 1:
                wait_time = (2 ** attempt) * 2  # 2, 4, 8 seconds
                print(f"    [Retry {attempt+1}/{retries}] Error: {str(e)[:50]}... waiting {wait_time}s")
                time.sleep(wait_time)
            else:
                return f"[ERROR] {str(e)}"


def call_deepinfra(messages: list[dict], model_id: str, retries: int = 3) -> str:
    """Call DeepInfra API with retry logic."""
    for attempt in range(retries):
        try:
            response = deepinfra_client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                timeout=120.0
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < retries - 1:
                wait_time = (2 ** attempt) * 2
                print(f"    [Retry {attempt+1}/{retries}] DeepInfra error... waiting {wait_time}s")
                time.sleep(wait_time)
            else:
                return f"[ERROR] {str(e)}"


def call_gemini(messages: list[dict], model_id: str, retries: int = 3) -> str:
    """Call Google Gemini API with retry logic."""
    for attempt in range(retries):
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
            if attempt < retries - 1:
                wait_time = (2 ** attempt) * 2
                print(f"    [Retry {attempt+1}/{retries}] Gemini error... waiting {wait_time}s")
                time.sleep(wait_time)
            else:
                return f"[ERROR] {str(e)}"


def call_model(messages: list[dict], model_name: str) -> str:
    """Universal model caller."""
    if model_name not in MODELS:
        return f"[ERROR] Unknown model: {model_name}"
    
    model_config = MODELS[model_name]
    provider = model_config["provider"]
    model_id = model_config["model_id"]
    
    if provider == "openai":
        return call_openai(messages, model_id)
    elif provider == "deepinfra":
        return call_deepinfra(messages, model_id)
    elif provider == "gemini":
        return call_gemini(messages, model_id)
    else:
        return f"[ERROR] Unknown provider: {provider}"


def run_hard_experiment(
    models: list[str] = None,
    samples_per_case: int = 5,
    modes: list[str] = None
):
    """Run hard test cases with detailed tracking."""
    if models is None:
        models = ACTIVE_MODELS
    if modes is None:
        modes = ["baseline", "pqr"]
    
    test_cases = ALL_HARD_CASES
    all_results = []
    
    total_calls = len(models) * len(test_cases) * len(modes) * samples_per_case
    current = 0
    
    print("=" * 80)
    print("HARD TEST CASES - MULTI-MODEL PQR EXPERIMENT")
    print("=" * 80)
    print(f"Models: {models}")
    print(f"Test cases: {len(test_cases)} (Format: 10, Brevity: 5, Disclosure: 10)")
    print(f"Samples per case: {samples_per_case}")
    print(f"Total API calls: {total_calls}")
    print("=" * 80)
    
    start_time = time.time()
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name} - {MODELS[model_name]['description']}")
        print(f"{'='*60}")
        
        for tc in test_cases:
            policy = POLICY_LEVELS.get(tc.policy_level, POLICY_LEVELS["MEGA"])
            
            for mode in modes:
                violations = 0
                for s in range(samples_per_case):
                    current += 1
                    
                    if mode == "baseline":
                        messages = build_baseline_prompt(policy, tc.query)
                    else:
                        messages = build_pqr_prompt(policy, tc.query)
                    
                    response = call_model(messages, model_name)
                    is_violation, reason = tc.violation_check(response)
                    
                    result = {
                        "model": model_name,
                        "test_id": tc.id,
                        "name": tc.name,
                        "category": tc.category,
                        "mode": mode,
                        "sample": s,
                        "is_violation": is_violation,
                        "reason": reason,
                        "response": response[:500] if response else ""
                    }
                    all_results.append(result)
                    
                    if is_violation:
                        violations += 1
                    
                    # Progress
                    if current % 10 == 0:
                        print(f"  Progress: {current}/{total_calls} ({100*current/total_calls:.1f}%)")
                    
                    # Rate limiting
                    time.sleep(0.3)
                
                status = "❌" if violations > samples_per_case // 2 else "✓"
                print(f"  {status} [{tc.id}] {tc.name[:30]:30s} | {mode}: {violations}/{samples_per_case}")
    
    elapsed = time.time() - start_time
    
    # Generate comprehensive summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total time: {elapsed:.1f}s\n")
    
    summary = {}
    overall_baseline_violations = 0
    overall_baseline_total = 0
    overall_pqr_violations = 0
    overall_pqr_total = 0
    
    for model in models:
        model_data = [r for r in all_results if r["model"] == model]
        baseline = [r for r in model_data if r["mode"] == "baseline"]
        pqr = [r for r in model_data if r["mode"] == "pqr"]
        
        b_viols = sum(1 for r in baseline if r["is_violation"])
        p_viols = sum(1 for r in pqr if r["is_violation"])
        
        overall_baseline_violations += b_viols
        overall_baseline_total += len(baseline)
        overall_pqr_violations += p_viols
        overall_pqr_total += len(pqr)
        
        b_rate = b_viols / len(baseline) * 100 if baseline else 0
        p_rate = p_viols / len(pqr) * 100 if pqr else 0
        improvement = ((b_viols - p_viols) / b_viols * 100) if b_viols > 0 else 0
        
        # Per-category breakdown
        categories = set(r["category"] for r in model_data)
        cat_breakdown = {}
        for cat in categories:
            cat_baseline = [r for r in baseline if r["category"] == cat]
            cat_pqr = [r for r in pqr if r["category"] == cat]
            cb_viols = sum(1 for r in cat_baseline if r["is_violation"])
            cp_viols = sum(1 for r in cat_pqr if r["is_violation"])
            cat_improvement = ((cb_viols - cp_viols) / cb_viols * 100) if cb_viols > 0 else 0
            cat_breakdown[cat] = {
                "baseline_violations": cb_viols,
                "baseline_total": len(cat_baseline),
                "pqr_violations": cp_viols,
                "pqr_total": len(cat_pqr),
                "improvement_pct": cat_improvement
            }
        
        summary[model] = {
            "baseline_violations": b_viols,
            "baseline_total": len(baseline),
            "baseline_rate": b_rate,
            "pqr_violations": p_viols,
            "pqr_total": len(pqr),
            "pqr_rate": p_rate,
            "improvement_pct": improvement,
            "by_category": cat_breakdown
        }
        
        arrow = "↑" if improvement > 0 else "↓" if improvement < 0 else "="
        print(f"\n{model}:")
        print(f"  Baseline: {b_viols}/{len(baseline)} violations ({b_rate:.1f}%)")
        print(f"  PQR:      {p_viols}/{len(pqr)} violations ({p_rate:.1f}%)")
        print(f"  {arrow} Improvement: {improvement:.1f}%")
        
        for cat, stats in cat_breakdown.items():
            cat_arrow = "↑" if stats["improvement_pct"] > 0 else "↓" if stats["improvement_pct"] < 0 else "="
            print(f"    {cat}: {stats['baseline_violations']}/{stats['baseline_total']} → {stats['pqr_violations']}/{stats['pqr_total']} ({cat_arrow}{stats['improvement_pct']:.0f}%)")
    
    # Overall summary
    overall_b_rate = overall_baseline_violations / overall_baseline_total * 100
    overall_p_rate = overall_pqr_violations / overall_pqr_total * 100
    overall_improvement = ((overall_baseline_violations - overall_pqr_violations) / overall_baseline_violations * 100) if overall_baseline_violations > 0 else 0
    
    print("\n" + "-" * 40)
    print("OVERALL ACROSS ALL MODELS:")
    print(f"  Baseline: {overall_baseline_violations}/{overall_baseline_total} violations ({overall_b_rate:.1f}%)")
    print(f"  PQR:      {overall_pqr_violations}/{overall_pqr_total} violations ({overall_p_rate:.1f}%)")
    print(f"  OVERALL IMPROVEMENT: {overall_improvement:.1f}%")
    print("-" * 40)
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_file = os.path.join(RESULTS_DIR, f"hard_experiment_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump({
            "results": all_results,
            "summary": summary,
            "overall": {
                "baseline_violations": overall_baseline_violations,
                "baseline_total": overall_baseline_total,
                "pqr_violations": overall_pqr_violations,
                "pqr_total": overall_pqr_total,
                "improvement_pct": overall_improvement
            },
            "config": {
                "models": models,
                "samples_per_case": samples_per_case,
                "test_cases": len(test_cases)
            },
            "timestamp": timestamp
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return all_results, summary, overall_improvement


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run hard test cases PQR experiment")
    parser.add_argument("--models", nargs="+", default=None, help="Models to test")
    parser.add_argument("--samples", type=int, default=5, help="Samples per test case")
    parser.add_argument("--quick", action="store_true", help="Quick test with 2 samples")
    
    args = parser.parse_args()
    
    samples = 2 if args.quick else args.samples
    
    run_hard_experiment(
        models=args.models,
        samples_per_case=samples
    )
