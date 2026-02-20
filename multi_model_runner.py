"""
Multi-model experiment runner for PQR.
Supports OpenAI, DeepInfra (Llama/Qwen), and Gemini models.
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
    MODELS, ACTIVE_MODELS, TEMPERATURE, MAX_TOKENS, RESULTS_DIR, DEFAULT_SAMPLES_PER_CASE
)
from policies import POLICY_LEVELS
from test_cases import TEST_CASES
from hard_test_cases import ALL_HARD_CASES


# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
deepinfra_client = OpenAI(api_key=DEEPINFRA_API_KEY, base_url=DEEPINFRA_BASE_URL)

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)


def build_baseline_prompt(policy: str, query: str) -> list[dict]:
    """Build baseline P+Q prompt."""
    return [
        {"role": "system", "content": policy},
        {"role": "user", "content": query}
    ]


def build_pqr_prompt(policy: str, query: str) -> list[dict]:
    """Build PQR P+Q+P+Q prompt."""
    repeated_user_content = f"""{query}

---POLICY REMINDER (RE-READ BEFORE RESPONDING)---
{policy}
---END POLICY REMINDER---

QUERY (RESPOND TO THIS):
{query}"""
    
    return [
        {"role": "system", "content": policy},
        {"role": "user", "content": repeated_user_content}
    ]


def call_openai(messages: list[dict], model_id: str) -> str:
    """Call OpenAI API."""
    try:
        # Handle o3-mini specially (reasoning model)
        if "o3" in model_id:
            response = openai_client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_completion_tokens=MAX_TOKENS
            )
        else:
            response = openai_client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR] {str(e)}"


def call_deepinfra(messages: list[dict], model_id: str) -> str:
    """Call DeepInfra API (OpenAI-compatible)."""
    try:
        response = deepinfra_client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            timeout=60.0  # 60 second timeout
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR] {str(e)}"


def call_gemini(messages: list[dict], model_id: str) -> str:
    """Call Google Gemini API."""
    try:
        model = genai.GenerativeModel(model_id)
        
        # Convert chat format to Gemini format
        # Gemini doesn't have system role, so we prepend it to first user message
        system_content = ""
        user_content = ""
        
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user":
                user_content = msg["content"]
        
        # Combine system and user for Gemini
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


def run_multi_model_experiment(
    models: list[str] = None,
    test_cases: list = None,
    samples_per_case: int = DEFAULT_SAMPLES_PER_CASE,
    modes: list[str] = None
):
    """
    Run experiment across multiple models.
    """
    if models is None:
        models = ACTIVE_MODELS
    if test_cases is None:
        test_cases = TEST_CASES[:10]  # Use subset for multi-model
    if modes is None:
        modes = ["baseline", "pqr"]
    
    all_results = []
    total_calls = len(models) * len(test_cases) * len(modes) * samples_per_case
    current = 0
    
    print("=" * 80)
    print("MULTI-MODEL PQR EXPERIMENT")
    print("=" * 80)
    print(f"Models: {models}")
    print(f"Test cases: {len(test_cases)}")
    print(f"Samples per case: {samples_per_case}")
    print(f"Total API calls: {total_calls}")
    print("=" * 80)
    
    start_time = time.time()
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name} - {MODELS[model_name]['description']}")
        print(f"{'='*60}")
        
        model_results = {"model": model_name, "baseline": [], "pqr": []}
        
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
                        "response": response[:300] if response else ""
                    }
                    all_results.append(result)
                    
                    if is_violation:
                        violations += 1
                    
                    # Rate limiting
                    time.sleep(0.5)
                
                status = "❌" if violations > samples_per_case // 2 else "✓"
                print(f"  {status} [{tc.id}] {mode}: {violations}/{samples_per_case} violations")
    
    elapsed = time.time() - start_time
    
    # Generate summary
    print("\n" + "=" * 80)
    print("CROSS-MODEL RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total time: {elapsed:.1f}s\n")
    
    # Per-model summary
    summary = {}
    for model in models:
        model_data = [r for r in all_results if r["model"] == model]
        baseline = [r for r in model_data if r["mode"] == "baseline"]
        pqr = [r for r in model_data if r["mode"] == "pqr"]
        
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
        print(f"{model}:")
        print(f"  Baseline: {b_viols}/{len(baseline)} violations ({b_rate:.1f}%)")
        print(f"  PQR:      {p_viols}/{len(pqr)} violations ({p_rate:.1f}%)")
        print(f"  {arrow} Improvement: {improvement:.1f}%")
        print()
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_file = os.path.join(RESULTS_DIR, f"multimodel_results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump({
            "results": all_results,
            "summary": summary,
            "models": models,
            "timestamp": timestamp
        }, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    
    return all_results, summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run multi-model PQR experiment")
    parser.add_argument("--models", nargs="+", default=None, help="Models to test")
    parser.add_argument("--samples", type=int, default=3, help="Samples per test case")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer cases")
    
    args = parser.parse_args()
    
    test_cases = TEST_CASES[:5] if args.quick else TEST_CASES[:10]
    
    run_multi_model_experiment(
        models=args.models,
        test_cases=test_cases,
        samples_per_case=args.samples
    )
