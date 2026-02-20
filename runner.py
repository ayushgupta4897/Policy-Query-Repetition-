"""
Main experiment runner for PQR (Policy-Query Repetition) experiments.
Compares baseline P+Q prompting with PQR P+Q+P+Q prompting.
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Optional

from openai import OpenAI

from config import OPENAI_API_KEY, MODEL, TEMPERATURE, MAX_TOKENS, RESULTS_DIR, DEFAULT_SAMPLES_PER_CASE
from policies import POLICY_LEVELS
from test_cases import ALL_TEST_CASES, TEST_CASES
from evaluator import evaluate_response, summarize_results, EvaluationResult


# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


def build_baseline_prompt(policy: str, query: str) -> list[dict]:
    """
    Build baseline P+Q prompt.
    
    Structure: [System: Policy] + [User: Query]
    """
    return [
        {"role": "system", "content": policy},
        {"role": "user", "content": query}
    ]


def build_pqr_prompt(policy: str, query: str) -> list[dict]:
    """
    Build PQR P+Q+P+Q prompt.
    
    Structure: [System: Policy] + [User: Query] + [System: Policy Again] + [User: Query Again]
    
    Note: For chat models, we simulate this by including policy in user messages
    since multiple system messages aren't standard. We structure it as:
    [System: Policy] + [User: Query + "---POLICY REMINDER---" + Policy + "---END REMINDER---" + Query]
    """
    # Method 1: Inline repetition in user message
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


def call_model(messages: list[dict], model: str = MODEL) -> str:
    """Call the OpenAI model and return the response."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR] {str(e)}"


def run_single_test(test_case, mode: str, verbose: bool = False) -> EvaluationResult:
    """
    Run a single test case in the specified mode.
    
    Args:
        test_case: The test case to run
        mode: Either "baseline" or "pqr"
        verbose: If True, print progress
    
    Returns:
        EvaluationResult
    """
    # Get the policy for this test case
    policy = POLICY_LEVELS.get(test_case.policy_level, POLICY_LEVELS["MEGA"])
    
    # Build the prompt based on mode
    if mode == "baseline":
        messages = build_baseline_prompt(policy, test_case.query)
    else:  # pqr
        messages = build_pqr_prompt(policy, test_case.query)
    
    if verbose:
        print(f"  [{mode.upper()}] Running: {test_case.id} - {test_case.name}")
    
    # Call the model
    response = call_model(messages)
    
    # Evaluate the response
    result = evaluate_response(test_case, response, mode)
    
    if verbose:
        status = "❌ VIOLATION" if result.is_violation else "✓ PASS"
        print(f"    {status}: {result.violation_reason}")
    
    return result


def run_experiment(
    test_cases: list = None,
    modes: list[str] = None,
    samples_per_case: int = DEFAULT_SAMPLES_PER_CASE,
    verbose: bool = True,
    save_results: bool = True
) -> list[EvaluationResult]:
    """
    Run the full experiment.
    
    Args:
        test_cases: List of test cases to run (defaults to all)
        modes: List of modes to test (defaults to ["baseline", "pqr"])
        samples_per_case: Number of times to run each test case
        verbose: Print progress
        save_results: Save results to file
    
    Returns:
        List of all evaluation results
    """
    if test_cases is None:
        test_cases = TEST_CASES  # Use core test cases by default
    if modes is None:
        modes = ["baseline", "pqr"]
    
    all_results = []
    total_tests = len(test_cases) * len(modes) * samples_per_case
    current = 0
    
    print(f"\n{'='*60}")
    print(f"PQR EXPERIMENT")
    print(f"{'='*60}")
    print(f"Test cases: {len(test_cases)}")
    print(f"Modes: {modes}")
    print(f"Samples per case: {samples_per_case}")
    print(f"Total API calls: {total_tests}")
    print(f"Model: {MODEL}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    for test_case in test_cases:
        print(f"\n[{test_case.category.upper()}] {test_case.name}")
        print(f"  Query: {test_case.query[:80]}...")
        
        for mode in modes:
            for sample in range(samples_per_case):
                current += 1
                if verbose:
                    print(f"  [{current}/{total_tests}] Mode: {mode}, Sample: {sample + 1}")
                
                result = run_single_test(test_case, mode, verbose=False)
                all_results.append(result)
                
                status = "❌" if result.is_violation else "✓"
                print(f"    {status} {mode}: {result.violation_reason[:50]}")
                
                # Small delay to avoid rate limits
                time.sleep(0.5)
    
    elapsed = time.time() - start_time
    
    # Generate summary
    summary = summarize_results(all_results)
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"\nBaseline (P+Q):")
    print(f"  Tests: {summary['overall']['baseline']['count']}")
    print(f"  Violations: {summary['overall']['baseline']['violations']}")
    print(f"  Pass Rate: {summary['overall']['baseline']['pass_rate']*100:.1f}%")
    print(f"\nPQR (P+Q+P+Q):")
    print(f"  Tests: {summary['overall']['pqr']['count']}")
    print(f"  Violations: {summary['overall']['pqr']['violations']}")
    print(f"  Pass Rate: {summary['overall']['pqr']['pass_rate']*100:.1f}%")
    if summary['overall']['improvement_percent'] is not None:
        print(f"\nImprovement: {summary['overall']['improvement_percent']:.1f}% reduction in violations")
    print(f"{'='*60}\n")
    
    # Save results
    if save_results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = os.path.join(RESULTS_DIR, f"results_{timestamp}.json")
        with open(results_file, "w") as f:
            json.dump([{
                "test_case_id": r.test_case_id,
                "test_case_name": r.test_case_name,
                "category": r.category,
                "policy_level": r.policy_level,
                "prompt_mode": r.prompt_mode,
                "response": r.response,
                "is_violation": r.is_violation,
                "violation_reason": r.violation_reason,
                "response_length": r.response_length,
                "is_valid_json": r.is_valid_json
            } for r in all_results], f, indent=2)
        print(f"Detailed results saved to: {results_file}")
        
        # Save summary
        summary_file = os.path.join(RESULTS_DIR, f"summary_{timestamp}.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run PQR experiment")
    parser.add_argument("--mode", choices=["baseline", "pqr", "both"], default="both",
                        help="Which mode to test")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES_PER_CASE,
                        help="Samples per test case")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with core cases only")
    parser.add_argument("--full", action="store_true",
                        help="Full test with all cases including volume tests")
    parser.add_argument("--category", type=str, default=None,
                        help="Only run tests from specific category")
    
    args = parser.parse_args()
    
    # Determine modes
    if args.mode == "both":
        modes = ["baseline", "pqr"]
    else:
        modes = [args.mode]
    
    # Determine test cases
    if args.full:
        from test_cases import ALL_TEST_CASES
        test_cases = ALL_TEST_CASES
    elif args.quick:
        test_cases = TEST_CASES[:5]  # Just first 5 for quick test
    else:
        test_cases = TEST_CASES
    
    # Filter by category if specified
    if args.category:
        test_cases = [tc for tc in test_cases if tc.category == args.category]
        if not test_cases:
            print(f"No test cases found for category: {args.category}")
            return
    
    # Run experiment
    run_experiment(
        test_cases=test_cases,
        modes=modes,
        samples_per_case=args.samples,
        verbose=True,
        save_results=True
    )


if __name__ == "__main__":
    main()
