"""
Evaluator module for analyzing LLM responses against policy requirements.
"""

import json
from dataclasses import dataclass
from typing import Optional
from test_cases import TestCase


@dataclass
class EvaluationResult:
    """Result of evaluating a single response."""
    test_case_id: str
    test_case_name: str
    category: str
    policy_level: str
    prompt_mode: str  # "baseline" (P+Q) or "pqr" (P+Q+P+Q)
    response: str
    is_violation: bool
    violation_reason: str
    response_length: int
    is_valid_json: bool


def evaluate_response(test_case: TestCase, response: str, prompt_mode: str) -> EvaluationResult:
    """
    Evaluate a model response against the test case's violation check.
    
    Args:
        test_case: The test case that was run
        response: The model's response
        prompt_mode: Either "baseline" or "pqr"
    
    Returns:
        EvaluationResult with all evaluation details
    """
    # Run the violation check
    is_violation, reason = test_case.violation_check(response)
    
    # Check if response is valid JSON (independent check)
    is_valid_json = False
    try:
        json.loads(response.strip())
        is_valid_json = True
    except:
        pass
    
    return EvaluationResult(
        test_case_id=test_case.id,
        test_case_name=test_case.name,
        category=test_case.category,
        policy_level=test_case.policy_level,
        prompt_mode=prompt_mode,
        response=response,
        is_violation=is_violation,
        violation_reason=reason,
        response_length=len(response),
        is_valid_json=is_valid_json
    )


def summarize_results(results: list[EvaluationResult]) -> dict:
    """
    Generate summary statistics from evaluation results.
    
    Returns:
        Dictionary with summary statistics
    """
    if not results:
        return {"error": "No results to summarize"}
    
    # Separate by mode
    baseline_results = [r for r in results if r.prompt_mode == "baseline"]
    pqr_results = [r for r in results if r.prompt_mode == "pqr"]
    
    def calc_stats(result_set: list[EvaluationResult]) -> dict:
        if not result_set:
            return {"count": 0, "violations": 0, "violation_rate": 0.0}
        
        violations = sum(1 for r in result_set if r.is_violation)
        return {
            "count": len(result_set),
            "violations": violations,
            "passes": len(result_set) - violations,
            "violation_rate": violations / len(result_set),
            "pass_rate": (len(result_set) - violations) / len(result_set)
        }
    
    # Overall stats
    baseline_stats = calc_stats(baseline_results)
    pqr_stats = calc_stats(pqr_results)
    
    # Per-category stats
    categories = set(r.category for r in results)
    category_stats = {}
    for cat in categories:
        cat_baseline = [r for r in baseline_results if r.category == cat]
        cat_pqr = [r for r in pqr_results if r.category == cat]
        category_stats[cat] = {
            "baseline": calc_stats(cat_baseline),
            "pqr": calc_stats(cat_pqr)
        }
    
    # Calculate improvement
    improvement = None
    if baseline_stats["violation_rate"] > 0:
        reduction = baseline_stats["violation_rate"] - pqr_stats.get("violation_rate", 0)
        improvement = reduction / baseline_stats["violation_rate"] * 100
    
    return {
        "overall": {
            "baseline": baseline_stats,
            "pqr": pqr_stats,
            "improvement_percent": improvement
        },
        "by_category": category_stats
    }
