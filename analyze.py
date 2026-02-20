"""
Analysis and visualization module for PQR experiment results.
"""

import json
import os
import argparse
from collections import defaultdict
from datetime import datetime

from config import RESULTS_DIR


def load_results(results_file: str) -> list[dict]:
    """Load results from a JSON file."""
    with open(results_file, "r") as f:
        return json.load(f)


def analyze_results(results: list[dict]) -> dict:
    """Perform comprehensive analysis of experiment results."""
    
    # Separate by mode
    baseline = [r for r in results if r["prompt_mode"] == "baseline"]
    pqr = [r for r in results if r["prompt_mode"] == "pqr"]
    
    analysis = {
        "total_samples": len(results),
        "baseline_samples": len(baseline),
        "pqr_samples": len(pqr),
    }
    
    # Overall violation rates
    baseline_violations = sum(1 for r in baseline if r["is_violation"])
    pqr_violations = sum(1 for r in pqr if r["is_violation"])
    
    analysis["baseline"] = {
        "violations": baseline_violations,
        "passes": len(baseline) - baseline_violations,
        "violation_rate": baseline_violations / len(baseline) if baseline else 0,
        "pass_rate": (len(baseline) - baseline_violations) / len(baseline) if baseline else 0
    }
    
    analysis["pqr"] = {
        "violations": pqr_violations,
        "passes": len(pqr) - pqr_violations,
        "violation_rate": pqr_violations / len(pqr) if pqr else 0,
        "pass_rate": (len(pqr) - pqr_violations) / len(pqr) if pqr else 0
    }
    
    # Improvement calculation
    if analysis["baseline"]["violation_rate"] > 0:
        reduction = analysis["baseline"]["violation_rate"] - analysis["pqr"]["violation_rate"]
        analysis["improvement"] = {
            "absolute_reduction": reduction,
            "relative_reduction_percent": (reduction / analysis["baseline"]["violation_rate"]) * 100,
            "passes_gained": analysis["pqr"]["passes"] - analysis["baseline"]["passes"]
        }
    else:
        analysis["improvement"] = {
            "absolute_reduction": 0,
            "relative_reduction_percent": 0,
            "passes_gained": 0
        }
    
    # Per-category analysis
    categories = set(r["category"] for r in results)
    analysis["by_category"] = {}
    
    for cat in categories:
        cat_baseline = [r for r in baseline if r["category"] == cat]
        cat_pqr = [r for r in pqr if r["category"] == cat]
        
        b_violations = sum(1 for r in cat_baseline if r["is_violation"])
        p_violations = sum(1 for r in cat_pqr if r["is_violation"])
        
        analysis["by_category"][cat] = {
            "baseline": {
                "total": len(cat_baseline),
                "violations": b_violations,
                "pass_rate": (len(cat_baseline) - b_violations) / len(cat_baseline) if cat_baseline else 0
            },
            "pqr": {
                "total": len(cat_pqr),
                "violations": p_violations,
                "pass_rate": (len(cat_pqr) - p_violations) / len(cat_pqr) if cat_pqr else 0
            }
        }
    
    # Per-test-case analysis (for identifying specific improvements)
    analysis["by_test_case"] = {}
    test_cases = set(r["test_case_id"] for r in results)
    
    for tc in test_cases:
        tc_baseline = [r for r in baseline if r["test_case_id"] == tc]
        tc_pqr = [r for r in pqr if r["test_case_id"] == tc]
        
        b_violations = sum(1 for r in tc_baseline if r["is_violation"])
        p_violations = sum(1 for r in tc_pqr if r["is_violation"])
        
        analysis["by_test_case"][tc] = {
            "name": tc_baseline[0]["test_case_name"] if tc_baseline else tc_pqr[0]["test_case_name"],
            "baseline_violations": b_violations,
            "baseline_total": len(tc_baseline),
            "pqr_violations": p_violations,
            "pqr_total": len(tc_pqr),
            "improved": p_violations < b_violations,
            "regressed": p_violations > b_violations
        }
    
    return analysis


def generate_report(analysis: dict, output_file: str = None) -> str:
    """Generate a human-readable report from analysis."""
    
    lines = []
    lines.append("=" * 70)
    lines.append("PQR EXPERIMENT ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Overall summary
    lines.append("-" * 70)
    lines.append("OVERALL RESULTS")
    lines.append("-" * 70)
    lines.append(f"Total samples: {analysis['total_samples']}")
    lines.append("")
    
    lines.append("BASELINE (P+Q):")
    lines.append(f"  Samples: {analysis['baseline_samples']}")
    lines.append(f"  Violations: {analysis['baseline']['violations']}")
    lines.append(f"  Passes: {analysis['baseline']['passes']}")
    lines.append(f"  Pass Rate: {analysis['baseline']['pass_rate']*100:.1f}%")
    lines.append("")
    
    lines.append("PQR (P+Q+P+Q):")
    lines.append(f"  Samples: {analysis['pqr_samples']}")
    lines.append(f"  Violations: {analysis['pqr']['violations']}")
    lines.append(f"  Passes: {analysis['pqr']['passes']}")
    lines.append(f"  Pass Rate: {analysis['pqr']['pass_rate']*100:.1f}%")
    lines.append("")
    
    lines.append("IMPROVEMENT:")
    lines.append(f"  Absolute violation rate reduction: {analysis['improvement']['absolute_reduction']*100:.2f}%")
    lines.append(f"  Relative improvement: {analysis['improvement']['relative_reduction_percent']:.1f}%")
    lines.append(f"  Additional passes: {analysis['improvement']['passes_gained']}")
    lines.append("")
    
    # Per-category breakdown
    lines.append("-" * 70)
    lines.append("RESULTS BY CATEGORY")
    lines.append("-" * 70)
    
    for cat, data in sorted(analysis["by_category"].items()):
        b_rate = data["baseline"]["pass_rate"] * 100
        p_rate = data["pqr"]["pass_rate"] * 100
        diff = p_rate - b_rate
        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
        
        lines.append(f"\n{cat.upper()}:")
        lines.append(f"  Baseline pass rate: {b_rate:.1f}% ({data['baseline']['total']} samples)")
        lines.append(f"  PQR pass rate:      {p_rate:.1f}% ({data['pqr']['total']} samples)")
        lines.append(f"  Change: {arrow} {abs(diff):.1f}%")
    
    lines.append("")
    
    # Test cases that improved
    lines.append("-" * 70)
    lines.append("TEST CASES THAT IMPROVED WITH PQR")
    lines.append("-" * 70)
    
    improved = [(tc, data) for tc, data in analysis["by_test_case"].items() if data["improved"]]
    if improved:
        for tc, data in improved:
            lines.append(f"  {tc}: {data['name']}")
            lines.append(f"    Baseline: {data['baseline_violations']}/{data['baseline_total']} violations")
            lines.append(f"    PQR: {data['pqr_violations']}/{data['pqr_total']} violations")
    else:
        lines.append("  No test cases showed improvement")
    
    lines.append("")
    
    # Test cases that regressed (if any)
    lines.append("-" * 70)
    lines.append("TEST CASES THAT REGRESSED WITH PQR")
    lines.append("-" * 70)
    
    regressed = [(tc, data) for tc, data in analysis["by_test_case"].items() if data["regressed"]]
    if regressed:
        for tc, data in regressed:
            lines.append(f"  {tc}: {data['name']}")
            lines.append(f"    Baseline: {data['baseline_violations']}/{data['baseline_total']} violations")
            lines.append(f"    PQR: {data['pqr_violations']}/{data['pqr_total']} violations")
    else:
        lines.append("  No test cases showed regression")
    
    lines.append("")
    lines.append("=" * 70)
    
    report = "\n".join(lines)
    
    if output_file:
        with open(output_file, "w") as f:
            f.write(report)
        print(f"Report saved to: {output_file}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze PQR experiment results")
    parser.add_argument("--results", type=str, help="Path to results JSON file")
    parser.add_argument("--output", type=str, default=RESULTS_DIR, help="Output directory")
    parser.add_argument("--latest", action="store_true", help="Use latest results file")
    
    args = parser.parse_args()
    
    # Find results file
    if args.latest:
        # Find the latest results file
        result_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("results_") and f.endswith(".json")]
        if not result_files:
            print("No results files found in", RESULTS_DIR)
            return
        results_file = os.path.join(RESULTS_DIR, sorted(result_files)[-1])
    elif args.results:
        results_file = args.results
    else:
        print("Specify --results <file> or --latest")
        return
    
    print(f"Analyzing: {results_file}")
    
    # Load and analyze
    results = load_results(results_file)
    analysis = analyze_results(results)
    
    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output, exist_ok=True)
    report_file = os.path.join(args.output, f"report_{timestamp}.txt")
    
    report = generate_report(analysis, report_file)
    print(report)
    
    # Save analysis JSON
    analysis_file = os.path.join(args.output, f"analysis_{timestamp}.json")
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to: {analysis_file}")


if __name__ == "__main__":
    main()
