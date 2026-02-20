# PQR Experiment - Policy-Query Repetition

Experimental framework to test whether P+Q+P+Q prompting reduces instruction violations compared to baseline P+Q prompting.

## Quick Start

```bash
# Install dependencies
pip install openai

# Run quick test (5 cases, 1 sample each)
python runner.py --quick --samples 1

# Run full experiment
python runner.py --samples 5

# Analyze results
python analyze.py --latest
```

## Files

- `config.py` - API keys and parameters
- `policies.py` - Multi-level policy definitions
- `test_cases.py` - Conflict queries with violation checks
- `runner.py` - Main experiment runner
- `analyze.py` - Results analysis
- `evaluator.py` - Response evaluation logic

## Hypothesis

Repeating Policy + Query (P+Q+P+Q) before generation forces the model to re-encode the user query after seeing policy constraints again, reducing stochastic precedence failures.
