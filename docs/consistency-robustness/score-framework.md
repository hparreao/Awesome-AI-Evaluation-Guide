# SCORE Framework: Systematic COnsistency and Robustness Evaluation

## Overview

The SCORE framework, introduced by NVIDIA Research in 2025 (arXiv:2503.00137), evaluates LLM consistency and robustness alongside traditional accuracy metrics.

### Key Finding

Models may show different levels of accuracy and consistency. A model with high accuracy might have lower consistency, which has implications for deployment decisions where predictable behavior is important.

## Mathematical Foundation

### Consistency Rate (CR@K)

The cornerstone metric of SCORE is the Consistency Rate at K:

```
CR@K = 1 if all k responses are correct
       0 otherwise
```

This differs fundamentally from Pass@k:
- **Pass@k**: At least one of k attempts succeeds (optimistic)
- **CR@K**: All k attempts must succeed (reliability)

### Robustness Dimensions

SCORE evaluates four types of robustness:

1. **Prompt Robustness (PR)**
   ```
   PR = Σ(correct_paraphrases) / total_paraphrases
   ```

2. **Sampling Robustness (SR)**
   ```
   SR = Σ(consistent_across_temperatures) / total_temperature_samples
   ```

3. **Order Robustness (OR)**
   ```
   OR = Σ(correct_regardless_of_choice_order) / total_orderings
   ```

4. **Non-greedy Robustness (NR)**
   ```
   NR = consistency_with_temperature>0 / baseline_consistency
   ```

## Implementation

### Basic Usage

```python
from examples.consistency_robustness import SCOREEvaluator

# Initialize evaluator
evaluator = SCOREEvaluator(
    model=your_model,
    k=5,  # Number of samples for CR@K
    temperature_range=(0.0, 1.0),
    temperature_steps=5
)

# Evaluate test cases
test_cases = [
    {
        "prompt": "What is the capital of France?",
        "expected": "Paris",
        "variations": [
            "Tell me the capital of France.",
            "France's capital is?",
            "Which city is the capital of France?"
        ]
    }
]

metrics = evaluator.evaluate(test_cases)
print(metrics)
```

### Output Interpretation

```python
SCORE Metrics:
  Accuracy: 85.0%
  Consistency Rate (CR@K): 62.0%  # ⚠️ 23% gap!
  Prompt Robustness: 71.0%
  Sampling Robustness: 68.0%
  Order Robustness: 83.0%
  Overall SCORE: 73.0%
```

## Production Implications

### When Consistency < Accuracy

This common scenario indicates:

1. **Unstable Behavior**: Model responses vary with minor input changes
2. **Temperature Sensitivity**: Performance degrades with sampling variation
3. **Deployment Risk**: Production behavior may differ from testing

### Policy considerations

SCORE reports accuracy, consistency, and robustness observations; it does not determine deployment readiness. The accountable owner must define any acceptance or escalation policy after reviewing baseline distributions, consequences of error, review capacity, and available mitigations. Keep that policy versioned separately from the evaluator.

## Advanced Features

### Parallel Evaluation

```python
evaluator = SCOREEvaluator(
    model=model,
    parallel_execution=True,
    max_workers=8  # Parallel processing
)
```

### Response Caching

```python
evaluator = SCOREEvaluator(
    model=model,
    cache_responses=True  # Avoid redundant API calls
)
```

### Custom Comparison Logic

```python
class CustomEvaluator(SCOREEvaluator):
    def _compare_responses(self, response: str, expected: str) -> bool:
        # Implement semantic similarity instead of exact match
        return semantic_similarity(response, expected) > 0.9
```

## Case Studies

### Medical Domain

In medical applications, consistency is paramount:

```python
medical_test = {
    "prompt": "What are symptoms of Type 2 diabetes?",
    "expected": "increased thirst, frequent urination, fatigue",
    "variations": [
        "List Type 2 diabetes symptoms",
        "What symptoms indicate Type 2 diabetes?",
        "How does Type 2 diabetes manifest?"
    ]
}

# The system owner supplies policy; the evaluator reports evidence.
policy = load_owner_policy()
if policy.requires_escalation(metrics):
    route_to_owner_review(metrics)
```

### Customer Support

Balance between consistency and flexibility:

```python
support_evaluator = SCOREEvaluator(
    model=model,
    k=3,  # Lower k for cost efficiency
    temperature_range=(0.3, 0.7)  # Allow some creativity
)
```

## Comparison with Other Metrics

| Aspect | Traditional Accuracy | SCORE Framework |
|--------|---------------------|-----------------|
| **Focus** | Correctness | Consistency + Correctness |
| **Reliability** | Not measured | Explicitly measured |
| **Production Ready** | Often misleading | Realistic assessment |
| **Cost** | Low | Medium (multiple samples) |
| **Insight** | Single dimension | Multi-dimensional |

## Best Practices

1. **Always Report Both**: Present accuracy alongside consistency
2. **Domain-Specific K**: Medical (k=10), Financial (k=7), General (k=5)
3. **Monitor Degradation**: Track consistency over time
4. **Combine with Confidence**: Use with ensemble voting for complete picture

## Common Pitfalls

### Pitfall 1: Ignoring the Gap
```python
# Report the dimensions separately and let an external policy decide.
report = {"accuracy": accuracy, "consistency_rate": consistency_rate}
publish_evaluation_report(report)
```

### Pitfall 2: Wrong K Selection
```python
# ❌ Too low for critical applications
medical_evaluator = SCOREEvaluator(k=2)

# ✅ Appropriate for domain
medical_evaluator = SCOREEvaluator(k=10)
```

### Pitfall 3: Single Temperature Testing
```python
# ❌ Missing temperature robustness
evaluator = SCOREEvaluator(temperature_range=(0.0, 0.0))

# ✅ Full temperature spectrum
evaluator = SCOREEvaluator(temperature_range=(0.0, 1.0))
```

## Performance Optimization

### Caching Strategy
```python
# Cache responses for repeated prompts
evaluator = SCOREEvaluator(
    model=model,
    cache_responses=True
)

# Results: 60% reduction in API calls for typical test suites
```

### Batch Processing
```python
# Process multiple test cases in parallel
metrics = evaluator.evaluate(
    test_cases,
    batch_size=10,
    parallel=True
)

# Results: 3x faster evaluation for large test sets
```

## Integration with CI/CD

```yaml
# .github/workflows/score-evaluation.yml
- name: Run SCORE Evaluation
  run: |
    python -m pytest tests/score_tests.py
  env:
    EVALUATION_POLICY_PATH: config/evaluation-policy.json
```

## References

1. NVIDIA Research (2025). "SCORE: Systematic COnsistency and Robustness Evaluation for Large Language Models." arXiv:2503.00137

2. Related frameworks:
   - Pass@k metrics for code generation
   - G-Eval for semantic evaluation
   - Confidence scoring via ensemble methods

## Code Examples

Full implementation: [examples/consistency_robustness/score_framework.py](../../examples/consistency_robustness/score_framework.py)

Test suite: [tests/test_score.py](../../tests/test_score.py)
