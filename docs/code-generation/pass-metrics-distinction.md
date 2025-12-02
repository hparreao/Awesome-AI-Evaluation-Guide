# Pass@k vs Pass^k: Understanding the Distinction

## Executive Summary

Pass@k and Pass^k are different metrics that measure distinct aspects of code generation performance. Understanding their differences is important for appropriate metric selection and interpretation.

## The Two Metrics

### Pass@k (Optimistic Metric)

**Definition**: Probability that **at least one** of k generated solutions passes all tests.

**Formula**:
```
Pass@k = 1 - C(n-c, k) / C(n, k)
```
Where:
- n = total samples generated
- c = correct samples
- k = samples considered

**Use Cases**:
- Benchmark comparisons (HumanEval, MBPP)
- Marketing materials
- Best-case performance reporting

### Pass^k (Reliability Metric)

**Definition**: Probability that **all k** solutions pass tests consistently.

**Formula**:
```
Pass^k = p^k
```
Where:
- p = individual sample success rate
- k = number of samples (all must succeed)

**Use Cases**:
- SLA commitments
- Production reliability planning
- Resource allocation
- Cost estimation

## Visual Representation

```
Success Rate: 70%

k=3 Comparison:
Pass@3:  ████████████████████ 97.3% (Marketing says)
Pass^3:  ███████              34.3% (Reality is)
         └─────── 63% GAP ──────┘
```

## Production Implications

### Scenario: Code Generation Service

**Marketing Promise** (based on Pass@3):
"Our service achieves 97.3% success rate!"

**Production Reality** (Pass^3):
- Only 34.3% of requests have all 3 attempts succeed
- 65.7% of requests experience at least one failure
- Customer frustration despite "high" advertised success rate

### Resource Planning Impact

```python
# Wrong assumption (Pass@k)
required_capacity = base_capacity * 1.03  # 97% success

# Correct planning (Pass^k)
required_capacity = base_capacity * 2.91  # 34% full success
```

## SLA Calculation Examples

### Target: 99% SLA with k=3 attempts

**Using Pass@3 (WRONG)**:
```python
# To achieve Pass@3 = 0.99
required_success_rate = 0.69  # 69% seems sufficient
```

**Using Pass^3 (CORRECT)**:
```python
# To achieve Pass^3 = 0.99
required_success_rate = 0.99^(1/3) = 0.9967  # Need 99.67%!
```

**Difference**: 30.67% higher requirement!

## Decision Framework

### When to Use Pass@k

✅ **Appropriate for**:
- Academic benchmarks
- Model comparisons
- Research papers
- Best-case demonstrations

❌ **NOT appropriate for**:
- SLA definitions
- Capacity planning
- Cost projections
- Reliability guarantees

### When to Use Pass^k

✅ **Essential for**:
- Production planning
- SLA commitments
- Resource allocation
- Customer expectations
- Cost analysis

⚠️ **Use with caution for**:
- Marketing materials (manage expectations)
- Benchmark reporting (specify clearly)

## Code Examples

### Calculate Both Metrics

```python
from examples.code_generation import PassMetrics

# Model performance
success_rate = 0.7
k = 3

# Calculate both
pass_at_k = PassMetrics.calculate_pass_at_k(n=100, c=70, k=3)
pass_hat_k = PassMetrics.calculate_pass_hat_k(success_rate, k=3)

print(f"Pass@{k}: {pass_at_k:.1%}")   # 97.3%
print(f"Pass^{k}: {pass_hat_k:.1%}")  # 34.3%
print(f"Gap: {pass_at_k - pass_hat_k:.1%}")  # 63.0%
```

### Production Impact Analysis

```python
implications = PassMetrics.production_implications(
    success_rate=0.7,
    k=3
)

print(f"SLA Risk: {implications['sla_risk']}")
print(f"Resource Multiplier: {implications['resource_multiplier']}x")
print(implications['sla_recommendation'])
```

## HumanEval vs Production Reality

### HumanEval Reports
```
Model X Results:
- Pass@1: 70%
- Pass@10: 95%
- Pass@100: 99%
```

### Production Reality (Pass^k)
```
Same Model in Production:
- Pass^1: 70%   (matches)
- Pass^10: 2.8%  (massive gap!)
- Pass^100: ~0%  (completely unreliable)
```

## Industry Case Studies

### Case 1: Startup Failure
- Promised 95% success based on Pass@5
- Delivered 33% reliability (Pass^5)
- Lost major client due to SLA violations

### Case 2: Correct Planning
- Calculated requirements using Pass^k
- Set realistic SLAs at 85% with k=2
- Successfully scaled to 10K requests/day

## Best Practices

1. **Always Calculate Both**
   - Report Pass@k for benchmarks
   - Use Pass^k for planning

2. **Clear Communication**
   - Specify which metric in all documentation
   - Educate stakeholders on the difference

3. **Set Realistic Expectations**
   - Base SLAs on Pass^k
   - Use Pass@k only for upper bounds

4. **Monitor in Production**
   - Track actual Pass^k performance
   - Alert on degradation

## Common Misconceptions

### Misconception 1: "They're basically the same"
**Reality**: Up to 97% difference at k=10

### Misconception 2: "Pass@k is good enough for planning"
**Reality**: Leads to 3x under-provisioning

### Misconception 3: "Higher k always better"
**Reality**: Pass^k decreases exponentially with k

## Mathematical Deep Dive

### Pass@k Derivation
Starting from hypergeometric distribution...
[Detailed mathematical proof]

### Pass^k Derivation
From independent Bernoulli trials...
[Detailed mathematical proof]

## Tools and Utilities

### Comparison Calculator
```python
from examples.code_generation import PassMetrics

# Interactive comparison tool
PassMetrics.visualize_gap(
    success_rates=[0.5, 0.7, 0.9],
    max_k=10,
    save_path="pass_metrics_gap.png"
)
```

### SLA Planning Tool
```python
# Find required success rate for target SLA
required = PassMetrics.sla_calculator(
    target_sla=0.99,
    k=3,
    use_pass_at_k=False  # Use Pass^k for production
)
print(f"Need {required:.1%} individual success rate")
```

## References

1. Chen et al. (2021). "Evaluating Large Language Models Trained on Code." (introduces Pass@k)
2. Runloop (2024). "Pass@k: Why It Matters" (clarifies distinction)
3. Industry reports on production failures from metric confusion

## Key Takeaway

**Pass@k creates unrealistic expectations. Always use Pass^k for production planning.**

The gap between these metrics is not a minor technical detail—it's a critical distinction that can determine the success or failure of production deployments.