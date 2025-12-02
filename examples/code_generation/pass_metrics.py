"""
Pass@k vs Pass^k: Critical Distinction for Code Generation Evaluation
=====================================================================

Based on research from Runloop and industry best practices.

CRITICAL INSIGHT: Pass@k and Pass^k measure fundamentally different things:
- Pass@k: Optimistic metric (at least one success) - used for marketing
- Pass^k: Reliability metric (all k succeed) - used for SLAs and production

The gap between these metrics has major implications for production planning,
resource allocation, and customer expectations.

Author: Hugo Parreão
License: CC0 1.0 Universal
"""

import numpy as np
from math import comb
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings


@dataclass
class PassMetricsComparison:
    """
    Container for Pass@k vs Pass^k comparison results.

    Attributes:
        k: Number of samples
        pass_at_k: Probability at least one succeeds
        pass_hat_k: Probability all succeed
        gap: Difference between metrics
        success_rate: Individual sample success rate
    """
    k: int
    pass_at_k: float
    pass_hat_k: float
    gap: float
    success_rate: float

    def __str__(self) -> str:
        return (
            f"k={self.k}:\n"
            f"  Pass@{self.k} (optimistic): {self.pass_at_k:.1%}\n"
            f"  Pass^{self.k} (reliability): {self.pass_hat_k:.1%}\n"
            f"  Gap: {self.gap:.1%}\n"
            f"  Success rate: {self.success_rate:.1%}"
        )


def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate Pass@k (optimistic metric).

    Pass@k = Probability that at least one of k samples passes all tests.

    Formula: Pass@k = 1 - C(n-c, k) / C(n, k)
    where:
        n = total samples generated
        c = number of correct samples
        k = number of samples to consider

    Args:
        n: Total number of samples generated
        c: Number of correct samples
        k: Number of samples to consider

    Returns:
        Pass@k probability (0-1)

    Example:
        >>> # 100 samples, 70 correct, looking at 3
        >>> pass_at_3 = calculate_pass_at_k(100, 70, 3)
        >>> print(f"Pass@3: {pass_at_3:.1%}")  # ~97.3%
    """
    if k > n:
        raise ValueError(f"k ({k}) cannot be larger than n ({n})")
    if c > n:
        raise ValueError(f"c ({c}) cannot be larger than n ({n})")

    # If we need more samples than incorrect ones exist, guaranteed success
    if n - c < k:
        return 1.0

    # Calculate using combination formula
    try:
        probability = 1.0 - (comb(n - c, k) / comb(n, k))
    except (ValueError, ZeroDivisionError):
        warnings.warn(f"Numerical issue with n={n}, c={c}, k={k}")
        return 0.0

    return probability


def calculate_pass_hat_k(success_rate: float, k: int) -> float:
    """
    Calculate Pass^k (reliability metric).

    Pass^k = Probability that ALL k samples pass tests.

    Formula: Pass^k = p^k
    where:
        p = individual sample success rate
        k = number of samples (all must succeed)

    Args:
        success_rate: Probability of individual sample success (0-1)
        k: Number of samples that must ALL succeed

    Returns:
        Pass^k probability (0-1)

    Example:
        >>> # 70% success rate, need all 3 to pass
        >>> pass_hat_3 = calculate_pass_hat_k(0.7, 3)
        >>> print(f"Pass^3: {pass_hat_3:.1%}")  # 34.3%
    """
    if not 0 <= success_rate <= 1:
        raise ValueError(f"Success rate must be between 0 and 1, got {success_rate}")

    return success_rate ** k


def compare_metrics(
    success_rate: float,
    k_values: Optional[List[int]] = None
) -> List[PassMetricsComparison]:
    """
    Compare Pass@k vs Pass^k for given success rate.

    Args:
        success_rate: Individual sample success rate (0-1)
        k_values: List of k values to compare (default: 1-10)

    Returns:
        List of comparison results

    Example:
        >>> comparisons = compare_metrics(0.7, k_values=[1, 3, 5, 10])
        >>> for comp in comparisons:
        ...     print(f"k={comp.k}: Gap = {comp.gap:.1%}")
    """
    if k_values is None:
        k_values = list(range(1, 11))

    # For Pass@k calculation, assume large sample size
    n = 10000
    c = int(n * success_rate)

    results = []
    for k in k_values:
        pass_at_k = calculate_pass_at_k(n, c, k)
        pass_hat_k = calculate_pass_hat_k(success_rate, k)
        gap = pass_at_k - pass_hat_k

        results.append(PassMetricsComparison(
            k=k,
            pass_at_k=pass_at_k,
            pass_hat_k=pass_hat_k,
            gap=gap,
            success_rate=success_rate
        ))

    return results


def production_implications(success_rate: float, k: int) -> Dict[str, any]:
    """
    Calculate production implications of Pass@k vs Pass^k gap.

    Args:
        success_rate: Individual sample success rate
        k: Number of samples

    Returns:
        Dictionary with production metrics and recommendations

    Example:
        >>> implications = production_implications(0.7, 3)
        >>> print(f"SLA risk: {implications['sla_risk']}")
    """
    pass_at_k = calculate_pass_at_k(10000, int(10000 * success_rate), k)
    pass_hat_k = calculate_pass_hat_k(success_rate, k)
    gap = pass_at_k - pass_hat_k

    # Calculate real-world implications
    implications = {
        "pass_at_k": pass_at_k,
        "pass_hat_k": pass_hat_k,
        "gap": gap,

        # SLA implications
        "sla_risk": "HIGH" if gap > 0.5 else "MEDIUM" if gap > 0.3 else "LOW",
        "sla_recommendation": (
            f"If promising {pass_at_k:.1%} success rate based on Pass@{k}, "
            f"only {pass_hat_k:.1%} of requests will have all {k} attempts succeed."
        ),

        # Resource implications
        "resource_multiplier": k,
        "effective_success_rate": pass_hat_k,
        "compute_cost_increase": f"{k}x",

        # Customer experience
        "customer_perception": {
            "marketed_performance": f"{pass_at_k:.1%}",
            "actual_reliability": f"{pass_hat_k:.1%}",
            "disappointment_risk": "HIGH" if gap > 0.5 else "MODERATE" if gap > 0.3 else "LOW"
        },

        # Recommendations
        "recommendations": []
    }

    # Add specific recommendations
    if gap > 0.5:
        implications["recommendations"].append(
            "⚠️ CRITICAL: Use Pass^k for SLA commitments, not Pass@k"
        )
        implications["recommendations"].append(
            "Consider reducing k or improving base success rate"
        )

    if pass_hat_k < 0.5:
        implications["recommendations"].append(
            "Reliability too low for production without fallbacks"
        )

    if k > 5:
        implications["recommendations"].append(
            f"High resource cost ({k}x) - consider optimization"
        )

    return implications


def visualize_gap(
    success_rates: List[float] = None,
    max_k: int = 10,
    save_path: Optional[str] = None
):
    """
    Visualize the gap between Pass@k and Pass^k.

    Args:
        success_rates: List of success rates to compare
        max_k: Maximum k value to plot
        save_path: Optional path to save figure

    Returns:
        matplotlib figure
    """
    if success_rates is None:
        success_rates = [0.3, 0.5, 0.7, 0.9]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Pass@k vs Pass^k: The Critical Gap", fontsize=16, fontweight='bold')

    k_values = list(range(1, max_k + 1))

    for idx, success_rate in enumerate(success_rates):
        ax = axes[idx // 2, idx % 2]

        comparisons = compare_metrics(success_rate, k_values)

        pass_at_k_values = [c.pass_at_k for c in comparisons]
        pass_hat_k_values = [c.pass_hat_k for c in comparisons]
        gap_values = [c.gap for c in comparisons]

        # Plot Pass@k and Pass^k
        ax.plot(k_values, pass_at_k_values, 'g-', marker='o', label='Pass@k (optimistic)', linewidth=2)
        ax.plot(k_values, pass_hat_k_values, 'r-', marker='s', label='Pass^k (reliability)', linewidth=2)

        # Fill the gap
        ax.fill_between(k_values, pass_at_k_values, pass_hat_k_values,
                        alpha=0.3, color='yellow', label='Gap (risk zone)')

        ax.set_xlabel('k (number of attempts)')
        ax.set_ylabel('Probability')
        ax.set_title(f'Success Rate = {success_rate:.0%}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

        # Annotate maximum gap
        max_gap_idx = gap_values.index(max(gap_values))
        ax.annotate(f'Max gap: {max(gap_values):.1%}',
                   xy=(k_values[max_gap_idx], (pass_at_k_values[max_gap_idx] + pass_hat_k_values[max_gap_idx]) / 2),
                   xytext=(k_values[max_gap_idx] + 1, (pass_at_k_values[max_gap_idx] + pass_hat_k_values[max_gap_idx]) / 2),
                   arrowprops=dict(arrowstyle='->', color='red'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def sla_calculator(
    target_sla: float,
    k: int,
    use_pass_at_k: bool = False
) -> float:
    """
    Calculate required success rate to meet SLA target.

    Args:
        target_sla: Target SLA (e.g., 0.99 for 99%)
        k: Number of attempts
        use_pass_at_k: Use Pass@k (True) or Pass^k (False)

    Returns:
        Required individual success rate

    Example:
        >>> # To achieve 99% SLA with k=3 using Pass^k
        >>> required = sla_calculator(0.99, 3, use_pass_at_k=False)
        >>> print(f"Need {required:.1%} individual success rate")
    """
    if use_pass_at_k:
        # For Pass@k, this is more complex - using approximation
        # Pass@k ≈ 1 - (1-p)^k for large n
        required_failure_rate = (1 - target_sla) ** (1/k)
        return 1 - required_failure_rate
    else:
        # For Pass^k: target = p^k, so p = target^(1/k)
        return target_sla ** (1/k)


def humaneval_vs_production(test_results: Dict[str, int]) -> Dict[str, any]:
    """
    Compare HumanEval Pass@k scores with production Pass^k reality.

    Args:
        test_results: Dict with 'total' and 'passed' counts

    Returns:
        Analysis of HumanEval vs production metrics

    Example:
        >>> results = {"total": 164, "passed": 115}  # 70% pass rate
        >>> analysis = humaneval_vs_production(results)
        >>> print(analysis["warning"])
    """
    total = test_results["total"]
    passed = test_results["passed"]
    success_rate = passed / total

    # Calculate various k values
    k_values = [1, 3, 5, 10]
    analysis = {
        "success_rate": success_rate,
        "humaneval_scores": {},
        "production_reality": {},
        "gaps": {}
    }

    for k in k_values:
        pass_at_k = calculate_pass_at_k(total, passed, min(k, total))
        pass_hat_k = calculate_pass_hat_k(success_rate, k)

        analysis["humaneval_scores"][f"Pass@{k}"] = pass_at_k
        analysis["production_reality"][f"Pass^{k}"] = pass_hat_k
        analysis["gaps"][f"k={k}"] = pass_at_k - pass_hat_k

    # Add warnings
    max_gap = max(analysis["gaps"].values())
    if max_gap > 0.5:
        analysis["warning"] = (
            "⚠️ CRITICAL: HumanEval scores are misleading for production! "
            f"Gap up to {max_gap:.1%} between reported and actual reliability."
        )
    else:
        analysis["warning"] = "Gap exists but manageable with proper expectations"

    return analysis


# Example usage and demonstrations
if __name__ == "__main__":
    print("=" * 80)
    print("PASS@K vs PASS^K: THE CRITICAL DISTINCTION")
    print("=" * 80)
    print()

    # Example 1: Basic comparison
    print("Example 1: The Shocking Gap")
    print("-" * 40)

    success_rate = 0.7
    comparisons = compare_metrics(success_rate, k_values=[1, 3, 5, 10])

    print(f"Model with {success_rate:.0%} individual success rate:\n")
    print(f"{'k':<5} {'Pass@k':<12} {'Pass^k':<12} {'Gap':<12} {'Impact'}")
    print("-" * 60)

    for comp in comparisons:
        impact = "🔴 CRITICAL" if comp.gap > 0.5 else "🟡 HIGH" if comp.gap > 0.3 else "🟢 MODERATE"
        print(f"{comp.k:<5} {comp.pass_at_k:<12.1%} {comp.pass_hat_k:<12.1%} {comp.gap:<12.1%} {impact}")

    print()

    # Example 2: Production implications
    print("Example 2: Production Reality Check")
    print("-" * 40)

    implications = production_implications(0.7, 3)

    print(f"Scenario: Code generation with 70% success rate, k=3 attempts\n")
    print(f"Marketing says: {implications['pass_at_k']:.1%} success (Pass@3)")
    print(f"Reality is: {implications['pass_hat_k']:.1%} reliability (Pass^3)")
    print(f"Gap: {implications['gap']:.1%} ⚠️\n")

    print("Production Implications:")
    for rec in implications["recommendations"]:
        print(f"  • {rec}")

    print()

    # Example 3: SLA Planning
    print("Example 3: SLA Planning")
    print("-" * 40)

    target_sla = 0.99
    k = 3

    print(f"Target: {target_sla:.1%} SLA with k={k} attempts\n")

    # Using Pass@k (wrong!)
    required_optimistic = sla_calculator(target_sla, k, use_pass_at_k=True)
    print(f"Using Pass@{k} (WRONG for SLAs):")
    print(f"  Required success rate: {required_optimistic:.1%}")

    # Using Pass^k (correct!)
    required_realistic = sla_calculator(target_sla, k, use_pass_at_k=False)
    print(f"\nUsing Pass^{k} (CORRECT for SLAs):")
    print(f"  Required success rate: {required_realistic:.1%}")

    print(f"\nDifference: {(required_realistic - required_optimistic):.1%} higher requirement!")
    print("This is why using Pass@k for SLAs leads to failures!")

    print()

    # Example 4: HumanEval vs Production
    print("Example 4: HumanEval Scores vs Production Reality")
    print("-" * 40)

    test_results = {"total": 164, "passed": 115}  # ~70% pass rate
    analysis = humaneval_vs_production(test_results)

    print("HumanEval-style reporting:")
    for metric, score in analysis["humaneval_scores"].items():
        print(f"  {metric}: {score:.1%}")

    print("\nProduction reality (Pass^k):")
    for metric, score in analysis["production_reality"].items():
        print(f"  {metric}: {score:.1%}")

    print(f"\n{analysis['warning']}")

    print()

    # Example 5: Decision Framework
    print("Example 5: Decision Framework")
    print("-" * 40)

    print("When to use each metric:\n")

    decisions = [
        ("Pass@k", "Marketing materials", "✅"),
        ("Pass@k", "Benchmark comparisons", "✅"),
        ("Pass@k", "Best-case scenarios", "✅"),
        ("Pass@k", "SLA commitments", "❌"),
        ("Pass@k", "Resource planning", "❌"),
        ("Pass@k", "Reliability guarantees", "❌"),
        ("", "", ""),
        ("Pass^k", "SLA commitments", "✅"),
        ("Pass^k", "Resource planning", "✅"),
        ("Pass^k", "Cost estimation", "✅"),
        ("Pass^k", "Reliability guarantees", "✅"),
        ("Pass^k", "Marketing materials", "⚠️"),
    ]

    print(f"{'Metric':<10} {'Use Case':<25} {'Appropriate'}")
    print("-" * 50)
    for metric, use_case, appropriate in decisions:
        if metric:  # Skip empty rows
            print(f"{metric:<10} {use_case:<25} {appropriate}")
        else:
            print()

    print("\n" + "=" * 80)
    print("KEY TAKEAWAY:")
    print("Pass@k creates unrealistic expectations. Use Pass^k for production planning!")
    print("=" * 80)