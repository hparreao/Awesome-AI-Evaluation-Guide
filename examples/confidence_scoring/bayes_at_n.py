"""Estimate benchmark performance and finite-trial uncertainty with Bayes@N.

Reference: Hariri, Samandar, Hinczewski, and Chaudhary (ICLR 2026),
"Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation".
https://arxiv.org/abs/2510.04265 (Algorithm 1, uniform-prior case).

Each question has its own categorical distribution and a Dirichlet(1, ..., 1)
prior. Trials are independent within a question; question posteriors are
independent. The benchmark is fixed. No model calls or external packages are used.
"""

from math import fsum, isfinite, sqrt
from typing import Dict, Sequence


def bayes_at_n(
    outcomes: Sequence[Sequence[int]], weights: Sequence[float]
) -> Dict[str, float]:
    """Return the posterior mean and standard deviation of benchmark performance.

    Args:
        outcomes: Nonempty rectangular question-by-trial matrix of category IDs.
        weights: At least two finite category scores, indexed by category ID.

    Returns:
        A dictionary with posterior_mean and posterior_std for the equally
        weighted average of question-level expected scores.

    Raises:
        ValueError: If the matrix is empty/ragged, scores are not finite, or
            categories are not integers between zero and len(weights) - 1.

    Examples:
        >>> result = bayes_at_n([[0, 1]], [0.0, 1.0])
        >>> result["posterior_mean"]
        0.5
    """
    rows = [list(row) for row in outcomes]
    scores = [float(weight) for weight in weights]
    if len(scores) < 2 or not all(isfinite(weight) for weight in scores):
        raise ValueError("Provide at least two finite category scores")
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Provide a nonempty rectangular question-by-trial matrix")
    if any(
        not isinstance(value, int) or not 0 <= value < len(scores)
        for row in rows
        for value in row
    ):
        raise ValueError("Category IDs must be integers indexing the scores")

    question_means = []
    question_variances = []
    for row in rows:
        counts = [1] * len(scores)  # One prior observation in each category.
        for category in row:
            counts[category] += 1
        total = sum(counts)
        probabilities = [count / total for count in counts]
        mean = fsum(prob * score for prob, score in zip(probabilities, scores))
        variance = fsum(
            prob * (score - mean) ** 2 for prob, score in zip(probabilities, scores)
        ) / (total + 1)
        question_means.append(mean)
        question_variances.append(variance)

    questions = len(rows)
    return {
        "posterior_mean": fsum(question_means) / questions,
        "posterior_std": sqrt(fsum(question_variances)) / questions,
    }


if __name__ == "__main__":
    binary = bayes_at_n([[1, 1, 0, 1], [0, 1, 0, 1]], [0.0, 1.0])
    graded = bayes_at_n([[0, 1, 2], [1, 2, 2]], [0.0, 0.5, 1.0])
    for name, result in [("Binary", binary), ("Graded", graded)]:
        print(
            f"{name}: posterior mean={result['posterior_mean']:.6f}, "
            f"posterior std={result['posterior_std']:.6f}"
        )
