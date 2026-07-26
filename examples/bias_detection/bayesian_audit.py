"""Summarise group outcomes with Beta posteriors; keep policy outside the module."""
from typing import Dict, Mapping, Sequence


def audit_group_rates(outcomes: Mapping[str, Sequence[int]], prior: Sequence[float] = (1.0, 1.0)) -> Dict[str, object]:
    """Return posterior parameters and means for a Bernoulli outcome per group."""
    if len(prior) != 2 or min(prior) <= 0:
        raise ValueError("prior must contain two positive Beta parameters")
    posterior = {}
    for group, values in outcomes.items():
        if values and all(value in {0, 1} for value in values):
            successes = sum(values)
            alpha, beta = prior[0] + successes, prior[1] + len(values) - successes
            posterior[group] = {"alpha": alpha, "beta": beta, "mean": alpha / (alpha + beta)}
    if len(posterior) < 2:
        raise ValueError("provide at least two non-empty groups")
    means = {group: item["mean"] for group, item in posterior.items()}
    return {"posterior": posterior, "posterior_mean_gap": max(means.values()) - min(means.values()), "sample_sizes": {group: len(values) for group, values in outcomes.items()}}


if __name__ == "__main__":
    print(audit_group_rates({"group_a": [1, 1, 0], "group_b": [1, 0, 0]}))
