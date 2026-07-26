---
name: bayesian-bias-audit
description: Audit group-level outcome differences and uncertainty in AI evaluations. Use when testing fairness hypotheses, comparing demographic or operational cohorts, or deciding whether a dataset needs deeper statistical review.
---

# Bayesian bias audit

Define comparison groups, outcome, inclusion criteria, and confounders before running the analysis. Report counts and group rates with uncertainty; small samples should widen conclusions rather than create certainty.

Treat an observed disparity as an audit finding. Investigate data quality, task validity, and causal explanations before making a normative conclusion. Use `examples/bias_detection/bayesian_audit.py` and apply certification or remediation policy externally.
