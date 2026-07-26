---
name: score-robustness
description: Measure accuracy and consistency of an LLM across prompt, sampling, and ordering perturbations using SCORE-style evaluation. Use when assessing reliability beyond a single model response.
---

# SCORE robustness

Choose semantically equivalent prompt variations and sampling settings that reflect the deployment surface. Preserve every response and comparison decision. Report accuracy, all-samples-correct rate, and per-variation failures separately.

Investigate a gap between correctness and consistency as a stability problem. Define any action rule only after measuring the local distribution and operational risk. Start from `examples/consistency_robustness/score_workflow.py`.
