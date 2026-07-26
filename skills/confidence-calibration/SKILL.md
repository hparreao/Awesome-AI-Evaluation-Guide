---
name: confidence-calibration
description: Audit confidence estimates against observed LLM or agent outcomes. Use when routing by confidence, comparing logprob or ensemble confidence, monitoring calibration drift, or validating uncertainty claims.
---

# Confidence calibration

Collect predictions with confidence, outcome labels, model version, and evaluation-set identity. Bin observations transparently and report sample count, predicted confidence, observed success, and calibration gap for each bin.

Do not turn confidence into an automatic action without a local policy that accounts for costs and error consequences. Recompute after material model, prompt, or data changes. Start from `examples/confidence_scoring/calibration_workflow.py`.
