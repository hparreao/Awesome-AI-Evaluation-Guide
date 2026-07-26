---
name: g-eval-workflow
description: Design evidence-based G-Eval or LLM-as-a-judge workflows for LLM and RAG outputs. Use when defining evaluator steps, critical dimensions, judge inputs, calibration samples, or reviewer-owned decision policies.
---

# G-Eval workflow

Define the claim, evidence context, criterion, and acceptable judge outputs before choosing a model. Keep critical dimensions separate from presentation-quality dimensions. Require the judge to return both a verdict and evidence-linked rationale.

Countercheck a sample against blinded human assessment before operational use. Version the evaluator prompt, model configuration, and rubric with each result. The owning team sets release or escalation policy outside the judge.

Use `examples/llm_as_judge/g_eval_workflow.py` as the provider-agnostic integration shape.
