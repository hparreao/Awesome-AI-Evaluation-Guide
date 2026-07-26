---
name: rag-component-evaluation
description: Evaluate RAG retrieval and generation as distinct components. Use when diagnosing answer failures, measuring retrieved evidence quality, checking citations, or assessing groundedness.
---

# RAG component evaluation

Construct cases with query, relevant document identifiers, retrieved documents, answer claims, and citations. Measure retrieval quality before diagnosing generated text. Measure grounding against the retrieved evidence, not world knowledge unless that is explicitly in scope.

Keep component results separate in reports because a strong generator cannot repair absent evidence. Use `examples/rag_evaluation/component_metrics.py` and let the application owner choose decision policy.
