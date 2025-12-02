# Awesome AI Evaluation Guide

[![License: CC0-1.0](https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg)](http://creativecommons.org/publicdomain/zero/1.0/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](CONTRIBUTING.md)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## About This Guide

A comprehensive guide for evaluating Large Language Models (LLMs) with practical implementations and clear guidance on metric selection.

### What This Repository Provides

- **Working Code Examples**: Complete implementations of evaluation metrics
- **Mathematical Foundations**: Understanding the theory behind each metric
- **Metric Selection Guidance**: When and why to use each evaluation method
- **Domain-Specific Considerations**: Tailored approaches for different applications
- **Observability Tools**: Integration with monitoring platforms

### Key Concepts from Recent Research

| Concept | Finding | Source | Implication |
|---------|---------|--------|-------------|
| Consistency vs Accuracy | Models can show high accuracy with low consistency | SCORE (NVIDIA 2025) | Evaluate both dimensions |
| Pass@k vs Pass^k | Different metrics measure different aspects | Code generation research | Choose based on use case |
| Confidence Scoring | Ensemble methods show correlation with accuracy | Industry studies | Consider multiple approaches |
| System Evaluation | Component interactions affect overall performance | RAG research | Evaluate holistically |

### Repository Structure

This guide organizes evaluation methods into clear categories with practical implementations for each.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Evaluation Metrics](#evaluation-metrics)
  - [Traditional Metrics](#traditional-metrics)
  - [Modern Metrics](#modern-metrics)
    - [Consistency & Robustness (SCORE)](#consistency--robustness-score)
    - [Probability-Based Metrics](#probability-based-metrics)
    - [LLM-as-a-Judge](#llm-as-a-judge)
  - [Production Metrics](#production-metrics)
    - [Confidence Scoring](#confidence-scoring)
    - [Calibration Methods](#calibration-methods)
  - [Safety & Bias](#safety--bias)
    - [Hallucination Detection](#hallucination-detection)
    - [Bias Detection](#bias-detection)
- [Domain-Specific Evaluation](#domain-specific-evaluation)
  - [RAG Systems](#rag-systems)
  - [Code Generation](#code-generation)
  - [Multi-Agent Systems](#multi-agent-systems)
- [Tools & Frameworks](#tools--frameworks)
- [Benchmarks & Datasets](#benchmarks--datasets)
- [Production Best Practices](#production-best-practices)
- [Contributing](#contributing)
- [Citation](#citation)

---

## Metric Selection Guide

### Quick Decision Table

| Task Type | Primary Metrics | Secondary Metrics | Key Considerations |
|-----------|----------------|-------------------|-------------------|
| **Text Generation** | Perplexity, G-Eval | BLEU, ROUGE | Need reference texts for BLEU/ROUGE |
| **Question Answering** | Answer Correctness, Faithfulness | BERTScore, Exact Match | Domain expertise affects threshold |
| **Code Generation** | Pass@k (benchmarks), Pass^k (reliability) | Syntax validity, Security | Pass@k ≠ Pass^k for planning |
| **RAG Systems** | Faithfulness, Context Relevance | Precision@k, NDCG | Evaluate retrieval and generation separately |
| **Translation** | BLEU, METEOR | BERTScore, Human eval | BLEU has known limitations |
| **Summarization** | ROUGE, Relevance | Coherence, Consistency | ROUGE may miss semantic equivalence |
| **Dialogue** | Coherence, Engagement | Response diversity | Context window important |
| **Multi-Agent** | Task completion, Coordination | Communication efficiency | System-level metrics needed |

### Domain-Specific Thresholds

| Domain | Metric Type | Typical Threshold | Rationale |
|--------|------------|------------------|-----------|
| **Medical** | Faithfulness | > 0.9 | Patient safety critical |
| **Legal** | Factual accuracy | > 0.95 | Regulatory compliance |
| **Financial** | Numerical precision | > 0.98 | Monetary implications |
| **Customer Support** | Response relevance | > 0.7 | User satisfaction |
| **Creative Writing** | Diversity score | > 0.6 | Avoid repetition |
| **Education** | Answer correctness | > 0.85 | Learning outcomes |

## Quick Start

### Installation

```bash
# Clone this repository
git clone https://github.com/hparreao/Awesome-AI-Evaluation-Guide.git
cd Awesome-AI-Evaluation-Guide

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage Example

```python
# Example: Evaluating text generation quality
from examples.llm_as_judge import evaluate_response

result = evaluate_response(
    question="What is the capital of France?",
    response="Paris is the capital of France.",
    criteria="factual_accuracy"
)

print(f"Score: {result.score}")
print(f"Reasoning: {result.reasoning}")
```

---

## Evaluation Metrics

### Traditional Metrics

Foundational metrics from NLP research, adapted for LLM evaluation.

#### Perplexity
**What it measures**: Model uncertainty in predicting the next token. Lower values indicate better performance.

**When to use**:
- Comparing language models on the same task
- Pre-training evaluation
- Domain adaptation assessment

**Implementation**: [examples/traditional_metrics/perplexity.py](examples/traditional_metrics/perplexity.py)

**Documentation**: [docs/traditional-metrics/perplexity.md](docs/traditional-metrics/perplexity.md)

#### BLEU Score
**What it measures**: Precision-based n-gram overlap between generated and reference text.

**When to use**:
- Machine translation evaluation
- Text generation with reference outputs
- Paraphrase quality assessment

**Limitations**:
- Doesn't account for semantic similarity
- Biased toward shorter outputs
- Requires reference text

**Implementation**: [examples/traditional_metrics/bleu_score.py](examples/traditional_metrics/bleu_score.py)

**Documentation**: [docs/traditional-metrics/bleu-score.md](docs/traditional-metrics/bleu-score.md)

#### ROUGE Score
**What it measures**: Recall-oriented n-gram overlap, primarily for summarization.

**When to use**:
- Summarization tasks
- Content coverage assessment
- Information preservation evaluation

**Implementation**: [examples/traditional_metrics/rouge_score.py](examples/traditional_metrics/rouge_score.py)

**Documentation**: [docs/traditional-metrics/rouge-score.md](docs/traditional-metrics/rouge-score.md)

---

### Probability-Based Metrics

Leverage model confidence through token probabilities.

#### Logprobs Analysis
**What it measures**: Log probabilities for each generated token.

**Applications**:
- Hallucination detection (low probability = potential hallucination)
- Confidence estimation
- Classification with uncertainty quantification

**Key Finding**: OpenAI research shows logprobs enable reliable confidence scoring for classification tasks.

**Implementation**: [examples/probability_based/logprobs.py](examples/probability_based/logprobs.py)

**Documentation**: [docs/probability-based/logprobs.md](docs/probability-based/logprobs.md)

#### Top-k Token Analysis
**What it measures**: Distribution of top-k most probable tokens at each position.

**Applications**:
- Diversity assessment
- Uncertainty quantification
- Alternative generation paths exploration

**Implementation**: [examples/probability_based/topk_analysis.py](examples/probability_based/topk_analysis.py)

**Documentation**: [docs/probability-based/topk-analysis.md](docs/probability-based/topk-analysis.md)

---

### LLM-as-a-Judge

Use LLMs to evaluate LLM outputs based on custom criteria.

#### G-Eval Framework
**What it is**: Chain-of-thought (CoT) based evaluation using LLMs with token probability normalization.

**Why it works**:
- Better human alignment than traditional metrics
- Flexible custom criteria
- Token probability weighting reduces bias

**Production Scale**: DeepEval processes 10M+ G-Eval metrics monthly.

**Core Use Cases**:

1. **Answer Correctness** - Validate factual accuracy
2. **Coherence & Clarity** - Assess text quality without references
3. **Tonality & Professionalism** - Domain-appropriate style
4. **Safety & Compliance** - PII detection, bias, toxicity
5. **Domain-Specific Faithfulness** - RAG evaluation with heavy hallucination penalties

**Implementation**: [examples/llm_as_judge/](examples/llm_as_judge/)

**Complete Guide**: [docs/llm-as-judge/g-eval-framework.md](docs/llm-as-judge/g-eval-framework.md)

**Quick Example**:

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

# Define custom evaluation
correctness = GEval(
    name="Correctness",
    evaluation_steps=[
        "Check for factual contradictions",
        "Penalize missing critical information",
        "Accept paraphrasing and style differences"
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    threshold=0.7
)

# Evaluate
test_case = LLMTestCase(
    input="What is Python?",
    actual_output="Python is a high-level programming language.",
    expected_output="Python is an interpreted, high-level programming language."
)

correctness.measure(test_case)
print(f"Score: {correctness.score}")  # 0-1 scale
```

---

## Modern Metrics

### Consistency & Robustness (SCORE)

The SCORE framework (NVIDIA 2025) evaluates model consistency alongside accuracy, providing insights into reliability.

#### Components of SCORE

| Metric | What it Measures | Use Case |
|--------|------------------|----------|
| **Consistency Rate (CR@K)** | If model gives same correct answer K times | Reliability assessment |
| **Prompt Robustness** | Stability across paraphrased prompts | Input variation handling |
| **Sampling Robustness** | Consistency under temperature changes | Deployment configuration |
| **Order Robustness** | Invariance to choice ordering | Multiple-choice tasks |

#### When to Use SCORE

- Evaluating model reliability beyond accuracy
- Testing robustness to input variations
- Assessing deployment readiness
- Comparing model stability

**Implementation**: [examples/consistency_robustness/score_framework.py](examples/consistency_robustness/score_framework.py)

**Documentation**: [docs/consistency-robustness/score-framework.md](docs/consistency-robustness/score-framework.md)

```python
from examples.consistency_robustness import SCOREEvaluator

evaluator = SCOREEvaluator(model=your_model)
metrics = evaluator.evaluate(test_cases)

# Compare accuracy vs consistency
print(f"Accuracy: {metrics.accuracy:.2%}")
print(f"Consistency Rate: {metrics.consistency_rate:.2%}")
```

---

### Confidence Scoring

Ensemble-based methods for reliable confidence estimation.

#### Majority Voting
**What it measures**: Consensus across multiple model generations.

**Key Finding**: Industry studies show **strong positive correlation between majority voting confidence and actual accuracy**, while "no clear correlation was found between logprob-based confidence score and accuracy."

**Optimal Configuration**: 4-7 diverse models (sweet spot for reliability vs. cost)

**Implementation**: [examples/confidence_scoring/majority_voting.py](examples/confidence_scoring/majority_voting.py)

**Documentation**: [docs/confidence-scoring/majority-voting.md](docs/confidence-scoring/majority-voting.md)

#### Weighted Ensemble
**What it does**: Weight model votes by historical accuracy.

**Weighting Strategy**: Linear weights preferred (`w_i = Accuracy_i`) over exponential to maintain ensemble diversity.

**Implementation**: [examples/confidence_scoring/weighted_ensemble.py](examples/confidence_scoring/weighted_ensemble.py)

#### Calibration (Platt Scaling)
**What it solves**: Aligns raw confidence scores with actual accuracy.

**Goal**: Expected Calibration Error (ECE) < 0.05 for production systems.

**Implementation**: [examples/confidence_scoring/calibration.py](examples/confidence_scoring/calibration.py)

**Complete Guide**: [docs/confidence-scoring/ensemble-methods.md](docs/confidence-scoring/ensemble-methods.md)

---

### Hallucination Detection

Methods for identifying fabricated or unsupported information.

#### SelfCheckGPT
**How it works**: Measures consistency across multiple samples from the same LLM. Factual statements remain consistent; hallucinations show high variance.

**Why it's better**: Unlike legacy NLP metrics (WER, METEOR), SelfCheckGPT is designed for Transformer-era LLMs and addresses hallucination problems that didn't exist in pre-Transformer systems.

**Zero-resource**: No external knowledge base required.

**Implementation**: [examples/hallucination_detection/selfcheck_gpt.py](examples/hallucination_detection/selfcheck_gpt.py)

**Documentation**: [docs/hallucination-detection/selfcheck-gpt.md](docs/hallucination-detection/selfcheck-gpt.md)

#### Logprobs-based Detection
**Method**: Identify low-confidence tokens as potential hallucinations.

**Threshold**: Typical cutoff at 0.3 probability for hallucination risk flagging.

**Implementation**: [examples/hallucination_detection/logprobs_detection.py](examples/hallucination_detection/logprobs_detection.py)

---

### Bias Detection

Systematic methods for identifying unfair treatment across demographic groups.

#### Correspondence Experiments
**Method**: Test model responses with demographic identifiers varied systematically.

**Example**: Same resume with different names (e.g., "John" vs. "Jamal") to detect hiring bias.

**Implementation**: [examples/bias_detection/correspondence.py](examples/bias_detection/correspondence.py)

#### Bayesian Hypothesis Testing
**What it measures**: Statistical evidence of bias using Bayesian inference.

**Advantage**: Quantifies uncertainty in bias detection, avoiding false positives from small sample sizes.

**Implementation**: [examples/bias_detection/bayesian_testing.py](examples/bias_detection/bayesian_testing.py)

#### QuaCer-B Certification
**What it provides**: Certified bounds on bias magnitude with statistical guarantees.

**Use case**: Regulatory compliance and high-stakes applications.

**Documentation**: [docs/bias-detection/quacer-b.md](docs/bias-detection/quacer-b.md)

**Complete Guide**: [docs/bias-detection/bias-methods.md](docs/bias-detection/bias-methods.md)

---

## Domain-Specific Evaluation

### RAG Systems

Retrieval-Augmented Generation requires specialized evaluation of both retrieval and generation components.

#### Component-Level Metrics

**Retrieval Quality**:
- Precision@k: Relevance of top-k retrieved documents
- Recall@k: Coverage of relevant documents in top-k
- MRR (Mean Reciprocal Rank): Position of first relevant document
- NDCG (Normalized Discounted Cumulative Gain): Graded relevance scoring

**Generation Faithfulness**:
- Groundedness: All claims supported by retrieval context
- Hallucination penalty: Severity weighting for fabricated information
- Attribution accuracy: Correct source citation

**End-to-End**:
- Answer correctness: Factual accuracy given context
- Completeness: Coverage of relevant information from context
- Conciseness: Avoiding unnecessary verbosity


**Implementation**: [examples/rag_evaluation/](examples/rag_evaluation/)

**Documentation**: [docs/rag-evaluation/rag-metrics.md](docs/rag-evaluation/rag-metrics.md)

#### Example: Medical RAG Evaluation

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

medical_faithfulness = GEval(
    name="Medical Faithfulness",
    evaluation_steps=[
        "Extract medical claims from output",
        "Verify each claim against clinical guidelines in context",
        "Identify contradictions or unsupported claims",
        "HEAVILY PENALIZE hallucinations that could cause patient harm",
        "Emphasize clinical accuracy and safety"
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT
    ],
    threshold=0.9  # High threshold for medical
)
```

---

### Code Generation

#### Understanding Pass@k vs Pass^k

These metrics measure different aspects of code generation performance and serve different purposes.

##### Metric Comparison

| Metric | Definition | Formula | Use Case |
|--------|-----------|---------|----------|
| **Pass@k** | At least one of k solutions passes | `1 - C(n-c,k)/C(n,k)` | Benchmark comparison |
| **Pass^k** | All k solutions pass | `p^k` | Reliability planning |

##### Practical Example

For a model with 70% individual success rate:

| k | Pass@k | Pass^k | Gap | Interpretation |
|---|--------|--------|-----|----------------|
| 1 | 70% | 70% | 0% | Single attempt baseline |
| 3 | 97% | 34% | 63% | Large gap between metrics |
| 5 | 99% | 17% | 82% | Gap increases with k |

##### When to Use Each

**Use Pass@k for:**
- Comparing models on benchmarks
- Reporting best-case performance
- Academic evaluation

**Use Pass^k for:**
- Planning system reliability
- Resource allocation
- SLA commitments

**Implementation**: [examples/code_generation/pass_metrics.py](examples/code_generation/pass_metrics.py)

**Documentation**: [docs/code-generation/pass-metrics-distinction.md](docs/code-generation/pass-metrics-distinction.md)

**Implementation**: [examples/code_generation/pass_at_k.py](examples/code_generation/pass_at_k.py)

#### Code Quality Metrics
- **Functional correctness**: Unit test passage
- **Code efficiency**: Runtime and memory benchmarks
- **Code style**: PEP8, linting scores
- **Security**: Vulnerability scanning (Bandit, CodeQL)

**Documentation**: [docs/code-generation/metrics.md](docs/code-generation/metrics.md)

---

### Multi-Agent Systems

Evaluation challenges unique to autonomous and cooperative agents.

#### Emergent Behavior Assessment
**Challenge**: Traditional metrics fail to capture dynamic, context-dependent agent behaviors.

**Approach**:
1. **Scenario-based testing**: Predefined interaction sequences
2. **Trace analysis**: Evaluate decision trees and communication patterns
3. **Goal achievement**: Success rate on complex multi-step objectives

#### Coordination Metrics
- **Communication efficiency**: Message volume vs. task complexity
- **Role adherence**: Agent specialization maintenance
- **Conflict resolution**: Time to consensus in disagreements
- **Distributed explainability**: Transparency across agent decisions

**Implementation**: [examples/multi_agent/](examples/multi_agent/)

---

## Tools & Frameworks

### Open Source Evaluation Frameworks

#### Core Frameworks
- **[DeepEval](https://github.com/confident-ai/deepeval)** - Production-grade G-Eval implementation, 10M+ metrics/month
- **[Ragas](https://github.com/explodinggradients/ragas)** - RAG-specific evaluation with pluggable scorers
- **[TruLens](https://github.com/truera/trulens)** - Feedback functions for chains and agents
- **[Promptfoo](https://github.com/promptfoo/promptfoo)** - Local-first CLI for prompt evaluation with regression detection
- **[OpenAI Evals](https://github.com/openai/evals)** - Reference harness with extensive eval registry

#### Observability Platforms
- **[Langfuse](https://github.com/langfuse/langfuse)** - Open-source tracing, eval dashboards, prompt analytics
- **[Arize Phoenix](https://github.com/Arize-ai/phoenix)** - OpenTelemetry-native observability for RAG and agents
- **[Opik](https://github.com/comet-ml/opik)** - Self-hostable evaluation hub with interactive traces

### Commercial Platforms

- **[Braintrust](https://www.braintrust.dev/)** - CI-style regression tests with agent sandboxes
- **[LangSmith](https://smith.langchain.com/)** - Hosted tracing + batched evals for LangChain apps
- **[Confident AI](https://www.confident-ai.com/)** - DeepEval-backed platform for production monitoring

### Cloud Platform Services

- **[Amazon Bedrock Evaluations](https://aws.amazon.com/bedrock/evaluations/)** - Managed model and RAG scoring
- **[Azure AI Foundry](https://learn.microsoft.com/en-us/azure/ai-foundry/)** - Evaluation flows integrated with Prompt Flow
- **[Vertex AI Evaluation](https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview)** - Adaptive rubric-based evaluation

**Complete List**: [tools-and-platforms.md](tools-and-platforms.md)

---

## Benchmarks & Datasets

### General Language Understanding

- **[MMLU](https://github.com/hendrycks/test)** - Massive multitask language understanding (57 subjects)
- **[MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro)** - Harder 10-choice variant focused on reasoning
- **[BIG-bench](https://github.com/google/BIG-bench)** - Collaborative benchmark for diverse reasoning tasks
- **[HELM](https://crfm.stanford.edu/helm/latest/)** - Holistic evaluation methodology emphasizing multi-criteria scoring

### Domain-Specific Benchmarks

**Code**:
- **[HumanEval](https://github.com/openai/human-eval)** - Unit-test-based code synthesis (164 problems)
- **[MBPP](https://github.com/google-research/google-research/tree/master/mbpp)** - Mostly Basic Programming Problems (974 problems)

**Mathematics**:
- **[MATH](https://github.com/hendrycks/math)** - Competition-level math (12,500 problems)
- **[GSM8K](https://github.com/openai/grade-school-math)** - Grade school math word problems

**Retrieval**:
- **[BEIR](https://github.com/beir-cellar/beir)** - Benchmark for information retrieval (18 datasets)
- **[MTEB](https://github.com/embeddings-benchmark/mteb)** - Massive text embedding benchmark

### Agent Benchmarks

- **[AgentBench](https://github.com/THUDM/AgentBench)** - LLMs as agents across simulated domains
- **[GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA)** - Tool-use benchmark with grounded reasoning

### Safety Benchmarks

- **[TruthfulQA](https://github.com/sylinrl/TruthfulQA)** - Hallucination and factuality measurement
- **[BBQ](https://github.com/nyu-mll/BBQ)** - Bias benchmark for QA
- **[ToxiGen](https://github.com/microsoft/ToxiGen)** - Toxic language generation and detection

**Complete List**: [benchmarks.md](benchmarks.md)

---

## Production Best Practices

### 1. Use Evaluation Steps, Not Criteria

```python
# ❌ Less consistent (regenerates steps each time)
metric = GEval(criteria="Check for correctness", ...)

# ✅ More consistent (fixed procedure)
metric = GEval(
    evaluation_steps=[
        "Verify factual accuracy",
        "Check for completeness",
        "Assess clarity"
    ],
    ...
)
```

### 2. Implement Calibration

```python
# Fit calibrator on validation set
calibrator = ConfidenceCalibrator()
calibrator.fit(validation_confidences, ground_truth)

# Apply to production
calibrated_score = calibrator.calibrate(raw_confidence)

# Monitor ECE < 0.05
```

### 3. Use Component-Level Tracing

```python
from deepeval.tracing import observe

@observe(metrics=[retrieval_quality])
def retrieve(query):
    # Retrieval logic
    return documents

@observe(metrics=[generation_faithfulness])
def generate(query, documents):
    # Generation logic
    return answer

# Separate scores for retrieval vs. generation
```

### 4. Set Domain-Appropriate Thresholds

```python
# Medical application: high threshold, strict mode
medical_metric = GEval(
    threshold=0.9,
    strict_mode=True,  # Binary: perfect or fail
    ...
)

# General chatbot: lower threshold, graded scoring
chatbot_metric = GEval(
    threshold=0.7,
    strict_mode=False,
    ...
)
```

### 5. Monitor Confidence-Accuracy Correlation

```python
# Validate Spearman ρ > 0.7
correlation = spearmanr(confidences, accuracies)
if correlation < 0.7:
    print("⚠️ Confidence scores unreliable - recalibrate")
```

### 6. Implement Human-in-the-Loop Thresholds

```python
def route_for_review(calibrated_confidence):
    if calibrated_confidence >= 0.85:
        return "AUTO_PROCESS"
    elif calibrated_confidence >= 0.60:
        return "SPOT_CHECK"  # 10% sampling
    else:
        return "HUMAN_REVIEW"  # 100% review
```

### 7. Cost Optimization

```python
# Use cheaper models for less critical evals
fast_metric = GEval(
    model="gpt-4o-mini",  # ~10x cheaper than GPT-4o
    ...
)

# Cache repeated evaluations
@lru_cache(maxsize=1000)
def cached_evaluate(input_hash, output_hash):
    return metric.measure(test_case)
```

---

## Research Context

This evaluation guide is developed in support of research on **Agentic AI Explainable-by-Design**, focusing on:

1. **Multi-agent systems for ethical analysis** of regulatory documents
2. **Behavior metrics in RAG systems** applied to sensitive domains (healthcare, legal, financial)
3. **Interpretive auditing frameworks** combining technical performance with human interpretability
4. **Global South perspectives** on AI evaluation in contexts of limited infrastructure and linguistic diversity

---

## Contributing

Contributions are welcome! This guide aims to be a living resource for the AI evaluation community.

**Ways to contribute**:
- Add new evaluation methods with code examples
- Improve documentation clarity
- Report issues or inaccuracies
- Share production case studies
- Translate content (especially Portuguese, Spanish for Latin American accessibility)

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## Citation

If you use this guide in your research or projects, please cite:

```bibtex
@misc{parreao2024awesome_ai_eval,
  author = {Parreão, Hugo},
  title = {Awesome AI Evaluation Guide: Implementation-Focused Methods for LLMs, RAG, and Agentic AI},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/hparreao/Awesome-AI-Evaluation-Guide}
}
```

---

## License

This work is released under [CC0 1.0 Universal](LICENSE) (Public Domain). You are free to use, modify, and distribute this content without attribution, though attribution is appreciated.

---

**Maintained by**: [Hugo Parreão](https://github.com/hparreao) | AI Engineering MSc 

**Contact**: Open an issue or reach out via GitHub for questions, suggestions, or collaboration opportunities.
