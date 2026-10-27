# Awesome AI Evaluation Guide [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

[![License: CC0-1.0](https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg)](http://creativecommons.org/publicdomain/zero/1.0/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](CONTRIBUTING.md)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> A comprehensive collection of evaluation methods, tools, and frameworks for assessing Large Language Models (LLMs), RAG systems, and AI agents in real-world applications.

## About This Guide

This repository provides practical implementations and detailed guidance for evaluating AI systems, with a focus on understanding when and how to apply different evaluation methods. Unlike simple metric collections, we offer working code, mathematical foundations, and domain-specific considerations.

## Runnable examples

The repository includes small Python examples for G-Eval, SCORE robustness, RAG component evaluation, confidence calibration, and group-outcome auditing. They are designed to be read, adapted, and connected to the provider or data source used by each project.

Run the test suite and any individual example from the repository root:

```bash
python -m unittest discover -s tests -v
python -m examples.rag_evaluation.component_metrics
```

The examples report observations and leave thresholds and release decisions to the team using them. [`config/evaluation-policy.example.json`](config/evaluation-policy.example.json) is a starting point for recording those decisions. The modules live beside their topics under [`examples`](examples); [`skills`](skills) contains the related Codex workflows.

### What Makes This Guide Different

- **Implementation-First**: Selected methods include runnable examples; the remaining sections provide references and design guidance
- **Decision Frameworks**: Clear tables and guides for metric selection
- **Mathematical Rigor**: Understanding the theory behind each evaluation method
- **Domain Expertise**: Tailored approaches for medical, legal, financial, and other specialized applications
- **System-Level Thinking**: Evaluation of components and their interactions

### Key Research Insights

| Concept | Finding | Source | Application |
|---------|---------|--------|------------|
| **Consistency vs Accuracy** | Models can have high accuracy but low consistency | SCORE (NVIDIA 2025) | Evaluate reliability alongside correctness |
| **Pass@k vs Pass^k** | Metrics measure different aspects (optimistic vs reliable) | Code generation research | Choose based on deployment needs |
| **Confidence Scoring** | Ensemble methods correlate better with accuracy than logprobs | Industry studies | Use majority voting for confidence |
| **Component Interaction** | System performance ≠ sum of component performance | RAG research | Evaluate end-to-end and per-component |
| **Bias Detection** | Bayes Factors superior to p-values | QuaCer-B research | Use Bayesian methods for statistical rigor |

---
- **[LevelBench-100 Cross-System Evaluation](https://levelcfo.com/benchmarks/agentic-accounting/)** ([GitHub](https://github.com/levelcfo/levelbench-100)): Across 75 live agentic accounting runs ($500K token study), frontier models score 96.0% on single-source tasks but drop to 63.6% on cross-system subledger-to-GL reconciliations unless paired with deterministic verification gates.

## Table of Contents

- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Quick Examples](#quick-examples)
  - [Metric Selection Guide](#metric-selection-guide)
- [Evaluation Methods](#evaluation-methods)
  - [Traditional Metrics](#traditional-metrics)
  - [Modern Metrics](#modern-metrics)
  - [Confidence & Calibration](#confidence--calibration)
  - [Safety & Bias Detection](#safety--bias-detection)
- [Domain-Specific Evaluation](#domain-specific-evaluation)
  - [RAG Systems](#rag-systems)
  - [Code Generation](#code-generation)
  - [Multi-Agent Systems](#multi-agent-systems)
  - [Medical & Healthcare](#medical--healthcare)
  - [Legal & Compliance](#legal--compliance)
- [Tools & Platforms](#tools--platforms)
  - [Open Source Frameworks](#open-source-frameworks)
  - [Observability Platforms](#observability-platforms)
  - [Commercial Solutions](#commercial-solutions)
- [Benchmarks & Datasets](#benchmarks--datasets)
  - [General Benchmarks](#general-benchmarks)
  - [Domain Benchmarks](#domain-benchmarks)
  - [Safety Benchmarks](#safety-benchmarks)
- [Implementation Guide](#implementation-guide)
  - [Best Practices](#best-practices)
  - [Performance Optimization](#performance-optimization)
  - [Common Pitfalls](#common-pitfalls)
- [Resources](#resources)
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

### Policy ownership

This repository does not prescribe domain thresholds. A threshold is a release-policy decision that depends on the consequences of failure, error distribution, review capacity, baseline performance, and the available mitigation. Define it with the accountable system owner, retain its rationale, and configure it outside the evaluator. [`config/evaluation-policy.example.json`](config/evaluation-policy.example.json) is a starting point.

## Getting Started

### Prerequisites

- Python 3.8+ for evaluation frameworks
- Node.js 14+ for JavaScript-based tools (optional)
- API keys for LLM providers (OpenAI, Anthropic, etc.)
- Docker for self-hosted solutions (optional)

### Installation

```bash
# Clone this repository
git clone https://github.com/hparreao/Awesome-AI-Evaluation-Guide.git
cd Awesome-AI-Evaluation-Guide

# Install core dependencies
pip install -r requirements.txt

# Optional: Install specific evaluation frameworks
pip install deepeval ragas langfuse trulens-eval  # Python frameworks
npm install -g promptfoo  # JavaScript CLI tool (optional)
```

### Quick Examples

#### Basic LLM Evaluation
```python
from examples.llm_as_judge import evaluate_response

result = evaluate_response(
    question="What is the capital of France?",
    response="Paris is the capital of France.",
    criteria="factual_accuracy"
)
print(f"Score: {result.score}, Reasoning: {result.reasoning}")
```

#### RAG Pipeline Evaluation
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# Evaluate your RAG system
result = evaluate(
    dataset=your_test_data,
    metrics=[faithfulness, answer_relevancy]
)
print(f"Faithfulness: {result['faithfulness']:.2f}")
```

#### Consistency Testing (SCORE Framework)
```python
from examples.consistency_robustness.score_framework import SCOREEvaluator

evaluator = SCOREEvaluator(model=your_model, k=5)
metrics = evaluator.evaluate(test_cases)
print(f"Accuracy: {metrics.accuracy:.2f}, Consistency: {metrics.consistency_rate:.2f}")
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

# Read from a versioned policy approved by the system owner.
owner_policy = load_owner_policy()

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
    threshold=owner_policy["correctness"]
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
    threshold=owner_policy["rag_faithfulness"]
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

## Tools & Platforms

### Open Source Frameworks

#### Evaluation Libraries
- **[DeepEval](https://github.com/confident-ai/deepeval)** - Comprehensive evaluation framework with G-Eval implementation and 14+ pre-built metrics
- **[Ragas](https://github.com/explodinggradients/ragas)** - Specialized for RAG evaluation with reference-free metrics (faithfulness, relevance, context quality)
- **[TruLens](https://github.com/truera/trulens)** - Custom feedback functions with LangChain/LlamaIndex integration
- **[Promptfoo](https://github.com/promptfoo/promptfoo)** - CLI tool for prompt testing with cost tracking and regression detection
- **[OpenAI Evals](https://github.com/openai/evals)** - Reference implementation with 100+ community-contributed evaluations
- **[LangCheck](https://github.com/citadel-ai/langcheck)** - Simple, composable evaluation metrics for LLM applications
- **[Athina AI](https://github.com/athina-ai/athina-sdk)** - Configurable evaluation metrics with focus on reliability

#### Observability & Monitoring
- **[Langfuse](https://github.com/langfuse/langfuse)** - Open-source LLM engineering platform with tracing and prompt management
- **[Langwatch](https://langwatch.ai)** - Real-time quality monitoring with custom evaluators and cost analytics
- **[Arize Phoenix](https://github.com/Arize-ai/phoenix)** - OpenTelemetry-based observability with embedding visualization
- **[Opik](https://github.com/comet-ml/opik)** - Self-hostable platform with dataset management and experiment tracking
- **[Helicone](https://www.helicone.ai/)** - Observability platform with request caching and rate limiting
- **[MLflow](https://mlflow.org/docs/latest/genai/tracing)** - OpenTelemetry-compatible tracing, evaluation, and experiment tracking for LLM applications and agents
- **[Traceloop / OpenLLMetry](https://www.traceloop.com/docs/openllmetry/introduction)** - OpenTelemetry instrumentation that can export traces to an existing observability stack
- **[Weights & Biases Weave](https://wandb.ai/site)** - Experiment tracking, tracing, and evaluation for LLM applications

### Commercial Solutions

#### Evaluation Platforms
- **[Braintrust](https://www.braintrust.dev/)** - CI/CD for AI with regression testing and agent sandboxes
- **[LangSmith](https://smith.langchain.com/)** - LangChain's hosted platform for tracing and evaluation
- **[Confident AI](https://www.confident-ai.com/)** - Production monitoring with scheduled evaluation suites
- **[HoneyHive](https://www.honeyhive.ai/)** - Enterprise platform with A/B testing and fine-tuning workflows
- **[Humanloop](https://humanloop.com/)** - Prompt engineering and evaluation with human-in-the-loop
- **[Galileo](https://www.rungalileo.io/)** - ML observability extended for GenAI applications
- **[Datadog Agent Observability](https://docs.datadoghq.com/llm_observability/)** - Agent traces, evaluations, cost/latency monitoring, and correlation with Datadog APM

### Cloud Services

- **[Amazon Bedrock Evaluations](https://aws.amazon.com/bedrock/evaluations/)** - AWS-native evaluation for foundation models
- **[Azure AI Foundry](https://learn.microsoft.com/en-us/azure/ai-foundry/)** - Integrated with Azure OpenAI and Prompt Flow
- **[Vertex AI Evaluation](https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview)** - Google Cloud's evaluation service with custom rubrics
- **[OpenAI Platform](https://platform.openai.com/)** - Built-in evaluation capabilities for GPT models

**Detailed Comparison**: [tools-and-platforms.md](tools-and-platforms.md)

---

## Benchmarks & Datasets

### General Language Understanding

#### Knowledge & Reasoning
- **[MMLU](https://github.com/hendrycks/test)** - Massive Multitask Language Understanding across 57 subjects from STEM to humanities
- **[MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro)** - Enhanced version with 10 choices per question, emphasizing reasoning over memorization
- **[BIG-bench](https://github.com/google/BIG-bench)** - Beyond the Imitation Game with 200+ diverse tasks testing emergent capabilities
- **[HELM](https://crfm.stanford.edu/helm/latest/)** - Holistic evaluation across 42 scenarios and 7 metrics (accuracy, calibration, robustness, fairness, bias, toxicity, efficiency)
- **[AGIEval](https://github.com/ruixiangcui/AGIEval)** - Human-centric standardized exams (SAT, LSAT, GRE, GMAT)

#### Common Sense & World Knowledge
- **[HellaSwag](https://rowanzellers.com/hellaswag/)** - Commonsense natural language inference with adversarial filtering
- **[WinoGrande](https://winogrande.allenai.org/)** - Large-scale Winograd schema challenge for commonsense reasoning
- **[ARC](https://allenai.org/data/arc)** - AI2 Reasoning Challenge with grade-school science questions

### Domain-Specific Benchmarks

#### Code Generation
- **[HumanEval](https://github.com/openai/human-eval)** - Function synthesis with 164 Python problems and unit tests
- **[MBPP](https://github.com/google-research/google-research/tree/master/mbpp)** - 974 crowd-sourced Python programming problems
- **[CodeContests](https://github.com/deepmind/code_contests)** - Competitive programming problems from Codeforces, TopCoder
- **[SWE-bench](https://www.swebench.com/)** - Real GitHub issues requiring repository-level understanding
- **[DS-1000](https://github.com/xlang-ai/DS-1000)** - Data science problems across NumPy, Pandas, TensorFlow, PyTorch

#### Mathematics
- **[MATH](https://github.com/hendrycks/math)** - 12,500 competition mathematics problems with step-by-step solutions
- **[GSM8K](https://github.com/openai/grade-school-math)** - 8,500 grade school math word problems
- **[MINERVA](https://github.com/google-research/google-research/tree/master/minerva)** - STEM problem-solving requiring quantitative reasoning

#### Scientific Understanding
- **[ScienceQA](https://scienceqa.github.io/)** - Multimodal science questions with explanations
- **[PubMedQA](https://pubmedqa.github.io/)** - Biomedical research question answering
- **[MedQA](https://github.com/jind11/MedQA)** - Medical examination questions (USMLE style)

#### Retrieval & RAG
- **[BEIR](https://github.com/beir-cellar/beir)** - Heterogeneous benchmark for information retrieval across 18 datasets
- **[MTEB](https://github.com/embeddings-benchmark/mteb)** - Massive Text Embedding Benchmark with 8 tasks and 58 datasets
- **[MS MARCO](https://microsoft.github.io/msmarco/)** - Machine reading comprehension with 1M+ real queries
- **[Natural Questions](https://ai.google.com/research/NaturalQuestions)** - Real Google search queries with Wikipedia answers

### Task-Specific Benchmarks

#### Dialogue & Conversation
- **[MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)** - Multi-turn conversation quality assessment
- **[ChatbotArena](https://chat.lmsys.org/)** - Crowd-sourced pairwise model comparisons
- **[DialogSum](https://github.com/cylnlp/dialogsum)** - Dialogue summarization dataset

#### Translation & Multilingual
- **[WMT](https://www.statmt.org/wmt23/)** - Annual shared tasks in machine translation
- **[FLORES-200](https://github.com/facebookresearch/flores)** - Translation between 200 languages
- **[XL-Sum](https://github.com/csebuetnlp/xl-sum)** - Multilingual abstractive summarization

#### Agent & Tool Use
- **[AgentBench](https://github.com/THUDM/AgentBench)** - Comprehensive agent evaluation across 8 environments
- **[GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA)** - General AI assistants with real-world questions requiring tools
- **[WebArena](https://webarena.dev/)** - Autonomous web agents in realistic environments
- **[ToolBench](https://github.com/OpenBMB/ToolBench)** - Tool learning with 16,000+ real-world APIs

### Safety & Alignment Benchmarks

#### Truthfulness & Hallucination
- **[TruthfulQA](https://github.com/sylinrl/TruthfulQA)** - Measures whether models generate truthful answers to questions
- **[HaluEval](https://github.com/RUCAIBox/HaluEval)** - Hallucination evaluation across diverse tasks
- **[FActScore](https://github.com/shmsw25/FActScore)** - Fine-grained atomic fact verification

#### Bias & Fairness
- **[BBQ](https://github.com/nyu-mll/BBQ)** - Bias Benchmark for Question Answering in ambiguous contexts
- **[BOLD](https://github.com/amazon-science/bold)** - Bias in Open-ended Language Generation Dataset
- **[WinoBias](https://github.com/uclanlp/corefBias)** - Gender bias in coreference resolution

#### Safety & Toxicity
- **[ToxiGen](https://github.com/microsoft/ToxiGen)** - Large-scale machine-generated toxicity dataset
- **[RealToxicityPrompts](https://github.com/allenai/real-toxicity-prompts)** - Toxic generation evaluation
- **[SafetyBench](https://github.com/thu-coai/SafetyBench)** - Chinese and English safety evaluation

### Creating Custom Benchmarks

When standard benchmarks don't fit your needs:

1. **Define Clear Evaluation Criteria**: Specify what success looks like
2. **Create Diverse Test Cases**: Cover edge cases and failure modes
3. **Establish Ground Truth**: Use expert annotations or automated validation
4. **Version Control**: Track benchmark changes over time
5. **Statistical Rigor**: Ensure sufficient sample size and significance testing

**Guide**: [docs/creating-custom-benchmarks.md](docs/creating-custom-benchmarks.md)

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

### 4. Apply an owner-approved policy

```python
# Define this outside the evaluator after studying the local error distribution,
# consequences, compensating controls, and reviewer workflow.
owner_policy = load_owner_policy()
metric = GEval(
    threshold=owner_policy["selected_metric"],
    strict_mode=owner_policy["strict_mode"],
    ...
)
```

### 5. Monitor Confidence-Accuracy Correlation

```python
# Report confidence-accuracy correlation and its uncertainty.
correlation = spearmanr(confidences, accuracies)
publish_calibration_report(correlation)
```

### 6. Implement Human-in-the-Loop Thresholds

```python
def route_for_review(calibrated_confidence, owner_policy):
    return owner_policy.route(calibrated_confidence)
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

## Resources

### Learning Materials

#### Tutorials & Guides
- **[LLM Evaluation: A Practical Guide](https://www.deeplearning.ai/short-courses/evaluating-debugging-generative-ai/)** - DeepLearning.AI course on evaluation fundamentals
- **[A Survey on Evaluation of LLMs](https://arxiv.org/abs/2307.03109)** - Comprehensive academic survey (2024)
- **[Holistic Evaluation of LLMs](https://crfm.stanford.edu/helm/latest/)** - Stanford's HELM methodology and learnings
- **[RAG Evaluation Guide](https://docs.ragas.io/en/latest/)** - Best practices for evaluating retrieval-augmented generation
- **[Prompt Engineering Guide](https://www.promptingguide.ai/)** - Includes evaluation strategies for prompts

#### Papers & Research
- **[G-Eval Paper (2023)](https://arxiv.org/abs/2303.16634)** - NLG evaluation using GPT-4 with chain-of-thought
- **[SCORE Framework (2025)](https://arxiv.org/abs/2503.00137)** - NVIDIA's consistency and robustness evaluation
- **[Judging LLM-as-a-Judge (2024)](https://arxiv.org/abs/2306.05685)** - Meta-evaluation of LLM judges
- **[Constitutional AI (2022)](https://arxiv.org/abs/2212.08073)** - Anthropic's approach to AI safety evaluation

### Implementation Examples

#### Code Repositories
- **[LangChain Evaluation](https://github.com/langchain-ai/langchain/tree/master/libs/langchain/langchain/evaluation)** - Evaluation chains and criteria
- **[OpenAI Cookbook](https://github.com/openai/openai-cookbook)** - Practical examples including evaluation techniques
- **[Hugging Face Evaluate](https://github.com/huggingface/evaluate)** - Library for easily evaluating ML models and datasets
- **[Microsoft Promptflow](https://github.com/microsoft/promptflow)** - Evaluation flows for LLM applications

#### Runnable examples
- **[G-Eval workflow](examples/llm_as_judge/g_eval_workflow.py)** - Separate critical dimensions from other evaluator outputs
- **[SCORE workflow](examples/consistency_robustness/score_workflow.py)** - Compare accuracy and consistency across variations
- **[RAG component metrics](examples/rag_evaluation/component_metrics.py)** - Attribute findings to retrieval or generation
- **[Calibration workflow](examples/confidence_scoring/calibration_workflow.py)** - Compare confidence with observed outcomes
- **[Bias audit](examples/bias_detection/bayesian_audit.py)** - Report Beta posteriors for group outcomes and their mean gap

### Communities & Discussions

#### Forums & Groups
- **[r/MachineLearning](https://www.reddit.com/r/MachineLearning/)** - Academic ML discussions including evaluation
- **[Hugging Face Forums](https://discuss.huggingface.co/)** - Community discussions on model evaluation
- **[LangChain Discord](https://discord.gg/langchain)** - Active community for LLM application development
- **[AI Alignment Forum](https://www.alignmentforum.org/)** - Safety and alignment evaluation discussions

#### Conferences & Workshops
- **[NeurIPS Datasets and Benchmarks Track](https://neurips.cc/Conferences/2024/CallForDatasetsBenchmarks)** - Annual benchmark proposals
- **[EMNLP Evaluation Track](https://2024.emnlp.org/)** - NLP evaluation methodologies
- **[ACL Workshop on Evaluation](https://aclanthology.org/)** - Specialized evaluation workshops

### Related Collections

#### Awesome Lists
- **[Awesome LLM](https://github.com/Hannibal046/Awesome-LLM)** - Comprehensive LLM resources
- **[Awesome RAG](https://github.com/siat-nlp/awesome-rag)** - RAG-specific tools and papers
- **[Awesome LLM Safety](https://github.com/ydli-ai/csl)** - Safety evaluation and alignment
- **[Awesome Production LLM](https://github.com/jihoo-kim/awesome-production-llm)** - Production deployment including monitoring

#### Benchmark Leaderboards
- **[Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)** - Hugging Face's model rankings
- **[LMSYS Chatbot Arena](https://chat.lmsys.org/)** - Human preference rankings
- **[Big Code Leaderboard](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard)** - Code generation benchmarks
- **[MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)** - Text embedding rankings

### Industry Reports & Case Studies

- **[State of AI Report](https://www.stateof.ai/)** - Annual industry overview including evaluation trends
- **[OpenAI System Card](https://cdn.openai.com/papers/gpt-4-system-card.pdf)** - GPT-4 evaluation methodology
- **[Anthropic Claude Evaluations](https://www.anthropic.com/news/claude-constitution)** - Constitutional AI evaluation approach
- **[Google PaLM Technical Report](https://arxiv.org/abs/2204.02311)** - Comprehensive evaluation across 150+ tasks

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
