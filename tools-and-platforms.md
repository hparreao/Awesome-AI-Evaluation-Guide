# AI Evaluation Tools & Platforms

A comprehensive guide to evaluation frameworks, observability platforms, and commercial services for LLM, RAG, and agent evaluation.

## Table of Contents

- [Open Source Frameworks](#open-source-frameworks)
- [Observability Platforms](#observability-platforms)
- [Commercial Platforms](#commercial-platforms)
- [Cloud Platform Services](#cloud-platform-services)
- [Comparison Guide](#comparison-guide)

---

## Open Source Frameworks

### Core Evaluation Frameworks

#### DeepEval
**Repository**: https://github.com/confident-ai/deepeval

**What it does**: Production-grade G-Eval implementation with 14+ pre-built metrics

**Key Features**:
- 10M+ evaluations per month (production-scale)
- G-Eval with token probability normalization
- RAG metrics (faithfulness, relevance, context precision/recall)
- Safety metrics (bias, toxicity, PII detection)
- Integration with pytest for CI/CD
- LLM observability and tracing

**When to use**:
- You need production-ready G-Eval
- You want pytest-style evaluation tests
- You need comprehensive RAG evaluation
- You're building in Python

**Installation**:
```bash
pip install deepeval
```

**Quick start**:
```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase

metric = GEval(name="Correctness", criteria="...")
test_case = LLMTestCase(input="...", actual_output="...")
metric.measure(test_case)
```

---

#### Ragas
**Repository**: https://github.com/explodinggradients/ragas

**What it does**: RAG-specific evaluation library with reference-free metrics

**Key Features**:
- Faithfulness: Answer grounded in context
- Answer relevance: Response addresses question
- Context precision/recall: Retrieval quality
- LangChain and LlamaIndex integration
- Synthetic test data generation

**When to use**:
- You're evaluating RAG pipelines
- You don't have reference answers
- You want component-level RAG metrics
- You need synthetic test generation

**Installation**:
```bash
pip install ragas
```

**Quick start**:
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy]
)
```

---

#### OpenAI Evals
**Repository**: https://github.com/openai/evals

**What it does**: Reference evaluation harness with extensive eval registry

**Key Features**:
- 100+ pre-built evaluations
- Model-graded evals (LLM-as-judge)
- Fact-based evals with scoring
- Community-contributed evals
- Standardized eval format

**When to use**:
- You want standardized eval formats
- You need reference implementations
- You're building custom evals from templates
- You want community-validated benchmarks

**Installation**:
```bash
pip install evals
```

---

#### Promptfoo
**Repository**: https://github.com/promptfoo/promptfoo

**What it does**: Local-first CLI and dashboard for prompt evaluation

**Key Features**:
- Cost tracking across providers
- Regression detection
- A/B testing for prompts
- Side-by-side comparison
- RAG flow evaluation
- Agent workflow testing
- No cloud dependencies (local-first)

**When to use**:
- You want local evaluation without cloud
- You need cost tracking
- You're testing prompt variations
- You want visual comparison dashboard

**Installation**:
```bash
npm install -g promptfoo
```

---

#### TruLens
**Repository**: https://github.com/truera/trulens

**What it does**: Feedback function framework for chains and agents

**Key Features**:
- Custom feedback functions
- Ground truth evaluation
- Guardrail metrics
- Interactive dashboard
- LangChain and LlamaIndex integration
- Production monitoring

**When to use**:
- You need custom evaluation logic
- You want observability + evaluation
- You're using LangChain/LlamaIndex
- You need production monitoring

**Installation**:
```bash
pip install trulens-eval
```

---

### Observability Platforms

### Langwatch 🔍 
**Website**: https://langwatch.ai
**GitHub**: https://github.com/langwatch/langwatch
**Pricing**: Free tier, Pro from $99/month

**What it does**: Comprehensive observability platform specifically designed for LLM applications, providing real-time monitoring, evaluation, and optimization capabilities.

**Key Features**:
- **Real-time Quality Monitoring**: Track response quality metrics live
- **Cost Analytics**: Monitor token usage and API costs
- **Custom Evaluators**: Define domain-specific evaluation metrics
- **Trace Analysis**: Debug conversation flows and multi-step reasoning
- **Guardrails**: Implement safety checks and quality thresholds
- **A/B Testing**: Compare model versions in production
- **Alert System**: Get notified of quality degradation or anomalies

**When to use**:
- Production monitoring of LLM applications
- Need real-time quality metrics
- Cost optimization tracking
- Compliance and audit requirements
- Scale beyond 1000+ requests/minute

**Installation**:
```bash
pip install langwatch
```

**Quick start**:
```python
import langwatch

langwatch.init(api_key="your-api-key")

@langwatch.trace()
def evaluate_response(prompt):
    response = llm.generate(prompt)
    # Automatic tracking of latency, tokens, cost
    return response

# Custom evaluation
langwatch.evaluate(
    response=response,
    evaluators=["coherence", "faithfulness", "toxicity"]
)
```

**Production Features**:
- Dashboard for real-time monitoring
- Historical trend analysis
- Export to data warehouses
- Integration with major LLM providers

---

## Observability Platforms

#### Langfuse
**Repository**: https://github.com/langfuse/langfuse

**What it does**: Open-source LLM engineering platform with tracing and evaluation

**Key Features**:
- Distributed tracing for LLM calls
- Prompt management and versioning
- Dataset creation from production traces
- Evaluation dashboards
- Cost tracking
- Self-hostable or cloud

**When to use**:
- You need both observability and evaluation
- You want prompt version control
- You need production trace analysis
- You prefer self-hosting

**Installation**:
```bash
pip install langfuse
```

**Quick start**:
```python
from langfuse import Langfuse

langfuse = Langfuse()
trace = langfuse.trace(name="rag-pipeline")
# ... your LLM calls
```

---

#### Arize Phoenix
**Repository**: https://github.com/Arize-ai/phoenix

**What it does**: OpenTelemetry-native observability for RAG, LLMs, and agents

**Key Features**:
- OpenTelemetry standard tracing
- Embedding visualization
- Retrieval analysis
- LLM evaluation metrics
- Real-time monitoring
- Open-source and self-hostable

**When to use**:
- You want OpenTelemetry compatibility
- You need embedding/retrieval analysis
- You prefer open standards
- You want real-time monitoring

**Installation**:
```bash
pip install arize-phoenix
```

---

#### Opik (Comet ML)
**Repository**: https://github.com/comet-ml/opik

**What it does**: Self-hostable evaluation and observability hub

**Key Features**:
- Dataset management
- Scoring jobs
- Interactive trace inspection
- Experiment tracking
- Model versioning
- Self-hostable

**When to use**:
- You need full control (self-hosting)
- You want dataset versioning
- You need experiment tracking
- You're already using Comet ML

---

## Commercial Platforms

### LangSmith (LangChain)
**Website**: https://smith.langchain.com/

**What it does**: Hosted tracing + evaluation for LangChain applications

**Key Features**:
- Zero-code tracing for LangChain
- Dataset management
- Batched evaluations
- Regression gating
- Prompt playground
- Production monitoring

**Pricing**: Free tier available, paid plans from $39/month

**When to use**:
- You're using LangChain
- You want zero-configuration setup
- You need production monitoring
- You prefer managed service

---

### Braintrust
**Website**: https://www.braintrust.dev/

**What it does**: Evaluation workspace with CI-style regression tests

**Key Features**:
- Agent sandboxes
- Token cost tracking
- Regression detection
- Dataset versioning
- A/B testing
- Team collaboration

**Pricing**: Free tier available, usage-based pricing

**When to use**:
- You need CI/CD integration
- You want cost tracking
- You need team collaboration
- You're testing agents

---

### Confident AI (DeepEval Backend)
**Website**: https://www.confident-ai.com/

**What it does**: DeepEval-backed platform for production evaluation

**Key Features**:
- Scheduled evaluation suites
- Production guardrails
- Real-time monitoring
- Alert system
- DeepEval integration
- Managed infrastructure

**Pricing**: Free tier, paid plans from $49/month

**When to use**:
- You're using DeepEval
- You need scheduled evaluations
- You want production guardrails
- You prefer managed service

---

### HoneyHive
**Website**: https://www.honeyhive.ai/

**What it does**: Evaluation + observability with A/B testing and fine-tuning

**Key Features**:
- Prompt versioning
- A/B testing
- Fine-tuning workflows
- Human-in-the-loop evaluation
- Production monitoring
- Dataset curation

**Pricing**: Enterprise (contact for pricing)

**When to use**:
- You need A/B testing
- You want fine-tuning integration
- You need human evaluation workflows
- You're in enterprise setting

---

## Cloud Platform Services

### Amazon Bedrock Evaluations
**Website**: https://aws.amazon.com/bedrock/evaluations/

**What it does**: Managed evaluation service for foundation models

**Key Features**:
- Automatic model evaluation
- RAG pipeline scoring
- Custom evaluation jobs
- Integration with Bedrock models
- AWS infrastructure

**When to use**:
- You're using AWS Bedrock
- You want AWS-native solution
- You need managed infrastructure
- You're evaluating foundation models

---

### Azure AI Foundry Evaluations
**Website**: https://learn.microsoft.com/en-us/azure/ai-foundry/

**What it does**: Evaluation flows integrated with Prompt Flow

**Key Features**:
- Visual evaluation designer
- Risk assessment reports
- Safety evaluations
- Integration with Azure OpenAI
- Prompt Flow integration

**When to use**:
- You're using Azure OpenAI
- You want visual workflow builder
- You need risk assessment
- You prefer Microsoft ecosystem

---

### Vertex AI Generative AI Evaluation
**Website**: https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview

**What it does**: Adaptive rubric-based evaluation for Google and third-party models

**Key Features**:
- Automatic evaluation
- Custom rubrics
- Model comparison
- Integration with Vertex AI
- Google infrastructure

**When to use**:
- You're using Google Cloud
- You want GCP-native solution
- You need model comparison
- You're using Vertex AI

---

## Comparison Guide

### Framework Selection Guide

**Choose DeepEval if**:
- You need production-ready G-Eval
- You want pytest integration
- You need comprehensive metrics library

**Choose Ragas if**:
- You're focusing on RAG evaluation
- You need reference-free metrics
- You want synthetic test generation

**Choose Promptfoo if**:
- You want local-first evaluation
- You need cost tracking
- You prefer CLI + dashboard

**Choose TruLens if**:
- You need custom feedback functions
- You want observability + evaluation
- You're using LangChain/LlamaIndex

**Choose Langfuse if**:
- You need trace-based evaluation
- You want prompt versioning
- You prefer self-hosting

---

## Integration Examples

### DeepEval + Langfuse

```python
from deepeval import evaluate
from langfuse import Langfuse

# Trace with Langfuse
langfuse = Langfuse()
trace = langfuse.trace(name="eval")

# Evaluate with DeepEval
result = evaluate(test_cases, metrics=[...])

# Log to Langfuse
trace.score(name="correctness", value=result.score)
```

### Ragas + LangChain

```python
from ragas import evaluate
from ragas.metrics import faithfulness
from langchain.chains import RetrievalQA

# RAG chain
qa_chain = RetrievalQA.from_chain_type(...)

# Evaluate
result = evaluate(
    qa_chain,
    metrics=[faithfulness],
    dataset=test_dataset
)
```

### Promptfoo + CI/CD

```yaml
# .github/workflows/eval.yml
name: Prompt Evaluation
on: [pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - run: npx promptfoo eval
      - run: npx promptfoo assert --threshold 0.8
```

---

## Recommendations by Use Case

### Production RAG System
**Recommended Stack**:
1. **Evaluation**: DeepEval (G-Eval + RAG metrics)
2. **Observability**: Langfuse (tracing + datasets)
3. **Monitoring**: Confident AI (scheduled evals + alerts)

### Prompt Engineering Workflow
**Recommended Stack**:
1. **Local Testing**: Promptfoo (cost tracking + regression)
2. **Team Collaboration**: Braintrust (version control + sharing)
3. **Production**: LangSmith (if using LangChain)

### Research & Experimentation
**Recommended Stack**:
1. **Evaluation**: OpenAI Evals (reference implementations)
2. **Custom Metrics**: TruLens (flexible feedback functions)
3. **Analysis**: Jupyter + Ragas (exploratory analysis)

### Enterprise Deployment
**Recommended Stack**:
1. **Evaluation**: Cloud platform service (Bedrock/Azure/Vertex)
2. **Governance**: HoneyHive (compliance + HITL)
3. **Observability**: Self-hosted Langfuse (data control)

---

## Further Reading

- [DeepEval Documentation](https://docs.confident-ai.com/)
- [Ragas Documentation](https://docs.ragas.io/)
- [Promptfoo Documentation](https://www.promptfoo.dev/docs/intro)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [Evaluation Best Practices](../docs/best-practices.md)
