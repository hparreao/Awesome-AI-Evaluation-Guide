# Bayes@N performance estimates

Bayes@N estimates a model's expected score on a fixed benchmark from repeated evaluation outcomes.
The [example] uses a uniform Dirichlet prior for each question and returns the posterior mean and standard deviation of the benchmark-average expected score.

Each row is a question, each column is a repeated trial, and each integer is a rubric category.
The weights map category IDs to scores: `[0, 1]` for binary correctness, or `[0, 0.5, 1]` for incorrect, partial-credit, and correct outcomes.
Weights define the evaluation target and should be chosen before comparing models.

```python
from examples.confidence_scoring.bayes_at_n import bayes_at_n

result = bayes_at_n([[1, 1], [0, 0]], [0.0, 1.0])
print(result)
# {'posterior_mean': 0.5, 'posterior_std': 0.13693063937629152}
```

Run the examples and tests from the repository root:

```bash
python examples/confidence_scoring/bayes_at_n.py
python -m unittest discover -s tests -v
```

The calculation assumes independent trials within each question and independent question posteriors.
It conditions on the selected benchmark, rubric, prompts, and inference settings.
Its uncertainty describes finite-trial performance estimation; it is not a per-answer confidence score or a measure of uncertainty from choosing the benchmark questions.
The example reports exact posterior moments and does not convert them into normal-approximation intervals.
Dataset bias, distribution shift, and rubric errors need separate evaluation.

The implementation follows Algorithm 1 of [Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation][paper] by Mohsen Hariri, Amirhossein Samandar, Michael Hinczewski, and Vipin Chaudhary (ICLR 2026).
Under a uniform prior and equal trial counts, its posterior-mean ranking matches the ranking by average rubric score.
[Scorio] is the authors' implementation with additional evaluation APIs and support for prior outcomes.
This example uses only Python's standard library, so `requirements.txt` needs no new dependency.

[example]: ../../examples/confidence_scoring/bayes_at_n.py
[paper]: https://arxiv.org/abs/2510.04265
[Scorio]: https://github.com/mohsenhariri/scorio
