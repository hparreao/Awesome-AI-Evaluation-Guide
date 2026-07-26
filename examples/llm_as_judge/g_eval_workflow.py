"""Evaluate a rubric with an injected assessor; no provider or threshold is fixed."""
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Sequence


@dataclass(frozen=True)
class Criterion:
    name: str
    instruction: str
    critical: bool = False


def evaluate(artifact: Mapping[str, str], criteria: Sequence[Criterion], assessor: Callable[[Criterion, Mapping[str, str]], Mapping[str, object]]) -> Dict[str, object]:
    judgments = {criterion.name: dict(assessor(criterion, artifact)) for criterion in criteria}
    return {"judgments": judgments, "critical_dimensions": [criterion.name for criterion in criteria if criterion.critical]}


if __name__ == "__main__":
    def demo_assessor(criterion, artifact):
        return {"verdict": "supported" if artifact["answer"] else "unknown", "reason": "replace with a model judge"}
    report = evaluate({"answer": "The context supports this claim."}, [Criterion("faithfulness", "Is each claim supported by context?", critical=True)], demo_assessor)
    print(report)
