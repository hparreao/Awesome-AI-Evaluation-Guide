"""Small SCORE-style consistency report across caller-selected perturbations."""
from statistics import mean
from typing import Callable, Dict, Mapping, Sequence


def evaluate(cases: Sequence[Mapping[str, object]], generate: Callable[[str, float], str], compare: Callable[[str, str], bool], temperatures: Sequence[float]) -> Dict[str, object]:
    rows = []
    for case in cases:
        outputs = [generate(str(variant), temperature) for variant in case["variations"] for temperature in temperatures]
        correct = [compare(output, str(case["expected"])) for output in outputs]
        rows.append({"id": case["id"], "accuracy": mean(correct), "consistent": all(correct), "samples": len(outputs)})
    return {"cases": rows, "mean_accuracy": mean(row["accuracy"] for row in rows) if rows else None, "consistency_rate": mean(row["consistent"] for row in rows) if rows else None}


if __name__ == "__main__":
    cases = [{"id": "capital", "variations": ["Capital of France?", "France's capital?"], "expected": "Paris"}]
    print(evaluate(cases, lambda prompt, temperature: "Paris", lambda output, expected: output == expected, [0.0, 0.7]))
