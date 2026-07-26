"""Report calibration evidence without defining an automatic-action boundary."""
from statistics import mean
from typing import Dict, Sequence, Tuple


def calibration_report(confidences: Sequence[float], outcomes: Sequence[int], bins: int = 10) -> Dict[str, object]:
    if len(confidences) != len(outcomes) or not confidences:
        raise ValueError("confidences and outcomes must be equally sized and non-empty")
    grouped = {index: [] for index in range(bins)}
    for confidence, outcome in zip(confidences, outcomes):
        if not 0 <= confidence <= 1 or outcome not in {0, 1}:
            raise ValueError("confidence must be in [0, 1] and outcome must be 0 or 1")
        grouped[min(int(confidence * bins), bins - 1)].append((confidence, outcome))
    rows = []
    for index, values in grouped.items():
        if values:
            predicted, observed = mean(value[0] for value in values), mean(value[1] for value in values)
            rows.append({"bin": index, "n": len(values), "predicted": predicted, "observed": observed, "gap": observed - predicted})
    ece = sum(row["n"] / len(confidences) * abs(row["gap"]) for row in rows)
    return {"bins": rows, "expected_calibration_error": ece}


if __name__ == "__main__":
    print(calibration_report([0.2, 0.7, 0.9], [0, 1, 1], bins=3))
