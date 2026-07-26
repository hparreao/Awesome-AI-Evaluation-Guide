import unittest

from examples.bias_detection.bayesian_audit import audit_group_rates
from examples.confidence_scoring.calibration_workflow import calibration_report
from examples.consistency_robustness.score_workflow import evaluate as score_evaluate
from examples.llm_as_judge.g_eval_workflow import Criterion, evaluate as judge_evaluate
from examples.rag_evaluation.component_metrics import evaluate as rag_evaluate


class EvaluationWorkflowTests(unittest.TestCase):
    def test_g_eval_returns_dimensions_without_aggregation(self):
        report = judge_evaluate({"answer": "x"}, [Criterion("grounded", "?", True)], lambda criterion, artifact: {"verdict": "yes"})
        self.assertEqual(report["critical_dimensions"], ["grounded"])

    def test_score_reports_accuracy_and_consistency(self):
        report = score_evaluate([{"id": "x", "variations": ["a"], "expected": "ok"}], lambda prompt, temperature: "ok", lambda output, expected: output == expected, [0.0, 0.5])
        self.assertEqual(report["consistency_rate"], 1)

    def test_calibration_reports_gap(self):
        self.assertEqual(calibration_report([0.5, 0.5], [1, 0], bins=2)["expected_calibration_error"], 0.0)

    def test_bias_audit_reports_posterior_gap_without_certification(self):
        report = audit_group_rates({"a": [1, 1], "b": [0, 1]})
        self.assertEqual(report["posterior"]["a"]["mean"], 0.75)
        self.assertEqual(report["posterior_mean_gap"], 0.25)

    def test_rag_keeps_components_separate(self):
        report = rag_evaluate(["d1"], {"d1", "d2"}, ["d1"], 1, 2)
        self.assertEqual(report["retrieval"]["recall"], 0.5)
        self.assertEqual(report["generation"]["groundedness"], 0.5)


if __name__ == "__main__":
    unittest.main()
