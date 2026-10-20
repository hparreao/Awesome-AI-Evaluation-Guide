"""Check Bayes@N against known Beta and Dirichlet posterior moments."""

from math import sqrt
import unittest

from examples.confidence_scoring.bayes_at_n import bayes_at_n


class BayesAtNTests(unittest.TestCase):
    def test_all_successes_retain_uncertainty(self):
        # Two successes with Beta(1, 1) give Beta(3, 1).
        result = bayes_at_n([[1, 1]], [0.0, 1.0])
        self.assertAlmostEqual(result["posterior_mean"], 0.75)
        self.assertAlmostEqual(result["posterior_std"], sqrt(3 / 80))

    def test_independent_question_average(self):
        result = bayes_at_n([[1, 1], [0, 0]], [0.0, 1.0])
        self.assertAlmostEqual(result["posterior_mean"], 0.5)
        self.assertAlmostEqual(result["posterior_std"], sqrt(3 / 160))

    def test_graded_dirichlet_moments(self):
        # Dirichlet(2, 2, 2), scored as 0, 0.5, 1, has variance 1/42.
        result = bayes_at_n([[0, 1, 2]], [0.0, 0.5, 1.0])
        self.assertAlmostEqual(result["posterior_mean"], 0.5)
        self.assertAlmostEqual(result["posterior_std"], sqrt(1 / 42))

    def test_invalid_inputs(self):
        for rows, weights in [
            ([], [0.0, 1.0]),
            ([[0], [0, 1]], [0.0, 1.0]),
            ([[2]], [0.0, 1.0]),
            ([[0.5]], [0.0, 1.0]),
            ([[0]], [0.0, float("nan")]),
        ]:
            with self.subTest(rows=rows, weights=weights):
                with self.assertRaises(ValueError):
                    bayes_at_n(rows, weights)


if __name__ == "__main__":
    unittest.main()
