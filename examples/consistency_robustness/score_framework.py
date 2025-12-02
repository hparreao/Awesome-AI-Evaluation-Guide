"""
SCORE: Systematic COnsistency and Robustness Evaluation Framework
==================================================================

Implementation based on NVIDIA Research (2025) - arXiv:2503.00137

Key Insight: Higher accuracy does not guarantee higher consistency.
A model can be 85% accurate but only 60% consistent.

This module provides production-ready implementation of the SCORE framework
for evaluating LLM consistency and robustness across multiple dimensions.

Author: Hugo Parreão
License: CC0 1.0 Universal
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import Counter
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SCOREMetrics:
    """
    Complete SCORE evaluation metrics.

    Attributes:
        accuracy: Traditional accuracy score (0-1)
        consistency_rate: CR@K - all K responses must be correct
        prompt_robustness: Stability across prompt variations
        sampling_robustness: Consistency under temperature changes
        order_robustness: Invariance to choice ordering
        overall_score: Weighted combination of all metrics
        detailed_results: Breakdown by test case
    """
    accuracy: float
    consistency_rate: float
    prompt_robustness: float
    sampling_robustness: float
    order_robustness: float
    overall_score: float
    detailed_results: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"SCORE Metrics:\n"
            f"  Accuracy: {self.accuracy:.2%}\n"
            f"  Consistency Rate (CR@K): {self.consistency_rate:.2%}\n"
            f"  Prompt Robustness: {self.prompt_robustness:.2%}\n"
            f"  Sampling Robustness: {self.sampling_robustness:.2%}\n"
            f"  Order Robustness: {self.order_robustness:.2%}\n"
            f"  Overall SCORE: {self.overall_score:.2%}"
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary for serialization."""
        return {
            "accuracy": self.accuracy,
            "consistency_rate": self.consistency_rate,
            "prompt_robustness": self.prompt_robustness,
            "sampling_robustness": self.sampling_robustness,
            "order_robustness": self.order_robustness,
            "overall_score": self.overall_score
        }


class SCOREEvaluator:
    """
    Production-ready SCORE framework evaluator.

    This evaluator implements the complete SCORE methodology for assessing
    LLM consistency and robustness beyond simple accuracy metrics.
    """

    def __init__(
        self,
        model: Any,
        k: int = 5,
        temperature_range: Tuple[float, float] = (0.0, 1.0),
        temperature_steps: int = 5,
        cache_responses: bool = True,
        parallel_execution: bool = True,
        max_workers: int = 4
    ):
        """
        Initialize SCORE evaluator.

        Args:
            model: LLM model instance with generate() method
            k: Number of samples for consistency rate (default: 5)
            temperature_range: Min and max temperature for robustness testing
            temperature_steps: Number of temperature points to test
            cache_responses: Whether to cache model responses
            parallel_execution: Enable parallel processing
            max_workers: Maximum parallel workers
        """
        self.model = model
        self.k = k
        self.temperature_range = temperature_range
        self.temperature_steps = temperature_steps
        self.cache_responses = cache_responses
        self.parallel_execution = parallel_execution
        self.max_workers = max_workers

        # Response cache for efficiency
        self._response_cache = {} if cache_responses else None

        logger.info(f"SCORE Evaluator initialized with k={k}")

    def _generate_cached(
        self,
        prompt: str,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate response with optional caching.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            Model response string
        """
        if self.cache_responses:
            # Create cache key
            cache_key = hashlib.sha256(
                f"{prompt}_{temperature}_{json.dumps(kwargs, sort_keys=True)}".encode()
            ).hexdigest()

            if cache_key in self._response_cache:
                return self._response_cache[cache_key]

            response = self.model.generate(prompt, temperature=temperature, **kwargs)
            self._response_cache[cache_key] = response
            return response

        return self.model.generate(prompt, temperature=temperature, **kwargs)

    def calculate_consistency_rate_at_k(
        self,
        prompt: str,
        expected: str,
        k: Optional[int] = None,
        temperature: float = 0.7
    ) -> float:
        """
        Calculate CR@K (Consistency Rate at K).

        CR@K = 1 if all k responses are correct, 0 otherwise.
        This measures reliability versus Pass@k which measures optimism.

        Args:
            prompt: Input prompt
            expected: Expected correct answer
            k: Number of samples (uses self.k if None)
            temperature: Sampling temperature

        Returns:
            1.0 if all k responses match expected, 0.0 otherwise
        """
        if k is None:
            k = self.k

        responses = []
        for _ in range(k):
            response = self._generate_cached(prompt, temperature=temperature)
            responses.append(response)

        # Check if all responses are correct
        correct_count = sum(1 for r in responses if self._compare_responses(r, expected))

        # CR@K requires ALL responses to be correct
        return 1.0 if correct_count == k else 0.0

    def evaluate_prompt_robustness(
        self,
        base_prompt: str,
        prompt_variations: List[str],
        expected: str
    ) -> float:
        """
        Evaluate robustness to prompt variations.

        Tests if semantically equivalent prompts produce consistent results.

        Args:
            base_prompt: Original prompt
            prompt_variations: List of paraphrased prompts
            expected: Expected answer

        Returns:
            Robustness score (0-1)
        """
        base_response = self._generate_cached(base_prompt, temperature=0.0)

        if not self._compare_responses(base_response, expected):
            return 0.0  # Base response is wrong

        consistent = 0
        for variation in prompt_variations:
            response = self._generate_cached(variation, temperature=0.0)
            if self._compare_responses(response, expected):
                consistent += 1

        return consistent / len(prompt_variations) if prompt_variations else 0.0

    def evaluate_sampling_robustness(
        self,
        prompt: str,
        expected: str,
        n_samples: int = 10
    ) -> float:
        """
        Evaluate robustness to temperature-based sampling.

        Tests consistency across different temperature settings.

        Args:
            prompt: Input prompt
            expected: Expected answer
            n_samples: Number of samples per temperature

        Returns:
            Sampling robustness score (0-1)
        """
        temperatures = np.linspace(
            self.temperature_range[0],
            self.temperature_range[1],
            self.temperature_steps
        )

        total_correct = 0
        total_samples = 0

        for temp in temperatures:
            for _ in range(n_samples):
                response = self._generate_cached(prompt, temperature=temp)
                if self._compare_responses(response, expected):
                    total_correct += 1
                total_samples += 1

        return total_correct / total_samples if total_samples > 0 else 0.0

    def evaluate_order_robustness(
        self,
        prompt_template: str,
        choices: List[str],
        correct_choice: str,
        n_permutations: int = 5
    ) -> float:
        """
        Evaluate robustness to choice ordering (for multiple-choice questions).

        Args:
            prompt_template: Template with {choices} placeholder
            choices: List of answer choices
            correct_choice: The correct choice
            n_permutations: Number of random orderings to test

        Returns:
            Order robustness score (0-1)
        """
        import random

        correct_count = 0

        for _ in range(n_permutations):
            # Shuffle choices
            shuffled = choices.copy()
            random.shuffle(shuffled)

            # Create prompt with shuffled choices
            choices_text = "\n".join(
                f"{chr(65+i)}. {choice}" for i, choice in enumerate(shuffled)
            )
            prompt = prompt_template.format(choices=choices_text)

            # Get response
            response = self._generate_cached(prompt, temperature=0.0)

            # Check if model selected correct answer
            correct_index = shuffled.index(correct_choice)
            correct_letter = chr(65 + correct_index)

            if correct_letter in response or correct_choice in response:
                correct_count += 1

        return correct_count / n_permutations if n_permutations > 0 else 0.0

    def _compare_responses(self, response: str, expected: str) -> bool:
        """
        Compare model response with expected answer.

        This is a simple comparison but can be extended with:
        - Semantic similarity
        - Fuzzy matching
        - Custom evaluation logic

        Args:
            response: Model response
            expected: Expected answer

        Returns:
            True if responses match
        """
        # Normalize for comparison
        response_normalized = response.strip().lower()
        expected_normalized = expected.strip().lower()

        # Exact match or contains check
        return (
            response_normalized == expected_normalized or
            expected_normalized in response_normalized
        )

    def evaluate(
        self,
        test_cases: List[Dict[str, Any]],
        verbose: bool = True
    ) -> SCOREMetrics:
        """
        Complete SCORE evaluation on test cases.

        Args:
            test_cases: List of test cases with format:
                {
                    "prompt": str,
                    "expected": str,
                    "variations": List[str] (optional),
                    "choices": List[str] (optional for multiple-choice),
                    "correct_choice": str (optional)
                }
            verbose: Print progress information

        Returns:
            SCOREMetrics object with complete evaluation results
        """
        start_time = time.time()

        # Initialize metric collectors
        accuracies = []
        consistency_rates = []
        prompt_robustness_scores = []
        sampling_robustness_scores = []
        order_robustness_scores = []
        detailed_results = []

        # Process test cases
        total_cases = len(test_cases)
        for idx, case in enumerate(test_cases):
            if verbose and idx % 10 == 0:
                logger.info(f"Processing test case {idx+1}/{total_cases}")

            case_results = {"case_id": idx}

            # 1. Accuracy
            response = self._generate_cached(case["prompt"], temperature=0.0)
            is_correct = self._compare_responses(response, case["expected"])
            accuracies.append(1.0 if is_correct else 0.0)
            case_results["accuracy"] = is_correct

            # 2. Consistency Rate (CR@K)
            cr = self.calculate_consistency_rate_at_k(
                case["prompt"],
                case["expected"]
            )
            consistency_rates.append(cr)
            case_results["consistency_rate"] = cr

            # 3. Prompt Robustness (if variations provided)
            if "variations" in case and case["variations"]:
                pr = self.evaluate_prompt_robustness(
                    case["prompt"],
                    case["variations"],
                    case["expected"]
                )
                prompt_robustness_scores.append(pr)
                case_results["prompt_robustness"] = pr

            # 4. Sampling Robustness
            sr = self.evaluate_sampling_robustness(
                case["prompt"],
                case["expected"],
                n_samples=3  # Reduced for efficiency
            )
            sampling_robustness_scores.append(sr)
            case_results["sampling_robustness"] = sr

            # 5. Order Robustness (for multiple-choice)
            if "choices" in case and "correct_choice" in case:
                or_score = self.evaluate_order_robustness(
                    case.get("prompt_template", case["prompt"]),
                    case["choices"],
                    case["correct_choice"]
                )
                order_robustness_scores.append(or_score)
                case_results["order_robustness"] = or_score

            detailed_results.append(case_results)

        # Calculate aggregate metrics
        metrics = SCOREMetrics(
            accuracy=np.mean(accuracies),
            consistency_rate=np.mean(consistency_rates),
            prompt_robustness=np.mean(prompt_robustness_scores) if prompt_robustness_scores else 0.0,
            sampling_robustness=np.mean(sampling_robustness_scores),
            order_robustness=np.mean(order_robustness_scores) if order_robustness_scores else 0.0,
            overall_score=np.mean([
                np.mean(accuracies),
                np.mean(consistency_rates),
                np.mean(sampling_robustness_scores)
            ]),
            detailed_results={"cases": detailed_results}
        )

        elapsed_time = time.time() - start_time
        if verbose:
            logger.info(f"Evaluation completed in {elapsed_time:.2f} seconds")
            logger.info(str(metrics))

        # Critical insight logging
        if metrics.consistency_rate < metrics.accuracy * 0.8:
            logger.warning(
                f"⚠️ Consistency significantly lower than accuracy! "
                f"CR@K={metrics.consistency_rate:.2%} vs Accuracy={metrics.accuracy:.2%}"
            )

        return metrics


# Example usage and testing
if __name__ == "__main__":
    # Mock model for demonstration
    class MockModel:
        def generate(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
            # Simulate varied responses based on temperature
            import random
            random.seed(hash(prompt) + int(temperature * 100))

            if "capital of France" in prompt:
                responses = ["Paris", "Paris", "Paris", "Lyon", "Paris"]
                return random.choice(responses) if temperature > 0 else "Paris"

            return "Mock response"

    # Initialize evaluator
    model = MockModel()
    evaluator = SCOREEvaluator(model=model, k=5)

    # Test cases
    test_cases = [
        {
            "prompt": "What is the capital of France?",
            "expected": "Paris",
            "variations": [
                "Tell me the capital of France.",
                "France's capital is?",
                "Which city is the capital of France?"
            ]
        },
        {
            "prompt": "Select the correct answer: What is 2+2?\nA. 3\nB. 4\nC. 5",
            "expected": "B",
            "prompt_template": "Select the correct answer: What is 2+2?\n{choices}",
            "choices": ["3", "4", "5"],
            "correct_choice": "4"
        }
    ]

    # Run evaluation
    print("Running SCORE Evaluation...")
    print("=" * 60)
    metrics = evaluator.evaluate(test_cases, verbose=True)

    print("\n" + "=" * 60)
    print("CRITICAL INSIGHT:")
    print(f"Accuracy alone is misleading! This model shows:")
    print(f"  - Accuracy: {metrics.accuracy:.2%}")
    print(f"  - Consistency: {metrics.consistency_rate:.2%}")
    print(f"  - Gap: {(metrics.accuracy - metrics.consistency_rate):.2%}")
    print("\nThis gap has major implications for production reliability!")