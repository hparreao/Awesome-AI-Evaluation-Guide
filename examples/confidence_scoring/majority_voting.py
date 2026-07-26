"""
Majority Voting for Confidence Estimation

Ensemble-based confidence scoring that leverages agreement across
multiple model generations.

Key Finding: Industry studies show STRONG positive correlation between
majority voting confidence and actual accuracy, while "no clear correlation
was found between logprob-based confidence score and accuracy."

Optimal Configuration: 4-7 diverse models (sweet spot for reliability + cost)

"""

import openai
import anthropic
from collections import Counter
from typing import List, Tuple, Dict, Optional
import numpy as np


def generate_responses(
    prompt: str,
    models: List[str],
    n_per_model: int = 1,
    temperature: float = 0.0
) -> List[str]:
    """
    Generate responses from multiple models.

    Args:
        prompt: Input prompt
        models: List of model identifiers
        n_per_model: Number of responses per model
        temperature: Sampling temperature (0 = deterministic)

    Returns:
        responses: List of generated responses

    Example:
        >>> models = ["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet"]
        >>> responses = generate_responses("What is 2+2?", models)
    """
    responses = []

    for model in models:
        for _ in range(n_per_model):
            try:
                if model.startswith('gpt') or model.startswith('o1'):
                    # OpenAI models
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature
                    )
                    text = response.choices[0].message.content.strip()

                elif model.startswith('claude'):
                    # Anthropic models
                    client = anthropic.Anthropic()
                    response = client.messages.create(
                        model=model,
                        max_tokens=1024,
                        temperature=temperature,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    text = response.content[0].text.strip()

                else:
                    raise ValueError(f"Unsupported model: {model}")

                responses.append(text)

            except Exception as e:
                print(f"⚠️  Error with {model}: {str(e)}")
                continue

    return responses


def majority_vote_confidence(
    prompt: str,
    models: List[str],
    temperature: float = 0.0
) -> Tuple[str, float, Dict]:
    """
    Calculate confidence using majority voting.

    Args:
        prompt: Input prompt
        models: List of model identifiers (recommended: 4-7 models)
        temperature: Sampling temperature

    Returns:
        (majority_response, confidence_score, details)

    Example:
        >>> models = ["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet"]
        >>> answer, confidence, details = majority_vote_confidence(
        ...     "What is the capital of France?",
        ...     models
        ... )
        >>> print(f"Answer: {answer} ({confidence:.0%} confidence)")
    """
    # Generate responses
    responses = generate_responses(prompt, models, n_per_model=1, temperature=temperature)

    if not responses:
        raise ValueError("No responses generated")

    # Count occurrences
    response_counts = Counter(responses)

    # Get majority response
    majority_response, majority_count = response_counts.most_common(1)[0]

    # Calculate confidence
    confidence = majority_count / len(responses)

    # Prepare details
    details = {
        'total_models': len(models),
        'successful_responses': len(responses),
        'majority_count': majority_count,
        'unique_responses': len(response_counts),
        'response_distribution': dict(response_counts),
        # Classification and routing belong to a caller-owned policy layer.
        'agreement_level': None
    }

    return majority_response, confidence, details


def majority_vote_with_normalization(
    prompt: str,
    models: List[str],
    normalize_fn: Optional[callable] = None
) -> Tuple[str, float, Dict]:
    """
    Majority voting with response normalization.

    Useful when models give semantically identical but syntactically
    different responses (e.g., "4" vs "four", "Paris" vs "Paris, France")

    Args:
        prompt: Input prompt
        models: List of model identifiers
        normalize_fn: Function to normalize responses before voting
                     (default: lowercase + strip)

    Returns:
        (majority_response, confidence_score, details)

    Example:
        >>> def normalize(text):
        ...     # Extract just the city name
        ...     return text.split(',')[0].lower().strip()
        >>> answer, conf, _ = majority_vote_with_normalization(
        ...     "What is the capital of France?",
        ...     models,
        ...     normalize_fn=normalize
        ... )
    """
    # Default normalization: lowercase and strip
    if normalize_fn is None:
        normalize_fn = lambda x: x.lower().strip()

    # Generate responses
    raw_responses = generate_responses(prompt, models)

    if not raw_responses:
        raise ValueError("No responses generated")

    # Normalize responses
    normalized_responses = [normalize_fn(r) for r in raw_responses]

    # Count normalized responses
    response_counts = Counter(normalized_responses)
    majority_normalized, majority_count = response_counts.most_common(1)[0]

    # Find original response corresponding to majority
    for raw, normalized in zip(raw_responses, normalized_responses):
        if normalized == majority_normalized:
            majority_response = raw
            break

    # Calculate confidence
    confidence = majority_count / len(raw_responses)

    details = {
        'total_responses': len(raw_responses),
        'majority_count': majority_count,
        'unique_normalized': len(response_counts),
        'normalization_applied': True,
        'response_distribution': dict(response_counts)
    }

    return majority_response, confidence, details


def adaptive_ensemble_size(
    prompt: str,
    models: List[str],
    target_confidence: float = 0.8,
    max_rounds: int = 3
) -> Tuple[str, float, Dict]:
    """
    Adaptively add models until target confidence is reached.

    Starts with small ensemble and adds models if confidence is too low,
    optimizing cost while maintaining reliability.

    Args:
        prompt: Input prompt
        models: List of available models
        target_confidence: Desired confidence threshold
        max_rounds: Maximum ensemble rounds

    Returns:
        (answer, confidence, details)

    Example:
        >>> models = ["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet", "claude-3-opus"]
        >>> answer, conf, details = adaptive_ensemble_size(
        ...     "What is 2+2?",
        ...     models,
        ...     target_confidence=0.75
        ... )
        >>> print(f"Used {details['models_used']} models to reach {conf:.0%} confidence")
    """
    if len(models) < 2:
        raise ValueError("Need at least 2 models for ensemble")

    responses = []
    models_used = []

    for round_num in range(max_rounds):
        # Add next batch of models
        start_idx = round_num * 2
        end_idx = min(start_idx + 2, len(models))

        if start_idx >= len(models):
            break

        batch_models = models[start_idx:end_idx]
        models_used.extend(batch_models)

        # Generate from new models
        new_responses = generate_responses(prompt, batch_models)
        responses.extend(new_responses)

        # Check current confidence
        response_counts = Counter(responses)
        majority_response, majority_count = response_counts.most_common(1)[0]
        confidence = majority_count / len(responses)

        # Stop if target reached
        if confidence >= target_confidence:
            break

    details = {
        'models_used': len(models_used),
        'total_responses': len(responses),
        'rounds_needed': round_num + 1,
        'final_confidence': confidence,
        'target_reached': confidence >= target_confidence
    }

    return majority_response, confidence, details


def confidence_intervals(
    prompt: str,
    models: List[str],
    n_bootstrap: int = 100
) -> Tuple[str, float, Tuple[float, float]]:
    """
    Calculate confidence with bootstrap confidence intervals.

    Provides uncertainty quantification for the confidence estimate itself.

    Args:
        prompt: Input prompt
        models: List of model identifiers
        n_bootstrap: Number of bootstrap samples

    Returns:
        (answer, confidence, (lower_bound, upper_bound))

    Example:
        >>> answer, conf, (lower, upper) = confidence_intervals(
        ...     "What is the capital of France?",
        ...     models
        ... )
        >>> print(f"Confidence: {conf:.1%} [{lower:.1%}, {upper:.1%}]")
    """
    # Generate responses
    responses = generate_responses(prompt, models)

    if not responses:
        raise ValueError("No responses generated")

    # Calculate observed confidence
    response_counts = Counter(responses)
    majority_response, majority_count = response_counts.most_common(1)[0]
    observed_confidence = majority_count / len(responses)

    # Bootstrap confidence intervals
    bootstrap_confidences = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(responses, size=len(responses), replace=True)
        counts = Counter(bootstrap_sample)
        _, max_count = counts.most_common(1)[0]
        bootstrap_conf = max_count / len(bootstrap_sample)
        bootstrap_confidences.append(bootstrap_conf)

    # Calculate 95% confidence interval
    lower_bound = np.percentile(bootstrap_confidences, 2.5)
    upper_bound = np.percentile(bootstrap_confidences, 97.5)

    return majority_response, observed_confidence, (lower_bound, upper_bound)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Example 1: Basic Majority Voting")
    print("=" * 70)

    # Simulated responses (replace with actual API calls)
    simulated_responses = [
        "Paris",
        "Paris",
        "Paris",
        "Paris",
        "Paris, France"
    ]

    response_counts = Counter(simulated_responses)
    majority, count = response_counts.most_common(1)[0]
    confidence = count / len(simulated_responses)

    print(f"Question: What is the capital of France?\n")
    print(f"Responses from 5 models:")
    for i, resp in enumerate(simulated_responses, 1):
        print(f"  Model {i}: {resp}")

    print(f"\nMajority Answer: {majority}")
    print(f"Confidence: {confidence:.0%} ({count}/{len(simulated_responses)} models agreed)")
    print(f"Agreement Level: {'HIGH' if confidence >= 0.8 else 'MEDIUM' if confidence >= 0.6 else 'LOW'}")
    print()

    # Example 2: Response distribution
    print("=" * 70)
    print("Example 2: Response Distribution Analysis")
    print("=" * 70)

    simulated_responses_2 = [
        "4",
        "4",
        "4",
        "Four",
        "4.0",
        "The answer is 4",
        "4"
    ]

    print(f"Question: What is 2+2?\n")
    print(f"Raw responses ({len(simulated_responses_2)} models):")

    counts = Counter(simulated_responses_2)
    for response, count in counts.most_common():
        percentage = count / len(simulated_responses_2)
        bar = '█' * int(percentage * 40)
        print(f"  '{response}': {count} ({percentage:.1%}) {bar}")

    print(f"\nUnique responses: {len(counts)}")
    print("Note: Response normalization would improve confidence here")
    print()

    # Example 3: Optimal ensemble size
    print("=" * 70)
    print("Example 3: Optimal Ensemble Size (4-7 models)")
    print("=" * 70)

    print("Cost-Reliability Trade-off:\n")
    print(f"{'Models':<10} {'Cost':<10} {'Reliability':<15} {'Recommendation'}")
    print("-" * 70)
    print(f"{'1':<10} {'1x':<10} {'Poor':<15} {'❌ Too unstable'}")
    print(f"{'2-3':<10} {'2-3x':<10} {'Fair':<15} {'⚠️  Insufficient'}")
    print(f"{'4-7':<10} {'4-7x':<10} {'Good':<15} {'✅ Optimal (recommended)'}")
    print(f"{'8-10':<10} {'8-10x':<10} {'Very Good':<15} {'⚠️  Diminishing returns'}")
    print(f"{'>10':<10} {'>10x':<10} {'Excellent':<15} {'❌ Excessive cost'}")
    print()

    # Example 4: Confidence with uncertainty
    print("=" * 70)
    print("Example 4: Confidence with Uncertainty Bounds")
    print("=" * 70)

    # Simulated bootstrap results
    observed_conf = 0.86
    lower_bound = 0.78
    upper_bound = 0.92

    print(f"Observed confidence: {observed_conf:.1%}")
    print(f"95% CI: [{lower_bound:.1%}, {upper_bound:.1%}]")
    print(f"\nInterpretation:")
    print(f"  We are 95% confident that the true confidence lies between")
    print(f"  {lower_bound:.1%} and {upper_bound:.1%}")
    print()

    # Example 5: Decision thresholds
    print("=" * 70)
    print("Example 5: Production Decision Thresholds")
    print("=" * 70)

    test_confidences = [0.95, 0.72, 0.45, 0.20]

    print(f"{'Confidence':<15} {'Decision':<20} {'Action'}")
    print("-" * 70)

    for conf in test_confidences:
        if conf >= 0.85:
            decision = "AUTO_PROCESS"
            action = "Automatic processing"
        elif conf >= 0.60:
            decision = "SPOT_CHECK"
            action = "10% sampling review"
        else:
            decision = "HUMAN_REVIEW"
            action = "100% human review"

        print(f"{conf:<15.0%} {decision:<20} {action}")

    print("\nRecommendation: Calibrate thresholds based on:")
    print("  - Domain criticality (medical: higher thresholds)")
    print("  - Cost of errors vs. cost of review")
    print("  - SLA requirements for automation rate")
