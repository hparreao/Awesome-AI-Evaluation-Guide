"""
Log Probabilities (Logprobs) for LLM Evaluation

Leverage token-level probabilities for:
- Confidence estimation
- Hallucination detection
- Classification with uncertainty
- Model comparison

References:
- OpenAI Cookbook: "Using Logprobs"
- "Runloop: Why Logprobs Matter for Production AI"
"""

import openai
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TokenAnalysis:
    """Container for token-level probability analysis."""
    token: str
    logprob: float
    probability: float
    top_alternatives: List[Dict[str, float]]
    position: int


def analyze_logprobs(
    prompt: str,
    model: str = "gpt-3.5-turbo",
    top_logprobs: int = 5,
    max_tokens: Optional[int] = None
) -> Dict:
    """
    Analyze token probabilities for confidence estimation.

    Args:
        prompt: Input prompt
        model: OpenAI model name
        top_logprobs: Number of top alternative tokens to return
        max_tokens: Maximum tokens to generate (optional)

    Returns:
        analysis: Dictionary with logprobs and confidence metrics

    Example:
        >>> prompt = "The capital of France is"
        >>> analysis = analyze_logprobs(prompt)
        >>> print(f"Mean confidence: {analysis['mean_confidence']:.2%}")
    """
    # Generate response with logprobs
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        logprobs=True,
        top_logprobs=top_logprobs,
        max_tokens=max_tokens
    )

    content = response.choices[0].logprobs.content

    # Extract token information
    tokens = []
    logprobs = []
    probabilities = []
    top_alternatives_list = []

    for i, token_data in enumerate(content):
        token = token_data.token
        logprob = token_data.logprob
        prob = np.exp(logprob)

        tokens.append(token)
        logprobs.append(logprob)
        probabilities.append(prob)

        # Extract top alternatives
        alternatives = [
            {
                'token': alt.token,
                'logprob': alt.logprob,
                'probability': np.exp(alt.logprob)
            }
            for alt in token_data.top_logprobs
        ]
        top_alternatives_list.append(alternatives)

    # Calculate aggregate metrics
    analysis = {
        'tokens': tokens,
        'logprobs': logprobs,
        'probabilities': probabilities,
        'top_alternatives': top_alternatives_list,
        'mean_confidence': np.mean(probabilities),
        'min_confidence': np.min(probabilities),
        'max_confidence': np.max(probabilities),
        'std_confidence': np.std(probabilities),
        'perplexity': np.exp(-np.mean(logprobs))
    }

    return analysis


def detect_hallucination_risk(
    text: str,
    model: str = "gpt-3.5-turbo",
    confidence_threshold: Optional[float] = None
) -> List[Dict]:
    """
    Detect potential hallucinations using low confidence scores.

    When the caller supplies a threshold, return tokens below it. With no
    threshold, return every token probability so the owner can apply policy.

    Args:
        text: Generated text to analyze
        model: Model used for generation
        confidence_threshold: Caller-owned probability boundary, if any

    Returns:
        risk_tokens: List of flagged tokens with alternatives

    Example:
        >>> text = "Paris is the capital of France"
        >>> risks = detect_hallucination_risk(text)
        >>> if risks:
        ...     print(f"⚠️ Found {len(risks)} low-confidence tokens")
    """
    # Analyze token probabilities
    analysis = analyze_logprobs(text, model=model)

    # Identify low-confidence tokens
    hallucination_risk = []

    for i, (token, prob) in enumerate(zip(analysis['tokens'], analysis['probabilities'])):
        if confidence_threshold is None or prob < confidence_threshold:
            hallucination_risk.append({
                'position': i,
                'token': token,
                'confidence': prob,
                'alternatives': analysis['top_alternatives'][i],
                'severity': None
            })

    return hallucination_risk


def calculate_sequence_confidence(logprobs: List[float]) -> Dict[str, float]:
    """
    Calculate confidence metrics from sequence of logprobs.

    Args:
        logprobs: List of log probabilities

    Returns:
        metrics: Dictionary with various confidence measures

    Example:
        >>> logprobs = [-0.5, -0.3, -0.8, -0.2]
        >>> metrics = calculate_sequence_confidence(logprobs)
        >>> print(f"Geometric mean confidence: {metrics['geometric_mean']:.2%}")
    """
    if not logprobs:
        raise ValueError("logprobs list cannot be empty")

    probs = [np.exp(lp) for lp in logprobs]

    metrics = {
        'arithmetic_mean': np.mean(probs),
        'geometric_mean': np.exp(np.mean(logprobs)),  # More robust
        'harmonic_mean': len(probs) / np.sum([1/p for p in probs if p > 0]),
        'min_confidence': np.min(probs),
        'max_confidence': np.max(probs),
        'std_dev': np.std(probs),
        'perplexity': np.exp(-np.mean(logprobs))
    }

    return metrics


def classify_with_confidence(
    text: str,
    labels: List[str],
    model: str = "gpt-3.5-turbo"
) -> Dict:
    """
    Classify text and get confidence scores using logprobs.

    Constrains model to output one of the specified labels and
    measures confidence in classification.

    Args:
        text: Text to classify
        labels: List of possible class labels
        model: Model to use for classification

    Returns:
        result: Classification with confidence distribution

    Example:
        >>> text = "This movie was absolutely terrible!"
        >>> labels = ["positive", "negative", "neutral"]
        >>> result = classify_with_confidence(text, labels)
        >>> print(f"Predicted: {result['predicted_label']}")
        >>> print(f"Confidence: {result['confidence']:.2%}")
    """
    # Create prompt
    prompt = (
        f"Classify the following text into one of these categories: {', '.join(labels)}\n\n"
        f"Text: {text}\n\n"
        f"Category:"
    )

    # Get response with logprobs
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        logprobs=True,
        top_logprobs=len(labels),
        max_tokens=1
    )

    # Extract first token (the classification)
    first_token = response.choices[0].logprobs.content[0]

    # Get probabilities for all labels
    label_probs = {}
    for alt in first_token.top_logprobs:
        token_text = alt.token.strip().lower()
        if token_text in [label.lower() for label in labels]:
            label_probs[token_text] = np.exp(alt.logprob)

    # Normalize probabilities
    total_prob = sum(label_probs.values())
    if total_prob > 0:
        label_probs = {k: v/total_prob for k, v in label_probs.items()}

    # Get predicted label
    predicted_label = first_token.token.strip()
    confidence = np.exp(first_token.logprob)

    result = {
        'predicted_label': predicted_label,
        'confidence': confidence,
        'label_distribution': label_probs,
        'alternatives': [
            {
                'label': alt.token.strip(),
                'probability': np.exp(alt.logprob)
            }
            for alt in first_token.top_logprobs
        ]
    }

    return result


def compare_generation_confidence(
    prompt: str,
    models: List[str] = ["gpt-3.5-turbo", "gpt-4"]
) -> Dict[str, Dict]:
    """
    Compare confidence scores across different models.

    Args:
        prompt: Input prompt
        models: List of model names to compare

    Returns:
        comparison: Dictionary mapping model names to their analyses

    Example:
        >>> prompt = "Explain quantum computing in simple terms."
        >>> comparison = compare_generation_confidence(prompt)
        >>> for model, analysis in comparison.items():
        ...     print(f"{model}: {analysis['mean_confidence']:.2%}")
    """
    comparison = {}

    for model in models:
        try:
            analysis = analyze_logprobs(prompt, model=model)
            comparison[model] = {
                'mean_confidence': analysis['mean_confidence'],
                'min_confidence': analysis['min_confidence'],
                'perplexity': analysis['perplexity'],
                'response': ''.join(analysis['tokens'])
            }
            print(f"✓ {model}: Mean confidence {analysis['mean_confidence']:.2%}")
        except Exception as e:
            print(f"✗ {model}: Error - {str(e)}")
            comparison[model] = None

    return comparison


# Example usage
if __name__ == "__main__":
    # Note: These examples require OpenAI API key
    # export OPENAI_API_KEY='your-key-here'

    print("=" * 70)
    print("Example 1: Basic Logprobs Analysis")
    print("=" * 70)

    prompt = "The capital of France is"
    print(f"Prompt: {prompt}\n")

    # Simulated analysis (replace with actual API call)
    print("Token-level probabilities:")
    print(f"{'Token':<15} {'Logprob':<12} {'Probability':<12}")
    print("-" * 70)

    # Simulated data
    tokens = [' Paris', '.', ' It', ' is']
    logprobs = [-0.1, -0.05, -1.2, -0.3]

    for token, logprob in zip(tokens, logprobs):
        prob = np.exp(logprob)
        print(f"{token:<15} {logprob:<12.4f} {prob:<12.4f}")

    print(f"\nMean confidence: {np.mean([np.exp(lp) for lp in logprobs]):.2%}")
    print()

    # Example 2: Hallucination detection
    print("=" * 70)
    print("Example 2: Hallucination Risk Detection")
    print("=" * 70)

    # Simulated low-confidence tokens
    risky_tokens = [
        {'token': ' obscure_fact', 'confidence': 0.15, 'position': 5},
        {'token': ' speculation', 'confidence': 0.22, 'position': 12}
    ]

    print("Detected low-confidence tokens:\n")
    for risk in risky_tokens:
        severity = '🔴 HIGH' if risk['confidence'] < 0.1 else '🟡 MEDIUM'
        print(f"{severity} RISK at position {risk['position']}")
        print(f"  Token: '{risk['token']}'")
        print(f"  Confidence: {risk['confidence']:.2%}")
    print()

    # Example 3: Sequence confidence metrics
    print("=" * 70)
    print("Example 3: Sequence Confidence Metrics")
    print("=" * 70)

    logprobs = [-0.5, -0.3, -0.8, -0.2, -0.6]
    metrics = calculate_sequence_confidence(logprobs)

    print("Confidence Metrics:")
    print(f"  Arithmetic mean: {metrics['arithmetic_mean']:.4f}")
    print(f"  Geometric mean:  {metrics['geometric_mean']:.4f} (recommended)")
    print(f"  Harmonic mean:   {metrics['harmonic_mean']:.4f}")
    print(f"  Min confidence:  {metrics['min_confidence']:.4f}")
    print(f"  Std deviation:   {metrics['std_dev']:.4f}")
    print(f"  Perplexity:      {metrics['perplexity']:.4f}")
    print()

    # Example 4: Classification with confidence
    print("=" * 70)
    print("Example 4: Classification with Confidence")
    print("=" * 70)

    # Simulated classification result
    result = {
        'predicted_label': 'negative',
        'confidence': 0.89,
        'label_distribution': {
            'negative': 0.89,
            'neutral': 0.08,
            'positive': 0.03
        }
    }

    print("Text: 'This movie was absolutely terrible!'\n")
    print(f"Predicted: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']:.2%}\n")
    print("Label Distribution:")
    for label, prob in sorted(result['label_distribution'].items(),
                              key=lambda x: x[1], reverse=True):
        bar = '█' * int(prob * 40)
        print(f"  {label:<10} {prob:>6.2%} {bar}")
    print()

    # Interpretation guide
    print("=" * 70)
    print("Interpretation Guide")
    print("=" * 70)
    print("Confidence Range | Interpretation | Action")
    print("-" * 70)
    print("> 0.8            | High           | Accept automatically")
    print("0.5 - 0.8        | Medium         | Human spot-check (10%)")
    print("0.3 - 0.5        | Low            | Human review (50%)")
    print("< 0.3            | Very Low       | Always review + flag hallucination risk")
    print("\nNote: Thresholds should be calibrated per domain and use case")
