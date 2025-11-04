"""
Multi-scale analysis for comparing feature activations across context window sizes.
"""

from typing import Any, Literal

import torch

from .latents import ActivatingExample, ScaleComparisonResult


def compare_scales(
    multi_scale_data: dict[int, list[ActivatingExample]],
) -> ScaleComparisonResult:
    """
    Compare feature activation behavior across different context scales.

    Args:
        multi_scale_data: Dictionary mapping context_size -> list of examples.

    Returns:
        ScaleComparisonResult with comparison metrics.
    """
    context_sizes = sorted(multi_scale_data.keys())

    avg_activation_by_scale = {}
    max_activation_by_scale = {}
    frequency_by_scale = {}

    for ctx_size in context_sizes:
        examples = multi_scale_data[ctx_size]

        if not examples:
            avg_activation_by_scale[ctx_size] = 0.0
            max_activation_by_scale[ctx_size] = 0.0
            frequency_by_scale[ctx_size] = 0.0
            continue

        # Compute average activation magnitude
        avg_acts = [ex.activations.mean().item() for ex in examples]
        avg_activation_by_scale[ctx_size] = float(torch.tensor(avg_acts).mean())

        # Compute max activation magnitude
        max_acts = [ex.max_activation for ex in examples]
        max_activation_by_scale[ctx_size] = float(torch.tensor(max_acts).mean())

        # Compute firing frequency (fraction of tokens with activation > 0)
        firing_counts = [
            (ex.activations > 0).float().mean().item() for ex in examples
        ]
        frequency_by_scale[ctx_size] = float(torch.tensor(firing_counts).mean())

    # Compute variance across scales
    avg_values = torch.tensor([avg_activation_by_scale[ctx] for ctx in context_sizes])
    activation_variance = float(avg_values.var())

    # Classify the feature scale
    scale_type = classify_feature_scale(
        context_sizes=context_sizes,
        avg_activation_by_scale=avg_activation_by_scale,
        max_activation_by_scale=max_activation_by_scale,
        activation_variance=activation_variance,
    )

    return ScaleComparisonResult(
        context_sizes=context_sizes,
        avg_activation_by_scale=avg_activation_by_scale,
        max_activation_by_scale=max_activation_by_scale,
        frequency_by_scale=frequency_by_scale,
        activation_variance=activation_variance,
        scale_type=scale_type,
    )


def classify_feature_scale(
    context_sizes: list[int],
    avg_activation_by_scale: dict[int, float],
    max_activation_by_scale: dict[int, float],
    activation_variance: float,
    variance_threshold: float = 0.1,
) -> Literal["token", "phrase", "sentence", "paragraph", "unknown"]:
    """
    Classify a feature's natural scale based on activation patterns.

    Args:
        context_sizes: List of context sizes.
        avg_activation_by_scale: Average activation at each scale.
        max_activation_by_scale: Maximum activation at each scale.
        activation_variance: Variance of activations across scales.
        variance_threshold: Threshold for classifying as token-level.

    Returns:
        The classified scale type.
    """
    if not context_sizes or not avg_activation_by_scale:
        return "unknown"

    # Get normalized activation curves for both avg and max
    avg_values = [avg_activation_by_scale[ctx] for ctx in context_sizes]
    max_values = [max_activation_by_scale[ctx] for ctx in context_sizes]

    max_avg_val = max(avg_values) if max(avg_values) > 0 else 1.0
    normalized_avg = [v / max_avg_val for v in avg_values]

    # Find peak scale for average activations
    peak_idx = normalized_avg.index(max(normalized_avg))
    peak_scale = context_sizes[peak_idx]

    # Compute max activation growth trend
    if len(context_sizes) >= 2 and max_values[0] > 0:
        max_growth_ratio = max_values[-1] / max_values[0]

        # Compute correlation between scale size and max activation
        sizes_mean = sum(context_sizes) / len(context_sizes)
        max_mean = sum(max_values) / len(max_values)

        covariance = sum(
            (ctx - sizes_mean) * (max_val - max_mean)
            for ctx, max_val in zip(context_sizes, max_values)
        ) / len(context_sizes)

        size_std = (sum((ctx - sizes_mean) ** 2 for ctx in context_sizes) / len(context_sizes)) ** 0.5
        max_std = (sum((m - max_mean) ** 2 for m in max_values) / len(max_values)) ** 0.5

        if size_std > 0 and max_std > 0:
            max_correlation = covariance / (size_std * max_std)
        else:
            max_correlation = 0.0
    else:
        max_growth_ratio = 1.0
        max_correlation = 0.0

    # Token-level: Low variance AND low max activation growth
    # Features that don't need more context will have similar max activations across scales
    if activation_variance < variance_threshold and max_growth_ratio < 1.2:
        return "token"

    # Phrase-level: Peak at small scales (8-16 tokens)
    # OR moderate growth but peaks early
    if peak_scale <= 16 and max_correlation < 0.7:
        return "phrase"

    # Paragraph-level: Strong correlation between scale and max activation
    # Features that need long context will activate more strongly with more tokens
    if max_correlation > 0.7 and max_growth_ratio > 1.5:
        return "paragraph"

    # Sentence-level: Peak at medium scales (32-64 tokens)
    if peak_scale <= 64:
        return "sentence"

    # Paragraph-level: Peak at large scales or increasing trend
    # Check if activations increase with context size
    if len(context_sizes) >= 3:
        # Compute trend: do activations generally increase?
        increasing_trend = sum(
            avg_values[i + 1] > avg_values[i] for i in range(len(avg_values) - 1)
        ) / (len(avg_values) - 1)
        if increasing_trend > 0.6:  # 60% of transitions are increases
            return "paragraph"

    return "sentence"  # Default for medium-high variance


def compute_scale_sensitivity(
    multi_scale_data: dict[int, list[ActivatingExample]],
) -> dict[int, float]:
    """
    Compute how sensitive the feature is to context size.

    Args:
        multi_scale_data: Dictionary mapping context_size -> examples.

    Returns:
        Dictionary mapping context_size -> sensitivity score.
    """
    context_sizes = sorted(multi_scale_data.keys())
    sensitivity = {}

    for i, ctx_size in enumerate(context_sizes):
        examples = multi_scale_data[ctx_size]
        if not examples:
            sensitivity[ctx_size] = 0.0
            continue

        # Compute coefficient of variation for activations
        max_acts = torch.tensor([ex.max_activation for ex in examples])
        mean_act = max_acts.mean()
        std_act = max_acts.std()

        if mean_act > 0:
            cv = std_act / mean_act
            sensitivity[ctx_size] = float(cv)
        else:
            sensitivity[ctx_size] = 0.0

    return sensitivity




def summarize_multi_scale(
    multi_scale_data: dict[int, list[ActivatingExample]],
) -> dict[str, Any]:
    """
    Generate a comprehensive summary of multi-scale behavior.

    Args:
        multi_scale_data: Dictionary mapping context_size -> examples.

    Returns:
        Dictionary with summary statistics.
    """
    comparison = compare_scales(multi_scale_data)
    sensitivity = compute_scale_sensitivity(multi_scale_data)

    # Compute max activation growth metrics
    context_sizes = sorted(multi_scale_data.keys())
    max_values = [comparison.max_activation_by_scale[ctx] for ctx in context_sizes if ctx in comparison.max_activation_by_scale]

    if len(max_values) >= 2 and max_values[0] > 0:
        max_growth_ratio = max_values[-1] / max_values[0]

        # Compute correlation
        sizes_mean = sum(context_sizes) / len(context_sizes)
        max_mean = sum(max_values) / len(max_values)
        covariance = sum(
            (ctx - sizes_mean) * (max_val - max_mean)
            for ctx, max_val in zip(context_sizes, max_values)
        ) / len(context_sizes)
        size_std = (sum((ctx - sizes_mean) ** 2 for ctx in context_sizes) / len(context_sizes)) ** 0.5
        max_std = (sum((m - max_mean) ** 2 for m in max_values) / len(max_values)) ** 0.5

        if size_std > 0 and max_std > 0:
            max_correlation = covariance / (size_std * max_std)
        else:
            max_correlation = 0.0
    else:
        max_growth_ratio = 1.0
        max_correlation = 0.0

    return {
        "scale_type": comparison.scale_type,
        "activation_variance": comparison.activation_variance,
        "avg_activation_by_scale": comparison.avg_activation_by_scale,
        "max_activation_by_scale": comparison.max_activation_by_scale,
        "frequency_by_scale": comparison.frequency_by_scale,
        "scale_sensitivity": sensitivity,
        "max_growth_ratio": max_growth_ratio,
        "max_correlation": max_correlation,
        "dominant_scale": max(
            comparison.avg_activation_by_scale.items(), key=lambda x: x[1]
        )[0]
        if comparison.avg_activation_by_scale
        else 0,
    }
