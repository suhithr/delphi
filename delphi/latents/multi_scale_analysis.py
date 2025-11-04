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
    activation_variance: float,
    variance_threshold: float = 0.1,
) -> Literal["token", "phrase", "sentence", "paragraph", "unknown"]:
    """
    Classify a feature's natural scale based on activation patterns.

    Args:
        context_sizes: List of context sizes.
        avg_activation_by_scale: Average activation at each scale.
        activation_variance: Variance of activations across scales.
        variance_threshold: Threshold for classifying as token-level.

    Returns:
        The classified scale type.
    """
    if not context_sizes or not avg_activation_by_scale:
        return "unknown"

    # Get normalized activation curve
    values = [avg_activation_by_scale[ctx] for ctx in context_sizes]
    max_val = max(values) if max(values) > 0 else 1.0
    normalized = [v / max_val for v in values]

    # Find peak scale
    peak_idx = normalized.index(max(normalized))
    peak_scale = context_sizes[peak_idx]

    # Token-level: Low variance, similar activations at all scales
    if activation_variance < variance_threshold:
        return "token"

    # Phrase-level: Peak at small scales (8-16 tokens)
    if peak_scale <= 16:
        return "phrase"

    # Sentence-level: Peak at medium scales (32-64 tokens)
    if peak_scale <= 64:
        return "sentence"

    # Paragraph-level: Peak at large scales or increasing trend
    # Check if activations increase with context size
    if len(context_sizes) >= 3:
        # Compute trend: do activations generally increase?
        increasing_trend = sum(
            values[i + 1] > values[i] for i in range(len(values) - 1)
        ) / (len(values) - 1)
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


def compute_position_consistency(
    multi_scale_data: dict[int, list[ActivatingExample]],
) -> float:
    """
    Compute how consistently the same relative positions activate across scales.

    Args:
        multi_scale_data: Dictionary mapping context_size -> examples.

    Returns:
        Consistency score between 0 and 1.
    """
    context_sizes = sorted(multi_scale_data.keys())

    if len(context_sizes) < 2:
        return 1.0

    # For each pair of consecutive scales, compute position overlap
    overlaps = []

    for i in range(len(context_sizes) - 1):
        smaller_ctx = context_sizes[i]
        larger_ctx = context_sizes[i + 1]

        smaller_examples = multi_scale_data[smaller_ctx]
        larger_examples = multi_scale_data[larger_ctx]

        if not smaller_examples or not larger_examples:
            continue

        # Compare activation positions (relative to center)
        smaller_positions = []
        for ex in smaller_examples:
            # Find positions with activation > 0
            active_pos = (ex.activations > 0).nonzero(as_tuple=True)[0]
            # Normalize to [-0.5, 0.5] relative to center
            center = ex.activations.shape[0] / 2
            rel_pos = [(p.item() - center) / center for p in active_pos]
            smaller_positions.extend(rel_pos)

        larger_positions = []
        for ex in larger_examples:
            active_pos = (ex.activations > 0).nonzero(as_tuple=True)[0]
            center = ex.activations.shape[0] / 2
            rel_pos = [(p.item() - center) / center for p in active_pos]
            larger_positions.extend(rel_pos)

        if smaller_positions and larger_positions:
            # Compute overlap (simple heuristic: both should have activations near center)
            smaller_near_center = sum(abs(p) < 0.3 for p in smaller_positions) / len(
                smaller_positions
            )
            larger_near_center = sum(abs(p) < 0.3 for p in larger_positions) / len(
                larger_positions
            )
            overlap = min(smaller_near_center, larger_near_center)
            overlaps.append(overlap)

    if not overlaps:
        return 1.0

    return float(torch.tensor(overlaps).mean())


def summarize_multi_scale(
    multi_scale_data: dict[int, list[ActivatingExample]],
) -> dict[str, any]:
    """
    Generate a comprehensive summary of multi-scale behavior.

    Args:
        multi_scale_data: Dictionary mapping context_size -> examples.

    Returns:
        Dictionary with summary statistics.
    """
    comparison = compare_scales(multi_scale_data)
    sensitivity = compute_scale_sensitivity(multi_scale_data)
    consistency = compute_position_consistency(multi_scale_data)

    return {
        "scale_type": comparison.scale_type,
        "activation_variance": comparison.activation_variance,
        "avg_activation_by_scale": comparison.avg_activation_by_scale,
        "max_activation_by_scale": comparison.max_activation_by_scale,
        "frequency_by_scale": comparison.frequency_by_scale,
        "scale_sensitivity": sensitivity,
        "position_consistency": consistency,
        "dominant_scale": max(
            comparison.avg_activation_by_scale.items(), key=lambda x: x[1]
        )[0]
        if comparison.avg_activation_by_scale
        else 0,
    }
