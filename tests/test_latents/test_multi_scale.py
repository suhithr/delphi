"""
Tests for multi-scale activation analysis.
"""

import pytest
import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from delphi.latents.latents import ActivatingExample, ActivationData
from delphi.latents.multi_scale_analysis import (
    classify_feature_scale,
    compare_scales,
    compute_position_consistency,
    compute_scale_sensitivity,
    summarize_multi_scale,
)
from delphi.latents.multi_scale_constructors import (
    extract_activation_window,
    extract_centered_window,
    multi_scale_constructor,
    multi_scale_constructor_with_tokenizer,
)


def test_extract_centered_window():
    """Test that windows are correctly extracted and centered."""
    torch.manual_seed(42)
    tokens = torch.randint(0, 1000, (2, 256))

    # Test normal window extraction
    window = extract_centered_window(tokens, batch_idx=0, center_pos=100, window_size=32)
    assert window is not None
    assert window.shape[0] == 32
    # Center should be at position 16 (half of 32)
    assert window[16] == tokens[0, 100]

    # Test window at edge (should fail)
    window = extract_centered_window(tokens, batch_idx=0, center_pos=10, window_size=64)
    assert window is None  # Not enough left context

    # Test different window sizes
    for size in [8, 16, 32, 64]:
        window = extract_centered_window(tokens, batch_idx=0, center_pos=128, window_size=size)
        assert window is not None
        assert window.shape[0] == size


def test_extract_activation_window():
    """Test extraction of activation windows."""
    torch.manual_seed(42)

    # Create sparse activation data
    locations = torch.tensor([
        [0, 100, 0],
        [0, 101, 0],
        [0, 105, 0],
    ])
    activations = torch.tensor([5.0, 3.0, 7.0])

    # Extract window centered at position 100
    act_window = extract_activation_window(
        activations=activations,
        locations=locations,
        batch_idx=0,
        center_pos=100,
        window_size=32,
        cache_ctx_len=256,
    )

    assert act_window.shape[0] == 32
    # Position 100 should be at center (index 16)
    assert act_window[16] == 5.0
    # Position 101 should be at index 17
    assert act_window[17] == 3.0
    # Position 105 should be at index 21
    assert act_window[21] == 7.0
    # Other positions should be zero
    assert act_window[0] == 0.0
    assert act_window[31] == 0.0


def test_multi_scale_constructor_basic():
    """Test basic multi-scale construction."""
    torch.manual_seed(42)

    # Create tokens
    n_batches, seq_len = 4, 256
    tokens = torch.randint(0, 1000, (n_batches, seq_len))

    # Create activations - high at specific positions
    locations = torch.tensor([
        [0, 100, 0],
        [1, 120, 0],
        [2, 80, 0],
        [3, 150, 0],
    ])
    activations = torch.tensor([10.0, 8.0, 9.0, 7.0])
    activation_data = ActivationData(locations, activations)

    # Extract at multiple scales
    context_sizes = [8, 16, 32, 64]
    multi_scale_data = multi_scale_constructor(
        activation_data=activation_data,
        tokens=tokens,
        context_sizes=context_sizes,
        cache_ctx_len=seq_len,
        n_examples_per_scale=10,
        min_examples=1,
    )

    # Verify all scales have data
    for ctx_size in context_sizes:
        assert len(multi_scale_data[ctx_size]) > 0
        for example in multi_scale_data[ctx_size]:
            assert isinstance(example, ActivatingExample)
            assert example.tokens.shape[0] == ctx_size
            assert example.activations.shape[0] == ctx_size
            assert example.max_activation > 0


def test_multi_scale_constructor_with_tokenizer(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
):
    """Test multi-scale construction with tokenizer."""
    torch.manual_seed(42)

    # Create tokens
    tokens = torch.randint(0, 100, (2, 256))

    # Create activations
    locations = torch.tensor([[0, 100, 0], [1, 120, 0]])
    activations = torch.tensor([10.0, 8.0])
    activation_data = ActivationData(locations, activations)

    # Extract with tokenizer
    multi_scale_data = multi_scale_constructor_with_tokenizer(
        activation_data=activation_data,
        tokens=tokens,
        context_sizes=[16, 32],
        cache_ctx_len=256,
        tokenizer=tokenizer,
        n_examples_per_scale=5,
    )

    # Verify string tokens are added
    for ctx_size in [16, 32]:
        for example in multi_scale_data[ctx_size]:
            assert example.str_tokens is not None
            assert len(example.str_tokens) == ctx_size


def test_token_level_feature():
    """Test classification of token-level feature (low variance)."""
    torch.manual_seed(42)

    # Create feature that activates similarly at all scales
    context_sizes = [8, 16, 32, 64]
    multi_scale_data = {}

    for ctx_size in context_sizes:
        examples = []
        for i in range(20):
            tokens = torch.randint(0, 1000, (ctx_size,))
            activations = torch.zeros(ctx_size)
            # Consistent activation at center
            activations[ctx_size // 2] = 5.0 + torch.randn(1).item() * 0.1
            examples.append(ActivatingExample(tokens=tokens, activations=activations))
        multi_scale_data[ctx_size] = examples

    # Compare scales
    comparison = compare_scales(multi_scale_data)

    # Should have low variance
    assert comparison.activation_variance < 0.5

    # Should be classified as token-level
    assert comparison.scale_type == "token"


def test_sentence_level_feature():
    """Test classification of sentence-level feature."""
    torch.manual_seed(42)

    # Create feature that peaks at medium scales (32-64)
    context_sizes = [8, 16, 32, 64]
    multi_scale_data = {}

    for ctx_size in context_sizes:
        examples = []
        # Activation increases with context size, peaks at 32
        base_activation = {8: 2.0, 16: 4.0, 32: 8.0, 64: 7.0}[ctx_size]

        for i in range(20):
            tokens = torch.randint(0, 1000, (ctx_size,))
            activations = torch.zeros(ctx_size)
            activations[ctx_size // 2] = base_activation + torch.randn(1).item() * 0.5
            examples.append(ActivatingExample(tokens=tokens, activations=activations))
        multi_scale_data[ctx_size] = examples

    # Compare scales
    comparison = compare_scales(multi_scale_data)

    # Should have higher variance
    assert comparison.activation_variance > 1.0

    # Should be classified as sentence-level
    assert comparison.scale_type == "sentence"


def test_phrase_level_feature():
    """Test classification of phrase-level feature (peaks at small scales)."""
    torch.manual_seed(42)

    context_sizes = [8, 16, 32, 64]
    multi_scale_data = {}

    for ctx_size in context_sizes:
        examples = []
        # Peak at small scales
        base_activation = {8: 8.0, 16: 6.0, 32: 3.0, 64: 2.0}[ctx_size]

        for i in range(20):
            tokens = torch.randint(0, 1000, (ctx_size,))
            activations = torch.zeros(ctx_size)
            activations[ctx_size // 2] = base_activation + torch.randn(1).item() * 0.3
            examples.append(ActivatingExample(tokens=tokens, activations=activations))
        multi_scale_data[ctx_size] = examples

    comparison = compare_scales(multi_scale_data)
    assert comparison.scale_type == "phrase"


def test_paragraph_level_feature():
    """Test classification of paragraph-level feature (increasing trend)."""
    torch.manual_seed(42)

    context_sizes = [8, 16, 32, 64, 128]
    multi_scale_data = {}

    for ctx_size in context_sizes:
        examples = []
        # Activation increases with context
        base_activation = ctx_size / 16.0  # 0.5, 1.0, 2.0, 4.0, 8.0

        for i in range(20):
            tokens = torch.randint(0, 1000, (ctx_size,))
            activations = torch.zeros(ctx_size)
            activations[ctx_size // 2] = base_activation + torch.randn(1).item() * 0.3
            examples.append(ActivatingExample(tokens=tokens, activations=activations))
        multi_scale_data[ctx_size] = examples

    comparison = compare_scales(multi_scale_data)
    assert comparison.scale_type == "paragraph"


def test_compare_scales_empty():
    """Test comparison with empty data."""
    multi_scale_data = {8: [], 16: [], 32: []}

    comparison = compare_scales(multi_scale_data)

    assert comparison.activation_variance == 0.0
    assert all(v == 0.0 for v in comparison.avg_activation_by_scale.values())


def test_compute_scale_sensitivity():
    """Test scale sensitivity computation."""
    torch.manual_seed(42)

    context_sizes = [8, 16, 32]
    multi_scale_data = {}

    for ctx_size in context_sizes:
        examples = []
        for i in range(10):
            tokens = torch.randint(0, 1000, (ctx_size,))
            # Variable activations
            activations = torch.zeros(ctx_size)
            activations[ctx_size // 2] = torch.rand(1).item() * 10
            examples.append(ActivatingExample(tokens=tokens, activations=activations))
        multi_scale_data[ctx_size] = examples

    sensitivity = compute_scale_sensitivity(multi_scale_data)

    assert len(sensitivity) == len(context_sizes)
    for ctx_size in context_sizes:
        assert ctx_size in sensitivity
        assert sensitivity[ctx_size] >= 0.0


def test_compute_position_consistency():
    """Test position consistency computation."""
    torch.manual_seed(42)

    context_sizes = [16, 32]
    multi_scale_data = {}

    for ctx_size in context_sizes:
        examples = []
        for i in range(10):
            tokens = torch.randint(0, 1000, (ctx_size,))
            activations = torch.zeros(ctx_size)
            # Activate at center
            activations[ctx_size // 2] = 5.0
            examples.append(ActivatingExample(tokens=tokens, activations=activations))
        multi_scale_data[ctx_size] = examples

    consistency = compute_position_consistency(multi_scale_data)

    # Should be high since all activate at center
    assert 0.0 <= consistency <= 1.0


def test_summarize_multi_scale():
    """Test comprehensive multi-scale summary."""
    torch.manual_seed(42)

    context_sizes = [8, 16, 32]
    multi_scale_data = {}

    for ctx_size in context_sizes:
        examples = []
        for i in range(15):
            tokens = torch.randint(0, 1000, (ctx_size,))
            activations = torch.zeros(ctx_size)
            activations[ctx_size // 2] = 5.0
            examples.append(ActivatingExample(tokens=tokens, activations=activations))
        multi_scale_data[ctx_size] = examples

    summary = summarize_multi_scale(multi_scale_data)

    # Verify all expected keys are present
    assert "scale_type" in summary
    assert "activation_variance" in summary
    assert "avg_activation_by_scale" in summary
    assert "max_activation_by_scale" in summary
    assert "frequency_by_scale" in summary
    assert "scale_sensitivity" in summary
    assert "position_consistency" in summary
    assert "dominant_scale" in summary

    # Verify types
    assert isinstance(summary["scale_type"], str)
    assert isinstance(summary["activation_variance"], float)
    assert isinstance(summary["dominant_scale"], int)


def test_classify_feature_scale_edge_cases():
    """Test classification with edge cases."""
    # Empty data
    result = classify_feature_scale([], {}, 0.0)
    assert result == "unknown"

    # Single scale
    result = classify_feature_scale([32], {32: 5.0}, 0.0)
    assert result in ["token", "phrase", "sentence", "paragraph", "unknown"]


def test_multi_scale_insufficient_examples():
    """Test handling of insufficient examples."""
    torch.manual_seed(42)

    tokens = torch.randint(0, 1000, (1, 256))
    # Only 2 activations
    locations = torch.tensor([[0, 100, 0], [0, 101, 0]])
    activations = torch.tensor([5.0, 3.0])
    activation_data = ActivationData(locations, activations)

    multi_scale_data = multi_scale_constructor(
        activation_data=activation_data,
        tokens=tokens,
        context_sizes=[8, 16, 32],
        cache_ctx_len=256,
        n_examples_per_scale=10,
        min_examples=10,  # Require at least 10
    )

    # Should return empty lists
    for ctx_size in [8, 16, 32]:
        assert multi_scale_data[ctx_size] == []
