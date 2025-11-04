"""
Multi-scale constructor for extracting activations at different context window sizes.
"""

import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from .latents import ActivatingExample, ActivationData


def extract_centered_window(
    tokens: Int[Tensor, "batch seq"],
    batch_idx: int,
    center_pos: int,
    window_size: int,
) -> Int[Tensor, "window_size"] | None:
    """
    Extract a window of tokens centered on a specific position.

    Args:
        tokens: The full token tensor.
        batch_idx: The batch index.
        center_pos: The center position in the sequence.
        window_size: The desired window size.

    Returns:
        Extracted window or None if not enough context.
    """
    seq_len = tokens.shape[1]

    # Calculate window boundaries
    left_context = window_size // 2
    right_context = window_size - left_context

    start_pos = center_pos - left_context
    end_pos = center_pos + right_context

    # Check if window fits within sequence
    if start_pos < 0 or end_pos > seq_len:
        return None

    return tokens[batch_idx, start_pos:end_pos]


def extract_activation_window(
    activations: Float[Tensor, "n_examples"],
    locations: Int[Tensor, "n_examples 3"],
    batch_idx: int,
    center_pos: int,
    window_size: int,
    cache_ctx_len: int,
) -> Float[Tensor, "window_size"]:
    """
    Extract activation window centered on a position.

    Args:
        activations: All activations for this latent.
        locations: All locations [batch_idx, seq_pos, latent_idx].
        batch_idx: The batch index.
        center_pos: The center position.
        window_size: The window size.
        cache_ctx_len: The original cache context length.

    Returns:
        Activation window (zero where no activation).
    """
    left_context = window_size // 2
    right_context = window_size - left_context

    start_pos = center_pos - left_context
    end_pos = center_pos + right_context

    # Create zero tensor for the window
    act_window = torch.zeros(window_size, dtype=activations.dtype)

    # Find activations in this window
    in_batch = locations[:, 0] == batch_idx
    in_window = (locations[:, 1] >= start_pos) & (locations[:, 1] < end_pos)
    mask = in_batch & in_window

    if mask.any():
        # Get positions and values
        positions = locations[mask, 1] - start_pos
        values = activations[mask]
        act_window[positions] = values

    return act_window


def multi_scale_constructor(
    activation_data: ActivationData,
    tokens: Int[Tensor, "batch seq"],
    context_sizes: list[int],
    cache_ctx_len: int,
    n_examples_per_scale: int = 50, # TODO: define this in config
    min_examples: int = 10, # TODO: use number from config
) -> dict[int, list[ActivatingExample]]:
    """
    Construct examples at multiple context window sizes.

    Args:
        activation_data: The activation data (locations and activations).
        tokens: The cached tokens.
        context_sizes: List of context sizes to extract (e.g., [8, 16, 32, 64]).
        cache_ctx_len: The cache context length.
        n_examples_per_scale: Number of examples to extract per scale.
        min_examples: Minimum examples required.

    Returns:
        Dictionary mapping context_size -> list of ActivatingExample.
    """
    locations = activation_data.locations
    activations = activation_data.activations

    # TODO: Doesn't correctly handle when activations don't have enough examples.
    if len(activations) < min_examples:
        return {ctx: [] for ctx in context_sizes}
    
    # TODO: Sample from top activations and also have options to respect random sampling.
    # Get top activations (sorted by magnitude)
    sorted_indices = torch.argsort(activations, descending=True)
    top_indices = sorted_indices[:n_examples_per_scale]

    # Build examples at each scale
    multi_scale_data: dict[int, list[ActivatingExample]] = {
        ctx: [] for ctx in context_sizes
    }

    for idx in top_indices:
        batch_idx = int(locations[idx, 0])
        seq_pos = int(locations[idx, 1])

        for ctx_size in context_sizes:
            # Extract token window
            token_window = extract_centered_window(
                tokens, batch_idx, seq_pos, ctx_size
            )

            if token_window is None:
                continue

            # Extract activation window
            act_window = extract_activation_window(
                activations, locations, batch_idx, seq_pos, ctx_size, cache_ctx_len
            )

            # Create example
            example = ActivatingExample(
                tokens=token_window,
                activations=act_window,
            )

            multi_scale_data[ctx_size].append(example)

    return multi_scale_data


def multi_scale_constructor_with_tokenizer(
    activation_data: ActivationData,
    tokens: Int[Tensor, "batch seq"],
    context_sizes: list[int],
    cache_ctx_len: int,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    n_examples_per_scale: int = 50,
    min_examples: int = 10,
) -> dict[int, list[ActivatingExample]]:
    """
    Construct examples at multiple scales with string tokens.

    Args:
        activation_data: The activation data.
        tokens: The cached tokens.
        context_sizes: List of context sizes.
        cache_ctx_len: The cache context length.
        tokenizer: The tokenizer for decoding.
        n_examples_per_scale: Examples per scale.
        min_examples: Minimum examples required.

    Returns:
        Dictionary mapping context_size -> list of ActivatingExample.
    """
    multi_scale_data = multi_scale_constructor(
        activation_data=activation_data,
        tokens=tokens,
        context_sizes=context_sizes,
        cache_ctx_len=cache_ctx_len,
        n_examples_per_scale=n_examples_per_scale,
        min_examples=min_examples,
    )

    # Add string tokens
    for ctx_size, examples in multi_scale_data.items():
        for example in examples:
            example.str_tokens = tokenizer.batch_decode(example.tokens)

    return multi_scale_data
