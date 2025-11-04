"""
End-to-end test for multi-scale analysis integrated with the full Delphi pipeline.

This runs a complete pipeline (cache -> construct -> explain -> score) and then
performs multi-scale analysis on the cached activations.
"""

import asyncio
import time
from pathlib import Path

import torch

from delphi.__main__ import run
from delphi.config import (
    CacheConfig,
    ConstructorConfig,
    MultiScaleConfig,
    RunConfig,
    SamplerConfig,
)
from delphi.utils import base_path_cfg_aware
from delphi.latents import LatentDataset
from delphi.latents.latents import ActivationData
from delphi.latents.loader import TensorBuffer
from delphi.latents.multi_scale_analysis import compare_scales, summarize_multi_scale
from delphi.latents.multi_scale_constructors import multi_scale_constructor
from delphi.log.result_analysis import get_agg_metrics, load_data


async def test():
    """Run full pipeline and test multi-scale analysis on the results."""

    # Configure with multi-scale enabled
    cache_cfg = CacheConfig(
        dataset_repo="EleutherAI/fineweb-edu-dedup-10b",
        dataset_split="train[:5%]",
        dataset_column="text",
        batch_size=8,
        cache_ctx_len=512,  # Large enough to test multiple scales
        n_splits=5,
        n_tokens=2_500_000,
    )

    sampler_cfg = SamplerConfig(
        train_type="quantiles",
        test_type="quantiles",
        n_examples_train=40,
        n_examples_test=50,
        n_quantiles=10,
    )

    constructor_cfg = ConstructorConfig(
        min_examples=90,
        example_ctx_len=32,
        n_non_activating=50,
        non_activating_source="random",
        faiss_embedding_cache_enabled=True,
        faiss_embedding_cache_dir=".embedding_cache",
    )

    # Configure multi-scale analysis
    multi_scale_cfg = MultiScaleConfig(
        # context_sizes=[16, 32, 64, 128],  # Must all divide cache_ctx_len=256
        n_examples_per_scale=50,
        min_examples=10,
        variance_threshold=0.1,
    )

    run_cfg = RunConfig(
        name="test_multi_scale",
        overwrite=["cache", "scores"],
        model="EleutherAI/pythia-160m",
        sparse_model="EleutherAI/sae-pythia-160m-32k",
        hookpoints=["layers.3.mlp"],
        explainer_model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        explainer_model_max_len=4208,
        max_latents=100,
        seed=22,
        num_gpus=torch.cuda.device_count(),
        filter_bos=True,
        verbose=False,
        sampler_cfg=sampler_cfg,
        constructor_cfg=constructor_cfg,
        cache_cfg=cache_cfg,
        multi_scale_cfg=multi_scale_cfg,

    )

    # Run the full pipeline
    print("Running full Delphi pipeline...")
    start_time = time.time()
    await run(run_cfg)
    base_path = base_path_cfg_aware(run_cfg)
    pipeline_time = time.time() - start_time
    print(f"Pipeline completed in {pipeline_time:.2f} seconds")

    # Validate scoring results (standard e2e check)
    scores_path = base_path / "scores"
    latent_df, counts = load_data(scores_path, run_cfg.hookpoints)
    processed_df = get_agg_metrics(latent_df, counts)

    # Performs better than random guessing
    for score_type, df in processed_df.groupby("score_type"):
        accuracy = df["accuracy"].mean()
        assert (
            accuracy > 0.55
        ), f"Score type {score_type} has an accuracy of {accuracy}"
        print(f"Score type {score_type}: accuracy = {accuracy:.3f}")

    # Now test multi-scale analysis on the cached data
    print("\nRunning multi-scale analysis on cached activations...")
    start_time = time.time()

    latents_path = base_path / "latents"

    # Use LatentDataset to properly load cached data
    dataset = LatentDataset(
        raw_dir=latents_path,
        sampler_cfg=sampler_cfg,
        constructor_cfg=constructor_cfg,
    )

    # Load tokens
    tokens = dataset.load_tokens()
    print(f"Loaded tokens: shape {tokens.shape}")

    # Analyze first N latents from the dataset
    n_latents_to_analyze = 10
    multi_scale_results = []

    for i, buffer in enumerate(dataset.buffers):
        if i >= n_latents_to_analyze:
            break

        # Load activations from this buffer
        latents, split_locations, split_activations = buffer.load_data_per_latent()

        for latent_idx, locations, activations in zip(latents, split_locations, split_activations):
            if len(multi_scale_results) >= n_latents_to_analyze:
                break

            # Create ActivationData
            activation_data = ActivationData(locations, activations)

            # Check if enough activations
            if len(activation_data.activations) < multi_scale_cfg.min_examples:
                continue

            # Run multi-scale constructor
            multi_scale_data = multi_scale_constructor(
                activation_data=activation_data,
                tokens=tokens,
                context_sizes=multi_scale_cfg.context_sizes,
                cache_ctx_len=cache_cfg.cache_ctx_len,
                n_examples_per_scale=multi_scale_cfg.n_examples_per_scale,
                min_examples=multi_scale_cfg.min_examples,
            )

            # Check if we got examples at multiple scales
            non_empty_scales = [
                ctx for ctx in multi_scale_cfg.context_sizes if multi_scale_data[ctx]
            ]

            if len(non_empty_scales) >= 2:
                # Run comparison
                comparison = compare_scales(multi_scale_data)
                summary = summarize_multi_scale(multi_scale_data)

                multi_scale_results.append(
                    {
                        "latent_idx": int(latent_idx),
                        "scale_type": summary["scale_type"],
                        "dominant_scale": summary["dominant_scale"],
                        "activation_variance": summary["activation_variance"],
                        "max_growth_ratio": summary["max_growth_ratio"],
                        "max_correlation": summary["max_correlation"],
                    }
                )

                print(f"\nLatent {int(latent_idx)}:")
                print(f"  Scale type: {summary['scale_type']}")
                print(f"  Dominant scale: {summary['dominant_scale']} tokens")
                print(f"  Activation variance: {summary['activation_variance']:.4f}")
                print(f"  Max activation growth ratio: {summary['max_growth_ratio']:.4f}")
                print(f"  Max activation correlation: {summary['max_correlation']:.4f}")

    multi_scale_time = time.time() - start_time
    print(f"\nMulti-scale analysis completed in {multi_scale_time:.2f} seconds")

    # Validate multi-scale results
    assert len(multi_scale_results) > 0, "No latents had sufficient activations for multi-scale analysis"

    print(f"\nSuccessfully analyzed {len(multi_scale_results)} latents with multi-scale")

    # Check diversity of scale types
    scale_types = [r["scale_type"] for r in multi_scale_results]
    print(f"Scale type distribution: {set(scale_types)}")

    # Verify all results have valid scale types
    for result in multi_scale_results:
        assert result["scale_type"] in [
            "token",
            "phrase",
            "sentence",
            "paragraph",
            "unknown",
        ]
        assert result["dominant_scale"] >= 0
        assert result["activation_variance"] >= 0.0
        assert result["max_growth_ratio"] >= 0.0
        assert -1.0 <= result["max_correlation"] <= 1.0

    print("\n✅ End-to-end multi-scale test passed!")
    print(f"Total time: {pipeline_time + multi_scale_time:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(test())
