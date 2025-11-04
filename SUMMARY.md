# Multi-Scale Analysis Summary

## Overview

The `multiscale` branch introduces functionality to analyze how SAE (Sparse Autoencoder) features behave across different context window sizes. The goal is to identify features which activate on different sequence lengths. This enables automatic classification of features as token-level, phrase-level, sentence-level, or paragraph-level based on their activation patterns.

- **Token-level features** activate consistently regardless of context size (e.g., detecting specific words or punctuation)
- **Phrase-level features** activate best with small context windows (8-16 tokens)
- **Sentence-level features** need medium context windows (16-64 tokens)
- **Paragraph-level features** require large context windows (64+ tokens) and show increasing activation with more context

By extracting activations at multiple scales and comparing the patterns, we can attempt to understand their scope.

Understanding the natural scale of SAE features helps:
1. Identify features that capture long-range dependencies vs. local patterns
2. Optimize context window sizes when analyzing specific features
3. Validate that features are capturing the semantic concepts we expect
4. Guide interpretability research by showing which features need more or less context

Note: This feature & document were written with the help of AI assistants like Claude Code.

## Results

We are able to identify that many of the features we tested mainly activate on the phrase level and do not make use of even a 32 token context window.
A small percentage ~10% on a specific layer activate at the sentence (16-64 tokens) level.

We have not tested any features which activate on the paragraph level although this is due to a lack of time.
This work needs to be cleaned up, and thresholds should be moved to the config files instead of sprinkled through the codebase as they are now. The thresholds are also very heuristic.

## How It Works

The implementation consists of three main components:

### 1. Multi-Scale Construction (`delphi/latents/multi_scale_constructors.py`)

This module extracts activation examples at multiple context window sizes. For each strong activation in the cached data:

For each context size (e.g., 8, 16, 32, 64 tokens), extract a window centered on the top activation position

The key functions are:
- `extract_centered_window()`: Extracts token windows centered on activation positions
- `extract_activation_window()`: Extracts sparse activation values for each window
- `multi_scale_constructor()`: Orchestrates the extraction process across all scales

### 2. Scale Comparison & Classification (`delphi/latents/multi_scale_analysis.py`)

This module analyzes how activation patterns change across scales and classifies features:

**Metrics computed for each scale:**
- Average activation magnitude across examples
- Maximum activation values
- Firing frequency (fraction of tokens that activate)

**Classification algorithm:**

The `classify_feature_scale()` function uses multiple signals:

1. **Activation variance**: How much do activations vary across scales?
   - Low variance → likely token-level
   - High variance → context-dependent feature

2. **Growth ratio**: How much does max activation increase from smallest to largest scale?
   - Ratio < 1.2 → doesn't need more context (token-level)
   - Ratio > 1.5 → benefits from more context (paragraph-level)

3. **Correlation**: How strongly do activations correlate with context size?
   - Strong correlation (>0.7) → paragraph-level feature
   - Weak correlation → phrase or sentence-level

4. **Peak detection**: At what context size do activations peak?
   - Peak ≤ 16 tokens → phrase-level
   - Peak ≤ 64 tokens → sentence-level
   - Peak > 64 tokens or increasing trend → paragraph-level

The classifier combines these signals with threshold-based rules to assign each feature to a category. These thresholds are heuristically determined.

### 3. Configuration & Data Structures

**New Configuration (`delphi/config.py`):**

Added `MultiScaleConfig` to make the analysis fully configurable:
- `context_sizes`: Which window sizes to compare (default: [8, 16, 32, 64])
- `n_examples_per_scale`: How many activation examples to extract per scale (default: 50)
- `min_examples`: Minimum examples needed for valid analysis (default: 10)
- `variance_threshold`: Classification threshold for token-level features (default: 0.1)

Integrated into `RunConfig` so multi-scale analysis can be enabled/configured on any Delphi run.

**New Data Structures (`delphi/latents/latents.py`):**

- `MultiScaleExample`: Stores a single activation's examples across all scales
  - Tracks the original position and activation windows at each context size
  - Provides computed properties like `activation_variance` and `dominant_scale`

- `ScaleComparisonResult`: Stores the full analysis results for a feature
  - Contains all metrics (avg/max activation, frequency) for each scale
  - Includes the final `scale_type` classification

## Testing & Validation

The implementation includes comprehensive tests to ensure correctness:

### Unit Tests (`tests/test_latents/test_multi_scale.py`)

Tests cover all core functionality:
- **Window extraction**: Verifies that centered windows are correctly extracted with proper alignment
- **Activation windows**: Tests sparse activation extraction and position mapping
- **Multi-scale construction**: Validates example building across multiple scales
- **Classification logic**: Tests the classification algorithm with synthetic data representing each feature type (token/phrase/sentence/paragraph)
- **Edge cases**: Handles insufficient examples, boundary conditions, etc.

### End-to-End Test (`tests/multi_scale_e2e.py`)

Validates integration with the full Delphi pipeline:
- Runs complete pipeline (cache → construct → explain → score) with multi-scale enabled
- Uses Pythia-160m model and real data (2.5M tokens from FineWeb-Edu)
- Tests multiple context sizes: [8, 16, 32, 64, 128] tokens
- Validates that features are classified into reasonable distributions
- Ensures metrics are computed correctly and fall within expected ranges

### Interactive Exploration (`tests/multi_scale_e2e.ipynb`)

Jupyter notebook for hands-on analysis:
- Demonstrates how to use multi-scale analysis on real latents
- Visualizes activation patterns across scales
- Shows examples of features at each classification level
- Useful for understanding classifier behavior and tuning thresholds


### Key Design Decisions

1. **Top activations**: By default, samples the top-N strongest activations rather than random sampling
2. **Multiple signals for classification**: Uses variance, growth, correlation, and peaks—no single metric determines the class

## Usage Example

Here's how to use multi-scale analysis in your code:

```python
from delphi.config import MultiScaleConfig, RunConfig
from delphi.latents.multi_scale_constructors import multi_scale_constructor
from delphi.latents.multi_scale_analysis import compare_scales, summarize_multi_scale

# Configure multi-scale analysis
multi_scale_cfg = MultiScaleConfig(
    context_sizes=[8, 16, 32, 64, 128],
    n_examples_per_scale=50,
    min_examples=10,
    variance_threshold=0.1,
)

# Extract examples at multiple scales
# (activation_data and tokens come from your cached SAE activations)
multi_scale_data = multi_scale_constructor(
    activation_data=activation_data,
    tokens=tokens,
    context_sizes=multi_scale_cfg.context_sizes,
    cache_ctx_len=512,
    n_examples_per_scale=multi_scale_cfg.n_examples_per_scale,
    min_examples=multi_scale_cfg.min_examples,
)

# Analyze and classify the feature
comparison = compare_scales(multi_scale_data)
summary = summarize_multi_scale(multi_scale_data)

# Results
print(f"Scale type: {summary['scale_type']}")
print(f"Dominant scale: {summary['dominant_scale']} tokens")
print(f"Activation variance: {summary['activation_variance']:.4f}")
print(f"Growth ratio: {summary['max_growth_ratio']:.2f}")
print(f"Correlation: {summary['max_correlation']:.2f}")
```
