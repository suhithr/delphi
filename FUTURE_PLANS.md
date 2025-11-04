# Future Plans for Multi-Scale Analysis Testing

## Additional Metrics for multi_scale_e2e.py

This document outlines potential metrics and validation checks that could be added to the multi-scale end-to-end test to better validate the results of multiple scale analysis.

### 1. Scale Coverage Metrics

Track how well features are distributed across different scale types and whether activations occur across multiple context sizes.

**Metrics to implement:**
- **Proportion of latents at each scale type**: Measure the distribution of features across token/phrase/sentence/paragraph classifications
  ```python
  scale_counts = {st: sum(1 for r in multi_scale_results if r["scale_type"] == st)
                  for st in ["token", "phrase", "sentence", "paragraph", "unknown"]}
  print(f"Scale distribution: {scale_counts}")
  assert scale_counts["unknown"] < len(multi_scale_results) * 0.5, "Too many unknown scales"
  ```

- **Number of scales with examples per latent**: Verify that latents actually activate across multiple scales (not just one)
  ```python
  # During multi-scale analysis, track:
  scales_per_latent = [len(non_empty_scales) for latent in analyzed_latents]
  avg_scales = sum(scales_per_latent) / len(scales_per_latent)
  assert avg_scales >= 2.0, f"Latents should activate at multiple scales, got {avg_scales}"
  ```

- **Empty scales tracking**: Count how many scales produce no activations per latent
  ```python
  empty_scales_per_latent = [
      len([ctx for ctx in context_sizes if not multi_scale_data[ctx]])
      for latent in analyzed_latents
  ]
  ```

### 2. Cross-Scale Activation Patterns

Analyze how activation magnitudes and frequencies change across different context window sizes.

**Metrics to implement:**
- **Activation trend consistency**: Verify that max_activation_by_scale follows expected patterns
  - Paragraph-level features should show increasing or peaked-at-large-scale trends
  - Token-level features should show flat trends (low variance)
  ```python
  for result in multi_scale_results:
      if result["scale_type"] == "token":
          assert result["activation_variance"] < multi_scale_cfg.variance_threshold
      elif result["scale_type"] == "paragraph":
          # Check for increasing trend in activation values
          activations = [result["avg_activation_by_scale"][ctx] for ctx in sorted(context_sizes)]
          # Verify trend is generally increasing
  ```

- **Frequency correlation**: Compare frequency_by_scale values to detect expected relationships
  ```python
  # Higher context sizes may show different firing patterns
  frequency_ratios = {}
  for result in multi_scale_results:
      freq = result["frequency_by_scale"]
      # Compare smallest to largest scale
      if min(context_sizes) in freq and max(context_sizes) in freq:
          ratio = freq[max(context_sizes)] / (freq[min(context_sizes)] + 1e-6)
          frequency_ratios[result["latent_idx"]] = ratio
  ```

- **Scale sensitivity scores**: Use the existing `compute_scale_sensitivity()` function
  ```python
  sensitivity = compute_scale_sensitivity(multi_scale_data)
  # Store and analyze sensitivity patterns
  for ctx_size, sens_score in sensitivity.items():
      print(f"Scale {ctx_size}: sensitivity = {sens_score:.4f}")
  ```

### 3. Statistical Validity Checks

Ensure that computed metrics fall within reasonable and meaningful ranges.

**Metrics to implement:**
- **Min/max activation ranges**: Verify activations are within reasonable bounds
  ```python
  for result in multi_scale_results:
      for ctx_size, max_act in result["max_activation_by_scale"].items():
          assert max_act >= 0.0, f"Negative max activation at scale {ctx_size}"
          assert max_act < 1000.0, f"Unreasonably large activation at scale {ctx_size}"
  ```

- **Variance distribution analysis**: Check that activation_variance has reasonable distribution
  ```python
  variances = [r["activation_variance"] for r in multi_scale_results]
  mean_variance = sum(variances) / len(variances)
  # Should have some variation but not all near zero or all very high
  assert 0.0 < mean_variance < 100.0
  assert min(variances) < max(variances)  # Some diversity expected
  ```

- **Position consistency statistics**: Aggregate and validate position consistency scores
  ```python
  position_scores = [r["position_consistency"] for r in multi_scale_results]
  avg_position_consistency = sum(position_scores) / len(position_scores)
  # Should be reasonably consistent but not perfect
  assert 0.3 < avg_position_consistency < 1.0
  print(f"Average position consistency: {avg_position_consistency:.3f}")
  ```

### 4. Inter-Scale Comparisons

Validate relationships between adjacent scales and dominant scale detection.

**Metrics to implement:**
- **Dominant scale alignment**: Verify dominant_scale matches scale_type classification
  ```python
  for result in multi_scale_results:
      scale_type = result["scale_type"]
      dominant = result["dominant_scale"]

      # Token-level should have consistent activations (no clear dominant)
      # or dominant at smallest scale
      if scale_type == "token":
          # Could be any scale since they're similar
          pass
      elif scale_type == "phrase":
          assert dominant <= 16, f"Phrase-level should peak at ≤16 tokens, got {dominant}"
      elif scale_type == "sentence":
          assert 16 < dominant <= 64, f"Sentence-level should peak at 16-64 tokens, got {dominant}"
      elif scale_type == "paragraph":
          assert dominant > 32, f"Paragraph-level should peak at >32 tokens, got {dominant}"
  ```

- **Activation ratio between adjacent scales**: Compare consecutive scale activations
  ```python
  for result in multi_scale_results:
      avg_acts = result["avg_activation_by_scale"]
      sorted_scales = sorted(avg_acts.keys())

      ratios = []
      for i in range(len(sorted_scales) - 1):
          small = sorted_scales[i]
          large = sorted_scales[i + 1]
          if avg_acts[small] > 0:
              ratio = avg_acts[large] / avg_acts[small]
              ratios.append(ratio)

      # Ratios should be reasonable (not 100x differences)
      for ratio in ratios:
          assert 0.01 < ratio < 100, f"Unreasonable activation ratio: {ratio}"
  ```

- **Peak detection validation**: Ensure features have clear peaks at their dominant scales
  ```python
  for result in multi_scale_results:
      avg_acts = result["avg_activation_by_scale"]
      dominant = result["dominant_scale"]

      # Dominant scale should be at or near the maximum
      max_activation = max(avg_acts.values())
      dominant_activation = avg_acts[dominant]

      # Allow small tolerance for near-ties
      assert dominant_activation >= max_activation * 0.95, \
          f"Dominant scale {dominant} not at peak activation"
  ```

### 5. Quality Assurance Metrics

Verify data quality and that the multi-scale analysis produces meaningful results.

**Metrics to implement:**
- **Example count per scale**: Verify sufficient examples at each scale
  ```python
  # Track during construction
  for latent_idx, multi_scale_data in analyzed_latents:
      for ctx_size, examples in multi_scale_data.items():
          if examples:  # Non-empty scale
              assert len(examples) >= multi_scale_cfg.min_examples, \
                  f"Scale {ctx_size} has insufficient examples: {len(examples)}"
  ```

- **Non-zero activation rate**: Check that features actually fire
  ```python
  for result in multi_scale_results:
      freq = result["frequency_by_scale"]
      # At least one scale should have meaningful firing rate
      assert max(freq.values()) > 0.01, \
          f"Latent {result['latent_idx']} has very low firing rate"
  ```

- **Scale type confidence metrics**: Validate classification decisions
  ```python
  # For token-level features
  token_features = [r for r in multi_scale_results if r["scale_type"] == "token"]
  if token_features:
      # All should have low variance
      for r in token_features:
          assert r["activation_variance"] < multi_scale_cfg.variance_threshold

  # For paragraph-level features
  para_features = [r for r in multi_scale_results if r["scale_type"] == "paragraph"]
  if para_features:
      # Should have high variance and late peak
      for r in para_features:
          assert r["activation_variance"] > multi_scale_cfg.variance_threshold
          assert r["dominant_scale"] >= 32
  ```

### 6. Comprehensive Multi-Scale Summary

Add aggregate statistics across all analyzed latents.

**Metrics to implement:**
- **Overall scale distribution**: Print summary of how many features fall into each category
- **Average metrics by scale type**: Group metrics by scale_type and compute averages
  ```python
  from collections import defaultdict

  metrics_by_type = defaultdict(list)
  for result in multi_scale_results:
      scale_type = result["scale_type"]
      metrics_by_type[scale_type].append({
          "variance": result["activation_variance"],
          "position_consistency": result["position_consistency"],
          "dominant_scale": result["dominant_scale"],
      })

  for scale_type, metrics_list in metrics_by_type.items():
      avg_variance = sum(m["variance"] for m in metrics_list) / len(metrics_list)
      avg_consistency = sum(m["position_consistency"] for m in metrics_list) / len(metrics_list)
      print(f"\n{scale_type.upper()} features:")
      print(f"  Average variance: {avg_variance:.4f}")
      print(f"  Average position consistency: {avg_consistency:.4f}")
  ```

- **Context size utilization**: Report which context sizes are most commonly dominant
  ```python
  dominant_scale_counts = {}
  for result in multi_scale_results:
      dom = result["dominant_scale"]
      dominant_scale_counts[dom] = dominant_scale_counts.get(dom, 0) + 1

  print("\nDominant scale distribution:")
  for ctx_size in sorted(dominant_scale_counts.keys()):
      count = dominant_scale_counts[ctx_size]
      print(f"  {ctx_size} tokens: {count} features ({100*count/len(multi_scale_results):.1f}%)")
  ```

## Implementation Priority

1. **High Priority** (critical for validation):
   - Scale coverage metrics (proportion of scale types, scales per latent)
   - Dominant scale alignment with scale_type
   - Statistical validity checks (ranges, variance distribution)

2. **Medium Priority** (valuable insights):
   - Activation trend consistency
   - Scale sensitivity scores
   - Example count validation

3. **Low Priority** (nice-to-have):
   - Frequency correlation analysis
   - Activation ratios between scales
   - Comprehensive aggregate summaries

## Notes

- Many of these metrics require storing the full `multi_scale_data` dictionary for each analyzed latent, not just the summary statistics
- Consider adding a `--detailed-metrics` flag to optionally run extended validation
- Some metrics may need tuning of thresholds based on empirical observations
- Consider creating a separate test file for these extended metrics if they make the main test too heavy
