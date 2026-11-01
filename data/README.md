# Measurements and their limits

The twelve files in [runs](runs) contain the measurements. Each pairs one Qwen3 Base model with one relative perplexity allowance. [shapes.csv](shapes.csv) contains only the matrix dimensions needed for the CPU calculation, transcribed from the pinned model configurations. The configuration files, model weights and WikiText-2 text are not redistributed. [checksums.json](checksums.json) fixes the included inputs.

| Field in a run | Meaning |
|---|---|
| `model`, `level`, `Ntot`, `tied` | Model, relative quality allowance, unique product-weight count, embedding sharing |
| `keys`, `s`, `N`, `v` | Aligned matrix views, mean squares, element counts and tuned noise variances |
| `iters`, `best_iter`, `tune_digital_ppl` | Validation allocation history and selected iterate |
| `test.digital_ppl`, `draws`, `unfavourable` | Per-run digital baseline, independent noisy perplexities, upper interval end |
| `test.zero_noise_ppl`, `zero_noise_rel` | Digital recovery through the zero-noise simulator |
| `test.g_test`, `g_test_se`, `slack_nll` | Measured gradient estimates, standard errors and NLL slack for the cut |
| `test.floor_b0`, `floor_1e-4_one`, `floor_honest_one` | Floors recomputable from saved measurements |
| `test.ranks_one_text`, `nq_texts` | Measured activation ranks on the random-token texts |
| `test.floor_honest_K` | Stored long-text honest floor; the activation spectra needed to recompute it were not retained |
| `test.bound`, `floor`, `ratio` | Reported leakage, raw floors before monotonicity, and corresponding raw ratios |
| `test.convexity_test` | Three sampled log-variance curvature diagnostics |

The two tied views have separate variances and appear separately in `N`; use `Ntot` for the unique denominator. The CPU command combines their SNRs for P0 and uses convex mixtures for the floor. It recalculates the raw floors, checks them, then applies monotonicity across budgets. Consequently the published [floor table](../results/floors.csv) can improve on a run's raw `test.floor`.

The [matrix atlas table](../results/matrices.csv) retains every optical view at every allowance. `noise_rms_pct` is `100*sqrt(v/s)`; `shared_weight_key` identifies tied lookup/head views and `N_view` is each view's size, not an additive unique-weight count. Blank layer indices denote the embedding and head, which sit outside the transformer layers. No missing layer is filled with a zero or an interpolated value.

[methods.json](methods.json) records the measurement settings, including 256 gradient probes and seed 9000. The run files do not store the probe count or seed, and the gradient vectors alone cannot verify those settings. New runs write both fields into their JSON output. The small-network context and GPU memory estimate are labelled separately from the regenerated results.

The CPU calculation checks arithmetic from saved measurements. It does not remeasure perplexity, gradients or ranks, or recreate the missing activation spectra. It preserves the stored honest-K floor as an input and labels every published row that uses it.
