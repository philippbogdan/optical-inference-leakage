# Reproduce the numbers

## CPU: check the included evidence

Use a recent Python interpreter and install the small CPU requirements:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
```

The final command needs no network or model files. It regenerates the reported CSV tables in `reproduced/`, checks them against `results/`, and exits unsuccessfully with a named mismatch if an input checksum, quality result, leakage, floor, ratio, attack estimate, matrix value or fitted prediction disagrees. Float comparisons use relative tolerance `1e-9` and absolute tolerance `1e-11`. It compares the complete top README and results README with prose generated from the calculations, including the long-text floor ratios, attack percentages and held-out prediction error. The complete set takes minutes on a laptop CPU.

To check one measured run:

```sh
python reproduce.py --run Qwen3-0.6B_0.005
```

That checks the selected run in full and checks the size fit using all four strict-quality runs. It writes the selected run's rows, the size-fit table and the optical moment check. All input checksums and both READMEs are still checked. In this partial mode, prose about other runs uses their saved floors and attack estimates; those are freshly recomputed only by a full run. The output directory can be changed with `--output`.

**What this regenerates:** P0 information from the measured noise variances and mean squares; the quality interval from saved perplexity draws; all zero-budget floors, one-text approximate-switching and honest floors from saved gradients and ranks; monotone budget floors and ratios; a new seeded simulation of the homodyne attack; the leave-one-model-out fit; matrix-type SNRs; and the small optical moment check. This is arithmetic reproduction of measured inputs, not a new measurement of the language models.

**One indispensable stored input:** `test.floor_honest_K` was calculated from activation spectra on eight random-token texts. Those spectra were not retained. The CPU command imports the number and checks downstream results; it cannot independently verify that floor. Rows depending on it say `stored_honest_K` in `results/floors.csv`. These determine the nonzero-budget floors at a million tokens. The reported floors used 256 gradient probes, with no completed higher-probe measurement.

The small failure and formula checks run separately:

```sh
python -m unittest discover -s tests -v
```

## GPU: remeasure from public inputs

One CUDA GPU with BF16 support is required for a full run. The 8B case is estimated to need about 48 to 64 GB of GPU memory. This is a capacity estimate, not a measured peak. Runs are sequential, one model and quality level at a time. The pinned inputs were publicly downloadable without sign-in when fetched for these measurements.

```sh
python -m pip install -r requirements-gpu.txt
python -m optical_inference_leakage.fetch
python -m optical_inference_leakage.gpu Qwen3-0.6B 0.005 reruns/llm_Qwen3-0.6B_0.005.json --final
```

`optical_inference_leakage.fetch` retrieves all four pinned Qwen3 Base snapshots and the pinned WikiText-2 validation and test files, verifying the test-file SHA-256 recorded in [optical_inference_leakage/paths.py](optical_inference_leakage/paths.py). Use `--model Qwen3-0.6B` to fetch a single model. Cache files live under `.cache/` by default. After downloading, set `HF_HUB_OFFLINE=1` to require cached inputs. No unpinned model fallback is used.

To run the complete measurement set:

```sh
for model in Qwen3-0.6B Qwen3-1.7B Qwen3-4B Qwen3-8B; do
  for level in 0.005 0.02 0.1; do
    python -m optical_inference_leakage.gpu "$model" "$level" "reruns/llm_${model}_${level}.json" --final || exit 1
  done
done
```

To remeasure a saved operating point, skipping allocation tuning:

```sh
python -m optical_inference_leakage.gpu Qwen3-0.6B 0.005 reruns/rechecked.json --final \
  --from-run data/runs/llm_Qwen3-0.6B_0.005.json
```

New measurements retain `test.spectra_K` as conservative eigenvalue values and multiplicities, alongside the ranks. They also write the gradient probe count and seed as `test.nhutf` and `test.gseed`. Preserve these files when comparing fresh measurements. The included run files do not store the probe count or seed; [data/methods.json](data/methods.json) records those settings. The gradient vectors alone cannot verify them. The GPU software environment used for the measurements was not recorded, so fresh runs may differ with software and numerical kernels. The versions pinned in [requirements-gpu.txt](requirements-gpu.txt) have been exercised with the offline small-model check; full-size runs with those versions have not been verified.

An offline code-path check creates a small random CPU model and random token sequences:

```sh
python -m optical_inference_leakage.gpu Qwen3-0.6B 0.005 reruns/smoke.json --smoke --final
```

It exercises tuning, noisy products, gradients, final scoring, ranks, spectra and floors. Its output is marked `random_cpu_smoke`; its model label is only a command-line choice and it is not a Qwen3 measurement. The CPU evidence checker never treats it as one.

## Measurement conventions

WikiText-2 raw lines are joined with `"\n\n"` and tokenised with each pinned model's tokenizer. The test split yields 299,078 tokens, scored as 146 complete windows of 2,048 tokens with the remainder discarded; these counts are recorded in [data/methods.json](data/methods.json). Each window scores its next-token predictions, without cross-window context. The digital reference is recomputed within each run, which explains small baseline differences between quality levels.

Tuning uses 24 validation windows, 64 gradient probes per allocation update, four updates, four noise draws and a target of 0.8 times the requested allowance. Gradients are shrunk towards their matrix-type mean; variance updates are damped and recalibrated. Only the final stage reads the test split. The reported test quality uses five independent noise draws and the upper end of the two-sided 95% t interval in perplexity, compared with that run's digital perplexity times one plus the allowance.

The final tangent cut uses antithetic Stein gradients of mean NLL, 256 probes, seed 9000 and Bonferroni normal intervals. NLL slack uses the lower normal interval of the log perplexities, with multiplier 1.96. These statistical approximations and the convexity assumption remain separate from the exact arithmetic of the dual bound. See [THEORY.md](THEORY.md).

A million tokens is bounded above for P0 by 489 complete texts. Floors use 488 complete texts that fit inside the token budget. These deliberately different counts are retained. Do not merge them or count the two optical links as separate measurements: back-to-back and 5 dB loss each way with 0.01 shot-noise units of excess noise give identical P0 values because nothing returns and loss rescales the client's signal in the bright regime.

The attack samples 20,000 modes per matrix, each with 64 Gaussian pairs, fitting the estimator on half and evaluating on half. Independent modes share one SNR under uniform encoding, allowing scaling to each matrix's full size. Tied views use inverse-variance combination. Reported attack intervals are approximate normal intervals across sampled modes. The log-MSE statistic is a finite-sample plug-in estimate and can slightly exceed capacity; it is not a certified extracted-information floor. The observed near-equality is a simulation result.

The optical check uses a three-mode instance, Fock cutoff 14 and four quadrature points per mode, with the seed and parameters in `data/methods.json`. Its relative error is the larger of the maximum absolute entry errors in the mean and covariance, each divided by the largest absolute reference entry in that array. This tests a small instance, not every network product.

The size fit uses only the four 0.5% runs. For each held-out model, least squares fits `log(Gamma_eff) = log(a) + b log(N)` to the other models, where `Gamma_eff = 2**(2*L1) - 1`. It predicts both counts with `0.5*log2(1 + K*a*N**b)`. The endpoint predictions extrapolate beyond the other three sizes. This is held-out prediction on a small family; it does not establish behaviour on other architectures or unmeasured sizes.

## Figure data

### Real-weight hero

This separate light CPU export needs the downloaded Qwen3-0.6B-Base model.safetensors at revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. Keep the downloaded weights and their source licence together outside the repository; an unchanged [Apache-2.0 licence copy](figures/QWEN-LICENSE.txt) is bundled with the rendered hero. This export is not part of the model-free check above.

```sh
python -m pip install torch==2.12.1 safetensors==0.8.0
python -m optical_inference_leakage.real_copy_data \
  --weights /path/to/qwen3-0.6b/model.safetensors \
  --output ../figure-pack
```

The output defaults to `../figure-pack`; destinations inside the repository are rejected, including symlink redirects and multiply linked output files. The command verifies the complete weight-file SHA-256 `cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba` and stored run checksum before extracting `model.layers.6.self_attn.k_proj.weight[384:512,128:256]`. Indices are zero-based and stop-exclusive. It checks the full 1024 by 1024 BF16 matrix against the stored size and mean square, with relative tolerance `1e-6` for the original floating-point reduction.

The crop was chosen for visible row bands using row/column RMS dispersion on contiguous 128-square grid crops of untied projections whose stored P0 level lies between 5.8 and 6.5 bits per weight. Selection used no simulated noise. NumPy PCG64 seed `20261101` generates one i.i.d. standard-normal field; copies are `W + sigma*G` with no shrinkage. Using the same field across levels isolates the change in noise strength. These are hypothetical channels, not a sequence of accumulated texts.

At 0.100, 1.000, 3.000, 5.895 bits per weight, the noise standard deviations are 0.0723110668, 0.0160989378, 0.00351307625, 0.000468745777, respectively, in the original weight units. The last variance is taken directly from the stored P0 one-text run at the 0.5% quality allowance, without rounding the bit level.

The export writes `real-weight-copy.npz` (`original`, `copy`, `bits_per_weight`, `noise_variance`, `standard_normal`), `real-weight-copy-errors.csv` (four panels' measured and expected errors) and `real-weight-copy-manifest.json` (identity, hashes, seed, shapes, channel formulas and shared colour scale). Load with `allow_pickle=False`. The crop is `(128,128)`, copies are `(4,128,128)`, and every panel has 16,384 entries. Copies remain float64 rather than being rounded to BF16. The plain-copy expected RMS is `sigma=sqrt(s)/sqrt(2**(2*b)-1)`; the final sigma is `sqrt(v)` from the stored run. The CSV reports both full-matrix and actual crop RMS normalisations. The sampled RMS must agree with sigma within `6/sqrt(2*16384)`, about six standard errors for Gaussian noise.

No plots or model evaluations are produced. The renderer uses `original` then `copy` for `figures/bits-per-weight.png`; the numerical arrays remain outside the repository. All panels use the fixed grayscale mapping `asinh(weight/sqrt(s))` between `asinh(-20)` and `asinh(20)`, without clipping or per-panel rescaling. The weight-derived image retains [Apache-2.0](figures/QWEN-LICENSE.txt) with [Qwen credit](THIRD_PARTY.md). The lower leak levels have no measured whole-model quality guarantee; the full-model information and quality results remain those of the stored runs.

### Expected-error animation and supporting figures

Export the motion data from the stored runs without rerunning floors or attack simulations:

```sh
python -m optical_inference_leakage.hero_data --output ../figure-pack
```

This checks input checksums, every value in results/matrices.csv, unique weight counts and both stored leakage endpoints, then writes weight-copy-matrices.csv, weight-copy-curves.csv, weight-copy-timeline.csv, weight-copy.npz and weight-copy-manifest.json. The NPZ contains K, rms_pct and leakage_bits_per_weight arrays; the matrix CSV identifies their columns and the manifest records dimensions, units, formulas and source hashes. The timeline fixes the frame-to-text mapping. No plots or model weights are generated.

To export that pack and the supporting matrix-noise.csv and p0-floor.csv while checking the full result set:

```sh
python reproduce.py --figure-data ../figure-pack
```

This full export also regenerates the available floors and attack estimates before checking all tables and prose. A single-run check still validates the motion arithmetic at every text count for every stored run, but does not export the complete figure pack.

The motion uses 100/sqrt(K*Gamma) for expected copy error as a percentage of matrix weight RMS, with Gamma=s/v for separate views and summed SNRs for tied observations. The static matrix atlas keeps both tied views separately and uses 100*sqrt(v/s); its public table is [matrices.csv](results/matrices.csv). The motion follows repeated independent texts, not the allocation-tuning history. Neither visual depicts actual weights. See [the interpretation](results/README.md#the-clients-copy).

Across 12 runs, the matrix table has 2,712 views and the motion pack has 2,703 unique matrix and allowance pairs. It covers 489 text counts and 5,868 leakage values; the supporting floor pack has 96 comparisons.
