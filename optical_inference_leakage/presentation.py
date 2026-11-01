"""Write the public results and their captions from checked numerical tables."""
from . import cases, real_copy_data


def headline(rows):
    strict = [r for r in rows if r['level'] == 0.005]
    values = [r['L1'] for r in strict]
    return (f'Sending Qwen3 Base weights as light exposes {min(values):.1f} to {max(values):.1f} '
            'bits per weight after one 2,048-token text while keeping WikiText-2 perplexity '
            'within 0.5% of digital BF16. Preliminary simulations.')


def figure_count_sentence(tables, scaling):
    unique = {(r['model'], r['level'], r['shared_weight_key']) for r in tables['matrices']}
    return (f"Across {len(tables['summary'])} runs, the matrix table has {len(tables['matrices']):,} views "
            f"and the motion pack has {len(unique):,} unique matrix and allowance pairs. "
            f"It covers 489 text counts and {len(tables['summary']) * 489:,} leakage values; "
            f"the supporting floor pack has {len(tables['floors'])} comparisons.")


def hero_schedule(tables):
    matrix = next(r for r in tables['matrices'] if r['model'] == real_copy_data.MODEL
                  and r['level'] == 0.005 and r['key'] == real_copy_data.MATRIX_KEY)
    return real_copy_data.noise_schedule(matrix['mean_square'], matrix['noise_variance'])


def hero_noise_sentence(tables):
    bits, variances = hero_schedule(tables)
    levels = ', '.join(f'{b:.3f}' for b in bits)
    sigmas = ', '.join(f'{v ** 0.5:.9g}' for v in variances)
    return (f'At {levels} bits per weight, the noise standard deviations are {sigmas}, '
            'respectively, in the original weight units. The last variance is taken directly '
            'from the stored P0 one-text run at the 0.5% quality allowance, without rounding the bit level.')


def readme_page(tables, scaling):
    strict = [r for r in tables['summary'] if r['level'] == 0.005]
    one = [r['L1'] for r in strict]
    many = [r['Lmillion'] for r in strict]
    ratios = {}
    for count in ('1', '1e6'):
        values = [r['ratio'] for r in tables['floors'] if r['model'] in cases.LLMS[:2] and r['count'] == count]
        ratios[count] = (min(values), max(values))
    attacks = [100 * r['fraction'] for r in tables['attacks']]
    fit_error = 100 * max(abs(r['relative_error']) for r in scaling)
    p0_bits = hero_schedule(tables)[0][-1]
    return f'''# Know what your optical inference client learns

Secure inference lets a client get a model's answers without revealing its data, while the owner wants to keep the model's learned parameters, or weights, private. Optical protocols send those weights as light and use the client's optics and detectors to multiply them by its data. Light's quantum noise, together with any added noise, limits what the client can learn about the weights under every measurement quantum mechanics allows. In the simple protocol studied here, bright light makes shot noise negligible, so deliberately added Gaussian noise sets the leakage. Our preliminary Qwen3 Base simulations keep WikiText-2 perplexity within 0.5% of digital BF16 while exposing {min(one):.1f} to {max(one):.1f} bits per weight after one 2,048-token text, already close to our conditional minimum for the smaller models.

![A 128 by 128 real Qwen3-0.6B crop with horizontal bands, followed by simulated noisy copies at 0.1, 1, 3 and {p0_bits:.3f} bits per weight. The bands emerge as the noise falls; the last panel is P0 after one text.](figures/bits-per-weight.png)

*A real Qwen3-0.6B crop (left) sits beside copies simulated at 0.1, 1, 3 and {p0_bits:.3f} bits per weight, all on the same colour scale. The last copy uses P0's stored noise for one text at the 0.5% quality allowance.*

[Matrix, noise and error checks](results/README.md#the-clients-copy)

The client's text stays private because nothing returns to the server.

For the 0.6B and 1.7B models, the simple protocol is within about **{100 * (ratios['1'][1] - 1):.0f}% of our conditional minimum after one text**, and **{100 * (ratios['1e6'][1] - 1):.0f}% after a million tokens**. Under the bound's assumptions, doing much better needs a different kind of protocol, such as one where light returns to the server.

## More use gives the client a sharper copy

After a million tokens, leakage reaches **{min(many):.2f} to {max(many):.2f} bits per weight** at the 0.5% quality allowance. Averaging the repeated copies cuts their expected error by about **{cases.counts()[-1][1] ** 0.5:.0f} times** over that span.

The server sends a fresh noisy copy of the weights for each text, so privacy comes at the cost of transmitting the model repeatedly. This no-return protocol, called P0, belongs to the coherent-light weight-delivery family of Sulimany et al., [Quantum-Secure Multiparty Deep Learning, PRX (2025)](https://doi.org/10.1103/k8wg-qmbh).

[![Expected copy-error bands narrow across all four models and three quality settings as the text count rises; leakage curves below rise with use.](figures/weight-copy.gif)](figures/weight-copy.mp4)

*Each band shows the expected error in the client's copy of every matrix, shrinking as texts accumulate; no model weights are pictured in this animation. The curves show how much information that copy exposes.*

[Watch the MP4](figures/weight-copy.mp4) · [Read the values and assumptions](results/README.md#the-clients-copy)

## How much room is left to improve?

For the smaller models, protocol leakage divided by the conditional minimum ranges from **{ratios['1'][0]:.3f} to {ratios['1'][1]:.3f} after one text**, and **{ratios['1e6'][0]:.3f} to {ratios['1e6'][1]:.3f} after a million tokens**, across every tested quality allowance and data budget. The larger-model bounds leave more room; they do not demonstrate a better protocol.

![Four model panels compare leakage with the conditional minimum across data budgets; the smaller models sit closest to one.](figures/p0-floor.png)

*Closer to 1 means less room to improve within the protocol family covered by the bound. Filled points show one text and open points a million tokens; colour and shape distinguish the quality allowances.*

[All comparisons and their limits](results/README.md#conditional-floor-gaps)

## Where the noise goes

![Four matrix-noise maps show layers downwards and projection types across, with brighter tiles for noisier copies.](figures/matrix-noise.png)

*Brighter tiles mean more noise in a matrix, with each triplet showing the 0.5%, 2% and 10% quality allowances from left to right. Layers run downwards, and embedding and output-head views sit below them.*

[Every matrix value](results/matrices.csv) · [How to read the map](results/README.md#matrix-noise-map)

A simulated client extracts **{min(attacks):.2f}% to {max(attacks):.2f}%** of the calculated information bound under its Gaussian weight model. A fit using the other three model sizes predicts each held-out model's leakage within **{fit_error:.1f}%**. [Attack and prediction evidence](results/README.md#attack-and-size-prediction)

## What this does not show

- A hardware demonstration or an independently checked proof: these are preliminary simulations and conditional arguments.
- A minimum for every optical protocol: the bound assumes fresh, weight-independent errors uniform within each matrix, restricted optical roles and quality assumptions detailed in [THEORY.md](THEORY.md); return-light, shaped-noise and client-encoded-light protocols sit outside it.
- Recovery of a full trained model file: the hero simulates one real crop, and the information attack uses a Gaussian weight model.
- Independent recovery of every floor input: the long-text floors at nonzero data budgets use stored measurements whose activation spectra were not retained; [the limitations](results/README.md#conditional-floor-gaps) also explain the weak larger-model bounds.

## How to use it

Use the [result tables](results/README.md) to compare leakage at your quality target, or start from the [measured runs](data/README.md). [REPRODUCE.md](REPRODUCE.md) gives the commands to check the numbers on a laptop, export figure data or rerun measurements on one GPU. Port the calculation through [optical_inference_leakage](optical_inference_leakage/README.md), retaining the assumptions. Our code, measurements, text and original figures are free to use under [CC0 1.0](LICENSE); the Qwen weight imagery retains [Apache-2.0](figures/QWEN-LICENSE.txt), with [Qwen credit and other third-party terms](THIRD_PARTY.md).
'''


def result_page(tables, scaling):
    lines = ['# Measured quality and leakage', '',
        'All values below are regenerated by [the CPU command](../REPRODUCE.md). The twelve rows are separate model and quality pairs, not separate optical links. Perplexity uses the upper end of a two-sided 95% t interval over five independent noise draws. Every row meets its own target. The baseline was recomputed per run.', '',
        '| Qwen3 Base | Allowance | Digital | Upper 95% | Target | Bits/weight, one text | Bits/weight, million tokens |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in tables['summary']:
        lines.append(f"| {r['model'].replace('Qwen3-', '')} | {100*r['level']:g}% | {r['digital_ppl']:.4f} | {r['upper95_ppl']:.4f} | {r['target_ppl']:.4f} | {r['L1']:.3f} | {r['Lmillion']:.3f} |")
    lines += ['', '## Conditional floor gaps', '',
        'Ranges of P0 divided by the monotone floor across all quality allowances and budgets. Ratio 1 means the bound meets P0 within the restricted class and assumptions in [THEORY.md](../THEORY.md). The long-text nonzero-budget floors depend on stored honest-client measurements with missing spectra.', '',
        'The floor covers fresh i.i.d. errors uniform within each matrix, weight-blind error laws, weights supplied only through server light and client data supplied only through its own optics and detection. Fresh output-preserving transformations have additional rank and matching limitations. Convex expected loss in variance and nonnegative true derivatives are required; three sampled log-variance directions cannot establish these assumptions. Extending the result to non-Gaussian i.i.d. laws assumes Gaussian noise is quality-optimal, which is untested on these models. Return-light, per-weight shaped-noise and client-encoded optical data protocols are excluded. [Full argument and assumptions](../THEORY.md).', '',
        'All floors use 256 gradient probes; no higher-probe measurement was completed. Uncertain gradients and rank-deficient views weaken the larger-model bounds. At a million tokens, P0 uses 489 texts and floors use 488 complete texts. The nonzero-budget floors at that count depend on test.floor_honest_K, a stored honest-client measurement from eight random-token texts whose activation spectra were not retained. Its downstream arithmetic can be checked on CPU, but that input cannot be independently regenerated from the included files.', '',
        'In the figure, the smaller models occupy the top row and the larger ones the bottom row. The equally spaced budget categories are 0, 0.0001, 0.01 and 1 bit per token, with small series offsets. The shared linear ratio axis has a reference at 1. Blue circles, orange squares and green triangles distinguish the allowances; filled points and solid lines mean one text, open points and dashed lines a million tokens. Larger gaps expose limits of the bound, not an observed protocol improvement.', '',
        '| Qwen3 Base | One text | Million tokens |', '|---|---:|---:|']
    for model in cases.LLMS:
        ranges = []
        for count in ('1', '1e6'):
            values = [r['ratio'] for r in tables['floors'] if r['model'] == model and r['count'] == count]
            ranges.append(f'{min(values):.3f} to {max(values):.3f}')
        lines.append(f"| {model.replace('Qwen3-', '')} | {' | '.join(ranges)} |")
    zero = max(r['ratio'] for r in tables['floors'] if r['budget'] == 0)
    worst = max(abs(r['relative_error']) for r in scaling)
    lines += ['', f'At zero data budget alone, the largest ratio is {zero:.3f} across both counts. For 4B, only the 0.5% one-text cases at nonzero budgets exceed 2. For 8B, one-text ratios exceed 2 at 0.5% for nonzero budgets, and at 2% and 10% for budgets 0.01 and 1.', '',
        f'The held-out size fit has maximum relative error {100*worst:.2f}%, at the 4B one-text case. It uses four sizes at the strictest quality allowance, with fitted exponents between {min(r["b"] for r in scaling):.2f} and {max(r["b"] for r in scaling):.2f}. This is a fit, not a derivation.', '',
        '## Files', '',
        '| File | Contents |', '|---|---|',
        '| [summary.csv](summary.csv) | Full-precision quality, targets and P0 leakage |',
        '| [floors.csv](floors.csv) | Every budget and count, raw and monotone floors, ratios, stored-input labels |',
        '| [attacks.csv](attacks.csv) | Simulated information, approximate intervals and fraction of P0 |',
        '| [scaling.csv](scaling.csv) | Held-out predictions, coefficients and relative errors |',
        '| [snr.csv](snr.csv) | Weight-weighted geometric SNR per matrix type and copy RMS error |',
        '| [matrices.csv](matrices.csv) | Every optical matrix view, its shape, noise variance and relative noise RMS |',
        '| [diagnostics.csv](diagnostics.csv) | Zero gradient endpoints with both counting conventions; matrix-type approximation errors |',
        '| [convexity.csv](convexity.csv) | Sampled curvature estimates, standard errors and significance flags |',
        '| [fidelity.csv](fidelity.csv) | Relative moment errors on the small optical instance |', '',
        'The attack estimates reach 99.99% to 100% of the P0 values when rounded to two decimal places. Sampling uses 20,000 modes per matrix and scales their i.i.d. contributions; the estimates and intervals are Monte Carlo quantities, not certified finite-sample bounds. One geometric-mean SNR per matrix type predicts leakage within 1.6% for every case. At the strictest allowance, type SNRs are approximately 1,200 to 83,000 except the separate 8B input embedding, at 14.1. The small optical moment check has maximum array-normalised relative error 0.00043.', '']
    lines += ['', "## The client's copy", '',
        'The hero uses rows [384:512] and columns [128:256] of model.layers.6.self_attn.k_proj.weight in pinned Qwen3-0.6B: a contiguous 128 by 128 crop with 16,384 weights in their original order. It was selected for visible horizontal bands by comparing row and column RMS dispersion across grid crops before drawing noise. It is a structural illustration, not a representative sample of all weights. Every panel uses the same monotone grayscale mapping, asinh(weight/sqrt(s)), with fixed bounds asinh(-20) and asinh(20); no entries are clipped.', '',
        'The displayed copies are plain observations W + sigma*G, without shrinkage or denoising. One fixed i.i.d. standard-normal field G is scaled across the four hypothetical leak levels, so differences isolate noise strength; the panels are not successive texts. For full-matrix mean square s, b = 0.5*log2(1+s/variance), giving sigma = sqrt(s)/sqrt(2**(2*b)-1). The smallest level is 0.1 because zero bits would require infinite noise in this unshrunk channel. Linear shrinkage would instead suppress the low-information copy towards zero, changing what is being illustrated.', '',
        hero_noise_sentence(tables), '',
        'The bit labels describe the Gaussian channel at the full matrix power, not measured mutual information in this selected trained crop. Its power differs from s. The separate [real-weight export](../REPRODUCE.md#real-weight-hero) retains the common noise field, reports errors relative to both matrix and crop RMS, and checks sampled RMS against sigma. The model-free CPU command checks the bit-to-noise arithmetic and numerical prose from the stored run; it does not open the downloaded weights.', '',
        'Only the final channel uses the stored whole-model quality operating point. The lower leak levels are illustrative noise settings, not measured quality-preserving protocols. The figure does not remeasure perplexity, demonstrate optical hardware or full-model recovery, or establish bitwise equality with BF16 weights. [Qwen](../THIRD_PARTY.md) supplies the underlying weights under [Apache-2.0](../figures/QWEN-LICENSE.txt).', '',
        'The motion shows expected error, calculated from the stored mean squares and noise variances at every integer text count from 1 to 489. It shows no trained weights, synthetic weights or sampled error realisations. Each band extends from minus to plus one RMS error about zero estimation error; it is not a confidence interval or a claim that every weight falls inside it.', '',
        'For an independent matrix view, Gamma = s/v and relative RMS error is 1/sqrt(K*Gamma), measured against the matrix weight RMS, not the magnitude of an individual weight. The copy is the unbiased average of repeated noisy observations. For tied embedding and head views, add their SNRs before computing the error: they are observations of the same weights. The motion has one band entry per unique matrix; its bands narrow by sqrt(489), about 22 times, from start to finish.', '',
        'The bands use fixed symmetric logarithmic error axes, linear within 0.01% of zero, spanning -60% to 60%. The horizontal matrix index follows layers and projection types, then the embedding and head; it is not a weight index. Every matrix has equal horizontal width, regardless of its weight count. The lower curves use L(K) = sum_l N_l/(2*N)*log2(1+K*Gamma_l), with weights counted once. All four models and all three allowances appear, with no smoothing, fitted values or inferred measurements between quality settings.', '',
        'K=489 is the upper text count used to bound a million tokens: 489 complete texts would contain 1,001,472 tokens. The floor uses 488 complete texts. The animation counter therefore displays texts, not an exact million-token total. Its 489 data frames run at 24 frames per second with a one-second hold at each end, for 537 playback frames and 22.375 seconds. This is playback time, not optical processing time. [Export the data](../REPRODUCE.md#figure-data).', '',
        'Leakage is a worst-case bright-limit Holevo upper bound over client strategies and weight priors with fixed matrix mean squares. Tied weights count once; biases and normalisation parameters are excluded. Bits per weight are information, not a fraction of a BF16 file recovered. These simulations do not establish a distribution over trained model weights or demonstrate optical hardware. [Definitions and derivation](../THEORY.md).', '',
        '## Matrix-noise map', '',
        'The four blocks run left to right through 0.6B, 1.7B, 4B and 8B. Rows are layer indices increasing downwards; blank space below the shorter models means those layers do not exist. Columns group q, k, v, o, gate, up and down projections. Triplets run left to right through 0.5%, 2% and 10% allowance, including the embedding and head below the layers.', '',
        'The shared logarithmic colour scale spans 0.2% to 60% noise RMS relative to weight RMS: brighter tiles mean more noise. All 2,712 optical views appear. Equal tile area counts views, not weights or leakage. Tied lookup and head views keep their separate variances here, unlike the combined copy estimate in the animation.', '',
        '## Attack and size prediction', '',
        'The homodyne attack fits a linear estimator on its own simulated samples. It uses 20,000 modes per matrix with 64 Gaussian pairs each, and scales their i.i.d. contributions. The percentages in the main README are rounded to two decimals; approximate 95% Monte Carlo intervals are in attacks.csv. The log-MSE estimate can cross capacity at finite sample size. Near-saturation under the worst-case Gaussian weight prior is not exact saturation, a certified finite-sample information floor or recovery of trained model files.', '',
        'Each size prediction uses the other three of only four model sizes at the 0.5% allowance, at both text counts. Error is 100*(predicted/measured - 1): positive values overpredict and negative values underpredict. There is no prediction interval. The endpoint predictions extrapolate beyond their three fitting sizes; this fitted relation is not a derived scaling law or evidence for other architectures or unmeasured sizes.', '']
    return '\n'.join(lines)


