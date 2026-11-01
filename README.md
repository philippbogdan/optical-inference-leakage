# Know what your optical inference client learns

Secure inference lets a client get a model's answers without revealing its data, while the owner wants to keep the model's learned parameters, or weights, private. Optical protocols send those weights as light and use the client's optics and detectors to multiply them by its data. Light's quantum noise, together with any added noise, limits what the client can learn about the weights under every measurement quantum mechanics allows. In the simple protocol studied here, bright light makes shot noise negligible, so deliberately added Gaussian noise sets the leakage. Our preliminary Qwen3 Base simulations keep WikiText-2 perplexity within 0.5% of digital BF16 while exposing 5.4 to 6.5 bits per weight after one 2,048-token text, already close to our conditional minimum for the smaller models.

![A 128 by 128 real Qwen3-0.6B crop with horizontal bands, followed by simulated noisy copies at 0.1, 1, 3 and 5.895 bits per weight. The bands emerge as the noise falls; the last panel is P0 after one text.](figures/bits-per-weight.png)

*A real Qwen3-0.6B crop (left) sits beside copies simulated at 0.1, 1, 3 and 5.895 bits per weight, all on the same colour scale. The last copy uses P0's stored noise for one text at the 0.5% quality allowance.*

[Matrix, noise and error checks](results/README.md#the-clients-copy)

The client's text stays private because nothing returns to the server.

For the 0.6B and 1.7B models, the simple protocol is within about **28% of our conditional minimum after one text**, and **10% after a million tokens**. Under the bound's assumptions, doing much better needs a different kind of protocol, such as one where light returns to the server.

## More use gives the client a sharper copy

After a million tokens, leakage reaches **9.85 to 10.95 bits per weight** at the 0.5% quality allowance. Averaging the repeated copies cuts their expected error by about **22 times** over that span.

The server sends a fresh noisy copy of the weights for each text, so privacy comes at the cost of transmitting the model repeatedly. This no-return protocol, called P0, belongs to the coherent-light weight-delivery family of Sulimany et al., [Quantum-Secure Multiparty Deep Learning, PRX (2025)](https://doi.org/10.1103/k8wg-qmbh).

[![Expected copy-error bands narrow across all four models and three quality settings as the text count rises; leakage curves below rise with use.](figures/weight-copy.gif)](figures/weight-copy.mp4)

*Each band shows the expected error in the client's copy of every matrix, shrinking as texts accumulate; no model weights are pictured in this animation. The curves show how much information that copy exposes.*

[Watch the MP4](figures/weight-copy.mp4) · [Read the values and assumptions](results/README.md#the-clients-copy)

## How much room is left to improve?

For the smaller models, protocol leakage divided by the conditional minimum ranges from **1.085 to 1.276 after one text**, and **1.043 to 1.100 after a million tokens**, across every tested quality allowance and data budget. The larger-model bounds leave more room; they do not demonstrate a better protocol.

![Four model panels compare leakage with the conditional minimum across data budgets; the smaller models sit closest to one.](figures/p0-floor.png)

*Closer to 1 means less room to improve within the protocol family covered by the bound. Filled points show one text and open points a million tokens; colour and shape distinguish the quality allowances.*

[All comparisons and their limits](results/README.md#conditional-floor-gaps)

## Where the noise goes

![Four matrix-noise maps show layers downwards and projection types across, with brighter tiles for noisier copies.](figures/matrix-noise.png)

*Brighter tiles mean more noise in a matrix, with each triplet showing the 0.5%, 2% and 10% quality allowances from left to right. Layers run downwards, and embedding and output-head views sit below them.*

[Every matrix value](results/matrices.csv) · [How to read the map](results/README.md#matrix-noise-map)

A simulated client extracts **99.99% to 100.00%** of the calculated information bound under its Gaussian weight model. A fit using the other three model sizes predicts each held-out model's leakage within **3.1%**. [Attack and prediction evidence](results/README.md#attack-and-size-prediction)

## What this does not show

- A hardware demonstration or an independently checked proof: these are preliminary simulations and conditional arguments.
- A minimum for every optical protocol: the bound assumes fresh, weight-independent errors uniform within each matrix, restricted optical roles and quality assumptions detailed in [THEORY.md](THEORY.md); return-light, shaped-noise and client-encoded-light protocols sit outside it.
- Recovery of a full trained model file: the hero simulates one real crop, and the information attack uses a Gaussian weight model.
- Independent recovery of every floor input: the long-text floors at nonzero data budgets use stored measurements whose activation spectra were not retained; [the limitations](results/README.md#conditional-floor-gaps) also explain the weak larger-model bounds.

## How to use it

Use the [result tables](results/README.md) to compare leakage at your quality target, or start from the [measured runs](data/README.md). [REPRODUCE.md](REPRODUCE.md) gives the commands to check the numbers on a laptop, export figure data or rerun measurements on one GPU. Port the calculation through [optical_inference_leakage](optical_inference_leakage/README.md), retaining the assumptions. Our code, measurements, text and original figures are free to use under [CC0 1.0](LICENSE); the Qwen weight imagery retains [Apache-2.0](figures/QWEN-LICENSE.txt), with [Qwen credit and other third-party terms](THIRD_PARTY.md).
