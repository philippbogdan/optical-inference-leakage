# Sources and third-party terms

Our code, measured numerical results, text and original figures are dedicated under [CC0 1.0](LICENSE). Qwen weight-derived imagery retains [Apache-2.0](figures/QWEN-LICENSE.txt). Numerical model weights and crops, tokenizer files, model configurations, dataset text and dependency source code are not included. Downloading them does not change their own terms.

## Public inputs

| Model | Pinned revision | Terms recorded at the source |
|---|---|---|
| [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base/tree/da87bfb608c14b7cf20ba1ce41287e8de496c0cd) | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` | [Apache-2.0](figures/QWEN-LICENSE.txt) |
| [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base/tree/ea980cb0a6c2ae4b936e82123acc929f1cec04c1) | `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` | Apache-2.0 |
| [Qwen3-4B-Base](https://huggingface.co/Qwen/Qwen3-4B-Base/tree/906bfd4b4dc7f14ee4320094d8b41684abff8539) | `906bfd4b4dc7f14ee4320094d8b41684abff8539` | Apache-2.0 |
| [Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base/tree/49e3418fbbbca6ecbdf9608b4d22e5a407081db4) | `49e3418fbbbca6ecbdf9608b4d22e5a407081db4` | Apache-2.0 |

Credit for the real-weight hero: the Qwen team, Qwen3-0.6B-Base, at the revision above, under [Apache-2.0](figures/QWEN-LICENSE.txt). It depicts a crop of the key projection in layer 6, with simulated Gaussian noise added at four leak levels to make the client copies. Those transformations are ours; they do not replace Qwen's terms with CC0. The source model.safetensors SHA-256 is `cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba`. Qwen's unchanged licence is bundled at [figures/QWEN-LICENSE.txt](figures/QWEN-LICENSE.txt); numerical crops and copies are exported outside this repository.

[WikiText-2 raw](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3) is pinned at `b08601e04326c79dfdd32d625aee71d232d685c3`. Its Hub tags state CC BY-SA 3.0 and GFDL, while its dataset-card prose says CC BY-SA 4.0. That inconsistency is unresolved here; no corpus text is redistributed. The paths fetched are `wikitext-2-raw-v1/validation-00000-of-00001.parquet` and `wikitext-2-raw-v1/test-00000-of-00001.parquet`. The test SHA-256 recorded in [optical_inference_leakage/paths.py](optical_inference_leakage/paths.py) is `5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`; no validation checksum is recorded, so its immutable revision is the retrieval constraint.

Credit: S. Merity, C. Xiong, J. Bradbury and R. Socher, [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843), 2016.

## Dependencies

| Dependency | Terms |
|---|---|
| NumPy, SciPy | BSD-3-Clause |
| PyTorch, pandas | BSD-3-Clause |
| Transformers, huggingface_hub, safetensors | Apache-2.0 |
| Apache Arrow / PyArrow | Apache-2.0 |

The requirements install these dependencies separately; their distributions carry the corresponding notices. The CPU requirements pin the versions used to check the included tables. GPU requirements pin the versions used for the small offline execution check. The GPU software environment used for the measurements was not recorded.

## Optical protocol context

K. Sulimany, S. K. Vadlamani, R. Hamerly, P. Iyengar and D. Englund, [Quantum-Secure Multiparty Deep Learning](https://doi.org/10.1103/k8wg-qmbh), *Physical Review X* 15, 041056 (2025), [arXiv:2408.05629](https://arxiv.org/abs/2408.05629), CC BY 4.0. P0 shares the use of coherent light to carry weights for client-side optical computation. The paper's return-light protocol lies outside the bound claimed here. No article text or figures are copied.
