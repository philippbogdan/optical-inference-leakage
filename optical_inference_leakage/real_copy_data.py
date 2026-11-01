"""Export a real weight crop and simulated client estimates outside the repository."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .hero_data import write_csv

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'Qwen3-0.6B'
REVISION = 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd'
WEIGHTS_SHA256 = 'cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba'
RUN_PATH = 'data/runs/llm_Qwen3-0.6B_0.005.json'
MATRIX_KEY = 'L14.q_proj'
TENSOR_KEY = 'model.layers.14.self_attn.q_proj.weight'
CROP = (0, 512, 0, 512)
COUNTS = (1, 10, 100, 489)
SEED = 20261101
FILES = ('real-weight-copy.npz', 'real-weight-copy-errors.csv', 'real-weight-copy-manifest.json')


def sha256(path):
    """Hash the complete file without loading it into memory."""
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_sha256(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f'SHA-256 mismatch: {Path(path).name}: {actual}')
    return actual


def output_directory(destination):
    """Reject repository destinations, including paths redirected by symlinks."""
    destination = Path(destination).resolve()
    if destination.is_relative_to(ROOT):
        raise ValueError('Weight-derived data must be written outside the repository')
    for name in FILES:
        path = destination / name
        if path.is_symlink() or path.resolve().is_relative_to(ROOT):
            raise ValueError(f'Unsafe output target: {name}')
        if path.exists() and path.stat().st_nlink > 1:
            raise ValueError(f'Output target has multiple hard links: {name}')
    return destination


def parameters(run):
    """Select the fixed, untied matrix and its stored quality operating point."""
    if (run['model'], run['level']) != (MODEL, 0.005):
        raise ValueError('The real-weight figure requires the pinned strict-quality run')
    index = run['keys'].index(MATRIX_KEY)
    s, v = float(run['s'][index]), float(run['v'][index])
    if not np.isfinite([s, v]).all() or min(s, v) <= 0:
        raise ValueError('Weight power and noise variance must be finite and positive')
    return s, v, int(run['N'][index])


def extract_crop(weights, s, count):
    """Read only the selected matrix after verifying the complete BF16 file."""
    verify_sha256(weights, WEIGHTS_SHA256)
    from safetensors import safe_open
    import torch

    with safe_open(weights, framework='pt', device='cpu') as tensors:
        tensor = tensors.get_tensor(TENSOR_KEY)
        if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != (2048, 1024):
            raise ValueError('Unexpected tensor dtype or shape')
        matrix = tensor.float().numpy().astype(np.float64)
    power = float(np.mean(matrix ** 2))
    if matrix.size != count or not np.isclose(power, s, rtol=1e-6, atol=0):
        raise ValueError('The pinned matrix disagrees with the stored weight power or size')
    r0, r1, c0, c1 = CROP
    return matrix[r0:r1, c0:c1].copy(), dict(
        tensor_shape=list(matrix.shape), dtype='BF16', measured_matrix_mean_square=power)


def simulate(crop, s, v, counts=COUNTS, seed=SEED):
    """Accumulate independent deliveries, retaining nested averages and linear estimates."""
    crop = np.asarray(crop, dtype=np.float64)
    if crop.ndim != 2 or not crop.size or not np.isfinite(crop).all():
        raise ValueError('The crop must be a finite, nonempty matrix')
    if not counts or any(int(k) != k or k <= 0 for k in counts) or tuple(sorted(set(counts))) != tuple(counts):
        raise ValueError('Text counts must be distinct, increasing positive integers')
    if not np.isfinite([s, v]).all() or min(s, v) <= 0:
        raise ValueError('Weight power and noise variance must be finite and positive')
    power = float(np.mean(crop ** 2))
    if power <= 0:
        raise ValueError('The crop must have positive weight power')
    rng = np.random.Generator(np.random.PCG64(seed))
    noise_sum = np.zeros_like(crop)
    averages, estimates, rows = [], [], []
    for k in range(1, int(counts[-1]) + 1):
        noise_sum += rng.normal(0, np.sqrt(v), size=crop.shape)
        if k not in counts:
            continue
        average = crop + noise_sum / k
        gain = s / (s + v / k)
        estimate = gain * average
        average_mse = float(np.mean((average - crop) ** 2))
        estimate_mse = float(np.mean((estimate - crop) ** 2))
        expected_average_mse = v / k
        expected_estimate_mse = (1 - gain) ** 2 * power + gain ** 2 * v / k
        rms_ratio = np.sqrt(average_mse / expected_average_mse)
        linear_ratio = np.sqrt(estimate_mse / expected_estimate_mse)
        if abs(rms_ratio - 1) > 6 / np.sqrt(2 * crop.size) or abs(linear_ratio - 1) > 6 / np.sqrt(2 * crop.size):
            raise ValueError(f'K={k}: sampled RMS disagrees with the noise model')
        rows.append(dict(K=k, entries=crop.size, stored_mean_square=s, noise_variance=v,
            gamma=s / v, crop_mean_square=power, shrinkage=gain,
            average_rmse=np.sqrt(average_mse), estimate_rmse=np.sqrt(estimate_mse),
            average_rms_pct_matrix=100 * np.sqrt(average_mse / s),
            estimate_rms_pct_matrix=100 * np.sqrt(estimate_mse / s),
            average_rms_pct_crop=100 * np.sqrt(average_mse / power),
            estimate_rms_pct_crop=100 * np.sqrt(estimate_mse / power),
            expected_average_rms_pct_matrix=100 * np.sqrt(v / (k * s)),
            expected_average_rms_pct_crop=100 * np.sqrt(v / (k * power)),
            expected_estimate_rms_pct_matrix=100 * np.sqrt(expected_estimate_mse / s),
            gaussian_prior_rms_pct_matrix=100 / np.sqrt(1 + k * s / v),
            average_observed_expected_ratio=rms_ratio,
            estimate_observed_expected_ratio=linear_ratio,
            estimator_change_rms_pct_matrix=100 * np.sqrt(np.mean((estimate - average) ** 2) / s),
            estimate_max_abs_error=float(np.max(abs(estimate - crop)))))
        averages.append(average)
        estimates.append(estimate)
    return dict(K=np.array(counts, dtype=np.int64), original=crop,
                average=np.stack(averages), estimate=np.stack(estimates)), rows


def export(weights, destination):
    """Write data and provenance only; no weights are saved inside the repository."""
    destination = output_directory(destination)
    checksums = json.loads((ROOT / 'data/checksums.json').read_text())
    run_hash = verify_sha256(ROOT / RUN_PATH, checksums[RUN_PATH])
    run = json.loads((ROOT / RUN_PATH).read_text())
    s, v, count = parameters(run)
    crop, metadata = extract_crop(weights, s, count)
    pack, rows = simulate(crop, s, v)
    display_max = float(max(np.max(abs(pack[name])) for name in ('original', 'estimate')) / np.sqrt(s))
    if display_max > 10:
        raise ValueError('The shared colour limits would clip a displayed weight')
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / FILES[0], **pack)
    write_csv(destination / FILES[1], rows)
    manifest = dict(
        model=MODEL, revision=REVISION, weights_sha256=WEIGHTS_SHA256,
        weights_credit='Qwen team', weights_terms='Apache-2.0',
        weights_license_file='figures/QWEN-LICENSE.txt',
        source_sha256={RUN_PATH: run_hash}, matrix_key=MATRIX_KEY, tensor_key=TENSOR_KEY,
        crop=dict(row_start=CROP[0], row_stop=CROP[1], column_start=CROP[2], column_stop=CROP[3],
                  indexing='zero-based, stop-exclusive; no reordering'),
        **metadata, stored_mean_square=s, crop_mean_square=float(np.mean(crop ** 2)),
        noise_variance=v, gamma=s / v, seed=SEED, generator='PCG64', numpy_version=np.__version__,
        text_counts=list(COUNTS), quality_allowance=run['level'],
        displayed_array='estimate', estimator='Population Gaussian-prior linear shrinkage s/(s+v/K) times the cumulative average.',
        sampling='One independent Gaussian noise matrix per text; all panels share the same cumulative trajectory.',
        interpretation='A simulated bright-limit copy of a real BF16 matrix crop, not an optical measurement or a full-model recovery.',
        formulas=dict(average_expected_rms='sqrt(v/K)',
                      average_relative_rms='1/sqrt(K*gamma), normalised by sqrt(stored s)',
                      estimate_expected_mse='(1-gain)^2*crop_mean_square + gain^2*v/K',
                      gaussian_prior_relative_rms='1/sqrt(1+K*gamma)',
                      validation='Both sampled RMS values within 6/sqrt(2*crop.size) of their respective expectations'),
        colour=dict(weight_scale=float(np.sqrt(s)), transform='asinh(weight/sqrt(s))',
                    limits_in_weight_rms=[-10, 10], linear_width_in_weight_rms=1,
                    max_abs_displayed_in_weight_rms=display_max, clipped_entries=0),
        array_shapes={name: list(array.shape) for name, array in pack.items()},
        pack_sha256={name: sha256(destination / name) for name in FILES[:2]},
        rendered_files=['figures/real-weight-copy.png'],
        rendering_terms='Retain Qwen credit and Apache-2.0 terms for weight-derived imagery.')
    (destination / FILES[2]).write_text(json.dumps(manifest, indent=2) + '\n')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True, help='Downloaded pinned model.safetensors')
    parser.add_argument('--output', type=Path, default=Path('../figure-pack'),
                        help='Data destination outside the repository (default: ../figure-pack)')
    args = parser.parse_args()
    for row in export(args.weights, args.output):
        print(f"K={row['K']:3d}: linear estimate RMS {row['estimate_rms_pct_matrix']:.6f}% of matrix RMS; "
              f"{row['estimate_rms_pct_crop']:.6f}% of crop RMS; "
              f"average/theory {row['average_observed_expected_ratio']:.6f}")
    print('PASS: pinned weights and run verified; all sampled errors agree with theory. Data written outside the repository.')


if __name__ == '__main__':
    main()
