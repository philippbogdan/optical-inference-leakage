"""Export a real weight crop and noisy copies at specified Gaussian leak levels."""
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
MATRIX_KEY = 'L6.k_proj'
TENSOR_KEY = 'model.layers.6.self_attn.k_proj.weight'
TENSOR_SHAPE = (1024, 1024)
CROP = (384, 512, 128, 256)
BIT_LEVELS = (0.1, 1.0, 3.0)
RENDERED_FILE = 'figures/bits-per-weight.png'
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
        if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != TENSOR_SHAPE:
            raise ValueError('Unexpected tensor dtype or shape')
        matrix = tensor.float().numpy().astype(np.float64)
    power = float(np.mean(matrix ** 2))
    if matrix.size != count or not np.isclose(power, s, rtol=1e-6, atol=0):
        raise ValueError('The pinned matrix disagrees with the stored weight power or size')
    r0, r1, c0, c1 = CROP
    return matrix[r0:r1, c0:c1].copy(), dict(
        tensor_shape=list(matrix.shape), dtype='BF16', measured_matrix_mean_square=power)


def noise_schedule(s, v, bits=BIT_LEVELS):
    """Append the stored one-text variance to the illustrative Gaussian channels."""
    if not np.isfinite([s, v]).all() or min(s, v) <= 0:
        raise ValueError('Weight power and noise variance must be finite and positive')
    bits = np.asarray(bits, dtype=np.float64)
    if bits.ndim != 1 or not bits.size or not np.isfinite(bits).all() or np.any(bits <= 0):
        raise ValueError('Bit levels must be finite and positive')
    p0_bits = 0.5 * np.log2(1 + s / v)
    if np.any(np.diff(bits) <= 0) or bits[-1] >= p0_bits:
        raise ValueError('Bit levels must increase strictly before the stored P0 level')
    variance = s / np.expm1(2 * bits * np.log(2))
    return np.append(bits, p0_bits), np.append(variance, v)


def simulate(crop, s, v, bits=BIT_LEVELS, seed=SEED):
    """Scale one fixed Gaussian field to isolate noise strength across leak levels."""
    crop = np.asarray(crop, dtype=np.float64)
    if crop.ndim != 2 or not crop.size or not np.isfinite(crop).all():
        raise ValueError('The crop must be a finite, nonempty matrix')
    power = float(np.mean(crop ** 2))
    if power <= 0:
        raise ValueError('The crop must have positive weight power')
    levels, variances = noise_schedule(s, v, bits)
    noise = np.random.Generator(np.random.PCG64(seed)).standard_normal(crop.shape)
    copies = crop[None, :, :] + np.sqrt(variances)[:, None, None] * noise
    rows = []
    for i, (level, variance, copy) in enumerate(zip(levels, variances, copies)):
        rmse = float(np.sqrt(np.mean((copy - crop) ** 2)))
        sigma = float(np.sqrt(variance))
        ratio = rmse / sigma
        if abs(ratio - 1) > 6 / np.sqrt(2 * crop.size):
            raise ValueError(f'b={level}: sampled RMS disagrees with the noise model')
        rows.append(dict(panel=i + 1, bits_per_weight=float(level),
            p0_one_text=i == len(levels) - 1, entries=crop.size,
            stored_mean_square=s, stored_p0_noise_variance=v, crop_mean_square=power,
            noise_variance=float(variance), noise_std=sigma,
            noise_rms_pct_matrix=100 * sigma / np.sqrt(s),
            noise_rms_pct_crop=100 * sigma / np.sqrt(power),
            measured_rmse=rmse, measured_rms_pct_matrix=100 * rmse / np.sqrt(s),
            measured_rms_pct_crop=100 * rmse / np.sqrt(power),
            observed_expected_ratio=ratio))
    return dict(original=crop, bits_per_weight=levels, noise_variance=variances,
                standard_normal=noise, copy=copies), rows


def export(weights, destination):
    """Write only data and provenance, with all numerical weights outside the repository."""
    destination = output_directory(destination)
    checksums = json.loads((ROOT / 'data/checksums.json').read_text())
    run_hash = verify_sha256(ROOT / RUN_PATH, checksums[RUN_PATH])
    run = json.loads((ROOT / RUN_PATH).read_text())
    s, v, count = parameters(run)
    crop, metadata = extract_crop(weights, s, count)
    pack, rows = simulate(crop, s, v)
    display_max = float(max(np.max(abs(pack[name])) for name in ('original', 'copy')) / np.sqrt(s))
    if display_max > 20:
        raise ValueError('The shared colour limits would clip a displayed weight')
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / FILES[0], **pack)
    write_csv(destination / FILES[1], rows)
    row_rms = np.sqrt(np.mean(crop ** 2, axis=1))
    column_rms = np.sqrt(np.mean(crop ** 2, axis=0))
    manifest = dict(
        model=MODEL, revision=REVISION, weights_sha256=WEIGHTS_SHA256,
        weights_credit='Qwen team', weights_terms='Apache-2.0',
        weights_license_file='figures/QWEN-LICENSE.txt',
        source_sha256={RUN_PATH: run_hash}, matrix_key=MATRIX_KEY, tensor_key=TENSOR_KEY,
        crop=dict(row_start=CROP[0], row_stop=CROP[1], column_start=CROP[2], column_stop=CROP[3],
                  indexing='zero-based, stop-exclusive; no reordering'),
        **metadata, stored_mean_square=s, crop_mean_square=float(np.mean(crop ** 2)),
        stored_p0_noise_variance=v, p0_bits_per_weight=float(pack['bits_per_weight'][-1]),
        seed=SEED, generator='PCG64', numpy_version=np.__version__,
        bits_per_weight=pack['bits_per_weight'].tolist(), quality_allowance=run['level'],
        displayed_array='copy', estimator='None: unshrunk noisy observations W + sigma(b)*G.',
        sampling='One fixed i.i.d. standard-normal field, scaled across hypothetical channels. These are not successive texts.',
        interpretation='Gaussian-channel information at full-matrix power s, not measured information in this selected trained crop.',
        crop_selection='Maximum row/column RMS dispersion on contiguous 128-square grid crops of untied projections with stored P0 levels 5.8 to 6.5 bits. No noise used in selection.',
        crop_structure=dict(row_rms_min=float(min(row_rms)), row_rms_max=float(max(row_rms)),
            column_rms_min=float(min(column_rms)), column_rms_max=float(max(column_rms)),
            score=float(max(np.std(row_rms), np.std(column_rms)) / np.sqrt(np.mean(crop ** 2)))),
        formulas=dict(bits='0.5*log2(1+s/variance)', variance='s/(2**(2*b)-1)',
                      p0_variance='Stored v used directly, without rounding b',
                      expected_rmse='sqrt(variance)', relative_rmse='sqrt(variance/s)',
                      validation='Sampled RMS within 6/sqrt(2*crop.size) of its expectation'),
        colour=dict(weight_scale=float(np.sqrt(s)), transform='asinh(weight/sqrt(s))',
                    limits_in_weight_rms=[-20, 20], linear_width_in_weight_rms=1,
                    max_abs_displayed_in_weight_rms=display_max, clipped_entries=0,
                    palette='Monotone grayscale, negative dark and positive light'),
        array_shapes={name: list(array.shape) for name, array in pack.items()},
        pack_sha256={name: sha256(destination / name) for name in FILES[:2]},
        rendered_files=[RENDERED_FILE],
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
        print(f"b={row['bits_per_weight']:.9f}: sigma={row['noise_std']:.12g}; "
              f"{row['noise_rms_pct_matrix']:.6f}% of matrix RMS; "
              f"sample/theory {row['observed_expected_ratio']:.6f}")
    print('PASS: pinned weights and run verified; sampled errors agree with theory. Data written outside the repository.')


if __name__ == '__main__':
    main()
