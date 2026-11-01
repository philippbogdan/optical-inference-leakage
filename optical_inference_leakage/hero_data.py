"""Derive the client's copy error and leakage from stored matrix measurements."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from . import cases, figure_data, recompute as rc

LAST_K = 489
FPS = 24
HOLD_FRAMES = 24


def copy_matrices(views):
    """Combine independent observations of shared weights by adding precision."""
    groups = {}
    for view in views:
        groups.setdefault(view['shared_weight_key'], []).append(view)
    rows = []
    for key, group in groups.items():
        first = group[0]
        for other in group[1:]:
            for field in ('model', 'level', 'rows', 'columns', 'N_view'):
                if other[field] != first[field]:
                    raise ValueError(f'{key}: inconsistent tied {field}')
            rc.close(other['mean_square'], first['mean_square'], f'{key}/tied mean square')
        gamma = sum(r['view_snr'] for r in group)
        rows.append(dict(model=first['model'], level=first['level'], key=key,
            layer=first['layer'], matrix_type='embed/head' if len(group) > 1 else first['matrix_type'],
            rows=first['rows'], columns=first['columns'], N=first['N_view'],
            views=';'.join(r['key'] for r in group), gamma=gamma,
            rms_pct_K1=100 / np.sqrt(gamma), rms_pct_K489=100 / np.sqrt(LAST_K * gamma)))
    def order(row):
        if row['layer'] != '':
            return (0, row['layer'], figure_data.PROJECTIONS.index(row['matrix_type']))
        return (1, 0 if row['matrix_type'] in ('embed', 'embed/head') else 1, 0)
    return sorted(rows, key=order)


def build(runs):
    """Return exact integer-count arrays, with no weight samples or fitted curves."""
    identities = [(model, level) for model in cases.LLMS for level in cases.LEVELS]
    by_id = {(r['model'], r['level']): r for r in runs}
    if len(runs) != len(identities) or set(by_id) != set(identities):
        raise ValueError('The motion data requires every model and allowance exactly once')
    matrices, curves, series = [], [], []
    counts = np.arange(1, LAST_K + 1, dtype=np.int64)
    errors, leakages = [], []
    for series_index, identity in enumerate(identities):
        run = by_id[identity]
        rows = copy_matrices(figure_data.matrix_rows(run))
        rc.close(sum(r['N'] for r in rows), run['Ntot'], f'{identity}/unique weights')
        gamma = np.array([r['gamma'] for r in rows])
        weights = np.array([r['N'] for r in rows], dtype=float) / run['Ntot']
        rms = 100 / np.sqrt(counts[:, None] * gamma[None, :])
        leakage = (0.5 * np.log2(1 + counts[:, None] * gamma[None, :])) @ weights
        for index, label in ((0, '1'), (-1, '1e6')):
            rc.close(leakage[index], run['test']['bound'][label], f'{identity}/hero/{label}')
        start = len(matrices)
        for matrix_index, row in enumerate(rows):
            matrices.append(dict(column=len(matrices), series_index=series_index,
                                 matrix_index=matrix_index, **row))
        series.append(dict(series_index=series_index, model=run['model'], level=run['level'],
                           column_start=start, column_stop=len(matrices), N=run['Ntot']))
        for K, value in zip(counts, leakage, strict=True):
            curves.append(dict(series_index=series_index, model=run['model'], level=run['level'],
                               K=int(K), bits_per_weight=float(value)))
        errors.append(rms)
        leakages.append(leakage)
    rms = np.concatenate(errors, axis=1)
    leakage = np.stack(leakages, axis=1)
    if not (np.isfinite(rms).all() and (rms > 0).all() and (np.diff(rms, axis=0) < 0).all()
            and np.isfinite(leakage).all() and (np.diff(leakage, axis=0) > 0).all()):
        raise ValueError('Non-finite or non-monotone copy arithmetic')
    indices = np.concatenate((np.zeros(HOLD_FRAMES, dtype=int), np.arange(LAST_K),
                              np.full(HOLD_FRAMES, LAST_K - 1, dtype=int)))
    timeline = [dict(frame=i, time_seconds=i / FPS, data_index=int(j), K=int(counts[j]))
                for i, j in enumerate(indices)]
    return dict(matrices=matrices, curves=curves, timeline=timeline, series=series,
                K=counts, rms_pct=rms, leakage_bits_per_weight=leakage)


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def export(runs, destination):
    """Write a data-only pack and its provenance; rendering is a separate step."""
    destination = Path(destination)
    for protected in ('data', 'results'):
        if destination.resolve().is_relative_to(rc.ROOT / protected):
            raise ValueError('Choose a destination outside preserved inputs and results')
    pack = build(runs)
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    for key in ('matrices', 'curves', 'timeline'):
        path = destination / f'weight-copy-{key}.csv'
        write_csv(path, pack[key])
        files.append(path)
    path = destination / 'weight-copy.npz'
    np.savez_compressed(path, K=pack['K'], rms_pct=pack['rms_pct'],
                        leakage_bits_per_weight=pack['leakage_bits_per_weight'])
    files.append(path)
    sources = ['data/shapes.csv'] + [f'data/runs/llm_{r["model"]}_{r["level"]}.json' for r in runs]
    manifest = dict(
        source_sha256={name: hashlib.sha256((rc.ROOT / name).read_bytes()).hexdigest() for name in sources},
        pack_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        series=pack['series'],
        array_shapes={key: list(pack[key].shape) for key in ('K', 'rms_pct', 'leakage_bits_per_weight')},
        csv_rows={key: len(pack[key]) for key in ('matrices', 'curves', 'timeline')},
        fps=FPS, duration_seconds=len(pack['timeline']) / FPS,
        formulas=dict(rms_pct='100/sqrt(K*gamma)', gamma='sum(s/v) over independent views of shared weights',
                      leakage='sum(N_l/(2*N)*log2(1+K*gamma_l))'),
        interpretation='Bright-limit expected RMS of the unbiased averaged copy, not a realised reconstruction or a confidence interval.',
        counts='K=489 is the upper text count for a million tokens; it represents 1,001,472 complete-text tokens. Floors use 488.',
        rendered_files=['figures/weight-copy.mp4', 'figures/weight-copy.gif'])
    (destination / 'weight-copy-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Destination for the data-only motion pack')
    args = parser.parse_args()
    checksums = json.loads((rc.ROOT / 'data/checksums.json').read_text())
    for relative, expected in checksums.items():
        if hashlib.sha256((rc.ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Input checksum mismatch: {relative}')
    runs = [json.loads((rc.ROOT / 'data/runs' / f'llm_{model}_{level}.json').read_text())
            for model in cases.LLMS for level in cases.LEVELS]
    matrix_values = [row for run in runs for row in figure_data.matrix_rows(run)]
    with (rc.ROOT / 'results/matrices.csv').open() as stream:
        expected = list(csv.DictReader(stream))
    if len(matrix_values) != len(expected):
        raise ValueError('Reported matrix table has a different row count')
    for actual, saved in zip(matrix_values, expected, strict=True):
        if set(actual) != set(saved):
            raise ValueError('Reported matrix table has different columns')
        for key, value in actual.items():
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                rc.close(value, float(saved[key]), f'matrices/{actual["key"]}/{key}')
            elif str(value) != saved[key]:
                raise ValueError(f'Matrix identity mismatch: {key}')
    manifest = export(runs, args.output)
    print(f'PASS: motion arithmetic from {len(runs)} stored runs; {manifest["csv_rows"]["matrices"]} matrix rows; '
          f'{LAST_K} integer text counts. No floors or attacks remeasured.')


if __name__ == '__main__':
    main()
