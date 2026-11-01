"""Regenerate the reported tables on a CPU and fail on any disagreement."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import time
from optical_inference_leakage import cases, figure_data, floors, hero_data, presentation, recompute as rc

ROOT = Path(__file__).resolve().parent


def load_runs():
    manifest = json.loads((ROOT / 'data/checksums.json').read_text())
    for relative, expected in manifest.items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Input checksum mismatch: {relative}')
    runs = []
    for model in cases.LLMS:
        for level in cases.LEVELS:
            path = ROOT / 'data/runs' / f'llm_{model}_{level}.json'
            run = json.loads(path.read_text())
            if run['model'] != model or run['level'] != level:
                raise ValueError(f'Run identity mismatch: {path.name}')
            runs.append(run)
    return runs


def table_rows(runs, detailed=True):
    tables = {key: [] for key in ('summary', 'floors', 'attacks', 'snr', 'convexity', 'diagnostics', 'matrices')}
    for run in runs:
        started = time.perf_counter()
        test, model, level = run['test'], run['model'], run['level']
        base = dict(model=model, level=level)
        tables['matrices'].extend(figure_data.matrix_rows(run))
        upper, target = rc.quality(run)
        bound = {label: rc.capacity(run, K) for label, K in cases.counts()}
        for label, value in bound.items():
            rc.close(value, test['bound'][label], f'{model}/{level}/bound/{label}')
        tables['summary'].append(dict(base, N=int(run['Ntot']), digital_ppl=test['digital_ppl'],
            upper95_ppl=upper, target_ppl=target, L1=bound['1'], Lmillion=bound['1e6'], zero_noise_rel=test['zero_noise_rel']))
        effective = rc.recompute_floors(run) if detailed else rc.monotone(test['floor'])
        for budget in cases.BUDGETS:
            for label, K in cases.counts():
                key = f'{budget}|{label}'
                stored = label == '1e6' and math.isclose(effective[key], test['floor_honest_K'], rel_tol=1e-10)
                tables['floors'].append(dict(base, budget=budget, count=label, K_bound=K,
                    K_floor=1 if label == '1' else 488, p0=bound[label], floor=effective[key],
                    ratio=bound[label] / effective[key], floor_source='stored_honest_K' if stored else 'cpu_recomputed',
                    floor_raw=test['floor'][key]))
        upper_th3 = (1 - floors.tau_bar(1e-4 * cases.WIN)) * test['floor_b0']['1e6']
        if upper_th3 > effective['0.0001|1e6']:
            raise ValueError('The long-text approximate-switching bound needs explicit evaluation')
        for name, values in test['convexity_test'].items():
            sig_concave = values['second_diff'] + 1.645 * values['se'] < 0
            if sig_concave:
                raise ValueError('A sampled direction is significantly concave')
            tables['convexity'].append(dict(base, direction=name, **values, concave_95=sig_concave))
        tables['diagnostics'].append(rc.diagnostics(run))
        types = rc.type_snrs(run)
        for typ, (count, gamma) in types.items():
            tables['snr'].append(dict(base, matrix_type=typ, N=int(count), gamma=gamma,
                relative_rms_one=1 / math.sqrt(gamma), relative_rms_489=1 / math.sqrt(489 * gamma)))
        for label, K in cases.counts():
            predicted = sum(n * 0.5 * math.log2(1 + K * g) for n, g in types.values()) / run['Ntot']
            if abs(predicted / bound[label] - 1) > 0.016:
                raise ValueError('Matrix-type approximation exceeds the reported tolerance')
            if detailed:
                estimate, low, high = rc.attack(run, K)
                fraction = estimate / bound[label]
                if not 0.9998 <= fraction <= 1.0002:
                    raise ValueError('Attack estimate differs from the reported near-saturation')
                tables['attacks'].append(dict(base, count=label, estimate=estimate, low95=low, high95=high,
                    p0=bound[label], fraction=fraction, modes_per_matrix=20_000, samples_per_mode=64))
        if detailed:
            print(f'{model} level={level:g}: verified quality, leakage, floors and attack in {time.perf_counter() - started:.2f} s', flush=True)
    return tables


def csv_text(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def compare_rows(actual, expected, name):
    if len(actual) != len(expected):
        raise ValueError(f'{name}: row count differs')
    for i, (got, want) in enumerate(zip(actual, expected)):
        if set(got) != set(want):
            raise ValueError(f'{name}: columns differ')
        for key, value in got.items():
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                rc.close(float(value), float(want[key]), f'{name}[{i}].{key}')
            elif str(value) != want[key]:
                raise ValueError(f'{name}[{i}].{key}: {value!r} differs from {want[key]!r}')


def check_prose(tables, scaling):
    expected = {'README.md': presentation.readme_page(tables, scaling),
                'results/README.md': presentation.result_page(tables, scaling)}
    for name, generated in expected.items():
        if generated != (ROOT / name).read_text():
            raise ValueError(f'{name}: prose differs from recomputed tables')
    if presentation.figure_count_sentence(tables, scaling) not in (ROOT / 'REPRODUCE.md').read_text():
        raise ValueError('REPRODUCE.md: figure counts differ from recomputed tables')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', choices=[f'{m}_{q}' for m in cases.LLMS for q in cases.LEVELS],
                        help='Verify one run; omit to verify the entire measured set')
    parser.add_argument('--output', type=Path, default=ROOT / 'reproduced', help='Destination for checked tables')
    parser.add_argument('--figure-data', type=Path, help='Export checked CSV and NPZ figure data; requires a full run')
    args = parser.parse_args()
    if args.figure_data and args.run:
        parser.error('--figure-data requires the complete measured set')
    for destination in (args.output, args.figure_data):
        if destination and any(destination.resolve().is_relative_to(ROOT / protected) for protected in ('data', 'results')):
            parser.error('Choose outputs outside the preserved data and reported results')
    started = time.perf_counter()
    runs = load_runs()
    hero_data.build(runs)
    cheap = table_rows(runs, detailed=False)
    sentence = presentation.headline(cheap['summary'])
    selected = [r for r in runs if not args.run or f'{r["model"]}_{r["level"]}' == args.run]
    tables = table_rows(selected)
    tables['scaling'] = rc.scaling(runs)
    tables['fidelity'] = rc.optical_check()
    for name, rows in tables.items():
        with (ROOT / 'results' / f'{name}.csv').open() as stream:
            expected = list(csv.DictReader(stream))
        if args.run and name not in ('scaling', 'fidelity'):
            expected = [r for r in expected if f'{r["model"]}_{float(r["level"])}' == args.run]
        compare_rows(rows, expected, name)
    if args.run:
        with (ROOT / 'results/attacks.csv').open() as stream:
            cheap['attacks'] = [dict(r, fraction=float(r['fraction'])) for r in csv.DictReader(stream)]
        check_prose(cheap, tables['scaling'])
    else:
        check_prose(tables, tables['scaling'])
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        (args.output / f'{name}.csv').write_text(csv_text(rows))
    if args.figure_data:
        args.figure_data.mkdir(parents=True, exist_ok=True)
        for name, rows in figure_data.figure_tables(tables).items():
            (args.figure_data / f'{name}.csv').write_text(csv_text(rows))
        hero_data.export(runs, args.figure_data)
    print(sentence)
    print(f'PASS: {len(selected)} run(s); {time.perf_counter() - started:.2f} s. '
          'The honest-K floor is a stored measurement, not recomputed from spectra.')


if __name__ == '__main__':
    main()
