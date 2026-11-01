"""Export complete matrix views and checked figure data without drawing plots."""
import csv
import math
from . import cases, recompute as rc

PROJECTIONS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')


def matrix_rows(run):
    """Keep both optical views of a tied weight matrix, with their identity explicit."""
    with (rc.ROOT / 'data/shapes.csv').open() as stream:
        shapes = {r['matrix_type']: r for r in csv.DictReader(stream) if r['model'] == run['model']}
    layers = int(shapes['embed']['layers'])
    expected = {'embed', 'head'} | {f'L{layer}.{kind}' for layer in range(layers) for kind in PROJECTIONS}
    keys = run['keys']
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError(f'{run["model"]}: incomplete or repeated matrix views')
    rows = []
    for key, s, v, count in zip(keys, run['s'], run['v'], run['N'], strict=True):
        kind = key.split('.')[-1]
        shape = shapes[kind]
        m, n = int(shape['rows']), int(shape['columns'])
        if count != m * n or not all(math.isfinite(x) and x > 0 for x in (s, v)):
            raise ValueError(f'{run["model"]}/{key}: invalid matrix shape or variance')
        tied = run['tied'] and key in ('embed', 'head')
        rows.append(dict(model=run['model'], level=run['level'], key=key,
            layer=int(key.split('.')[0][1:]) if '.' in key else '', matrix_type=kind,
            layers_in_model=layers, rows=m, columns=n, N_view=count,
            shared_weight_key='embed/head' if tied else key, tied_view=tied,
            mean_square=s, noise_variance=v, view_snr=s / v,
            noise_rms_pct=100 * math.sqrt(v / s)))
    unique = {r['shared_weight_key']: r['N_view'] for r in rows}
    rc.close(sum(unique.values()), run['Ntot'], f'{run["model"]}/matrix unique count')
    return rows


def figure_tables(tables):
    """Return every plotted observation with its units and original identifiers."""
    packed = {'matrix-noise': tables['matrices'], 'p0-floor': tables['floors']}
    expected = {'matrix-noise': sum(len(r['keys']) for r in _identities(tables)),
                'p0-floor': len(cases.LLMS) * len(cases.LEVELS) * len(cases.BUDGETS) * len(cases.counts())}
    for name, rows in packed.items():
        if len(rows) != expected[name]:
            raise ValueError(f'{name}: incomplete figure data')
    return packed


def _identities(tables):
    """Require every model and allowance before exporting the complete visual set."""
    for model in cases.LLMS:
        for level in cases.LEVELS:
            rows = [r for r in tables['matrices'] if r['model'] == model and r['level'] == level]
            if not rows:
                raise ValueError(f'{model}/{level}: missing matrix data')
            keys = [r['key'] for r in rows]
            expected = {'embed', 'head'} | {f'L{i}.{p}' for i in range(rows[0]['layers_in_model']) for p in PROJECTIONS}
            if len(keys) != len(expected) or set(keys) != expected:
                raise ValueError(f'{model}/{level}: incomplete matrix data')
            yield {'keys': keys}
