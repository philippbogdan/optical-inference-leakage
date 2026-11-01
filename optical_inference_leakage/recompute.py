"""Recompute leakage, attacks and conditional floors from the included measurements."""
import csv
import json
import math
from pathlib import Path
import numpy as np
from . import attacks, calib, cases, cutfloor as cf, floors

ROOT = Path(__file__).resolve().parents[1]
INPUT_OF = {'q_proj': 'attn_in', 'k_proj': 'attn_in', 'v_proj': 'attn_in',
            'o_proj': 'o_in', 'gate_proj': 'mlp_in', 'up_proj': 'mlp_in', 'down_proj': 'down_in'}


def close(actual, expected, label, rtol=1e-9, atol=1e-11):
    """Raise on a non-finite value or a numerical mismatch, including the field name."""
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol):
        raise ValueError(f'{label}: recomputed {actual!r}, reported {expected!r}')


def shape_map():
    with (ROOT / 'data/shapes.csv').open() as stream:
        return {(r['model'], r['matrix_type']): (int(r['rows']), int(r['columns'])) for r in csv.DictReader(stream)}


def shapes(run):
    table = shape_map()
    return [table[run['model'], k.split('.')[-1]] for k in run['keys']]


def physical_matrices(run):
    """Yield one SNR and weight count per physical matrix, combining tied views."""
    keys = run['keys']
    for i, key in enumerate(keys):
        if run['tied'] and key == 'embed':
            continue
        gamma = run['s'][i] / run['v'][i]
        if run['tied'] and key == 'head':
            j = keys.index('embed')
            gamma += run['s'][j] / run['v'][j]
        yield i, key, run['N'][i], gamma


def capacity(run, K):
    return sum(n * 0.5 * math.log2(1 + K * g) for _, _, n, g in physical_matrices(run)) / run['Ntot']


def quality(run):
    test = run['test']
    upper, _, _ = calib.unfavourable(test['draws'])
    close(upper, test['unfavourable'], 'test.unfavourable')
    target = test['digital_ppl'] * (1 + run['level'])
    if not (upper <= target and test['meets'] and len(test['draws']) == 5):
        raise ValueError('Quality or independent-draw count mismatch')
    close(abs(test['zero_noise_ppl'] / test['digital_ppl'] - 1), test['zero_noise_rel'], 'test.zero_noise_rel')
    close(test['zero_noise_rel'], 0, 'zero-noise identity')
    return upper, target


def raw_floor_table(bound0, th3, honest1, honestK):
    """Construct the stored pre-monotonicity table, retaining the measured honest-K input."""
    result = {}
    for budget in cases.BUDGETS:
        for label in ('1', '1e6'):
            if budget == 0:
                value = bound0[label]
            elif budget == 1e-4:
                value = th3 if label == '1' else max(th3, honestK)
            else:
                value = honest1 if label == '1' else max(honest1, honestK)
            result[f'{budget}|{label}'] = value
    return result


def monotone(raw):
    result = dict(raw)
    for label in ('1', '1e6'):
        best = 0.0
        for budget in reversed(cases.BUDGETS):
            key = f'{budget}|{label}'
            best = max(best, raw[key])
            result[key] = best
    return result


def recompute_floors(run):
    """Rebuild all available cut floors; import only the lost-spectrum honest-K measurement."""
    test = run['test']
    keys, dims = run['keys'], shapes(run)
    s, v, n = (np.asarray(run[k], float) for k in ('s', 'v', 'N'))
    total = run['Ntot']
    if any(a * b != count for (a, b), count in zip(dims, n)):
        raise ValueError('Matrix shapes disagree with the measured weight counts')
    close(sum(count for _, _, count, _ in physical_matrices(run)), total, 'Ntot')
    if test['K_bound'] != 489 or test['K_floor'] != 488 or test.get('nhutf', 256) != 256 or test.get('gseed', 9000) != 9000:
        raise ValueError('Unexpected text counts or gradient settings')
    z = cf.bonferroni_z(len(keys))
    g, se = (np.asarray(test[k], float) for k in ('g_test', 'g_test_se'))
    lo, hi = g - z * se, g + z * se
    nll = np.log(test['draws'])
    slack = max(math.log(test['digital_ppl'] * (1 + run['level'])) -
                (nll.mean() - 1.96 * nll.std(ddof=1) / math.sqrt(len(nll))), 0.0)
    close(slack, test['slack_nll'], 'test.slack_nll')
    weights = []
    for theta in ([0, 0.25, 0.5, 0.75, 1] if run['tied'] else [None]):
        w = np.ones(len(keys))
        if run['tied']:
            w[keys.index('embed')], w[keys.index('head')] = theta, 1 - theta
        weights.append(w)

    def cut(tabs):
        return cf.cut_dual_floor(tabs, v, lo, hi, slack, weights=weights) / total

    b0 = {label: cut([cf.capacity_tab(n[i], s[i], v[i], K) for i in range(len(keys))])
          for label, K in [('1', 1), ('1e6', 488)]}
    for label, value in b0.items():
        close(value, test['floor_b0'][label], f'test.floor_b0.{label}')
    J = sum(dims[i][0] if key == 'embed' else dims[i][1] for i, key in enumerate(keys))
    th3 = floors.theorem3_floor_info(b0['1'] * total, J, 1e-4 * cases.WIN) / total
    close(th3, test['floor_1e-4_one'], 'test.floor_1e-4_one')
    tabs = []
    for i, key in enumerate(keys):
        gamma = s[i] / (v[i] * cf.GRID)
        rows, columns = dims[i]
        if key == 'embed':
            tab = cases.WIN * columns / 2 * np.log2(1 + gamma * rows / cases.WIN)
        else:
            source = 'head_in' if key == 'head' else f"{key.split('.')[0]}.{INPUT_OF[key.split('.')[1]]}"
            rank = max(test['ranks_one_text'][source])
            if rank <= 0 or rank > min(columns, cases.WIN):
                raise ValueError(f'Invalid activation rank: {source}')
            tab = rows * rank / 2 * np.log2(1 + gamma * columns / rank)
        tabs.append(tab)
    honest1 = cut(tabs)
    close(honest1, test['floor_honest_one'], 'test.floor_honest_one')
    raw = raw_floor_table(b0, th3, honest1, test['floor_honest_K'])
    for key, value in raw.items():
        close(value, test['floor'][key], f'test.floor.{key}')
        close(test['bound'][key.split('|')[1]] / value, test['ratio'][key], f'test.ratio.{key}')
    return monotone(raw)


def attack(run, K):
    """Scale a simulated subset of identical-SNR modes to each full matrix."""
    total, variance = 0.0, 0.0
    for i, key, count, gamma in physical_matrices(run):
        noise = run['v'][i]
        if run['tied'] and key == 'head':
            j = run['keys'].index('embed')
            noise = 1 / (1 / noise + 1 / run['v'][j])
        subset = int(min(count, 20_000))
        value, (low, high) = attacks.p0_attack_all_modes(np.full(subset, noise),
            float(subset * run['s'][i]), K=K, S=64, seed=1000 + i)
        scale = count / subset
        total += value * scale
        variance += (scale * (high - low) / (2 * 1.96)) ** 2
    se = math.sqrt(variance)
    return total / run['Ntot'], (total - 1.96 * se) / run['Ntot'], (total + 1.96 * se) / run['Ntot']


def type_snrs(run):
    groups = {}
    for _, key, count, gamma in physical_matrices(run):
        typ = 'embed/head' if run['tied'] and key == 'head' else key.split('.')[-1]
        n, logsum = groups.get(typ, (0, 0))
        groups[typ] = (n + count, logsum + count * math.log(gamma))
    return {key: (n, math.exp(logsum / n)) for key, (n, logsum) in groups.items()}


def scaling(runs):
    selected = [r for r in runs if r['level'] == 0.005]
    rows = []
    for held in selected:
        fitted = [r for r in selected if r['model'] != held['model']]
        x = [math.log(r['Ntot']) for r in fitted]
        y = [math.log(2 ** (2 * capacity(r, 1)) - 1) for r in fitted]
        exponent, loga = np.polyfit(x, y, 1)
        gamma = math.exp(loga) * held['Ntot'] ** exponent
        for label, K in cases.counts():
            predicted = 0.5 * math.log2(1 + K * gamma)
            measured = capacity(held, K)
            error = predicted / measured - 1
            if abs(error) > 0.031:
                raise ValueError(f'Scaling error exceeds reported limit: {held["model"]}')
            rows.append(dict(model=held['model'], count=label, N=int(held['Ntot']), a=math.exp(loga),
                             b=exponent, measured=measured, predicted=predicted, relative_error=error))
    return rows


def diagnostics(run):
    test = run['test']
    lo = np.array(test['g_test']) - cf.bonferroni_z(len(run['keys'])) * np.array(test['g_test_se'])
    n = np.array(run['N'])
    unique_fraction = sum(count for i, _, count, _ in physical_matrices(run) if lo[i] <= 0) / run['Ntot']
    view_fraction = n[lo <= 0].sum() / n.sum()
    types = type_snrs(run)
    errors = [sum(count * 0.5 * math.log2(1 + K * gamma) for count, gamma in types.values()) /
              run['Ntot'] / capacity(run, K) - 1 for _, K in cases.counts()]
    return dict(model=run['model'], level=run['level'], zero_gradient_unique_weight_fraction=unique_fraction,
                zero_gradient_encoded_view_fraction=view_fraction, type_error_one=errors[0], type_error_million=errors[1])


def optical_check():
    """Compare array-normalised moment errors with the retained small-instance specification."""
    from . import fidelity
    settings = json.loads((ROOT / 'data/methods.json').read_text())['fidelity']
    rng = np.random.RandomState(settings['seed'])
    weights = rng.randn(settings['modes']) * settings['weight_scale']
    inputs = rng.randn(settings['modes'], settings['products'])
    mean, covariance = fidelity.exact_readout(weights, inputs, settings['alpha'], settings['variance'],
        cut=settings['fock_cut'], nq=settings['quadrature_points_per_mode'])
    target_mean, target_covariance = fidelity.model_readout(weights, inputs, settings['alpha'], settings['variance'])
    return [dict(mean_relative_error=float(abs(mean - target_mean).max() / abs(target_mean).max()),
                 covariance_relative_error=float(abs(covariance - target_covariance).max() / abs(target_covariance).max()))]
