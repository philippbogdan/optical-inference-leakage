"""Validation margin and the test perplexity interval."""
import math
import numpy as np
from scipy import stats

TUNE = 0.8


def unfavourable(values, kind='ppl'):
    """Upper end, mean and standard error of a two-sided t interval over noise draws."""
    if kind != 'ppl' or len(values) < 2:
        raise ValueError('Perplexity requires at least two independent draws')
    values = np.asarray(values, float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    return mean + stats.t.ppf(0.975, len(values) - 1) * se, mean, se


def meets(values, digital, kind, level):
    upper, _, _ = unfavourable(values, kind)
    return bool(upper <= digital * (1 + level)), float(upper)
