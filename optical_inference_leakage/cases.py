"""Models, quality allowances and text counts used in the measured comparison."""
import math

LLMS = ['Qwen3-0.6B', 'Qwen3-1.7B', 'Qwen3-4B', 'Qwen3-8B']
LEVELS = [0.005, 0.02, 0.1]
BUDGETS = [0.0, 1e-4, 1e-2, 1.0]
WIN = 2048
TOKENS = 1_000_000


def counts(name=None):
    """Upper counts for the protocol; the floor instead uses complete windows."""
    return [('1', 1), ('1e6', math.ceil(TOKENS / WIN))]
