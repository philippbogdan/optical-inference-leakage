"""Approximate switching for the restricted uniform-noise class. All information quantities are in bits."""
import math
import numpy as np

def h2(p):
    """Binary entropy with finite endpoints."""
    p = min(max(p, 1e-300), 1 - 1e-16)
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

def tau_bar(eps_bits):
    """Upper bound on the mean total-variation distance for a text."""
    return math.sqrt(max(0.0, 1.0 - 2.0 ** (-eps_bits)))

def theorem3_floor_info(C0_bits, J, eps_bits):
    """Subtract the coupling penalty from C0 using J chain terms and a variance bound on Gaussian information density."""
    t = tau_bar(eps_bits)
    if t == 0:
        return C0_bits
    a = 2 * math.log2(math.e) * t * (1 - t)
    return max(0.0, (1 - t) * C0_bits - math.sqrt(a * J * C0_bits) - J * (h2(t) + t * math.log2(1 / t)))
