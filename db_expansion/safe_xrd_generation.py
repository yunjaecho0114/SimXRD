"""Safe and optimized SimXRD generation utilities.

Features:
1. Fast nearest-grid intensity mapping.
2. Axial-divergence exact-center NaN protection.

The original SimXRD source files remain unchanged.

WARNING:
The temporary patches are process-global.
Do not use concurrently from multiple threads.
"""

from contextlib import contextmanager

import numpy as np

import Pysimxrd.utils.funs as funs

from db_expansion.fast_xrd_mapping import (
    optimized_xrd_mapping,
)


_ORIGINAL_AXIAL_DIV = funs.axial_div


def safe_axial_div(x, mu, L=500, H=50, S=25):
    """Protect exact peak-center points from singular division."""

    x = np.asarray(x)

    # Preserve original computation for normal inputs.
    if not np.any(x == mu):
        return _ORIGINAL_AXIAL_DIV(x, mu, L, H, S)

    x_safe = np.array(x, dtype=float, copy=True)

    # normal_cdf() only uses positions x < mu.
    # Exclude exact-center points from axial_div's x <= mu.
    x_safe[x_safe == mu] = np.nextafter(
        float(mu),
        np.inf
    )

    return _ORIGINAL_AXIAL_DIV(
        x_safe,
        mu,
        L,
        H,
        S
    )


@contextmanager
def safe_xrd_generation():
    """Temporarily enable both validated XRD improvements."""

    original_axial = funs.axial_div

    try:
        funs.axial_div = safe_axial_div

        with optimized_xrd_mapping():
            yield

    finally:
        funs.axial_div = original_axial
