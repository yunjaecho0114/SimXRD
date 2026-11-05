"""Fast grid mapping for SimXRD DB generation.

This module preserves the original map_int nearest-grid
and last-write behavior for the validated XRD grid.

It does not modify the original SimXRD source files.

The temporary monkey-patch is process-global and should
not be used concurrently with other threads generating XRDs.
"""

from contextlib import contextmanager

import numpy as np
import Pysimxrd.utils.funs as funs


def fast_map_int(peak, x, twotheta):
    """Map local peak intensities to the nearest global grid."""

    result = np.zeros_like(twotheta)

    mask = (
        (x >= twotheta[0])
        & (x <= twotheta[-1])
    )

    angles = x[mask]
    values = peak[mask]

    if len(angles) == 0:
        return result

    right = np.searchsorted(twotheta, angles)

    left = np.clip(
        right - 1, 0, len(twotheta) - 1
    )

    right = np.clip(
        right, 0, len(twotheta) - 1
    )

    use_right = (
        np.abs(twotheta[right] - angles)
        < np.abs(angles - twotheta[left])
    )

    indices = np.where(use_right, right, left)

    # Preserve the original last-write behavior.
    unique_indices, reverse_positions = np.unique(
        indices[::-1],
        return_index=True
    )

    last_positions = (
        len(indices) - 1 - reverse_positions
    )

    result[unique_indices] = values[last_positions]

    return result


@contextmanager
def optimized_xrd_mapping():
    """Temporarily enable optimized mapping within a process."""

    original_map = funs.map_int

    try:
        funs.map_int = fast_map_int
        yield
    finally:
        funs.map_int = original_map
