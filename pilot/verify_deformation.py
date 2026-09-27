"""Compare parser and direct prim2conv for one structure; never write a DB.

Reuse meeting1/06_internal_stress/internal_stress_test.py's seed and one
stretch bound. Other stochastic intensity/background effects are disabled;
the same fixed profile broadening and instrument settings remain in both paths.
"""

import json
from pathlib import Path
import sys

import numpy as np
from ase.db import connect
from pymatgen.core import Lattice, Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from Pysimxrd import generator
from Pysimxrd.utils.funs import prim2conv
from Pysimxrd.utils.MatgenKit import get_diff, matgen_pxrdsim


def compare(a, b):
    a, b = np.asarray(a), np.asarray(b)
    same_shape = a.shape == b.shape
    return {
        "shapes": [list(a.shape), list(b.shape)],
        "max_absolute_difference": float(np.max(np.abs(a - b))) if same_shape else None,
        "array_equal": bool(np.array_equal(a, b)),
        "allclose_default": bool(np.allclose(a, b)) if same_shape else False,
        "allclose_rtol_1e-10_atol_1e-10": bool(np.allclose(a, b, rtol=1e-10, atol=1e-10)) if same_shape else False,
    }


def same_rng(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


def main():
    seed, stretch, shear = 0, 0.01, 0.0
    db = connect(str(ROOT / "sim" / "demo_mp.db"))
    atoms = db.get_atoms(id=8)
    assert atoms.get_chemical_symbols().count("Ag") > 0
    # This baseline call consumes no deformation random draws.
    before, before_matrix, before_atoms = prim2conv(atoms, False, stretch, shear)

    settings = dict(
        grainsize=20, prefect_orientation=[0.0, 0.0], thermo_vibration=0.0,
        zero_shift=0.0, dis_detector2sample=500,
        half_height_slit_detector=5, half_height_sample=2.5,
        background_order=6, background_ratio=0.0, mixture_noise_ratio=0.0,
        lattice_extinction_ratio=stretch, lattice_torsion_ratio=shear,
        sim_model=None, xrd="reciprocal",
    )
    np.random.seed(seed)
    reference_x, reference_y = generator.parser(db, entry_id=8, deformation=True, **settings)
    reference_rng = np.random.get_state()

    # Reset once for the independent direct comparison. Do not reset again
    # between prim2conv and matgen_pxrdsim: retain the post-deformation RNG state.
    np.random.seed(seed)
    after, after_matrix, deformed = prim2conv(atoms, True, stretch, shear)
    direct_x, direct_y = matgen_pxrdsim(
        deformed, settings["grainsize"], settings["prefect_orientation"],
        settings["thermo_vibration"], False,
        settings["dis_detector2sample"] / 10,
        settings["half_height_slit_detector"], settings["half_height_sample"],
        settings["background_order"], settings["background_ratio"],
        settings["mixture_noise_ratio"], stretch, shear, settings["xrd"],
    )
    direct_rng = np.random.get_state()

    # Inspect precisely the extra conventionalization performed by get_diff.
    reconv, reconv_matrix, reconv_atoms = prim2conv(deformed, False, stretch, shear)
    raw_structure = Structure(
        Lattice(deformed.cell), deformed.get_chemical_symbols(), deformed.get_scaled_positions()
    )
    raw_peaks = XRDCalculator().get_pattern(raw_structure, two_theta_range=(10, 80))
    reconv_theta, reconv_intensity = get_diff(deformed, False, stretch, shear)
    physical = dict(zip(
        ["strain_a", "strain_b", "strain_c", "delta_alpha", "delta_beta", "delta_gamma"],
        np.concatenate(((after[:3] - before[:3]) / before[:3], after[3:] - before[3:])).tolist(),
    ))
    report = {
        "entry_id": 8, "formula": atoms.get_chemical_formula(),
        "random_seed": seed, "lattice_stretch_bound": stretch, "lattice_shear_bound": shear,
        "reference_settings": settings,
        "before_cellpar_A_deg": before.tolist(), "after_cellpar_A_deg": after.tolist(),
        "physical_targets": physical,
        "before_lattice_matrix": before_matrix.tolist(),
        "after_lattice_matrix": after_matrix.tolist(),
        "reconventionalized_cellpar_A_deg": reconv.tolist(),
        "reconventionalized_matrix": reconv_matrix.tolist(),
        "atom_counts": [len(before_atoms), len(deformed), len(reconv_atoms)],
        "extra_conventionalization_cellpar": compare(after, reconv),
        "extra_conventionalization_peak_positions": compare(raw_peaks.x, reconv_theta),
        "extra_conventionalization_peak_intensities": compare(raw_peaks.y, reconv_intensity),
        "x_grid": compare(reference_x, direct_x),
        "reference_vs_required_grid": compare(reference_x, np.arange(10.0, 80.0, 0.02)),
        "intensity": compare(reference_y, direct_y),
        "rng_states_equal_after_profiles": same_rng(reference_rng, direct_rng),
        "all_values_finite": bool(np.all(np.isfinite(reference_y)) and np.all(np.isfinite(direct_y))),
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
