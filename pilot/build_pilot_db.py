"""Write and reopen exactly one AgTlTe2 condition in an ASE SQLite DB.

Run in the existing HANS SimXRD environment. Existing output is never replaced.
Scalar schema-v1 fields are ASE key/value pairs except the intrinsic formula.
XRD arrays are stored in row.data; the shared grid is database metadata.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import spglib
from ase.db import connect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from Pysimxrd.utils.MatgenKit import get_diff, matgen_pxrdsim
from Pysimxrd.utils.funs import prim2conv
from Pysimxrd.utils.WPEMsim import space_group_to_crystal_system
from verify_deformation import compare


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=ROOT / "pilot" / "agtlte2_pilot.db")
    output = cli.parse_args().output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to append to or overwrite {output}")

    source = ROOT / "sim" / "demo_mp.db"
    if not source.is_file():
        raise FileNotFoundError(source)
    with connect(str(source)) as source_db:
        source_row = source_db.get(id=8)
        atoms = source_row.toatoms()
        mp_id = str(source_row.mpid)

    # Compare reduced composition; ASE's formula ordering need not be AgTlTe2.
    counts = {symbol: atoms.get_chemical_symbols().count(symbol)
              for symbol in set(atoms.get_chemical_symbols())}
    if set(counts) != {"Ag", "Tl", "Te"} or not (
        counts["Ag"] == counts["Tl"] and counts["Te"] == 2 * counts["Ag"]
    ):
        raise ValueError(f"Entry 8 is not AgTlTe2: {counts}")

    # meeting1/01_structure/check_symmetry.py: original structure, symprec=1e-3.
    symmetry = spglib.get_symmetry_dataset(
        (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers), symprec=1e-3
    )
    if symmetry is None:
        raise ValueError("Could not determine the original structure's space group")
    sg = int(symmetry.number)
    system_names = {1: "cubic", 2: "hexagonal", 3: "tetragonal",
                    4: "orthorhombic", 5: "trigonal", 6: "monoclinic", 7: "triclinic"}

    # Exact numerical logic from meeting1/00_ideal_XRD/ideal_xrd.py and
    # ideal_to_grid.py. Those scripts perform plotting/file I/O at import time,
    # so only their calculation block is reused here, without executing them.
    theta, intensity = get_diff(atoms, False, 0.0, 0.0)
    grid = np.arange(10.0, 80.0, 0.02)
    ideal = np.zeros(len(grid))
    for angle, value in zip(theta, intensity):
        idx = np.argmin(np.abs(grid - angle))
        ideal[idx] += value
    if ideal.max() > 0:
        ideal = ideal / ideal.max() * 100

    seed, stretch, shear = 0, 0.01, 0.0
    grain_size_A = 20.0  # verify_deformation.py; matgen_pxrdsim uses Angstroms.
    before, _, _ = prim2conv(atoms, False, stretch, shear)
    np.random.seed(seed)
    after, _, deformed = prim2conv(atoms, True, stretch, shear)
    # Verified direct path: no reseeding and no second random deformation.
    x, perturbed = matgen_pxrdsim(
        deformed, grain_size_A, [0.0, 0.0], 0.0, False,
        500 / 10, 5, 2.5, 6, 0.0, 0.0, stretch, shear, "reciprocal",
    )
    perturbed = np.asarray(perturbed, dtype=float)
    if ideal.shape != (3500,) or perturbed.shape != (3500,):
        raise ValueError("Both XRD vectors must have shape (3500,)")
    if not np.array_equal(x, grid):
        raise ValueError("Perturbed x-grid differs from the shared ideal grid")
    if not (np.isfinite(ideal).all() and np.isfinite(perturbed).all()):
        raise ValueError("Non-finite XRD values")

    physical = dict(zip(
        ["strain_a", "strain_b", "strain_c", "delta_alpha", "delta_beta", "delta_gamma"],
        np.concatenate(((after[:3] - before[:3]) / before[:3], after[3:] - before[3:])).tolist(),
    ))
    scalars = dict(
        sample_id="AgTlTe2-entry8-lattice-seed0", structure_id=8, mp_id=mp_id,
        space_group=sg, crystal_system=system_names[space_group_to_crystal_system(sg)],
        grain_size_nm=grain_size_A / 10, thermal_displacement_A=0.0,
        **physical,
        orientation_randomness_bound=0.0, lattice_stretch_bound=stretch,
        lattice_shear_bound=shear, zero_shift_deg=0.0,
        background_ratio=0.0, background_order=6, noise_ratio=0.0,
        detector_sample_distance=500.0, slit_half_height=5.0, sample_half_height=2.5,
        random_seed=seed, sim_model="pymatgen", split="unassigned",
    )
    metadata = {
        "schema_version": 1, "x_grid": grid.tolist(), "x_grid_unit": "degree_2theta",
        "structure_storage": "original source Atoms; labels use conventional cells",
        "source_db": "sim/demo_mp.db", "sim_model_mapping": "pymatgen = parser sim_model=None",
        "units": {"grain_size_nm": "nm", "thermal_displacement_A": "angstrom",
                  "strain_a/b/c": "dimensionless", "delta_alpha/beta/gamma": "degree",
                  "detector_sample_distance": "mm (divide by 10 for matgen_pxrdsim L)",
                  "slit_half_height": "unchanged SimXRD H input",
                  "sample_half_height": "unchanged SimXRD S input"},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with connect(str(output)) as output_db:
        output_db.metadata = metadata
        row_id = output_db.write(atoms, key_value_pairs=scalars,
                                 data={"ideal_xrd": ideal, "perturbed_xrd": perturbed})
    # Context exit commits and closes the connection; use a new connection.
    with connect(str(output)) as reloaded_db:
        count = reloaded_db.count()
        row = reloaded_db.get(id=row_id)
        restored = row.toatoms()
        restored_ideal = np.asarray(row.data["ideal_xrd"])
        restored_perturbed = np.asarray(row.data["perturbed_xrd"])
        restored_metadata = reloaded_db.metadata
        restored_scalars = row.key_value_pairs
        restored_formula = row.formula

    scalar_checks = {name: restored_scalars.get(name) == value for name, value in scalars.items()}
    scalar_checks["formula"] = restored_formula == atoms.get_chemical_formula()
    structure_checks = {
        "atomic_numbers": np.array_equal(atoms.numbers, restored.numbers),
        "positions": np.array_equal(atoms.positions, restored.positions),
        "cell": np.array_equal(atoms.cell.array, restored.cell.array),
        "pbc": np.array_equal(atoms.pbc, restored.pbc),
    }
    ideal_check = compare(ideal, restored_ideal)
    perturbed_check = compare(perturbed, restored_perturbed)
    passed = bool(
        count == 1 and all(scalar_checks.values()) and all(structure_checks.values())
        and restored_metadata == metadata and restored_scalars == scalars
        and restored_ideal.shape == (3500,) and restored_perturbed.shape == (3500,)
        and ideal_check["array_equal"] and perturbed_check["array_equal"]
    )
    report = {
        "output_db": str(output), "row_count": count, "row_id": row_id,
        "before_cellpar_A_deg": before.tolist(), "after_cellpar_A_deg": after.tolist(),
        "stored_scalar_values": dict(scalars, formula=atoms.get_chemical_formula()),
        "restored_scalar_values": dict(restored_scalars, formula=restored_formula),
        "scalar_matches": scalar_checks,
        "structure_matches": {name: bool(value) for name, value in structure_checks.items()},
        "ideal_xrd_length": len(restored_ideal), "perturbed_xrd_length": len(restored_perturbed),
        "ideal_xrd_roundtrip": ideal_check, "perturbed_xrd_roundtrip": perturbed_check,
        "database_metadata_equal": restored_metadata == metadata,
        "x_grid_equal": bool(np.array_equal(grid, restored_metadata["x_grid"])),
        "verification_passed": passed,
    }
    print(json.dumps(report, indent=2, allow_nan=False))
    if not passed:
        raise RuntimeError("Pilot DB round-trip verification failed; inspect the report")


if __name__ == "__main__":
    main()
