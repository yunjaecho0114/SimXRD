"""Copy verified Condition #1 and generate only Condition #2, on HANS.

Condition #1 is grain size 2 nm + lattice deformation (not deformation-only).
The existing one-row DB, scripts, and Pysimxrd sources remain untouched.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from ase.db import connect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from Pysimxrd.utils.funs import prim2conv
from Pysimxrd.utils.MatgenKit import matgen_pxrdsim
from verify_deformation import compare


def structure_checks(original, restored):
    return {
        "atomic_numbers": bool(np.array_equal(original.numbers, restored.numbers)),
        "positions": bool(np.array_equal(original.positions, restored.positions)),
        "cell": bool(np.array_equal(original.cell.array, restored.cell.array)),
        "pbc": bool(np.array_equal(original.pbc, restored.pbc)),
        "canonical_formula": original.get_chemical_formula(mode="hill") == restored.get_chemical_formula(mode="hill"),
        "composition": Counter(original.get_chemical_symbols()) == Counter(restored.get_chemical_symbols()),
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--condition1-db", type=Path, required=True,
                     help="Existing verified one-row DB; read only, never regenerated")
    cli.add_argument("--output", type=Path, default=ROOT / "pilot" / "agtlte2_two_conditions.db")
    args = cli.parse_args()
    source, output = args.condition1_db.resolve(), args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Refusing to append to or overwrite {output}")

    with connect(str(source)) as db:
        if db.count() != 1:
            raise ValueError("Condition #1 input must contain exactly one verified row")
        row = db.get()
        atoms = row.toatoms()
        first = dict(row.key_value_pairs)
        first_data = dict(row.data)
        metadata = db.metadata
    grid = np.arange(10.0, 80.0, 0.02)
    if metadata.get("schema_version") != 1 or not np.array_equal(metadata.get("x_grid"), grid):
        raise ValueError("Input DB must use schema v1 and the shared 3500-point grid")
    expected_first = dict(
        structure_id=8, grain_size_nm=2.0, thermal_displacement_A=0.0,
        random_seed=0, lattice_stretch_bound=0.01, lattice_shear_bound=0.0,
        orientation_randomness_bound=0.0, zero_shift_deg=0.0,
        background_ratio=0.0, background_order=6, noise_ratio=0.0,
        detector_sample_distance=500.0, slit_half_height=5.0, sample_half_height=2.5,
        sim_model="pymatgen",
    )
    if any(first.get(key) != value for key, value in expected_first.items()):
        raise ValueError("Input metadata does not match verified Condition #1")
    target_names = ["strain_a", "strain_b", "strain_c", "delta_alpha", "delta_beta", "delta_gamma"]
    for key in ["sample_id", "mp_id", "space_group", "crystal_system", "split", *target_names]:
        if key not in first:
            raise ValueError(f"Missing schema-v1 field: {key}")
    original_db = ROOT / "sim" / "demo_mp.db"
    if not original_db.is_file():
        raise FileNotFoundError(original_db)
    with connect(str(original_db)) as db:
        original_row = db.get(id=8)
        if not all(structure_checks(original_row.toatoms(), atoms).values()):
            raise ValueError("Condition #1 structure differs from source entry 8")
        if str(original_row.mpid) != first["mp_id"]:
            raise ValueError("Condition #1 MP ID differs from source entry 8")

    # Reuse the already verified ideal calculation without recomputing it.
    ideal = np.asarray(first_data["ideal_xrd"])
    old_perturbed = np.asarray(first_data["perturbed_xrd"])
    for array in (ideal, old_perturbed):
        if array.shape != (3500,) or not np.isfinite(array).all():
            raise ValueError("Invalid Condition #1 XRD vector")

    second = dict(first)
    second.update(sample_id="AgTlTe2-entry8-multiphysical-seed1",
                  grain_size_nm=10.0, thermal_displacement_A=0.2, random_seed=1)
    if second["sample_id"] == first["sample_id"]:
        raise ValueError("Condition sample IDs must be distinct")
    stretch, shear = second["lattice_stretch_bound"], second["lattice_shear_bound"]
    before, _, _ = prim2conv(atoms, False, stretch, shear)
    np.random.seed(second["random_seed"])
    after, _, deformed = prim2conv(atoms, True, stretch, shear)
    actual = dict(zip(target_names, np.concatenate((
        (after[:3] - before[:3]) / before[:3], after[3:] - before[3:]
    )).tolist()))
    if not np.isfinite(list(actual.values())).all():
        raise ValueError("Non-finite actual lattice targets")
    second.update(actual)
    # Same direct path as build_pilot_db.py. Convert nm to Angstroms only here;
    # retain the RNG state after the single deformation realization.
    x, perturbed = matgen_pxrdsim(
        deformed, second["grain_size_nm"] * 10.0,
        [second["orientation_randomness_bound"]] * 2, second["thermal_displacement_A"],
        False, second["detector_sample_distance"] / 10,
        second["slit_half_height"], second["sample_half_height"],
        second["background_order"], second["background_ratio"], second["noise_ratio"],
        stretch, shear, "reciprocal",
    )
    perturbed = np.asarray(perturbed, dtype=float)
    if not np.array_equal(x, grid) or perturbed.shape != (3500,) or not np.isfinite(perturbed).all():
        raise ValueError("Invalid Condition #2 XRD/grid")

    output.parent.mkdir(parents=True, exist_ok=True)
    with connect(str(output)) as db:
        db.metadata = metadata
        id1 = db.write(atoms, key_value_pairs=first, data=first_data)
        id2 = db.write(atoms, key_value_pairs=second,
                       data={"ideal_xrd": ideal, "perturbed_xrd": perturbed})
    # Commit/close, then explicitly open a fresh connection and reload both rows.
    with connect(str(output)) as db:
        row_count = db.count()
        rows = [db.get(id=id1), db.get(id=id2)]
        restored_metadata = db.metadata

    reports = []
    for restored, expected, expected_y in zip(rows, [first, second], [old_perturbed, perturbed]):
        recovered = restored.toatoms()
        structural = structure_checks(atoms, recovered)
        scalars = {key: restored.key_value_pairs.get(key) == value for key, value in expected.items()}
        reloaded_ideal = np.asarray(restored.data["ideal_xrd"])
        reloaded_y = np.asarray(restored.data["perturbed_xrd"])
        ideal_check, perturbed_check = compare(ideal, reloaded_ideal), compare(expected_y, reloaded_y)
        reports.append({
            "sample_id": expected["sample_id"], "structure_matches": structural,
            "metadata_matches": scalars, "metadata_equal": restored.key_value_pairs == expected,
            "stored_metadata": expected, "restored_metadata": restored.key_value_pairs,
            "ideal_xrd_length": len(reloaded_ideal), "perturbed_xrd_length": len(reloaded_y),
            "ideal_roundtrip": ideal_check, "perturbed_roundtrip": perturbed_check,
            "passed": bool(all(structural.values()) and all(scalars.values())
                           and restored.key_value_pairs == expected
                           and reloaded_ideal.shape == (3500,) and reloaded_y.shape == (3500,)
                           and ideal_check["array_equal"] and perturbed_check["array_equal"]),
        })
    shared_ideal = bool(np.array_equal(rows[0].data["ideal_xrd"], rows[1].data["ideal_xrd"]))
    same_structure_id = rows[0].structure_id == rows[1].structure_id == 8
    actual_match = {name: rows[1].key_value_pairs[name] == value for name, value in actual.items()}
    passed = bool(row_count == 2 and all(item["passed"] for item in reports)
                  and shared_ideal and same_structure_id and all(actual_match.values())
                  and restored_metadata == metadata)
    print(json.dumps({
        "input_condition1_db": str(source), "output_db": str(output), "row_count": row_count,
        "same_structure_id": same_structure_id, "shared_ideal_array_equal": shared_ideal,
        "database_metadata_equal": restored_metadata == metadata,
        "condition2_before_cellpar_A_deg": before.tolist(),
        "condition2_after_cellpar_A_deg": after.tolist(),
        "condition2_actual_targets": actual, "condition2_actual_targets_match": actual_match,
        "conditions": reports, "verification_passed": passed,
    }, indent=2, allow_nan=False))
    if not passed:
        raise RuntimeError("Two-condition round-trip verification failed; inspect the report")


if __name__ == "__main__":
    main()
