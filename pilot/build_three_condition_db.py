"""Preserve two verified rows and add the seed-2 shear condition on HANS."""

import argparse
import json
from pathlib import Path

import numpy as np
from ase.db import connect

from build_two_condition_db import ROOT, structure_checks
from verify_deformation import compare
from Pysimxrd.utils.funs import prim2conv
from Pysimxrd.utils.MatgenKit import matgen_pxrdsim

ANGLE_TOLERANCE_DEG = 1e-8
TARGET_NAMES = ["strain_a", "strain_b", "strain_c", "delta_alpha", "delta_beta", "delta_gamma"]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--two-condition-db", type=Path, required=True)
    cli.add_argument("--output", type=Path, default=ROOT / "pilot" / "agtlte2_three_conditions.db")
    args = cli.parse_args()
    source, output = args.two_condition_db.resolve(), args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Refusing to append to or overwrite {output}")

    with connect(str(source)) as db:
        if db.count() != 2:
            raise ValueError("Input must be the verified two-condition DB")
        original_rows = [db.get(random_seed=0), db.get(random_seed=1)]
        metadata = db.metadata
        original_atoms = [row.toatoms() for row in original_rows]
        expected_metadata = [dict(row.key_value_pairs) for row in original_rows]
        expected_data = [dict(row.data) for row in original_rows]

    grid = np.arange(10.0, 80.0, 0.02)
    if metadata.get("schema_version") != 1 or not np.array_equal(metadata.get("x_grid"), grid):
        raise ValueError("Expected schema v1 and the existing fixed grid")
    atoms = original_atoms[0]
    if not all(structure_checks(atoms, original_atoms[1]).values()):
        raise ValueError("Input structures differ")
    common = dict(structure_id=8, lattice_stretch_bound=0.01, lattice_shear_bound=0.0,
                  orientation_randomness_bound=0.0, zero_shift_deg=0.0,
                  background_ratio=0.0, background_order=6, noise_ratio=0.0,
                  detector_sample_distance=500.0, slit_half_height=5.0,
                  sample_half_height=2.5, sim_model="pymatgen")
    for index, scalars in enumerate(expected_metadata):
        required = dict(common, random_seed=index,
                        grain_size_nm=2.0 if index == 0 else 10.0,
                        thermal_displacement_A=0.0 if index == 0 else 0.2)
        if any(scalars.get(key) != value for key, value in required.items()):
            raise ValueError(f"Unexpected Condition #{index + 1} metadata")
        for key in ["sample_id", "mp_id", "space_group", "crystal_system", "split", *TARGET_NAMES]:
            if key not in scalars:
                raise ValueError(f"Missing schema-v1 field: {key}")
        for name in ["ideal_xrd", "perturbed_xrd"]:
            array = np.asarray(expected_data[index][name])
            if array.shape != (3500,) or not np.isfinite(array).all():
                raise ValueError(f"Invalid input {name}")
    ideal = np.asarray(expected_data[0]["ideal_xrd"])
    if not np.array_equal(ideal, expected_data[1]["ideal_xrd"]):
        raise ValueError("Input ideal XRDs differ")

    third = dict(expected_metadata[1])
    third.update(sample_id="AgTlTe2-entry8-shear-seed2", grain_size_nm=10.0,
                 thermal_displacement_A=0.2, lattice_stretch_bound=0.01,
                 lattice_shear_bound=0.01, random_seed=2)
    if len({m["sample_id"] for m in [*expected_metadata, third]}) != 3:
        raise ValueError("Sample IDs must be unique")
    stretch, shear = third["lattice_stretch_bound"], third["lattice_shear_bound"]
    # Same validated direct path. Only the final cell defines the actual targets.
    before, _, _ = prim2conv(atoms, False, stretch, shear)
    np.random.seed(third["random_seed"])
    after, _, deformed = prim2conv(atoms, True, stretch, shear)
    actual = dict(zip(TARGET_NAMES, np.concatenate((
        (after[:3] - before[:3]) / before[:3], after[3:] - before[3:]
    )).tolist()))
    if not np.isfinite(list(actual.values())).all():
        raise ValueError("Non-finite actual lattice targets")
    third.update(actual)
    # No RNG reset or further random lattice deformation; nm -> Angstrom here.
    x, perturbed = matgen_pxrdsim(
        deformed, third["grain_size_nm"] * 10.0,
        [third["orientation_randomness_bound"]] * 2, third["thermal_displacement_A"],
        False, third["detector_sample_distance"] / 10,
        third["slit_half_height"], third["sample_half_height"],
        third["background_order"], third["background_ratio"], third["noise_ratio"],
        stretch, shear, "reciprocal",
    )
    perturbed = np.asarray(perturbed, dtype=float)
    if not np.array_equal(x, grid) or perturbed.shape != (3500,) or not np.isfinite(perturbed).all():
        raise ValueError("Invalid Condition #3 XRD/grid")
    expected_metadata.append(third)
    expected_data.append({"ideal_xrd": ideal, "perturbed_xrd": perturbed})
    output.parent.mkdir(parents=True, exist_ok=True)
    with connect(str(output)) as db:
        db.metadata = metadata
        ids = [db.write(structure, key_value_pairs=scalars, data=data)
               for structure, scalars, data in zip(
                   [*original_atoms, atoms], expected_metadata, expected_data)]
    # Close the writer before creating a fresh connection for round-trip checks.
    with connect(str(output)) as db:
        row_count = db.count()
        rows = [db.get(id=row_id) for row_id in ids]
        restored_metadata = db.metadata

    reports = []
    for restored, expected, data in zip(rows, expected_metadata, expected_data):
        structural = structure_checks(atoms, restored.toatoms())
        scalars = {key: restored.key_value_pairs.get(key) == value for key, value in expected.items()}
        reloaded_ideal = np.asarray(restored.data["ideal_xrd"])
        reloaded_y = np.asarray(restored.data["perturbed_xrd"])
        ideal_check = compare(data["ideal_xrd"], reloaded_ideal)
        perturbed_check = compare(data["perturbed_xrd"], reloaded_y)
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
    shared_ideal = all(np.array_equal(rows[0].data["ideal_xrd"], row.data["ideal_xrd"]) for row in rows)
    same_structure_id = all(row.structure_id == 8 for row in rows)
    actual_match = {name: rows[2].key_value_pairs[name] == value for name, value in actual.items()}
    angle_changes = {name: rows[2].key_value_pairs[name] for name in TARGET_NAMES[3:]}
    nonzero_angles = {name: abs(value) > ANGLE_TOLERANCE_DEG for name, value in angle_changes.items()}
    passed = bool(row_count == 3 and all(item["passed"] for item in reports)
                  and shared_ideal and same_structure_id and all(actual_match.values())
                  and restored_metadata == metadata and any(nonzero_angles.values()))
    print(json.dumps({
        "input_two_condition_db": str(source), "output_db": str(output), "row_count": row_count,
        "same_structure_id": same_structure_id, "shared_ideal_array_equal": shared_ideal,
        "condition1_metadata_preserved": reports[0]["metadata_equal"],
        "condition2_metadata_preserved": reports[1]["metadata_equal"],
        "database_metadata_equal": restored_metadata == metadata,
        "cellpar_order": ["a_A", "b_A", "c_A", "alpha_deg", "beta_deg", "gamma_deg"],
        "condition3_before_cellpar_A_deg": before.tolist(),
        "condition3_after_cellpar_A_deg": after.tolist(),
        "condition3_actual_targets": actual, "condition3_actual_targets_match": actual_match,
        "condition3_restored_delta_angles_deg": angle_changes,
        "angle_tolerance_deg": ANGLE_TOLERANCE_DEG,
        "delta_angles_exceed_tolerance": nonzero_angles,
        "at_least_one_nonzero_delta_angle": any(nonzero_angles.values()),
        "conditions": reports, "verification_passed": passed,
    }, indent=2, allow_nan=False))
    if not passed:
        raise RuntimeError("Three-condition shear/round-trip verification failed; inspect the report")


if __name__ == "__main__":
    main()
