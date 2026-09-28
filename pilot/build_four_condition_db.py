"""Add only the seed-3 nuisance pilot condition to a verified three-row DB.

Run on HANS. Parameters are pilot checks, not a production distribution.
Repository noise is positive uniform additive noise, NOT Gaussian noise.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from ase.db import connect

from build_two_condition_db import ROOT, structure_checks
from build_three_condition_db import TARGET_NAMES
from verify_deformation import compare
from Pysimxrd.utils.funs import prim2conv
from Pysimxrd.utils.MatgenKit import matgen_pxrdsim

STEP_DEG = 0.02
SETTINGS = dict(
    grain_size_nm=10.0, thermal_displacement_A=0.2,
    lattice_stretch_bound=0.01, lattice_shear_bound=0.0,
    orientation_randomness_bound=0.0, zero_shift_deg=0.10,
    background_ratio=0.05, background_order=6, noise_ratio=0.02,
    random_seed=3, detector_sample_distance=500.0,
    slit_half_height=5.0, sample_half_height=2.5, sim_model="pymatgen",
)


def integer_bin_shift(profile, shift_deg, step_deg=STEP_DEG):
    """Positive shift moves y[i] to y[i + bins]; zero fill, never wrap.

    No interpolation or renormalization. Reject non-integer grid shifts.
    """
    profile = np.asarray(profile)
    if profile.ndim != 1 or step_deg <= 0 or not np.isfinite(shift_deg):
        raise ValueError("Expected a 1-D profile and a finite grid shift")
    bins = int(round(shift_deg / step_deg))
    if not np.isclose(shift_deg, bins * step_deg, rtol=0, atol=1e-12):
        raise ValueError("Zero shift must be an integer number of grid bins")
    shifted = np.zeros_like(profile)
    if bins == 0:
        shifted[:] = profile
    elif 0 < bins < len(profile):
        shifted[bins:] = profile[:-bins]
    elif -len(profile) < bins < 0:
        shifted[:bins] = profile[-bins:]
    return shifted, bins


def generate_condition4(atoms):
    """One full realization: seed exactly once, then preserve RNG ordering."""
    p = SETTINGS
    np.random.seed(p["random_seed"])
    stretch, shear = p["lattice_stretch_bound"], p["lattice_shear_bound"]
    before, _, _ = prim2conv(atoms, False, stretch, shear)
    after, _, deformed = prim2conv(atoms, True, stretch, shear)
    actual = dict(zip(TARGET_NAMES, np.concatenate((
        (after[:3] - before[:3]) / before[:3], after[3:] - before[3:]
    )).tolist()))
    # Reuse repository background/noise as-is. Even disabled orientation consumes
    # random draws internally. Do not reseed between deformation and this call.
    x, pre_shift = matgen_pxrdsim(
        deformed, p["grain_size_nm"] * 10.0,
        [p["orientation_randomness_bound"]] * 2, p["thermal_displacement_A"],
        False, p["detector_sample_distance"] / 10,
        p["slit_half_height"], p["sample_half_height"],
        p["background_order"], p["background_ratio"], p["noise_ratio"],
        stretch, shear, "reciprocal",
    )
    pre_shift = np.asarray(pre_shift, dtype=float)
    shifted, bins = integer_bin_shift(pre_shift, p["zero_shift_deg"])
    return dict(x=np.asarray(x), before=before, after=after, actual=actual,
                pre_shift=pre_shift, shifted=shifted, bins=bins)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--three-condition-db", type=Path, required=True)
    cli.add_argument("--output", type=Path, default=ROOT / "pilot" / "agtlte2_four_conditions.db")
    args = cli.parse_args()
    source, output = args.three_condition_db.resolve(), args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Refusing to append to or overwrite {output}")
    with connect(str(source)) as db:
        if db.count() != 3:
            raise ValueError("Input must contain exactly three verified conditions")
        originals = [db.get(random_seed=seed) for seed in (0, 1, 2)]
        structures = [row.toatoms() for row in originals]
        expected_metadata = [dict(row.key_value_pairs) for row in originals]
        expected_data = [dict(row.data) for row in originals]
        metadata = db.metadata
    grid = np.arange(10.0, 80.0, STEP_DEG)
    if metadata.get("schema_version") != 1 or not np.array_equal(metadata.get("x_grid"), grid):
        raise ValueError("Expected schema v1 and the existing grid")
    atoms = structures[0]
    ideal = np.asarray(expected_data[0]["ideal_xrd"])
    for index, (structure, scalars, data) in enumerate(zip(structures, expected_metadata, expected_data)):
        if scalars.get("structure_id") != 8 or not all(structure_checks(atoms, structure).values()):
            raise ValueError("Input rows must share the verified entry-8 structure")
        for name in ["sample_id", "mp_id", "space_group", "crystal_system", "split", *TARGET_NAMES, *SETTINGS]:
            if name not in scalars:
                raise ValueError(f"Missing schema-v1 field: {name}")
        for name in ("ideal_xrd", "perturbed_xrd"):
            array = np.asarray(data[name])
            if array.shape != (3500,) or not np.isfinite(array).all():
                raise ValueError(f"Invalid Condition #{index + 1} {name}")
        if not np.array_equal(ideal, data["ideal_xrd"]):
            raise ValueError("Input ideal XRDs differ")
        for name in ("detector_sample_distance", "slit_half_height", "sample_half_height"):
            if scalars[name] != SETTINGS[name]:
                raise ValueError("Input instrument settings differ from Condition #4")

    run = generate_condition4(atoms)
    repeat = generate_condition4(atoms)
    for generated in (run, repeat):
        for name in ("pre_shift", "shifted"):
            if generated[name].shape != (3500,) or not np.isfinite(generated[name]).all():
                raise ValueError("Invalid generated profile")
        if not np.array_equal(generated["x"], grid) or not np.isfinite(list(generated["actual"].values())).all():
            raise ValueError("Invalid generated grid/actual targets")
    reproducibility = {name: compare(run[name], repeat[name])
                       for name in ("before", "after", "x", "pre_shift", "shifted")}
    reproducibility["actual_targets"] = compare(
        [run["actual"][name] for name in TARGET_NAMES],
        [repeat["actual"][name] for name in TARGET_NAMES])
    reproducibility_passed = all(result["array_equal"] for result in reproducibility.values()) and run["bins"] == repeat["bins"]
    shift_checks = {
        "zero_shift_bins_is_5": run["bins"] == 5,
        "profile_changed": not np.array_equal(run["pre_shift"], run["shifted"]),
        "shifted_tail_equals_pre_shift_prefix": bool(np.array_equal(run["shifted"][5:], run["pre_shift"][:-5])),
        "left_five_bins_zero": bool(np.array_equal(run["shifted"][:5], np.zeros(5))),
        "no_wraparound": bool(np.array_equal(run["shifted"], np.concatenate((np.zeros(5), run["pre_shift"][:-5])))),
    }
    fourth = dict(expected_metadata[1])
    fourth.update(SETTINGS)
    fourth.update(run["actual"])
    fourth["sample_id"] = "AgTlTe2-entry8-nuisance-seed3"
    expected_metadata.append(fourth)
    expected_data.append({"ideal_xrd": ideal, "perturbed_xrd": run["shifted"]})
    structures.append(atoms)
    if len({m["sample_id"] for m in expected_metadata}) != 4:
        raise ValueError("Sample IDs must be unique")
    output.parent.mkdir(parents=True, exist_ok=True)
    with connect(str(output)) as db:
        db.metadata = metadata
        ids = [db.write(structure, key_value_pairs=scalars, data=data)
               for structure, scalars, data in zip(structures, expected_metadata, expected_data)]
    # Writer is closed before opening a new connection, as in the verified builder.
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
    roundtrip_passed = bool(row_count == 4 and all(item["passed"] for item in reports)
                           and shared_ideal and same_structure_id and restored_metadata == metadata)
    passed = bool(roundtrip_passed and reproducibility_passed and all(shift_checks.values()))
    print(json.dumps({
        "input_three_condition_db": str(source), "output_db": str(output), "row_count": row_count,
        "same_structure_id": same_structure_id, "shared_ideal_array_equal": shared_ideal,
        "existing_three_metadata_preserved": all(item["metadata_equal"] for item in reports[:3]),
        "database_metadata_equal": restored_metadata == metadata,
        "cellpar_order": ["a_A", "b_A", "c_A", "alpha_deg", "beta_deg", "gamma_deg"],
        "condition4_before_cellpar_A_deg": run["before"].tolist(),
        "condition4_after_cellpar_A_deg": run["after"].tolist(),
        "condition4_actual_targets": run["actual"], "condition4_settings": SETTINGS,
        "noise_model": "positive uniform additive noise; repository implementation, not Gaussian",
        "parameter_scope": "pilot validation only; not a production parameter distribution",
        "zero_shift_bins": run["bins"], "zero_shift_checks": shift_checks,
        "reproducibility": reproducibility, "reproducibility_passed": bool(reproducibility_passed),
        "conditions": reports, "roundtrip_passed": roundtrip_passed, "verification_passed": passed,
    }, indent=2, allow_nan=False))
    if not passed:
        raise RuntimeError("Four-condition nuisance/reproducibility/round-trip verification failed")


if __name__ == "__main__":
    main()
