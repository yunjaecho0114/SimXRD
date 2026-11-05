import csv
import json
import time
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from ase.db import connect

from Pysimxrd.utils.funs import prim2conv
from Pysimxrd.utils.MatgenKit import get_diff, matgen_pxrdsim
from db_expansion.safe_xrd_generation import safe_xrd_generation

ROOT = Path("db_expansion/dataset_v1")
CONFIG_PATH = ROOT / "generation_config.json"
OUT = ROOT / "smoke_3x30_v2"

with CONFIG_PATH.open(encoding="utf-8") as f:
    cfg = json.load(f)

with Path(cfg["source"]["selection_csv"]).open(
    newline="", encoding="utf-8"
) as f:
    candidates = list(csv.DictReader(f))

schedule = cfg["strain"]["schedule"]
control = cfg["controlled_variables"]

N = int(cfg["xrd"]["length"])
START = float(cfg["xrd"]["two_theta_start_deg"])
STOP = float(cfg["xrd"]["two_theta_stop_exclusive_deg"])
STEP = float(cfg["xrd"]["step_deg"])

# IMPORTANT: Use the same grid construction as SimXRD.
grid = np.arange(START, STOP, STEP)

assert len(grid) == N == 3500
assert cfg["strain"]["shear_bound"] == 0.0
assert sum(g["count"] for g in schedule) == 30

targets = [
    ("train", "cubic"),
    ("validation", "monoclinic"),
    ("test", "triclinic"),
]

selected = []

for split, system in targets:
    eligible = [
        r for r in candidates
        if r["split"] == split
        and r["crystal_system"] == system
    ]
    assert eligible

    selected.append(min(
        eligible,
        key=lambda r: (int(r["natoms"]), int(r["db_id"]))
    ))

assert len({r["mp_id"] for r in selected}) == 3

# Never overwrite a previous smoke test.
OUT.mkdir(parents=True, exist_ok=False)

ideal = open_memmap(
    OUT / "ideal_xrd.npy",
    mode="w+", dtype="float64", shape=(3, N)
)
perturbed = open_memmap(
    OUT / "perturbed_xrd.npy",
    mode="w+", dtype="float64", shape=(90, N)
)
lattice_before = open_memmap(
    OUT / "lattice_before.npy",
    mode="w+", dtype="float64", shape=(90, 3, 3)
)
lattice_after = open_memmap(
    OUT / "lattice_after.npy",
    mode="w+", dtype="float64", shape=(90, 3, 3)
)

db = connect(cfg["source"]["structure_db"])
structure_rows = []
sample_rows = []
row_id = 0
start_time = time.perf_counter()

print("=== STEP 3-40B: SMOKE DATASET V2 ===", flush=True)

with safe_xrd_generation():

    for structure_index, record in enumerate(selected):
        db_id = int(record["db_id"])
        atoms = db.get(id=db_id).toatoms()

        before, A0, conventional = prim2conv(
            atoms, False, 0.0, 0.0
        )

        before = np.asarray(before, dtype=float)
        A0 = np.asarray(A0, dtype=float)

        # Generate ideal stick XRD once per structure.
        theta, weights = get_diff(
            atoms, False, 0.0, 0.0
        )

        ideal_pattern = np.zeros(N, dtype=np.float64)

        for angle, weight in zip(theta, weights):
            index = int(np.argmin(np.abs(grid - angle)))
            ideal_pattern[index] += float(weight)

        assert np.isfinite(ideal_pattern).all()
        assert ideal_pattern.max() > 0

        ideal_pattern *= 100.0 / ideal_pattern.max()
        ideal[structure_index] = ideal_pattern

        structure_rows.append({
            "structure_index": structure_index,
            "source_db_id": db_id,
            "mp_id": record["mp_id"],
            "split": record["split"],
            "crystal_system": record["crystal_system"],
            "space_group_number": record["space_group_number"],
            "original_natoms": len(atoms),
            "conventional_natoms": len(conventional),
            "ideal_xrd_row": structure_index,
            "ideal_peak_count": len(theta),
        })

        print(
            f"\nSTRUCTURE {structure_index + 1}/3: "
            f"{record['mp_id']} | {record['split']} | "
            f"{record['crystal_system']}",
            flush=True
        )

        sample_in_structure = 0

        for group_index, group in enumerate(schedule):
            name = group["name"]
            low = float(group["low"])
            high = float(group["high"])
            bound = float(group["bound"])

            for group_sample_index in range(int(group["count"])):
                accepted = False

                for attempt in range(
                    int(cfg["strain"]["max_attempts"])
                ):
                    seed = int(
                        np.random.SeedSequence([
                            int(cfg["strain"]["base_seed"]),
                            db_id,
                            group_index,
                            group_sample_index,
                            attempt,
                        ]).generate_state(1)[0]
                    )

                    np.random.seed(seed)

                    after, A1, deformed = prim2conv(
                        atoms, True, bound, 0.0
                    )

                    after = np.asarray(after, dtype=float)
                    A1 = np.asarray(A1, dtype=float)

                    strain = (
                        (after[:3] - before[:3]) / before[:3]
                    )

                    delta_angles = after[3:] - before[3:]

                    rms = float(
                        np.sqrt(np.mean(strain ** 2))
                    )

                    assert np.isfinite(rms)
                    assert np.isfinite(A1).all()
                    assert np.all(
                        np.abs(strain) <= bound + 1e-10
                    )
                    assert np.allclose(
                        delta_angles, 0.0, atol=1e-8
                    )

                    accepted = (
                        rms == 0.0
                        if name == "zero"
                        else low <= rms < high
                    )

                    if accepted:
                        break

                if not accepted:
                    raise RuntimeError(
                        f"Sampling failed: {db_id}, {name}"
                    )

                # Preserve the RNG state after deformation.
                x, y = matgen_pxrdsim(
                    deformed,
                    float(control["grain_size_input_A"]),
                    [float(control["orientation_randomness_bound"])] * 2,
                    float(control["thermal_displacement_A"]),
                    False,
                    float(control["simxrd_L"]),
                    float(control["simxrd_H"]),
                    float(control["simxrd_S"]),
                    int(control["background_order"]),
                    float(control["background_ratio"]),
                    float(control["noise_ratio"]),
                    bound,
                    0.0,
                    control["xrd_mode"],
                )

                x = np.asarray(x)
                y = np.asarray(y, dtype=np.float64)

                # Corrected validation: canonical SimXRD grid.
                if not np.array_equal(x, grid):
                    raise ValueError(
                        f"Grid mismatch: db_id={db_id}, "
                        f"sample={sample_in_structure}"
                    )

                if y.shape != (N,) or not np.isfinite(y).all():
                    raise ValueError(
                        f"Invalid intensity: db_id={db_id}, "
                        f"sample={sample_in_structure}"
                    )

                if y.max() <= 0:
                    raise ValueError("Non-positive XRD maximum")

                perturbed[row_id] = y
                lattice_before[row_id] = A0
                lattice_after[row_id] = A1

                sample_rows.append({
                    "xrd_row": row_id,
                    "structure_index": structure_index,
                    "sample_in_structure": sample_in_structure,
                    "source_db_id": db_id,
                    "mp_id": record["mp_id"],
                    "split": record["split"],
                    "rms_group": name,
                    "stretch_bound": bound,
                    "shear_bound": 0.0,
                    "random_seed": seed,
                    "sampling_attempt": attempt + 1,
                    "strain_rms": rms,
                    "strain_a": float(strain[0]),
                    "strain_b": float(strain[1]),
                    "strain_c": float(strain[2]),
                    "delta_alpha": float(delta_angles[0]),
                    "delta_beta": float(delta_angles[1]),
                    "delta_gamma": float(delta_angles[2]),
                    "grain_size_nm": control["grain_size_nm"],
                    "thermal_displacement_A": control["thermal_displacement_A"],
                    "orientation_randomness_bound": control["orientation_randomness_bound"],
                    "background_ratio": control["background_ratio"],
                    "noise_ratio": control["noise_ratio"],
                    "zero_shift_deg": control["zero_shift_deg"],
                })

                row_id += 1
                sample_in_structure += 1

                if row_id % 10 == 0:
                    print(
                        f"Generated {row_id}/90",
                        flush=True
                    )

        assert sample_in_structure == 30

assert row_id == 90

for arr in [ideal, perturbed, lattice_before, lattice_after]:
    arr.flush()

def write_csv(filename, records):
    with (OUT / filename).open(
        "x", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(records[0].keys())
        )
        writer.writeheader()
        writer.writerows(records)

write_csv("structures.csv", structure_rows)
write_csv("samples.csv", sample_rows)

# ==========================================
# Readback verification
# ==========================================

ideal_read = np.load(OUT / "ideal_xrd.npy", mmap_mode="r")
pert_read = np.load(OUT / "perturbed_xrd.npy", mmap_mode="r")
a0_read = np.load(OUT / "lattice_before.npy", mmap_mode="r")
a1_read = np.load(OUT / "lattice_after.npy", mmap_mode="r")

with (OUT / "samples.csv").open(
    newline="", encoding="utf-8"
) as f:
    saved_rows = list(csv.DictReader(f))

assert ideal_read.shape == (3, 3500)
assert pert_read.shape == (90, 3500)
assert a0_read.shape == (90, 3, 3)
assert a1_read.shape == (90, 3, 3)
assert len(saved_rows) == 90
assert np.isfinite(ideal_read).all()
assert np.isfinite(pert_read).all()

max_rms_error = 0.0

for i, row in enumerate(saved_rows):
    assert int(row["xrd_row"]) == i

    lengths0 = np.linalg.norm(a0_read[i], axis=1)
    lengths1 = np.linalg.norm(a1_read[i], axis=1)

    actual_strain = (lengths1 - lengths0) / lengths0
    actual_rms = float(
        np.sqrt(np.mean(actual_strain ** 2))
    )

    saved_rms = float(row["strain_rms"])
    error = abs(actual_rms - saved_rms)
    max_rms_error = max(max_rms_error, error)

    assert error < 1e-10
    assert np.allclose(
        actual_strain,
        [float(row[f"strain_{axis}"]) for axis in "abc"],
        atol=1e-10,
        rtol=0,
    )

elapsed = time.perf_counter() - start_time

# Write completion marker only after all checks pass.
manifest = {
    "status": "complete",
    "structures": 3,
    "perturbed_xrd": 90,
    "ideal_xrd_shape": [3, 3500],
    "perturbed_xrd_shape": [90, 3500],
    "generation_seconds": round(elapsed, 3),
    "max_rms_readback_error": max_rms_error,
    "configuration_source": str(CONFIG_PATH),
}

with (OUT / "smoke_manifest.json").open(
    "x", encoding="utf-8"
) as f:
    json.dump(manifest, f, indent=2)

print("\n=== FINAL VALIDATION ===")
print("Structures:", len(structure_rows))
print("Perturbed XRD:", len(sample_rows))
print("Ideal shape:", ideal_read.shape)
print("Perturbed shape:", pert_read.shape)
print("A0 shape:", a0_read.shape)
print("A1 shape:", a1_read.shape)
print("Maximum RMS label error:", max_rms_error)
print("Elapsed seconds:", round(elapsed, 3))
print("Output:", OUT)
print("STATUS: PASS")
print("=== STEP 3-40B COMPLETE ===")
