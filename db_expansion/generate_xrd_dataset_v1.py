"""Resumable, structure-level SimXRD dataset generation."""

import argparse
import csv
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
from ase.db import connect

from Pysimxrd.utils.funs import prim2conv
from Pysimxrd.utils.MatgenKit import get_diff, matgen_pxrdsim
from db_expansion.safe_xrd_generation import safe_xrd_generation


ROOT = Path("db_expansion/dataset_v1")
CONFIG = ROOT / "generation_config.json"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def atomic_json(path, data):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def validate_chunk(path, structure_index, record, plan_rows):
    """Reject incomplete or inconsistent existing chunks."""
    with np.load(path, allow_pickle=False) as z:
        ideal = z["ideal_xrd"]
        y = z["perturbed_xrd"]
        a0 = z["lattice_before"]
        a1 = z["lattice_after"]
        indices = z["xrd_rows"]
        seeds = z["random_seeds"]
        stored_id = int(z["source_db_id"])
        stored_mp = str(z["mp_id"])

    assert ideal.shape == (3500,)
    assert y.shape == (30, 3500)
    assert a0.shape == (30, 3, 3)
    assert a1.shape == (30, 3, 3)
    assert np.isfinite(ideal).all()
    assert np.isfinite(y).all()
    assert np.isfinite(a0).all()
    assert np.isfinite(a1).all()
    assert np.all(np.max(y, axis=1) > 0)

    expected_indices = [
        int(r["xrd_row"]) for r in plan_rows
    ]
    expected_seeds = [
        int(r["random_seed"]) for r in plan_rows
    ]

    assert indices.tolist() == expected_indices
    assert seeds.tolist() == expected_seeds
    assert stored_id == int(record["db_id"])
    assert stored_mp == record["mp_id"]

    lengths0 = np.linalg.norm(a0, axis=2)
    lengths1 = np.linalg.norm(a1, axis=2)
    strains = (lengths1 - lengths0) / lengths0
    rms = np.sqrt(np.mean(strains ** 2, axis=1))

    for j, r in enumerate(plan_rows):
        assert abs(rms[j] - float(r["strain_rms"])) < 1e-10
        assert np.allclose(
            strains[j],
            [float(r[f"strain_{axis}"]) for axis in "abc"],
            atol=1e-10,
            rtol=0,
        )

    return True


def generate_structure(db, record, plan_rows, cfg, grid):
    db_id = int(record["db_id"])
    atoms = db.get(id=db_id).toatoms()

    before, A0, _ = prim2conv(atoms, False, 0.0, 0.0)

    before = np.asarray(before, dtype=np.float64)
    A0 = np.asarray(A0, dtype=np.float64)

    theta, weights = get_diff(atoms, False, 0.0, 0.0)

    ideal = np.zeros(len(grid), dtype=np.float64)
    start = float(cfg["xrd"]["two_theta_start_deg"])
    stop = float(cfg["xrd"]["two_theta_stop_exclusive_deg"])

    # Prevent peaks outside the requested range from
    # being incorrectly assigned to the endpoint bins.
    for angle, weight in zip(theta, weights):
        angle = float(angle)
        weight = float(weight)

        if not (np.isfinite(angle) and np.isfinite(weight)):
            raise ValueError("Non-finite ideal diffraction peak")

        if start <= angle < stop:
            idx = int(np.argmin(np.abs(grid - angle)))
            ideal[idx] += weight

    if not np.isfinite(ideal).all() or ideal.max() <= 0:
        raise ValueError(f"Invalid ideal XRD: db_id={db_id}")

    ideal *= 100.0 / ideal.max()

    y_all = np.empty((30, len(grid)), dtype=np.float64)
    A0_all = np.empty((30, 3, 3), dtype=np.float64)
    A1_all = np.empty((30, 3, 3), dtype=np.float64)

    control = cfg["controlled_variables"]

    for j, row in enumerate(plan_rows):
        assert int(row["sample_in_structure"]) == j
        assert int(row["source_db_id"]) == db_id

        seed = int(row["random_seed"])
        bound = float(row["stretch_bound"])
        shear = float(row["shear_bound"])

        np.random.seed(seed)

        after, A1, deformed = prim2conv(
            atoms, True, bound, shear
        )

        after = np.asarray(after, dtype=np.float64)
        A1 = np.asarray(A1, dtype=np.float64)

        actual_strain = (
            (after[:3] - before[:3]) / before[:3]
        )
        actual_rms = float(
            np.sqrt(np.mean(actual_strain ** 2))
        )

        expected_rms = float(row["strain_rms"])
        expected_strain = np.asarray([
            float(row[f"strain_{axis}"]) for axis in "abc"
        ])

        actual_angle_delta = after[3:] - before[3:]
        expected_angle_delta = np.asarray([
            float(row[f"delta_{axis}"])
            for axis in ("alpha", "beta", "gamma")
        ])

        if (
            not np.isfinite(A1).all()
            or abs(actual_rms - expected_rms) >= 1e-10
            or not np.allclose(
                actual_strain,
                expected_strain,
                atol=1e-10,
                rtol=0,
            )
            or not np.allclose(
                actual_angle_delta,
                expected_angle_delta,
                atol=1e-8,
                rtol=0,
            )
        ):
            raise ValueError(
                f"Strain plan mismatch: "
                f"db_id={db_id}, sample={j}"
            )

        # Keep RNG state after applying deformation.
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
            shear,
            control["xrd_mode"],
        )

        x = np.asarray(x)
        y = np.asarray(y, dtype=np.float64)

        if not np.array_equal(x, grid):
            raise ValueError(
                f"Grid mismatch: db_id={db_id}, sample={j}"
            )

        if (
            y.shape != (3500,)
            or not np.isfinite(y).all()
            or y.max() <= 0
        ):
            raise ValueError(
                f"Invalid XRD: db_id={db_id}, sample={j}"
            )

        y_all[j] = y
        A0_all[j] = A0
        A1_all[j] = A1

    return {
        "ideal_xrd": ideal,
        "perturbed_xrd": y_all,
        "lattice_before": A0_all,
        "lattice_after": A1_all,
        "xrd_rows": np.asarray(
            [int(r["xrd_row"]) for r in plan_rows],
            dtype=np.int64,
        ),
        "random_seeds": np.asarray(
            [int(r["random_seed"]) for r in plan_rows],
            dtype=np.int64,
        ),
        "source_db_id": np.asarray(db_id, dtype=np.int64),
        "mp_id": np.asarray(record["mp_id"]),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--structures", type=int, choices=[3, 1000],
        required=True,
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    output = Path(args.output)

    with CONFIG.open(encoding="utf-8") as f:
        cfg = json.load(f)

    selection_path = Path(cfg["source"]["selection_csv"])
    plan_path = ROOT / "strain_plan_1000x30_v1.csv"

    structures = read_csv(selection_path)
    plan = read_csv(plan_path)

    assert len(structures) == 1000
    assert len(plan) == 30000
    assert cfg["strain"]["shear_bound"] == 0.0

    for i in range(args.structures):
        record = structures[i]
        rows = plan[i * 30:(i + 1) * 30]

        assert len(rows) == 30
        assert [
            int(r["sample_in_structure"]) for r in rows
        ] == list(range(30))
        assert [
            int(r["xrd_row"]) for r in rows
        ] == list(range(i * 30, (i + 1) * 30))

        for row in rows:
            assert int(row["structure_index"]) == i
            assert int(row["source_db_id"]) == int(record["db_id"])
            assert row["mp_id"] == record["mp_id"]
            assert row["split"] == record["split"]

    grid = np.arange(
        float(cfg["xrd"]["two_theta_start_deg"]),
        float(cfg["xrd"]["two_theta_stop_exclusive_deg"]),
        float(cfg["xrd"]["step_deg"]),
    )
    assert grid.shape == (3500,)

    inputs = {
        "config_sha256": sha256(CONFIG),
        "selection_sha256": sha256(selection_path),
        "plan_sha256": sha256(plan_path),
        "generator_sha256": sha256(Path(__file__)),
        "source_db_size": Path(
            cfg["source"]["structure_db"]
        ).stat().st_size,
        "source_db_mtime_ns": Path(
            cfg["source"]["structure_db"]
        ).stat().st_mtime_ns,
        "structure_count": args.structures,
    }

    print("=== PRODUCTION GENERATOR ===", flush=True)
    print("Structures:", args.structures)
    print("Expected XRD:", args.structures * 30)
    print("Output:", output)
    print("Mode:", "DRY RUN" if args.dry_run else "RUN")

    if args.dry_run:
        print("Input validation: PASS")
        print("No dataset files created")
        print("STATUS: PASS")
        return

    # Prevent two generators from writing to the same dataset.
    import fcntl

    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".generator.lock"

    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(
                lock.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError:
            raise RuntimeError(
                "Another generator is using this output directory"
            )

        manifest_path = output / "generation_manifest.json"
        chunks = output / "chunks"

        if manifest_path.exists():
            with manifest_path.open() as f:
                previous = json.load(f)
            if previous != inputs:
                raise RuntimeError(
                    "Existing dataset has different source "
                    "files, parameters or generator version"
                )
        else:
            existing = [
                p for p in output.iterdir()
                if p.name != ".generator.lock"
            ]
            if existing:
                raise RuntimeError(
                    "Output is not empty and has no manifest"
                )
            atomic_json(manifest_path, inputs)

        chunks.mkdir(exist_ok=True)
        db = connect(cfg["source"]["structure_db"])

        created = 0
        skipped = 0
        started = time.perf_counter()

        with safe_xrd_generation():
            for i in range(args.structures):
                record = structures[i]
                rows = plan[i * 30:(i + 1) * 30]

                target = chunks / f"structure_{i:04d}.npz"

                if target.exists():
                    validate_chunk(target, i, record, rows)
                    skipped += 1
                    continue

                if created >= args.batch_size:
                    break

                tmp = chunks / (
                    f".structure_{i:04d}.{os.getpid()}.tmp.npz"
                )

                try:
                    data = generate_structure(
                        db, record, rows, cfg, grid
                    )

                    # Finish writing and fsync before atomic rename.
                    with tmp.open("xb") as f:
                        np.savez(f, **data)
                        f.flush()
                        os.fsync(f.fileno())

                    os.replace(tmp, target)
                    validate_chunk(target, i, record, rows)

                except Exception:
                    error_path = output / "failures.jsonl"
                    with error_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "structure_index": i,
                            "source_db_id": int(record["db_id"]),
                            "mp_id": record["mp_id"],
                            "traceback": traceback.format_exc(),
                        }, ensure_ascii=False) + "\n")
                        f.flush()
                        os.fsync(f.fileno())
                    raise

                created += 1

                print(
                    f"COMPLETE structure={i+1}/{args.structures} "
                    f"mp_id={record['mp_id']} "
                    f"new={created} "
                    f"elapsed={time.perf_counter()-started:.1f}s",
                    flush=True,
                )

        completed = sum(
            (chunks / f"structure_{i:04d}.npz").exists()
            for i in range(args.structures)
        )

        print("\n=== RUN SUMMARY ===")
        print("New structures:", created)
        print("Previously completed:", skipped)
        print("Total completed:", completed)
        print("Expected:", args.structures)
        print(
            "STATUS:",
            "COMPLETE" if completed == args.structures
            else "PARTIAL - RESUMABLE",
        )


if __name__ == "__main__":
    main()
