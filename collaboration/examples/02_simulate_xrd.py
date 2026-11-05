"""Generate paired ideal and perturbed XRD patterns.

This example uses the pymatgen-based SimXRD engine.

Physical-variable inputs specify simulation settings or bounds,
not measured physical-variable labels.

Example:
    python collaboration/examples/02_simulate_xrd.py \
        --entry-id 8 --grain-size-nm 10 --stretch-bound 0.01
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.db import connect

from Pysimxrd import generator
from Pysimxrd.utils.MatgenKit import get_diff


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "sim" / "demo_mp.db"
DEFAULT_OUTPUT = ROOT / "collaboration" / "outputs"

# Make the repository-local optimization module importable.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db_expansion.safe_xrd_generation import safe_xrd_generation


def generate_ideal(atoms):
    """Calculate the undeformed ideal stick XRD on the common grid."""

    theta, weights = get_diff(
        atoms,
        deformation=False,
        lattice_extinction_ratio=0.0,
        lattice_torsion_ratio=0.0,
    )

    theta = np.asarray(theta, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    grid = np.arange(10.0, 80.0, 0.02)
    ideal = np.zeros(grid.size, dtype=np.float64)

    if (
        theta.ndim != 1
        or theta.shape != weights.shape
        or not np.isfinite(theta).all()
        or not np.isfinite(weights).all()
    ):
        raise ValueError("Invalid ideal diffraction peaks.")

    for angle, weight in zip(theta, weights):
        if 10.0 <= angle < 80.0:
            index = int(np.argmin(np.abs(grid - angle)))
            ideal[index] += weight

    if ideal.max() <= 0:
        raise ValueError("Ideal XRD has no positive peaks.")

    ideal *= 100.0 / ideal.max()

    return grid, ideal


def validate_settings(args):
    """Reject invalid physical-variable settings."""

    if args.entry_id < 1:
        raise ValueError("entry-id must be positive.")

    if args.grain_size_nm <= 0:
        raise ValueError("grain-size-nm must be positive.")

    settings = {
        "thermal-displacement-A": args.thermal_displacement_A,
        "orientation-bound": args.orientation_bound,
        "stretch-bound": args.stretch_bound,
        "shear-bound": args.shear_bound,
        "background-ratio": args.background_ratio,
        "noise-ratio": args.noise_ratio,
    }

    for name, value in settings.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")

    if args.stretch_bound >= 1:
        raise ValueError("stretch-bound must be smaller than 1.")

    if args.orientation_bound > 1:
        raise ValueError("orientation-bound must not exceed 1.")

    if not 0 <= args.seed < 2**32:
        raise ValueError("seed must be in [0, 2**32).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--entry-id", type=int, default=8)

    parser.add_argument("--grain-size-nm", type=float, default=10.0)
    parser.add_argument(
        "--thermal-displacement-A",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--orientation-bound",
        type=float,
        default=0.0,
    )
    parser.add_argument("--stretch-bound", type=float, default=0.01)
    parser.add_argument("--shear-bound", type=float, default=0.0)
    parser.add_argument("--background-ratio", type=float, default=0.0)
    parser.add_argument("--noise-ratio", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()
    validate_settings(args)

    if not args.db.is_file():
        raise FileNotFoundError(args.db)

    name = args.name or f"simulation_entry_{args.entry_id}_seed_{args.seed}"

    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError(
            "Output name may contain only letters, digits, _ and -."
        )

    output_dir = args.output_dir.resolve()

    npz_path = output_dir / f"{name}.npz"
    json_path = output_dir / f"{name}.json"
    png_path = output_dir / f"{name}.png"

    # Never silently overwrite existing simulation results.
    for path in (npz_path, json_path, png_path):
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")

    db = connect(str(args.db))
    row = db.get(id=args.entry_id)
    atoms = row.toatoms()

    grid, ideal = generate_ideal(atoms)

    # SimXRD expects grain size in Angstroms.
    grain_size_A = args.grain_size_nm * 10.0

    # Fix the random stream for reproducible simulation.
    np.random.seed(args.seed)

    # The optimization patches are process-global.
    # Do not use this context concurrently across threads.
    with safe_xrd_generation():
        x, perturbed = generator.parser(
            database=db,
            entry_id=args.entry_id,
            grainsize=grain_size_A,
            prefect_orientation=[
                args.orientation_bound,
                args.orientation_bound,
            ],
            thermo_vibration=args.thermal_displacement_A,
            zero_shift=0.0,
            dis_detector2sample=500.0,
            half_height_slit_detector=5.0,
            half_height_sample=2.5,
            deformation=True,
            sim_model=None,
            xrd="reciprocal",
            background_order=6,
            background_ratio=args.background_ratio,
            mixture_noise_ratio=args.noise_ratio,
            lattice_extinction_ratio=args.stretch_bound,
            lattice_torsion_ratio=args.shear_bound,
        )

    x = np.asarray(x, dtype=np.float64)
    perturbed = np.asarray(perturbed, dtype=np.float64)

    if not np.array_equal(x, grid):
        raise ValueError("Simulated XRD grid mismatch.")

    if (
        perturbed.shape != (3500,)
        or not np.isfinite(perturbed).all()
        or perturbed.max() <= 0
    ):
        raise ValueError("Invalid perturbed XRD intensity.")

    metadata = {
        "structure_formula": row.formula,
        "ase_db_entry_id": args.entry_id,
        "random_seed": args.seed,
        "simulation_engine": "pymatgen-based SimXRD",
        "grain_size_nm": args.grain_size_nm,
        "grain_size_input_A": grain_size_A,
        "thermal_displacement_A": args.thermal_displacement_A,
        "orientation_bound": args.orientation_bound,
        "stretch_bound": args.stretch_bound,
        "shear_bound": args.shear_bound,
        "background_ratio": args.background_ratio,
        "noise_ratio": args.noise_ratio,
        "zero_shift_deg": 0.0,
        "two_theta_start_deg": 10.0,
        "two_theta_stop_exclusive_deg": 80.0,
        "two_theta_step_deg": 0.02,
        "label_status": "settings_only_no_actual_strain_label",
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        npz_path,
        two_theta_deg=grid,
        ideal_xrd=ideal,
        perturbed_xrd=perturbed,
    )

    with json_path.open("x", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True
    )

    ax1.plot(grid, perturbed, linewidth=1.0)
    ax1.set_title("Perturbed XRD")
    ax1.set_ylabel("Intensity (max = 100)")

    positions = grid[ideal > 0]
    intensities = ideal[ideal > 0]

    ax2.vlines(positions, 0, intensities)
    ax2.set_title("Ideal Reference XRD (stick)")
    ax2.set_xlabel("2-theta (degree)")
    ax2.set_ylabel("Intensity (max = 100)")
    ax2.set_xlim(10.0, 80.0)

    fig.suptitle(
        f"{row.formula} | Grain size = {args.grain_size_nm:g} nm"
    )

    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    print("=== PERTURBED XRD SIMULATION COMPLETE ===")
    print("Structure:", row.formula)
    print("Random seed:", args.seed)
    print("Grid shape:", perturbed.shape)
    print("Ideal maximum:", ideal.max())
    print("Perturbed maximum:", perturbed.max())
    print("Stretch bound:", args.stretch_bound)
    print("Shear bound:", args.shear_bound)
    print("Actual strain label: NOT CALCULATED")
    print("Saved XRD pair:", npz_path)
    print("Saved settings:", json_path)
    print("Saved plot:", png_path)


if __name__ == "__main__":
    main()
