"""Generate a paired XRD sample with its actual lattice strain label.

The strain label is computed from the lattice geometry after applying
a random deformation, not from the requested deformation bound.

The same deformed structure is used to calculate the perturbed XRD.

Example:
    python collaboration/examples/03_strain_labeling.py \
        --entry-id 8 --stretch-bound 0.01 --seed 0
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

from Pysimxrd.utils.funs import prim2conv
from Pysimxrd.utils.MatgenKit import get_diff, matgen_pxrdsim


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "sim" / "demo_mp.db"
DEFAULT_OUTPUT = ROOT / "collaboration" / "outputs"

# Import repository-local numerical stability utilities.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db_expansion.safe_xrd_generation import safe_xrd_generation


def generate_ideal(atoms):
    """Generate the undeformed ideal stick pattern on a common grid."""

    theta, weights = get_diff(
        atoms, False, 0.0, 0.0
    )

    theta = np.asarray(theta, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if (
        theta.ndim != 1
        or theta.shape != weights.shape
        or not np.isfinite(theta).all()
        or not np.isfinite(weights).all()
    ):
        raise ValueError("Invalid ideal diffraction peaks.")

    grid = np.arange(10.0, 80.0, 0.02)
    ideal = np.zeros(grid.size, dtype=np.float64)

    for angle, weight in zip(theta, weights):
        if 10.0 <= angle < 80.0:
            index = int(np.argmin(np.abs(grid - angle)))
            ideal[index] += weight

    if ideal.max() <= 0:
        raise ValueError("Ideal XRD contains no positive peaks.")

    ideal *= 100.0 / ideal.max()

    return grid, ideal


def calculate_actual_strain(before, after, A0, A1):
    """Compute actual axis strains and RMS from lattice lengths."""

    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    A0 = np.asarray(A0, dtype=np.float64)
    A1 = np.asarray(A1, dtype=np.float64)

    if (
        before.shape != (6,)
        or after.shape != (6,)
        or A0.shape != (3, 3)
        or A1.shape != (3, 3)
        or not np.isfinite(before).all()
        or not np.isfinite(after).all()
        or not np.isfinite(A0).all()
        or not np.isfinite(A1).all()
    ):
        raise ValueError("Invalid lattice geometry.")

    if np.any(before[:3] <= 0):
        raise ValueError("Original lattice lengths must be positive.")

    # Axis strains from conventional-cell lattice parameters.
    strain_abc = (after[:3] - before[:3]) / before[:3]

    strain_rms = float(
        np.sqrt(np.mean(strain_abc ** 2))
    )

    delta_angles = after[3:] - before[3:]

    # Independently verify labels against the stored lattice matrices.
    lengths0 = np.linalg.norm(A0, axis=1)
    lengths1 = np.linalg.norm(A1, axis=1)

    matrix_strain = (lengths1 - lengths0) / lengths0

    if not np.allclose(
        strain_abc,
        matrix_strain,
        atol=1e-10,
        rtol=0,
    ):
        raise ValueError(
            "Strain labels do not match the lattice matrices."
        )

    return strain_abc, strain_rms, delta_angles


def validate_args(args):
    """Validate simulation parameters before generating any output."""

    if args.entry_id < 1:
        raise ValueError("entry-id must be positive.")

    if (
        not np.isfinite(args.grain_size_nm)
        or args.grain_size_nm <= 0
    ):
        raise ValueError("grain-size-nm must be finite and positive.")

    parameters = {
        "stretch-bound": args.stretch_bound,
        "shear-bound": args.shear_bound,
        "thermal-displacement-A": args.thermal_displacement_A,
        "orientation-bound": args.orientation_bound,
        "background-ratio": args.background_ratio,
        "noise-ratio": args.noise_ratio,
    }

    for name, value in parameters.items():
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
    parser.add_argument("--stretch-bound", type=float, default=0.01)
    parser.add_argument("--shear-bound", type=float, default=0.0)

    parser.add_argument(
        "--thermal-displacement-A",
        type=float,
        default=0.0,
    )
    parser.add_argument("--orientation-bound", type=float, default=0.0)
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
    validate_args(args)

    if not args.db.is_file():
        raise FileNotFoundError(args.db)

    name = args.name or f"strain_entry_{args.entry_id}_seed_{args.seed}"

    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Invalid output name.")

    output_dir = args.output_dir.resolve()

    npz_path = output_dir / f"{name}.npz"
    json_path = output_dir / f"{name}.json"
    png_path = output_dir / f"{name}.png"

    for path in (npz_path, json_path, png_path):
        if path.exists():
            raise FileExistsError(f"Output exists: {path}")

    db = connect(str(args.db))
    row = db.get(id=args.entry_id)
    atoms = row.toatoms()

    # 1. Undeformed ideal reference XRD.
    grid, ideal = generate_ideal(atoms)

    # 2. Original conventional lattice.
    before, A0, _ = prim2conv(
        atoms, False, 0.0, 0.0
    )

    # 3. Apply exactly one random lattice deformation.
    np.random.seed(args.seed)

    after, A1, deformed = prim2conv(
        atoms,
        True,
        args.stretch_bound,
        args.shear_bound,
    )

    # 4. Calculate actual strain labels.
    strain_abc, strain_rms, delta_angles = (
        calculate_actual_strain(before, after, A0, A1)
    )

    # 5. Preserve RNG state after deformation.
    #    The XRD must be calculated from the SAME deformed lattice.
    #    deformation=False prevents a second lattice deformation.
    with safe_xrd_generation():
        x, perturbed = matgen_pxrdsim(
            deformed,
            args.grain_size_nm * 10.0,
            [
                args.orientation_bound,
                args.orientation_bound,
            ],
            args.thermal_displacement_A,
            False,
            50.0,
            5.0,
            2.5,
            6,
            args.background_ratio,
            args.noise_ratio,
            args.stretch_bound,
            args.shear_bound,
            "reciprocal",
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

    # 6. Store the XRD pair and its ACTUAL physical label.
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        npz_path,
        two_theta_deg=grid,
        ideal_xrd=ideal,
        perturbed_xrd=perturbed,
        lattice_before=np.asarray(A0),
        lattice_after=np.asarray(A1),
        strain_abc=strain_abc,
        strain_rms=np.asarray(strain_rms),
        delta_angles_deg=delta_angles,
    )

    metadata = {
        "structure_formula": row.formula,
        "ase_db_entry_id": args.entry_id,
        "random_seed": args.seed,
        "simulation_engine": "pymatgen-based SimXRD",
        "label_definition": "lattice_axis_length_rms",
        "grain_size_nm": args.grain_size_nm,
        "stretch_bound": args.stretch_bound,
        "shear_bound": args.shear_bound,
        "thermal_displacement_A": args.thermal_displacement_A,
        "orientation_bound": args.orientation_bound,
        "background_ratio": args.background_ratio,
        "noise_ratio": args.noise_ratio,
        "strain_a": float(strain_abc[0]),
        "strain_b": float(strain_abc[1]),
        "strain_c": float(strain_abc[2]),
        "strain_rms": strain_rms,
        "delta_alpha_deg": float(delta_angles[0]),
        "delta_beta_deg": float(delta_angles[1]),
        "delta_gamma_deg": float(delta_angles[2]),
        "actual_label_calculated": True,
    }

    with json_path.open("x", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 7. Plot the perturbed profile and ideal reference.
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True
    )

    ax1.plot(grid, perturbed, linewidth=1)
    ax1.set_title("Perturbed XRD")
    ax1.set_ylabel("Intensity (max = 100)")

    ax2.vlines(grid[ideal > 0], 0, ideal[ideal > 0])
    ax2.set_title("Ideal reference XRD")
    ax2.set_xlabel("2-theta (degree)")
    ax2.set_ylabel("Intensity (max = 100)")
    ax2.set_xlim(10.0, 80.0)

    fig.suptitle(
        f"{row.formula} | Actual strain RMS = {strain_rms:.6f}"
    )

    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    print("=== ACTUAL STRAIN LABELING COMPLETE ===")
    print("Structure:", row.formula)
    print("Random seed:", args.seed)
    print("Stretch bound:", args.stretch_bound)
    print("Shear bound:", args.shear_bound)
    print("Actual strain_a:", float(strain_abc[0]))
    print("Actual strain_b:", float(strain_abc[1]))
    print("Actual strain_c:", float(strain_abc[2]))
    print("Actual strain_rms:", strain_rms)
    print("Angle changes (deg):", delta_angles.tolist())
    print("XRD shape:", perturbed.shape)
    print("Saved labeled pair:", npz_path)
    print("Saved metadata:", json_path)
    print("Saved plot:", png_path)


if __name__ == "__main__":
    main()
