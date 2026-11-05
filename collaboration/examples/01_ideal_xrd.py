"""Generate an ideal stick XRD pattern on the shared 3,500-point grid.

The reference pattern is calculated from an undeformed crystal structure.
It is a stick pattern mapped to a grid, not an experimental XRD profile.

Example:
    python collaboration/examples/01_ideal_xrd.py --entry-id 8
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.db import connect

from Pysimxrd.utils.MatgenKit import get_diff


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "sim" / "demo_mp.db"
DEFAULT_OUTPUT = ROOT / "collaboration" / "outputs"

START_DEG = 10.0
STOP_DEG = 80.0
STEP_DEG = 0.02


def generate_ideal_xrd(atoms):
    """Return ideal peak positions, intensities, grid, and binned pattern."""

    # Calculate the undeformed reference diffraction peaks.
    theta, intensity = get_diff(
        atoms,
        deformation=False,
        lattice_extinction_ratio=0.0,
        lattice_torsion_ratio=0.0,
    )

    theta = np.asarray(theta, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)

    if (
        theta.shape != intensity.shape
        or theta.ndim != 1
        or not np.isfinite(theta).all()
        or not np.isfinite(intensity).all()
    ):
        raise ValueError("Invalid ideal diffraction peaks.")

    # Use the same grid construction as the SimXRD generator.
    grid = np.arange(START_DEG, STOP_DEG, STEP_DEG)
    ideal = np.zeros(grid.size, dtype=np.float64)

    # Exclude peaks outside the requested range.
    valid = (theta >= START_DEG) & (theta < STOP_DEG)

    for angle, weight in zip(theta[valid], intensity[valid]):
        index = int(np.argmin(np.abs(grid - angle)))
        ideal[index] += weight

    if grid.size != 3500 or ideal.max() <= 0:
        raise ValueError("Invalid ideal XRD grid or intensity.")

    # Normalize the maximum intensity to 100.
    ideal *= 100.0 / ideal.max()

    return theta[valid], intensity[valid], grid, ideal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--entry-id", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()

    if not args.db.is_file():
        raise FileNotFoundError(args.db)

    db = connect(str(args.db))
    row = db.get(id=args.entry_id)
    atoms = row.toatoms()

    theta, intensity, grid, ideal = generate_ideal_xrd(atoms)

    output_dir = args.output_dir.resolve()
    prefix = f"ideal_entry_{args.entry_id}"

    npy_path = output_dir / f"{prefix}.npy"
    dat_path = output_dir / f"{prefix}.dat"
    png_path = output_dir / f"{prefix}.png"

    # Avoid overwriting existing results.
    for path in (npy_path, dat_path, png_path):
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the array used by ML models.
    np.save(npy_path, ideal)

    # Save the angle-intensity pairs for inspection.
    np.savetxt(
        dat_path,
        np.column_stack((grid, ideal)),
        header="two_theta_deg intensity",
    )

    # Plot the raw stick peaks and the binned reference.
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True
    )

    ax1.vlines(theta, 0, intensity)
    ax1.set_title(f"Ideal stick XRD: {row.formula}")
    ax1.set_ylabel("Intensity (a.u.)")

    ax2.plot(grid, ideal)
    ax2.set_title("Ideal XRD on the 3,500-point grid")
    ax2.set_xlabel("2-theta (degree)")
    ax2.set_ylabel("Intensity (max = 100)")
    ax2.set_xlim(START_DEG, STOP_DEG)

    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    print("=== IDEAL XRD COMPLETE ===")
    print("Structure:", row.formula)
    print("ASE DB entry:", args.entry_id)
    print("Number of ideal peaks:", len(theta))
    print("Grid shape:", ideal.shape)
    print("Maximum intensity:", ideal.max())
    print("Saved array:", npy_path)
    print("Saved data:", dat_path)
    print("Saved plot:", png_path)


if __name__ == "__main__":
    main()
