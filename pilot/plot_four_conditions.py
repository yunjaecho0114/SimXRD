"""Plot the four verified pilot conditions from an existing ASE DB.

Reads the stored grid and arrays only. No simulation or database writes.
Run on HANS from the repository root:
    python -B pilot/plot_four_conditions.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.db import connect


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "pilot" / "agtlte2_four_conditions.db"
DEFAULT_PLOTS = ROOT / "pilot" / "plots"
COLORS = ["#4477AA", "#EE6677", "#228833", "#AA3377"]


def read_pilot(database_path):
    if not database_path.is_file():
        raise FileNotFoundError(database_path)
    with connect(str(database_path)) as db:
        if db.count() != 4:
            raise ValueError("Expected exactly four stored pilot conditions")
        grid = np.asarray(db.metadata["x_grid"], dtype=float)
        rows = [db.get(random_seed=seed) for seed in range(4)]
        conditions = []
        for number, row in enumerate(rows, start=1):
            conditions.append({
                "number": number,
                "structure_id": row.structure_id,
                "sample_id": row.sample_id,
                "metadata": dict(row.key_value_pairs),
                "ideal": np.asarray(row.data["ideal_xrd"], dtype=float),
                "perturbed": np.asarray(row.data["perturbed_xrd"], dtype=float),
            })
    if grid.shape != (3500,) or not np.isfinite(grid).all() or not np.all(np.diff(grid) > 0):
        raise ValueError("Invalid stored x-grid")
    if len({item["structure_id"] for item in conditions}) != 1:
        raise ValueError("Conditions do not share a structure")
    reference = conditions[0]["ideal"]
    for item in conditions:
        if any(array.shape != grid.shape or not np.isfinite(array).all()
               for array in (item["ideal"], item["perturbed"])):
            raise ValueError(f"Invalid XRD array for Condition #{item['number']}")
        if not np.array_equal(item["ideal"], reference):
            raise ValueError("Ideal XRD arrays differ between conditions")
    return grid, reference, conditions


def condition_title(item):
    m = item["metadata"]
    return (
        f"Condition #{item['number']} | grain {m['grain_size_nm']:g} nm, "
        f"thermal {m['thermal_displacement_A']:g} Å, "
        f"stretch {m['lattice_stretch_bound']:g}, shear {m['lattice_shear_bound']:g}\n"
        f"zero shift {m['zero_shift_deg']:+g}°, background {m['background_ratio']:g}, "
        f"noise {m['noise_ratio']:g}"
    )


def save_overview(grid, ideal, conditions, plots):
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    # Fixed offsets make five curves legible without changing stored values.
    offset = 110.0
    ax.plot(grid, ideal, color="black", lw=1.0, label="Ideal")
    for item, color in zip(conditions, COLORS):
        ax.plot(grid, item["perturbed"] + offset * item["number"],
                color=color, lw=0.9, label=f"Condition #{item['number']}")
    ax.set(xlabel="2θ (degree)", ylabel="Intensity (a.u.) + display offset",
           title="AgTlTe2: ideal and four stored perturbed XRD patterns",
           xlim=(grid[0], grid[-1]))
    ax.set_yticks([0, 110, 220, 330, 440],
                  ["Ideal", "#1", "#2", "#3", "#4"])
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", ncol=5, fontsize=8)
    fig.savefig(plots / "figure1_ideal_vs_conditions.png", dpi=200)
    plt.close(fig)


def save_panels(grid, conditions, plots):
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True,
                             constrained_layout=True)
    for item, ax, color in zip(conditions, axes, COLORS):
        ax.plot(grid, item["perturbed"], color=color, lw=0.9)
        ax.set_title(condition_title(item), loc="left", fontsize=9)
        ax.set_ylabel("Intensity (a.u.)")
        ax.grid(alpha=0.2)
    axes[-1].set(xlabel="2θ (degree)", xlim=(grid[0], grid[-1]))
    fig.savefig(plots / "figure2_condition_panels.png", dpi=200)
    plt.close(fig)


def save_nuisance_zoom(grid, ideal, conditions, plots):
    no_nuisance = conditions[1]["perturbed"]  # #2 has the same control bounds as #4.
    nuisance = conditions[3]["perturbed"]
    # Center on the strongest stored ideal peak away from plot boundaries.
    interior = (grid >= grid[0] + 2) & (grid <= grid[-1] - 2)
    peak_index = np.flatnonzero(interior)[np.argmax(ideal[interior])]
    peak_center = grid[peak_index]
    # A low-ideal-intensity region makes the raised background/noise visible.
    window_points = max(1, int(round(2.0 / np.median(np.diff(grid)))))
    score = np.convolve(ideal, np.ones(window_points), mode="valid")
    starts = np.arange(len(score))
    eligible = (grid[starts] >= grid[0] + 2) & (grid[starts + window_points - 1] <= grid[-1] - 2)
    quiet_start = starts[eligible][np.argmin(score[eligible])]
    quiet_center = (grid[quiet_start] + grid[quiet_start + window_points - 1]) / 2

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    for ax, center, title in zip(
        axes, [peak_center, quiet_center],
        [f"Strongest ideal peak near {peak_center:.2f}°",
         f"Low-ideal-intensity window near {quiet_center:.2f}°"],
    ):
        selected = np.abs(grid - center) <= 1.5
        ax.plot(grid[selected], ideal[selected], color="black", lw=1.1, label="Ideal")
        ax.plot(grid[selected], no_nuisance[selected], color=COLORS[1], lw=1.1,
                label="Condition #2: nuisance off, seed 1")
        ax.plot(grid[selected], nuisance[selected], color=COLORS[3], lw=1.1,
                label="Condition #4: nuisance on, seed 3")
        ax.set(title=title, xlabel="2θ (degree)", ylabel="Intensity (a.u.)")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Condition #4 nuisance inspection — seeds and deformation realizations also differ",
                 fontsize=11)
    fig.savefig(plots / "figure3_condition4_nuisance_zoom.png", dpi=200)
    plt.close(fig)
    return peak_center, quiet_center


def print_statistics(ideal, conditions):
    print("Condition | min | max | mean | nonzero | MAE vs ideal | RMSE vs ideal")
    for item in conditions:
        y = item["perturbed"]
        difference = y - ideal
        mae = float(np.mean(np.abs(difference)))
        rmse = float(np.sqrt(np.mean(difference**2)))
        print(f"#{item['number']:d} | {y.min():.6g} | {y.max():.6g} | "
              f"{y.mean():.6g} | {np.count_nonzero(y)} | {mae:.6g} | {rmse:.6g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS)
    args = parser.parse_args()
    grid, ideal, conditions = read_pilot(args.db.resolve())
    plots = args.plots_dir.resolve()
    plots.mkdir(parents=True, exist_ok=True)
    save_overview(grid, ideal, conditions, plots)
    save_panels(grid, conditions, plots)
    peak_center, quiet_center = save_nuisance_zoom(grid, ideal, conditions, plots)
    print_statistics(ideal, conditions)
    print(f"Figure 3 windows: peak {peak_center:.2f}°, quiet {quiet_center:.2f}°")
    print("Condition #2 and #4 differ in random seed and actual deformation as well as nuisance effects.")
    for filename in ("figure1_ideal_vs_conditions.png", "figure2_condition_panels.png",
                     "figure3_condition4_nuisance_zoom.png"):
        print(plots / filename)


if __name__ == "__main__":
    main()
