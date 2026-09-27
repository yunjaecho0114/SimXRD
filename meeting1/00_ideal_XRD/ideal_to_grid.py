import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------
# 1. Load ideal stick XRD
# ---------------------------------------
data = np.loadtxt("AgTlTe2_ideal.dat")

two_theta = data[:, 0]
intensity = data[:, 1]

# ---------------------------------------
# 2. Make the same grid as generator.parser()
# ---------------------------------------
grid = np.arange(10.0, 80.0, 0.02)

ideal_grid = np.zeros(len(grid))

# ---------------------------------------
# 3. Put each ideal peak onto nearest grid point
# ---------------------------------------
indices = []

for theta, inten in zip(two_theta, intensity):

    idx = np.argmin(np.abs(grid - theta))

    # += is used in case multiple peaks fall into the same grid bin
    ideal_grid[idx] += inten

    indices.append(idx)

# ---------------------------------------
# 4. Normalize maximum intensity to 100
# ---------------------------------------
if ideal_grid.max() > 0:
    ideal_grid = ideal_grid / ideal_grid.max() * 100

# ---------------------------------------
# 5. Check results
# ---------------------------------------
print("Original number of peaks :", len(two_theta))
print("Grid length              :", len(grid))
print("Non-zero grid points      :", np.count_nonzero(ideal_grid))
print("Grid min                  :", grid.min())
print("Grid max                  :", grid.max())

unique_indices = len(set(indices))
print("Unique peak bins          :", unique_indices)
print("Peak-bin collisions       :", len(indices) - unique_indices)

# ---------------------------------------
# 6. Save grid XRD
# ---------------------------------------
np.savetxt(
    "AgTlTe2_ideal_grid.dat",
    np.column_stack((grid, ideal_grid)),
    header="2theta(degree) Intensity(a.u.)"
)

# ---------------------------------------
# 7. Plot
# ---------------------------------------
plt.figure(figsize=(9, 6))

plt.plot(grid, ideal_grid)

plt.xlabel("2theta (degree)")
plt.ylabel("Intensity (a.u.)")
plt.title("Ideal XRD on 3500-point grid")
plt.xlim(10, 80)

plt.tight_layout()

plt.savefig(
    "AgTlTe2_ideal_grid.png",
    dpi=300
)

plt.close()

print("Saved : AgTlTe2_ideal_grid.dat")
print("Saved : AgTlTe2_ideal_grid.png")
