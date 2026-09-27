from ase.db import connect
from Pysimxrd import generator
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------
# 1. Open database
# ----------------------------------
db = connect("../../sim/demo_mp.db")

# Stretching/compression ratios
strain_ratios = [0.0, 0.005, 0.01, 0.02]

plt.figure(figsize=(9, 6))

# ----------------------------------
# 2. Generate XRD patterns
# ----------------------------------
for ratio in strain_ratios:

    # Keep the random deformation direction
    # and other random effects reproducible
    np.random.seed(0)

    if ratio == 0.0:

        # Original structure: no lattice deformation
        x, y = generator.parser(
            db,
            entry_id=8,
            deformation=False
        )

    else:

        # Apply lattice stretching/compression
        x, y = generator.parser(
            db,
            entry_id=8,
            deformation=True,
            lattice_extinction_ratio=ratio,
            lattice_torsion_ratio=0.0
        )

    plt.plot(
        x,
        y,
        label=f"strain range = ±{ratio*100:.1f}%"
    )

# ----------------------------------
# 3. Plot
# ----------------------------------
plt.xlabel("2theta (degree)")
plt.ylabel("Intensity (a.u.)")
plt.title("Effect of Lattice Deformation on AgTlTe2 XRD")
plt.legend()
plt.tight_layout()

plt.savefig(
    "AgTlTe2_internal_stress_comparison.png",
    dpi=300
)

plt.close()

print("Internal-stress deformation test completed")
print("strain ratios :", strain_ratios)
print("Saved : AgTlTe2_internal_stress_comparison.png")
# ----------------------------------
# 4. Zoom around the main peak
# ----------------------------------
plt.figure(figsize=(8, 5))

for ratio in strain_ratios:

    np.random.seed(0)

    if ratio == 0.0:
        x, y = generator.parser(
            db,
            entry_id=8,
            deformation=False
        )
    else:
        x, y = generator.parser(
            db,
            entry_id=8,
            deformation=True,
            lattice_extinction_ratio=ratio,
            lattice_torsion_ratio=0.0
        )

    plt.plot(
        x,
        y,
        label=f"±{ratio*100:.1f}%"
    )

plt.xlim(27, 30)
plt.xlabel("2theta (degree)")
plt.ylabel("Intensity (a.u.)")
plt.title("Main Peak Shift by Lattice Deformation")
plt.legend()
plt.tight_layout()

plt.savefig(
    "AgTlTe2_internal_stress_zoom.png",
    dpi=300
)

plt.close()
