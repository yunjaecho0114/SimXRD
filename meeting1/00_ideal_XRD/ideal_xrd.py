from ase.db import connect
from Pysimxrd.utils.MatgenKit import get_diff
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------
# 1. Open database
# ----------------------------------
db = connect("../../sim/demo_mp.db")

# AgTlTe2 = ID 8
atoms = db.get_atoms(id=8)

# ----------------------------------
# 2. Calculate ideal diffraction peaks
#    No lattice deformation
# ----------------------------------
two_theta, intensity = get_diff(
    atoms,
    deformation=False,
    lattice_extinction_ratio=0.0,
    lattice_torsion_ratio=0.0
)

# ----------------------------------
# 3. Save ideal peak data
# ----------------------------------
np.savetxt(
    "AgTlTe2_ideal.dat",
    np.column_stack((two_theta, intensity)),
    header="2theta(degree) Intensity(a.u.)"
)

# ----------------------------------
# 4. Plot ideal stick XRD
# ----------------------------------
plt.figure(figsize=(9, 6))

plt.vlines(
    two_theta,
    0,
    intensity
)

plt.xlabel("2theta (degree)")
plt.ylabel("Intensity (a.u.)")
plt.title("Ideal XRD of AgTlTe2")
plt.xlim(10, 80)
plt.tight_layout()

plt.savefig(
    "AgTlTe2_ideal.png",
    dpi=300
)

plt.close()

print("Ideal XRD calculation completed")
print("Number of diffraction peaks :", len(two_theta))
print("Saved : AgTlTe2_ideal.dat")
print("Saved : AgTlTe2_ideal.png")
