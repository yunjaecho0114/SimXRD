from ase.db import connect
from Pysimxrd import generator
import numpy as np
import matplotlib.pyplot as plt

# Open database
db = connect("../../sim/demo_mp.db")

# Idealized XRD using the same PysimXRD parser
x, y = generator.parser(
    db,
    entry_id=8,

    # Make grain-size broadening almost negligible
    grainsize=1000000,

    # Remove preferred-orientation effect
    prefect_orientation=[0.0, 0.0],

    # Remove thermal vibration
    thermo_vibration=0.0,

    # No zero shift
    zero_shift=0.0,

    # No lattice deformation
    deformation=False,

    # No background
    background_ratio=0.0,

    # No noise
    mixture_noise_ratio=0.0
)

# Save data
np.savetxt(
    "AgTlTe2_idealized_parser.dat",
    np.column_stack((x, y)),
    header="2theta(degree) Intensity(a.u.)"
)

# Plot
plt.figure(figsize=(9, 6))
plt.plot(x, y)

plt.xlabel("2theta (degree)")
plt.ylabel("Intensity (a.u.)")
plt.title("Idealized XRD of AgTlTe2 - PysimXRD parser")
plt.tight_layout()

plt.savefig(
    "AgTlTe2_idealized_parser.png",
    dpi=300
)

plt.close()

print("Idealized parser XRD completed")
print("Number of points :", len(x))
print("Saved : AgTlTe2_idealized_parser.png")
