from ase.db import connect
from Pysimxrd import generator
import numpy as np
import matplotlib.pyplot as plt

# Open demo database
db = connect("../../sim/demo_mp.db")

# AgTlTe2 = database ID 8
x, y = generator.parser(
    db,
    entry_id=8
)

# Save numerical data
np.savetxt(
    "AgTlTe2_default.dat",
    np.column_stack((x, y)),
    header="2theta(degree) Intensity(a.u.)"
)

# Plot XRD
plt.figure(figsize=(8, 5))
plt.plot(x, y)
plt.xlabel("2theta (degree)")
plt.ylabel("Intensity (a.u.)")
plt.title("AgTlTe2 - PysimXRD Default")
plt.tight_layout()
plt.savefig("AgTlTe2_default.png", dpi=300)
plt.close()

print("Simulation completed")
print("Number of points :", len(x))
print("Saved:")
print("AgTlTe2_default.dat")
print("AgTlTe2_default.png")
