from ase.db import connect
from Pysimxrd import generator
import numpy as np
import matplotlib.pyplot as plt

# Open database
db = connect("../../sim/demo_mp.db")

# Preferred-orientation values to compare
orientation_values = [0.0, 0.1, 0.2, 0.4]

plt.figure(figsize=(9, 6))

for ori in orientation_values:

    # Reset random numbers so all other
    # random conditions remain identical
    np.random.seed(0)

    x, y = generator.parser(
        db,
        entry_id=8,
        prefect_orientation=[ori, ori]
    )

    plt.plot(
        x,
        y,
        label=f"orientation = {ori}"
    )

plt.xlabel("2theta (degree)")
plt.ylabel("Intensity (a.u.)")
plt.title("Effect of Preferred Orientation on AgTlTe2 XRD")
plt.legend()
plt.tight_layout()

plt.savefig(
    "AgTlTe2_orientation_comparison.png",
    dpi=300
)

plt.close()

print("Preferred orientation comparison completed")
print("orientation values :", orientation_values)
print("Default value      : 0.1")
print("Saved : AgTlTe2_orientation_comparison.png")
