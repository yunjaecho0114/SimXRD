from ase.db import connect
from Pysimxrd import generator
import numpy as np
import matplotlib.pyplot as plt

# Open database
db = connect("../../sim/demo_mp.db")

# Thermal-vibration values to compare
thermal_values = [0.0, 0.1, 0.2, 0.3]

plt.figure(figsize=(9, 6))

for tv in thermal_values:

    # Keep all random conditions identical
    np.random.seed(0)

    x, y = generator.parser(
        db,
        entry_id=8,
        thermo_vibration=tv
    )

    plt.plot(
        x,
        y,
        label=f"thermal vibration = {tv}"
    )

plt.xlabel("2theta (degree)")
plt.ylabel("Intensity (a.u.)")
plt.title("Effect of Thermal Vibration on AgTlTe2 XRD")
plt.legend()
plt.tight_layout()

plt.savefig(
    "AgTlTe2_thermal_comparison.png",
    dpi=300
)

plt.close()

print("Thermal vibration comparison completed")
print("thermal values :", thermal_values)
print("Default value  : 0.1")
print("Saved : AgTlTe2_thermal_comparison.png")
