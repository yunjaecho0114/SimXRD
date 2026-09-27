from ase.db import connect
from Pysimxrd import generator
import numpy as np
import matplotlib.pyplot as plt

# Open database
db = connect("../../sim/demo_mp.db")

# Grain sizes to compare
grain_sizes = [5, 10, 20, 40]

plt.figure(figsize=(9, 6))

for size in grain_sizes:

    np.random.seed(0)

    x, y = generator.parser(
        db,
        entry_id=8,
        grainsize=size
    )

    plt.plot(
        x,
        y,
        label=f"grain size = {size}"
    )

plt.xlabel("2theta (degree)")
plt.ylabel("Intensity (a.u.)")
plt.title("Effect of Grain Size on AgTlTe2 XRD")
plt.legend()
plt.tight_layout()

plt.savefig(
    "AgTlTe2_grainsize_comparison.png",
    dpi=300
)

plt.close()

print("Grain size comparison completed")
print("grain sizes :", grain_sizes)
print("Saved : AgTlTe2_grainsize_comparison.png")
