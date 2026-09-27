from ase.db import connect
from Pysimxrd import generator

import csv
import os
import random
import numpy as np

from tqdm import tqdm


# =========================================================
# 1. File paths
# =========================================================

SOURCE_DB = "../../sim/demo_mp.db"
LABEL_FILE = "../01_labels/demo_structure_labels.csv"

TRAIN_DB = "train.db"
VAL_DB = "val.db"
TEST_DB = "test.db"


# =========================================================
# 2. Number of XRD patterns per structure
# =========================================================

TRAIN_PER_STRUCTURE = 30
VAL_PER_STRUCTURE = 1
TEST_PER_STRUCTURE = 2


# =========================================================
# 3. Fix random seeds for reproducibility
# =========================================================

random.seed(0)
np.random.seed(0)


# =========================================================
# 4. Read symmetry labels
# =========================================================

labels = {}

with open(LABEL_FILE, "r") as f:

    reader = csv.DictReader(f)

    for row in reader:

        structure_id = int(row["ID"])

        labels[structure_id] = {
            "spacegroup": int(row["SpaceGroup_Number"]),
            "crysystem": int(row["CrystalSystem_Label"])
        }


print("Loaded labels for", len(labels), "structures")


# =========================================================
# 5. Open source crystal database
# =========================================================

source_db = connect(SOURCE_DB)

print("Source structures :", source_db.count())


# =========================================================
# 6. Remove old output DBs
#    This prevents duplicated data when rerunning the script
# =========================================================

for filename in [TRAIN_DB, VAL_DB, TEST_DB]:

    if os.path.exists(filename):
        os.remove(filename)
        print("Removed old file :", filename)


# =========================================================
# 7. Open new output databases
# =========================================================

train_db = connect(TRAIN_DB)
val_db = connect(VAL_DB)
test_db = connect(TEST_DB)


# =========================================================
# 8. Function for generating one simulated XRD
# =========================================================

def generate_pattern(structure_id):

    # -----------------------------------------------------
    # Random simulation environment
    # based on the author's ht_simxrd.py
    # -----------------------------------------------------

    grainsize = random.uniform(2, 20)

    orientation = [
        random.uniform(0.0, 0.4),
        random.uniform(0.0, 0.4)
    ]

    thermo_vib = random.uniform(0.0, 0.3)

    zero_shift = random.uniform(-1.5, 1.5)

    deformation = True

    lattice_extinction_ratio = 0.01
    lattice_torsion_ratio = 0.01

    background_order = 6
    background_ratio = 0.05
    mixture_noise_ratio = 0.02


    # -----------------------------------------------------
    # Generate d-I XRD pattern
    # -----------------------------------------------------

    x, y = generator.parser(
        database=source_db,
        entry_id=structure_id,

        deformation=deformation,

        grainsize=grainsize,

        prefect_orientation=orientation,

        thermo_vibration=thermo_vib,

        zero_shift=zero_shift,

        lattice_extinction_ratio=lattice_extinction_ratio,
        lattice_torsion_ratio=lattice_torsion_ratio,

        background_order=background_order,
        background_ratio=background_ratio,
        mixture_noise_ratio=mixture_noise_ratio,

        xrd="real"
    )


    # Convert arrays to ordinary Python lists
    x = [float(value) for value in x]
    y = [float(value) for value in y]


    simulation_param = [
        grainsize,
        orientation,
        thermo_vib,
        zero_shift
    ]


    return x, y, simulation_param


# =========================================================
# 9. Function for writing patterns into a DB
# =========================================================

def write_patterns(output_db, structure_id, number_of_patterns):

    row = source_db.get(id=structure_id)

    atoms = row.toatoms()

    spacegroup = labels[structure_id]["spacegroup"]
    crysystem = labels[structure_id]["crysystem"]

    target = [
        spacegroup,
        crysystem
    ]


    for _ in range(number_of_patterns):

        x, y, simulation_param = \
            generate_pattern(structure_id)


        output_db.write(
            atoms,

            latt_dis=str(x),

            intensity=str(y),

            tager=str(target),

            chem_form=atoms.get_chemical_formula(),

            simulation_param=str(simulation_param),

            source_id=structure_id
        )


# =========================================================
# 10. Generate Train / Validation / Test data
# =========================================================

structure_ids = list(range(1, source_db.count() + 1))


for structure_id in tqdm(
    structure_ids,
    desc="Generating structures"
):

    write_patterns(
        train_db,
        structure_id,
        TRAIN_PER_STRUCTURE
    )

    write_patterns(
        val_db,
        structure_id,
        VAL_PER_STRUCTURE
    )

    write_patterns(
        test_db,
        structure_id,
        TEST_PER_STRUCTURE
    )


# =========================================================
# 11. Print final summary
# =========================================================

print("\n====================================")
print(" In-library dataset completed")
print("====================================")

print("Train      :", train_db.count())
print("Validation :", val_db.count())
print("Test       :", test_db.count())

print("\nExpected:")
print("Train      : 300")
print("Validation : 10")
print("Test       : 20")
