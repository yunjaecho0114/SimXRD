from ase.db import connect
import spglib
import csv


# =========================================================
# 1. Connect to SimXRD demo crystal database
# =========================================================

db = connect("../../sim/demo_mp.db")


# =========================================================
# 2. Convert space-group number to crystal system
# =========================================================

def get_crystal_system(spacegroup_number):

    if 1 <= spacegroup_number <= 2:
        return "triclinic", 7

    elif 3 <= spacegroup_number <= 15:
        return "monoclinic", 6

    elif 16 <= spacegroup_number <= 74:
        return "orthorhombic", 5

    elif 75 <= spacegroup_number <= 142:
        return "tetragonal", 4

    elif 143 <= spacegroup_number <= 167:
        return "trigonal", 3

    elif 168 <= spacegroup_number <= 194:
        return "hexagonal", 2

    elif 195 <= spacegroup_number <= 230:
        return "cubic", 1

    else:
        return "unknown", 0


# =========================================================
# 3. Prepare CSV output
# =========================================================

output_file = "demo_structure_labels.csv"

with open(output_file, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "ID",
        "Formula",
        "MP_ID",
        "SpaceGroup_Number",
        "SpaceGroup_Symbol",
        "CrystalSystem",
        "CrystalSystem_Label"
    ])


    # =====================================================
    # 4. Analyze every structure in demo_mp.db
    # =====================================================

    for row in db.select():

        atoms = row.toatoms()

        # Convert ASE structure to Spglib format
        cell = (
            atoms.cell.array,
            atoms.get_scaled_positions(),
            atoms.get_atomic_numbers()
        )

        # Symmetry analysis
        dataset = spglib.get_symmetry_dataset(
            cell,
            symprec=1e-3
        )

        sg_number = dataset.number
        sg_symbol = dataset.international

        crystal_system, crystal_label = \
            get_crystal_system(sg_number)

        mpid = getattr(row, "mpid", "N/A")

        # Print results to terminal
        print(
            f"ID={row.id:2d} | "
            f"{row.formula:10s} | "
            f"SG={sg_number:3d} {sg_symbol:8s} | "
            f"{crystal_system:12s} | "
            f"label={crystal_label}"
        )

        # Save results to CSV
        writer.writerow([
            row.id,
            row.formula,
            mpid,
            sg_number,
            sg_symbol,
            crystal_system,
            crystal_label
        ])


print("\nSaved :", output_file)
