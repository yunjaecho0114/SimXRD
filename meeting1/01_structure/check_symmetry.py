from ase.db import connect
import spglib

# ----------------------------------
# 1. Open SimXRD demo database
# ----------------------------------
db = connect("../../sim/demo_mp.db")

# Select AgTlTe2 structure
row = db.get(id=8)
atoms = row.toatoms()

# ----------------------------------
# 2. Convert ASE structure
#    to Spglib cell format
# ----------------------------------
cell = (
    atoms.cell.array,
    atoms.get_scaled_positions(),
    atoms.get_atomic_numbers(),
)

# ----------------------------------
# 3. Analyze crystal symmetry
# ----------------------------------
dataset = spglib.get_symmetry_dataset(
    cell,
    symprec=1e-3
)

# ----------------------------------
# 4. Print results
# ----------------------------------
print("====================================")
print(" Symmetry Analysis of AgTlTe2")
print("====================================")

print("Formula            :", atoms.get_chemical_formula())
print("Database ID        :", row.id)
print("MP ID              :", row.mpid)

print("\nCrystal symmetry")
print("Space group symbol :", dataset.international)
print("Space group number :", dataset.number)
print("Point group        :", dataset.pointgroup)
print("Hall symbol        :", dataset.hall)

print("\nCell information")
print("Lattice parameters :", atoms.cell.cellpar())
