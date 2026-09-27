from ase.db import connect
import spglib

# -----------------------------
# 1. Open SimXRD demo database
# -----------------------------
db = connect("../../sim/demo_mp.db")

# Official PysimXRD tutorial commonly uses entry_id = 8
entry_id = 8

row = db.get(id=entry_id)
atoms = row.toatoms()

# -----------------------------
# 2. Basic crystal information
# -----------------------------
formula = atoms.get_chemical_formula()
num_atoms = len(atoms)

a, b, c, alpha, beta, gamma = atoms.cell.cellpar()

# -----------------------------
# 3. Space-group analysis
# -----------------------------
cell = (
    atoms.cell.array,
    atoms.get_scaled_positions(),
    atoms.get_atomic_numbers(),
)

symmetry = spglib.get_symmetry_dataset(cell)

# -----------------------------
# 4. Print results
# -----------------------------
print("====================================")
print(" SimXRD Demo Structure Information")
print("====================================")
print("Entry ID :", entry_id)
print("Formula  :", formula)
print("Atoms    :", num_atoms)

print("\nLattice parameters")
print("a     =", a, "Angstrom")
print("b     =", b, "Angstrom")
print("c     =", c, "Angstrom")
print("alpha =", alpha, "degree")
print("beta  =", beta, "degree")
print("gamma =", gamma, "degree")

print("\nSymmetry")
print("Space group symbol :", symmetry.international)
print("Space group number :", symmetry.number)
