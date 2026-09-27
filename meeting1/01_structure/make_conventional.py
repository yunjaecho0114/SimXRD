from ase.db import connect
from ase import Atoms
from ase.io import write
import spglib

# ----------------------------------
# 1. Read original structure
# ----------------------------------
db = connect("../../sim/demo_mp.db")
row = db.get(id=8)
atoms = row.toatoms()

cell = (
    atoms.cell.array,
    atoms.get_scaled_positions(),
    atoms.get_atomic_numbers(),
)

# ----------------------------------
# 2. Generate standardized
#    conventional cell
# ----------------------------------
std_cell = spglib.standardize_cell(
    cell,
    to_primitive=False,
    no_idealize=False,
    symprec=1e-3
)

lattice, positions, numbers = std_cell

# ----------------------------------
# 3. Convert to ASE Atoms
# ----------------------------------
conventional = Atoms(
    numbers=numbers,
    scaled_positions=positions,
    cell=lattice,
    pbc=True
)

# ----------------------------------
# 4. Save conventional structure
# ----------------------------------
write("AgTlTe2_conventional.cif", conventional)

# ----------------------------------
# 5. Print comparison
# ----------------------------------
print("====================================")
print(" Primitive Cell")
print("====================================")
print(atoms.cell.cellpar())
print("Number of atoms :", len(atoms))

print("\n====================================")
print(" Conventional Cell")
print("====================================")
print(conventional.cell.cellpar())
print("Number of atoms :", len(conventional))

print("\nSaved: AgTlTe2_conventional.cif")
