from ase.db import connect
from ase.io import write

db = connect("../../sim/demo_mp.db")

row = db.get(id=8)
atoms = row.toatoms()

write("demo_structure_8.cif", atoms)

print("Saved: demo_structure_8.cif")
print("Formula:", atoms.get_chemical_formula())
print("Cell:", atoms.cell.cellpar())

