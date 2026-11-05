import csv
import time
from pathlib import Path
from collections import Counter

import spglib
from ase.db import connect

# =====================================
# 1. Settings
# =====================================

DB_PATH = "CrystDB.db"
OUTPUT = Path("db_expansion/crystdb_structure_inventory.csv")
SYMPREC = 1e-3

if OUTPUT.exists():
    raise SystemExit(f"Output already exists: {OUTPUT}")

# =====================================
# 2. Crystal system classification
# =====================================

def get_crystal_system(number):
    if number <= 2:
        return "triclinic"
    elif number <= 15:
        return "monoclinic"
    elif number <= 74:
        return "orthorhombic"
    elif number <= 142:
        return "tetragonal"
    elif number <= 167:
        return "trigonal"
    elif number <= 194:
        return "hexagonal"
    else:
        return "cubic"

# =====================================
# 3. Read CrystDB
# =====================================

db = connect(DB_PATH)

systems = Counter()
space_groups = Counter()
failed = []

start = time.perf_counter()

print("=== CrystDB Full Inventory ===", flush=True)
print("Total structures:", db.count(), flush=True)
print("spglib version:", spglib.__version__, flush=True)
print("symprec:", SYMPREC, flush=True)

# =====================================
# 4. Analyze all structures
# =====================================

with OUTPUT.open("x", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow([
        "db_id",
        "mp_id_raw",
        "mp_id",
        "formula",
        "natoms",
        "n_elements",
        "elements",
        "volume_A3",
        "formation_energy",
        "space_group_number",
        "space_group_symbol",
        "crystal_system",
        "symprec_A",
        "symmetry_status",
        "error"
    ])

    for i, row in enumerate(db.select(), 1):

        raw_mpid = str(row.get("mpid", ""))
        mpid = raw_mpid.removesuffix(".cif")

        formula = ""
        n_elements = ""
        elements = ""
        volume = ""
        sg_number = ""
        sg_symbol = ""
        crystal_system = ""
        status = "success"
        error = ""

        try:
            atoms = row.toatoms()

            formula = atoms.get_chemical_formula()

            symbols = sorted(set(atoms.get_chemical_symbols()))
            n_elements = len(symbols)
            elements = ",".join(symbols)

            volume = atoms.get_volume()

            cell = (
                atoms.cell.array,
                atoms.get_scaled_positions(),
                atoms.get_atomic_numbers()
            )

            dataset = spglib.get_symmetry_dataset(
                cell,
                symprec=SYMPREC
            )

            if dataset is None:
                raise ValueError("spglib returned None")

            sg_number = int(dataset.number)
            sg_symbol = str(dataset.international)

            crystal_system = get_crystal_system(sg_number)

            systems[crystal_system] += 1
            space_groups[sg_number] += 1

        except Exception as exc:
            status = "failed"
            error = str(exc)
            failed.append((row.id, error))

        writer.writerow([
            row.id,
            raw_mpid,
            mpid,
            formula,
            row.natoms,
            n_elements,
            elements,
            volume,
            row.get("formation_energy"),
            sg_number,
            sg_symbol,
            crystal_system,
            SYMPREC,
            status,
            error
        ])

        if i % 5000 == 0:
            print(
                f"Processed: {i}/{db.count()}",
                flush=True
            )

# =====================================
# 5. Summary
# =====================================

elapsed = time.perf_counter() - start

print("\n=== Crystal System Distribution ===")

for system in [
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic"
]:
    print(system, systems[system])

print("\n=== Summary ===")
print("Successful:", sum(systems.values()))
print("Failed:", len(failed))
print("Unique space groups:", len(space_groups))
print("Elapsed seconds:", round(elapsed, 2))
print("CSV saved:", OUTPUT)

print("\nFailed examples:")
for item in failed[:10]:
    print(item)

print("\n=== INVENTORY COMPLETE ===")
