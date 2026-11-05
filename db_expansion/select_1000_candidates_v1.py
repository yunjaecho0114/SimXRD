import csv
import math
import random
from pathlib import Path
from collections import defaultdict, Counter

# =========================================
# 1. Settings
# =========================================

INPUT = Path("db_expansion/crystdb_structure_inventory.csv")
OUTPUT = Path("db_expansion/selected_1000_candidates_v1.csv")

SEED = 20261101
MAX_ATOMS = 100

QUOTAS = {
    "triclinic": 181,
    "monoclinic": 200,
    "orthorhombic": 182,
    "tetragonal": 109,
    "trigonal": 109,
    "hexagonal": 95,
    "cubic": 124,
}

assert sum(QUOTAS.values()) == 1000

if OUTPUT.exists():
    raise SystemExit(f"Output already exists: {OUTPUT}")

rng = random.Random(SEED)

# =========================================
# 2. Load eligible structures
# =========================================

candidates = defaultdict(lambda: defaultdict(list))

with INPUT.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    original_fields = reader.fieldnames

    for row in reader:

        if row["symmetry_status"] != "success":
            continue

        if int(row["natoms"]) > MAX_ATOMS:
            continue

        if not row["mp_id"]:
            continue

        if not row["formula"]:
            continue

        if float(row["volume_A3"]) <= 0:
            continue

        system = row["crystal_system"]
        sg = int(row["space_group_number"])

        if system not in QUOTAS:
            continue

        candidates[system][sg].append(row)

# =========================================
# 3. Allocate quota by sqrt(SG count)
# =========================================

selected = []

print("=== STRUCTURE SELECTION ===")
print("Random seed:", SEED)
print("Max atoms:", MAX_ATOMS)

for system, target in QUOTAS.items():

    groups = candidates[system]

    available = sum(len(rows) for rows in groups.values())

    if available < target:
        raise RuntimeError(
            f"Insufficient candidates in {system}"
        )

    weights = {
        sg: math.sqrt(len(rows))
        for sg, rows in groups.items()
    }

    weight_sum = sum(weights.values())

    ideal = {
        sg: target * weight / weight_sum
        for sg, weight in weights.items()
    }

    # Initial integer allocations
    allocations = {
        sg: min(len(groups[sg]), math.floor(ideal[sg]))
        for sg in groups
    }

    # Distribute remaining slots
    while sum(allocations.values()) < target:

        eligible_sg = [
            sg for sg in groups
            if allocations[sg] < len(groups[sg])
        ]

        if not eligible_sg:
            raise RuntimeError(
                f"Cannot fill quota: {system}"
            )

        best_sg = max(
            eligible_sg,
            key=lambda sg: (
                ideal[sg] - allocations[sg],
                -sg
            )
        )

        allocations[best_sg] += 1

    # Sample individual structures
    system_selected = []

    for sg in sorted(groups):

        pool = sorted(
            groups[sg],
            key=lambda row: int(row["db_id"])
        )

        count = allocations[sg]

        system_selected.extend(
            rng.sample(pool, count)
        )

    assert len(system_selected) == target

    selected.extend(system_selected)

    print(
        f"{system:15s} "
        f"selected={len(system_selected):3d} "
        f"space_groups={sum(n > 0 for n in allocations.values())}"
    )

# =========================================
# 4. Validate selected structures
# =========================================

selected.sort(key=lambda row: int(row["db_id"]))

mpids = [row["mp_id"] for row in selected]
dbids = [row["db_id"] for row in selected]

assert len(selected) == 1000
assert len(set(mpids)) == 1000
assert len(set(dbids)) == 1000
assert all(int(row["natoms"]) <= MAX_ATOMS for row in selected)

# =========================================
# 5. Save provisional manifest
# =========================================

additional_fields = [
    "selection_seed",
    "selection_method",
    "selection_status",
]

with OUTPUT.open("x", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=additional_fields + original_fields
    )

    writer.writeheader()

    for row in selected:

        writer.writerow({
            "selection_seed": SEED,
            "selection_method": "system_50_50_sg_sqrt",
            "selection_status": "candidate",
            **row,
        })

# =========================================
# 6. Summary
# =========================================

sg_counts = Counter(
    int(row["space_group_number"])
    for row in selected
)

element_counts = Counter(
    int(row["n_elements"])
    for row in selected
)

atom_counts = [
    int(row["natoms"])
    for row in selected
]

print("\n=== SELECTION SUMMARY ===")
print("Selected structures:", len(selected))
print("Unique MP-IDs:", len(set(mpids)))
print("Unique space groups:", len(sg_counts))
print("Minimum atoms:", min(atom_counts))
print("Maximum atoms:", max(atom_counts))
print("Mean atoms:", round(sum(atom_counts) / len(atom_counts), 2))

print("\nMost common space groups:")
for sg, count in sg_counts.most_common(10):
    print(f"SG {sg}: {count}")

print("\nElement-count distribution:")
for n, count in sorted(element_counts.items()):
    print(f"{n} elements: {count}")

print("\n=== OUTPUT ===")
print("Saved:", OUTPUT)
print("Status: CANDIDATE - NOT FINAL")
print("=== COMPLETE ===")
