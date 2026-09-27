from ase.db import connect

import torch
from torch.utils.data import Dataset, DataLoader


# =========================================================
# 1. Define PyTorch Dataset
# =========================================================

class XRDDataset(Dataset):

    def __init__(self, db_path):

        self.db = connect(db_path)

        # Save database row IDs
        self.ids = [
            row.id for row in self.db.select()
        ]


    def __len__(self):

        return len(self.ids)


    def __getitem__(self, index):

        # ---------------------------------------------
        # Read one row from ASE database
        # ---------------------------------------------

        row_id = self.ids[index]

        row = self.db.get(id=row_id)


        # ---------------------------------------------
        # Read XRD intensity
        # ---------------------------------------------

        intensity = eval(row.intensity)


        # ---------------------------------------------
        # Read target
        #
        # tager = [space_group, crystal_system]
        # ---------------------------------------------

        target = eval(row.tager)

        space_group = target[0]
        crystal_system = target[1]


        # ---------------------------------------------
        # Convert to PyTorch tensors
        # ---------------------------------------------

        x = torch.tensor(
            intensity,
            dtype=torch.float32
        )

        # DB label: 1~7
        # PyTorch label: 0~6
        y = torch.tensor(
            crystal_system - 1,
            dtype=torch.long
        )


        return x, y


# =========================================================
# 2. Load training database
# =========================================================

train_dataset = XRDDataset(
    "../02_inlibrary_data/train.db"
)


print("====================================")
print(" PyTorch Dataset Check")
print("====================================")

print("Number of samples :", len(train_dataset))


# =========================================================
# 3. Check first sample
# =========================================================

x, y = train_dataset[0]

print("\nFirst sample")
print("XRD tensor shape :", x.shape)
print("PyTorch label    :", y.item())


# =========================================================
# 4. Create DataLoader
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)


# =========================================================
# 5. Check one batch
# =========================================================

x_batch, y_batch = next(iter(train_loader))

print("\nFirst batch")
print("X batch shape :", x_batch.shape)
print("Y batch shape :", y_batch.shape)

print("Labels :", y_batch)
