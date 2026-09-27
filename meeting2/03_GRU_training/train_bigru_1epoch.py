from ase.db import connect

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =========================================================
# 1. Dataset
# =========================================================

class XRDDataset(Dataset):

    def __init__(self, db_path):

        self.db = connect(db_path)

        self.ids = [
            row.id for row in self.db.select()
        ]


    def __len__(self):

        return len(self.ids)


    def __getitem__(self, index):

        row = self.db.get(
            id=self.ids[index]
        )

        # XRD intensity
        intensity = eval(row.intensity)

        # tager = [space_group, crystal_system]
        target = eval(row.tager)

        crystal_system = target[1]


        # Convert XRD to float tensor
        x = torch.tensor(
            intensity,
            dtype=torch.float32
        )


        # Convert DB label 1~7
        # into PyTorch label 0~6
        y = torch.tensor(
            crystal_system - 1,
            dtype=torch.long
        )


        return x, y


# =========================================================
# 2. DataLoader
# =========================================================

train_dataset = XRDDataset(
    "../02_inlibrary_data/train.db"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)


# =========================================================
# 3. Bidirectional GRU
#    Following SimXRD tutorial
# =========================================================

class BiGRU(nn.Module):

    def __init__(self):

        super(BiGRU, self).__init__()

        self.hidden_size = 64
        self.num_layers = 4


        self.gru = nn.GRU(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )


        # 64 forward + 64 backward
        # -> 7 crystal-system scores
        self.fc = nn.Linear(
            self.hidden_size * 2,
            7
        )


    def forward(self, x):

        # Input:
        # [batch, 3500]
        #
        # GRU requires:
        # [batch, 3500, 1]

        x = x.unsqueeze(-1)


        # GRU output:
        # [batch, 3500, 128]
        out, _ = self.gru(x)


        # Take output of the last sequence position
        # [batch, 128]
        out = out[:, -1, :]


        # Classification
        # [batch, 128]
        # ->
        # [batch, 7]
        out = self.fc(out)


        return out


# =========================================================
# 4. Device
# =========================================================

device = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device :", device)


# =========================================================
# 5. Model / Loss / Optimizer
# =========================================================

model = BiGRU().to(device)


criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(
    model.parameters(),
    lr=0.00025
)


# =========================================================
# 6. Training - one epoch
# =========================================================

model.train()

total_loss = 0.0
correct = 0
total = 0


for x_batch, y_batch in tqdm(
    train_loader,
    desc="Training"
):

    # Move data to CPU or GPU
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)


    # ---------------------------------------------
    # Reset gradients
    # ---------------------------------------------

    optimizer.zero_grad()


    # ---------------------------------------------
    # Forward pass
    # ---------------------------------------------

    outputs = model(x_batch)


    # ---------------------------------------------
    # Calculate loss
    # ---------------------------------------------

    loss = criterion(
        outputs,
        y_batch
    )


    # ---------------------------------------------
    # Backpropagation
    # ---------------------------------------------

    loss.backward()


    # ---------------------------------------------
    # Update model parameters
    # ---------------------------------------------

    optimizer.step()


    # ---------------------------------------------
    # Accumulate loss
    # ---------------------------------------------

    total_loss += loss.item()


    # ---------------------------------------------
    # Prediction
    # ---------------------------------------------

    _, predicted = torch.max(
        outputs,
        dim=1
    )


    total += y_batch.size(0)

    correct += (
        predicted == y_batch
    ).sum().item()


# =========================================================
# 7. Epoch result
# =========================================================

average_loss = (
    total_loss / len(train_loader)
)

accuracy = (
    100.0 * correct / total
)


print("\n====================================")
print(" One Epoch Training Completed")
print("====================================")

print("Samples       :", total)
print("Average loss  :", average_loss)
print("Train accuracy:", f"{accuracy:.2f}%")
