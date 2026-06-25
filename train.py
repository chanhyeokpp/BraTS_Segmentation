import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CHECKPOINT_PATH
from data_loader import DATA_DIR, BraTS2DDataset, PatientBatchSampler, get_split_lists
from model import UNet2D


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def validate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss_sum += criterion(model(x), y).item()
    return loss_sum / max(len(loader), 1)


def dice_score(pred, target, num_classes=5):
    pred = pred.view(-1)
    target = target.view(-1)
    dice = 0.0
    for c in range(1, num_classes):
        p = pred == c
        t = target == c
        union = p.sum() + t.sum()
        if union == 0:
            continue
        dice += (2 * (p & t).sum().float()) / union
    return dice / (num_classes - 1)


def main():
    device = get_device()
    print("=" * 60)
    print(f"device: {device}")
    print("=" * 60)

    train_ids, val_ids, test_ids = get_split_lists()
    train_ids = train_ids[:20]
    val_ids = val_ids[:5]
    test_ids = test_ids[:5]

    train_ds = BraTS2DDataset(train_ids, DATA_DIR)
    val_ds = BraTS2DDataset(val_ids, DATA_DIR)
    test_ds = BraTS2DDataset(test_ids, DATA_DIR)

    train_loader = DataLoader(
        train_ds,
        batch_sampler=PatientBatchSampler(train_ds, batch_size=2, shuffle=True),
    )
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    model = UNet2D().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 5

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = validate(model, val_loader, criterion, device)
        print("=" * 60)
        print(f"Epoch {epoch + 1}")
        print(f"Train Loss: {train_loss / max(len(train_loader), 1):.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        torch.save(model.state_dict(), CHECKPOINT_PATH)

    model.eval()
    dice_list = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = torch.argmax(model(x), dim=1)
            dice_list.append(dice_score(pred, y).item())
    print("=" * 60)
    print(f"TEST DICE: {sum(dice_list) / max(len(dice_list), 1):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
