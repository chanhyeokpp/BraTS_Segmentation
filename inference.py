import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

from config import CHECKPOINT_PATH, DATA_DIR
from model import UNet2D


MODALITIES = ["flair", "t1", "t1ce", "t2"]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize(img):
    mask = img > 0
    if np.any(mask):
        img[mask] = (img[mask] - img[mask].mean()) / (img[mask].std() + 1e-8)
    return img


def main():
    device = get_device()
    model = UNet2D().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    patients = sorted(p for p in os.listdir(DATA_DIR) if p.startswith("BraTS2021") and os.path.isdir(os.path.join(DATA_DIR, p)))
    patient_id = os.environ.get("BRATS2021_PATIENT_ID", patients[0])
    z = int(os.environ.get("BRATS2021_SLICE", "80"))
    patient_path = os.path.join(DATA_DIR, patient_id)

    channels = []
    for modality in MODALITIES:
        img = nib.load(os.path.join(patient_path, f"{patient_id}_{modality}.nii.gz")).get_fdata(dtype=np.float32)[:, :, z]
        channels.append(normalize(img.copy()))
    x = torch.tensor(np.stack(channels), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = torch.argmax(model(x), dim=1).squeeze().cpu().numpy()

    seg = nib.load(os.path.join(patient_path, f"{patient_id}_seg.nii.gz")).get_fdata(dtype=np.float32)[:, :, z]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(channels[0], cmap="gray")
    axes[0].set_title("MRI (FLAIR)")
    axes[1].imshow(seg, cmap="jet", vmin=0, vmax=4)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred, cmap="jet", vmin=0, vmax=4)
    axes[2].set_title("Prediction")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
