import os

import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CHECKPOINT_PATH, DATA_DIR
from data_loader import BraTS2DDataset
from model import UNet2D


REQUIRED_MODALITIES = ("flair", "t1", "t1ce", "t2", "seg")


def patient_dirs():
    return sorted(
        name for name in os.listdir(DATA_DIR)
        if name.startswith("BraTS2021") and os.path.isdir(os.path.join(DATA_DIR, name))
    )


def main():
    patients = patient_dirs()
    if not patients:
        raise SystemExit(f"No BraTS2021 patient directories found in DATA_DIR={DATA_DIR}")

    for patient_id in patients[:10]:
        missing = [
            f"{patient_id}_{m}.nii.gz"
            for m in REQUIRED_MODALITIES
            if not os.path.exists(os.path.join(DATA_DIR, patient_id, f"{patient_id}_{m}.nii.gz"))
        ]
        if missing:
            raise SystemExit(f"Missing files for {patient_id}: {missing}")

    sample_patient = patients[0]
    sample_flair = os.path.join(DATA_DIR, sample_patient, f"{sample_patient}_flair.nii.gz")
    data_shape = nib.load(sample_flair).shape

    dataset = BraTS2DDataset([sample_patient], DATA_DIR)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    x, y = next(iter(loader))

    model = UNet2D()
    with torch.no_grad():
        out = model(x)
        loss = nn.CrossEntropyLoss()(out, y)

    print(f"DATA_DIR={DATA_DIR}")
    print(f"patients={len(patients)}")
    print(f"sample_patient={sample_patient}")
    print(f"nifti_shape={data_shape}")
    print(f"batch_x_shape={tuple(x.shape)}")
    print(f"batch_y_shape={tuple(y.shape)}")
    print(f"model_out_shape={tuple(out.shape)}")
    print(f"loss={loss.item():.6f}")
    print(f"checkpoint_path={CHECKPOINT_PATH}")
    print("smoke_test=PASS")


if __name__ == "__main__":
    main()
