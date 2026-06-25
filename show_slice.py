import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from config import DATA_DIR, OUTPUT_DIR


MODALITIES = ("flair", "t1", "t1ce", "t2")


def patient_dirs():
    return sorted(name for name in os.listdir(DATA_DIR) if name.startswith("BraTS2021") and os.path.isdir(os.path.join(DATA_DIR, name)))


def load_volume(patient_id, suffix):
    return nib.load(os.path.join(DATA_DIR, patient_id, f"{patient_id}_{suffix}.nii.gz")).get_fdata(dtype=np.float32)


def display_scale(img):
    mask = img > 0
    values = img[mask] if np.any(mask) else img.reshape(-1)
    low, high = np.percentile(values, [1, 99])
    scaled = np.clip(img, low, high)
    scaled = (scaled - low) / (high - low + 1e-8)
    return scaled * mask


def best_slice(patient_id):
    seg = load_volume(patient_id, "seg")
    return int(np.argmax(np.sum(seg > 0, axis=(0, 1))))


def main():
    parser = argparse.ArgumentParser(description="Save one BraTS patient slice image.")
    parser.add_argument("--patient-id", default=None)
    parser.add_argument("--slice", type=int, default=None)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR / "slices"))
    args = parser.parse_args()

    patients = patient_dirs()
    patient_id = args.patient_id or patients[0]
    z = args.slice if args.slice is not None else best_slice(patient_id)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{patient_id}_z{z}.png"

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, modality in zip(axes[:4], MODALITIES):
        ax.imshow(display_scale(load_volume(patient_id, modality)[:, :, z]), cmap="gray")
        ax.set_title(modality.upper())
        ax.axis("off")
    axes[4].imshow(load_volume(patient_id, "seg")[:, :, z], cmap="jet", vmin=0, vmax=4)
    axes[4].set_title("SEG")
    axes[4].axis("off")
    fig.suptitle(f"{patient_id} | z={z}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
