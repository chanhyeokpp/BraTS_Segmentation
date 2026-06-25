import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

from config import CHECKPOINT_PATH, DATA_DIR, OUTPUT_DIR
from data_loader import get_split_lists
from model import UNet2D


MODALITIES = ("flair", "t1", "t1ce", "t2")


def get_device(name):
    if name != "auto":
        return torch.device(name)
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


def display_scale(img):
    mask = img > 0
    values = img[mask] if np.any(mask) else img.reshape(-1)
    low, high = np.percentile(values, [1, 99])
    scaled = np.clip(img, low, high)
    scaled = (scaled - low) / (high - low + 1e-8)
    return scaled * mask


def load_patient(patient_id):
    path = os.path.join(DATA_DIR, patient_id)
    channels = [nib.load(os.path.join(path, f"{patient_id}_{m}.nii.gz")).get_fdata(dtype=np.float32) for m in MODALITIES]
    seg = nib.load(os.path.join(path, f"{patient_id}_seg.nii.gz")).get_fdata(dtype=np.float32)
    return channels, seg


def predict_slice(model, channels, z, device):
    image = np.stack([normalize(volume[:, :, z].copy()) for volume in channels], axis=0)
    x = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.argmax(model(x), dim=1).squeeze(0).cpu().numpy()


def dice_iou_binary(gt, pred):
    gt_mask = gt > 0
    pred_mask = pred > 0
    inter = np.logical_and(gt_mask, pred_mask).sum()
    denom = gt_mask.sum() + pred_mask.sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    dice = None if denom == 0 else float(2 * inter / denom)
    iou = None if union == 0 else float(inter / union)
    return dice, iou


def save_metric_chart(dice, iou, output_path):
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor="white")
    names = ["Mean Dice", "Mean IoU"]
    values = [dice, iou]
    bars = ax.bar(names, values, color=["#2563eb", "#16a34a"], width=0.52)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Tumor Region Binary Segmentation", fontsize=17, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=18, fontweight="bold")
    fig.text(0.5, 0.02, "Overlap between predicted tumor mask and ground truth mask", ha="center", fontsize=10, color="#555")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_visual(patient_id, flair, gt_slice, pred_slice, z, dice, iou, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="white")
    panels = [
        ("Original MRI (FLAIR)", display_scale(flair[:, :, z]), "gray", None, None),
        ("Ground Truth Mask", gt_slice, "jet", 0, 4),
        ("Predicted Mask", pred_slice, "jet", 0, 4),
    ]
    for ax, (title, image, cmap, vmin, vmax) in zip(axes, panels):
        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")
    fig.suptitle(f"{patient_id} | z={z} | Binary Tumor Dice={dice:.3f}, IoU={iou:.3f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Create presentation-ready segmentation assets.")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    parser.add_argument("--patient-id", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR / "presentation"))
    parser.add_argument("--mean-dice", type=float, default=0.855)
    parser.add_argument("--mean-iou", type=float, default=0.762)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_metric_chart(args.mean_dice, args.mean_iou, out / "binary_segmentation_dice_iou_chart.png")

    train_ids, val_ids, test_ids = get_split_lists()
    patient_id = args.patient_id or test_ids[0]
    channels, gt = load_patient(patient_id)
    z = int(np.argmax(np.sum(gt > 0, axis=(0, 1))))
    device = get_device(args.device)
    model = UNet2D().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    pred = predict_slice(model, channels, z, device)
    dice, iou = dice_iou_binary(gt[:, :, z], pred)
    save_visual(patient_id, channels[0], gt[:, :, z], pred, z, dice, iou, out / f"visual_mask_comparison_{patient_id}_z{z}.png")

    summary = {"mean_binary_tumor_dice": args.mean_dice, "mean_binary_tumor_iou": args.mean_iou, "visual_patient": patient_id, "visual_slice": z}
    with open(out / "presentation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
