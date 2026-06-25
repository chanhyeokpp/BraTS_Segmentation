import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

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


def first_test_patient():
    _, _, test_ids = get_split_lists()
    return test_ids[0]


def load_slice(patient_id, z):
    path = os.path.join(DATA_DIR, patient_id)
    channels, brain_masks, flair_raw = [], [], None
    for modality in MODALITIES:
        img = nib.load(os.path.join(path, f"{patient_id}_{modality}.nii.gz")).get_fdata(dtype=np.float32)[:, :, z]
        if modality == "flair":
            flair_raw = img
        brain_masks.append(img > 0)
        channels.append(normalize(img.copy()))
    seg = nib.load(os.path.join(path, f"{patient_id}_seg.nii.gz")).get_fdata(dtype=np.float32)[:, :, z]
    brain_mask = np.any(np.stack(brain_masks), axis=0)
    return np.stack(channels), seg, display_scale(flair_raw), brain_mask


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, x, class_id, target_mask=None, brain_mask=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        pred = torch.argmax(logits, dim=1)
        if target_mask is None:
            target_mask = pred == class_id
        if target_mask.sum() == 0:
            target_mask = torch.ones_like(logits[:, class_id], dtype=torch.bool)
        score = logits[:, class_id][target_mask].mean()
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        if brain_mask is not None:
            cam = cam * brain_mask
            valid = brain_mask.astype(bool)
        else:
            valid = np.ones_like(cam, dtype=bool)
        cam_valid = cam[valid]
        if cam_valid.size:
            cam[valid] = (cam_valid - cam_valid.min()) / (cam_valid.max() - cam_valid.min() + 1e-8)
            cam[~valid] = 0
        return cam, pred.squeeze().detach().cpu().numpy()


def save_presentation(flair, cam, path, patient_id, z, class_id):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="white")
    titles = ["Original MRI", "Grad-CAM Heatmap", "Overlay Result"]
    axes[0].imshow(flair, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(flair, cmap="gray", vmin=0, vmax=1, alpha=0.25)
    axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1, alpha=0.9)
    axes[2].imshow(flair, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(cam, cmap="jet", vmin=0, vmax=1, alpha=0.45)
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")
        ax.set_facecolor("black")
    fig.suptitle(f"{patient_id} | z={z} | class {class_id}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM overlay for BraTS UNet.")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    parser.add_argument("--patient-id", default=None)
    parser.add_argument("--slice", type=int, default=None)
    parser.add_argument("--class-id", type=int, default=4)
    parser.add_argument("--target", choices=["gt", "pred"], default="gt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR / "grad_cam"))
    args = parser.parse_args()

    patient_id = args.patient_id or first_test_patient()
    if args.slice is None:
        seg_path = os.path.join(DATA_DIR, patient_id, f"{patient_id}_seg.nii.gz")
        seg_vol = nib.load(seg_path).get_fdata(dtype=np.float32)
        z = int(np.argmax(np.sum(seg_vol > 0, axis=(0, 1))))
    else:
        z = args.slice

    device = get_device(args.device)
    model = UNet2D().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    image, seg, flair, brain_mask = load_slice(patient_id, z)
    x = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)
    target_mask = torch.tensor(seg == args.class_id).unsqueeze(0).to(device) if args.target == "gt" else None
    cam_runner = GradCAM(model, model.enc3.block[-1])
    try:
        cam, pred = cam_runner(x, args.class_id, target_mask=target_mask, brain_mask=brain_mask)
    finally:
        cam_runner.close()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"grad_cam_{patient_id}_z{z}_class{args.class_id}_{args.target}.png"
    save_presentation(flair, cam, path, patient_id, z, args.class_id)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
