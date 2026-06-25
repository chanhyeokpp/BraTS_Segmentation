import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve
from torch.utils.data import DataLoader

from config import CHECKPOINT_PATH, DATA_DIR, OUTPUT_DIR
from data_loader import BraTS2DDataset, get_split_lists
from model import UNet2D


CLASS_NAMES = ["background", "necrotic/core", "edema", "unused", "enhancing"]
TUMOR_LABELS = [1, 2, 4]


def get_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def row_percent(cm):
    values = cm.astype(np.float64)
    row_sums = values.sum(axis=1, keepdims=True)
    return np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums != 0) * 100


def plot_cm(cm, labels, path, title, percent=False):
    display = row_percent(cm) if percent else cm
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(display, cmap="Blues", vmin=0, vmax=100 if percent else None)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title + (" (row %)" if percent else ""))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{display[i, j]:.1f}%" if percent else str(cm[i, j])
            ax.text(j, i, text, ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def dice_score(y_true, y_pred, label):
    t = y_true == label
    p = y_pred == label
    denom = t.sum() + p.sum()
    if denom == 0:
        return None
    return float(2 * np.logical_and(t, p).sum() / denom)


def iou_score(y_true, y_pred, label):
    t = y_true == label
    p = y_pred == label
    union = np.logical_or(t, p).sum()
    if union == 0:
        return None
    return float(np.logical_and(t, p).sum() / union)


def main():
    parser = argparse.ArgumentParser(description="Evaluate BraTS segmentation checkpoint.")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--max-patients", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR / "evaluation"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_ids, val_ids, test_ids = get_split_lists()
    ids = {"train": train_ids, "val": val_ids, "test": test_ids}[args.split][:args.max_patients]

    device = get_device(args.device)
    model = UNet2D().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    dataset = BraTS2DDataset(ids, DATA_DIR)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    y_true_all, y_pred_all, y_true_bin_all, y_score_bin_all = [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred = np.argmax(probs, axis=1)
            y_np = y.numpy()
            y_true_all.append(y_np.reshape(-1))
            y_pred_all.append(pred.reshape(-1))
            y_true_bin_all.append((y_np.reshape(-1) > 0).astype(np.uint8))
            y_score_bin_all.append((1.0 - probs[:, 0]).reshape(-1))

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    y_true_bin = np.concatenate(y_true_bin_all)
    y_score_bin = np.concatenate(y_score_bin_all)
    y_pred_bin = (y_pred > 0).astype(np.uint8)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    fg_mask = y_true > 0
    fg_cm = confusion_matrix(y_true[fg_mask], y_pred[fg_mask], labels=TUMOR_LABELS)
    plot_cm(cm, CLASS_NAMES, output_dir / "confusion_matrix_percent.png", "Pixel-level Confusion Matrix", percent=True)
    plot_cm(fg_cm, ["necrotic/core", "edema", "enhancing"], output_dir / "confusion_matrix_tumor_classes_percent.png", "Tumor-class Confusion Matrix", percent=True)

    fpr, tpr, _ = roc_curve(y_true_bin, y_score_bin)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"Tumor ROC AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Foreground Tumor ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close(fig)

    per_class = {str(label): {"dice": dice_score(y_true, y_pred, label), "iou": iou_score(y_true, y_pred, label)} for label in TUMOR_LABELS}
    dice_values = [v["dice"] for v in per_class.values() if v["dice"] is not None]
    iou_values = [v["iou"] for v in per_class.values() if v["iou"] is not None]

    metrics = {
        "patients": len(ids),
        "slices": len(dataset),
        "background_pixel_ratio": float((y_true == 0).mean()),
        "tumor_pixel_ratio": float((y_true > 0).mean()),
        "macro_precision_multiclass": precision_score(y_true, y_pred, labels=TUMOR_LABELS, average="macro", zero_division=0),
        "macro_recall_multiclass": recall_score(y_true, y_pred, labels=TUMOR_LABELS, average="macro", zero_division=0),
        "macro_f1_multiclass": f1_score(y_true, y_pred, labels=TUMOR_LABELS, average="macro", zero_division=0),
        "mean_dice_tumor_classes": float(np.mean(dice_values)),
        "mean_iou_tumor_classes": float(np.mean(iou_values)),
        "per_class": per_class,
        "tumor_precision_binary": precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "tumor_recall_binary": recall_score(y_true_bin, y_pred_bin, zero_division=0),
        "tumor_f1_binary": f1_score(y_true_bin, y_pred_bin, zero_division=0),
        "tumor_roc_auc": roc_auc,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
