# -*- coding: utf-8 -*-
"""
V-JEPA 2 -- Complete Pipeline for Human Activity Recognition (HAR)

Fine-tuning, Evaluation, Metrics, and Inference

This script fine-tunes a pre-trained V-JEPA 2 model for 11-class Human Activity
Recognition on video data. It includes data loading, training with TensorBoard
logging, evaluation on test set, confusion matrix generation, model saving,
single video inference, and real-time webcam inference.

**Dataset:** `data_demo/` with classes: clapping, meet_and_split, sitting, still, walking
**Hardware:** RTX 3060 12 GB
**Base model:** facebook/vjepa2-vitl-fpc16-256-ssv2

Notebook structure (original Colab cells):
1.  Installation and dependencies
2.  Imports and global configuration
3.  Automatic dataset split
4.  Dataset and DataLoaders
5.  Model loading
6.  Training with TensorBoard
7.  Complete evaluation on test set
8.  Confusion matrix and per-class metrics
9.  Save and reload model
10. Inference on MP4 video
11. Real-time inference with camera

Original Colab: https://colab.research.google.com/drive/1izaQoj3inBzXYkLJ23ghUp-mR1GOc7yP
"""

# Installation cell -- run only the first time
# Uncomment and run if dependencies are not yet installed:
"""
import subprocess, sys

pkgs = [
    "torch==2.6.0",
    "torchvision==0.21.0",
    "torchcodec==0.2.1",
    "transformers>=4.51.0",
    "accelerate>=0.30.0",
    "seaborn>=0.13.0",
    "scikit-learn>=1.4.0",
    "tensorboard>=2.16.0",
    "opencv-python>=4.9.0",
    "matplotlib>=3.8.0",
]

for pkg in pkgs:
    print(f"Installing {pkg}...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

print("\nInstallation complete")
"""

# Ejecutar solo la primera vez

""""
import subprocess, sys

pkgs = [
    "torch==2.6.0",
    "torchvision==0.21.0",
    "torchcodec==0.2.1",
    "transformers>=4.51.0",
    "accelerate>=0.30.0",
    "seaborn>=0.13.0",
    "scikit-learn>=1.4.0",
    "tensorboard>=2.16.0",
    "opencv-python>=4.9.0",
    "matplotlib>=3.8.0",
]

for pkg in pkgs:
    print(f"Instalando {pkg}...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

print("\n Instalación completa")

"""

"""---
## Imports and Global Configuration
"""

import os
import sys
import pathlib
import random
import shutil
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from IPython.display import Image as IPImage, display, clear_output

from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, precision_score, recall_score
)

from transformers import (
    VJEPA2ForVideoClassification,
    VJEPA2VideoProcessor,
)
from torchcodec.decoders import VideoDecoder
from torchcodec.samplers import clips_at_random_indices
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter

# ── Global Configuration ──────────────────────────────────────────────────
# Reference: https://github.com/facebookresearch/vjepa2
# Available models:
#   facebook/vjepa2-vitl-fpc64-256
#   facebook/vjepa2-vitl-fpc16-256-ssv2
#   vjepa2_1_vit_base_384


CFG = {
    # Model
    "model_name":       "facebook/vjepa2-vitl-fpc64-256",
    # Dataset
    "data_root":        "HAR_data",       # Change to "data_demo" for this repo
    "video_ext":        "mp4",
    "train_ratio":      0.70,
    "val_ratio":        0.10,
    # Training
    "num_epochs":       15,
    "batch_size":       1,
    "accum_steps":      8,      # effective batch = 8
    "lr":               1e-4,
    "weight_decay":     1e-4,
    "num_workers":      4,
    "freeze_backbone":  True,   # True = only train the head (recommended)
    "frames_per_clip":  32,
    "frame_gap":        3,
    "patience":         5,
    # Outputs
    "output_dir":       "vjepa2_output",
    "checkpoint_best":  "vjepa2_output/best_model",
    "log_dir":          "vjepa2_output/runs",
}

os.makedirs(CFG["output_dir"], exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"{'='*55}")
print(f"  Device : {DEVICE}")
if DEVICE == "cuda":
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free  = torch.cuda.mem_get_info()[0] / 1e9
    print(f"  VRAM   : {free:.1f} GB libres / {total:.1f} GB total")
print(f"  PyTorch: {torch.__version__}")
print(f"{'='*55}")

"""---
## Automatic Dataset Split

If `data` contains class folders directly (without train/val/test split),
this cell automatically splits the data. If already split, it detects this
and skips the splitting step.
"""

def split_dataset(data_root, train_r=0.70, val_r=0.15, seed=42):
    """
    Split dataset into train/val/test splits.

    Args:
        data_root: Path to dataset root directory.
        train_r: Ratio of data to use for training (default: 0.70).
        val_r: Ratio of data to use for validation (default: 0.15).
        seed: Random seed for reproducibility.

    Returns:
        None. Creates train/, val/, test/ directories with copied videos.
    """
    root   = pathlib.Path(data_root)
    splits = {"train", "val", "test"}
    classes = [d.name for d in root.iterdir()
                if d.is_dir() and d.name not in splits]

    if not classes:
        print("[INFO] Dataset already split into train/val/test")
        # Show counts
        for sp in ["train", "val", "test"]:
            paths = list(root.glob(f"{sp}/**/*.mp4"))
            print(f"  {sp}: {len(paths)} videos")
        return

    print(f"[SPLIT] Classes found: {classes}")
    random.seed(seed)
    summary = {}

    for cls in classes:
        videos = sorted((root / cls).glob(f"*.{CFG['video_ext']}"))
        random.shuffle(videos)
        n       = len(videos)
        n_train = max(1, int(n * train_r))
        n_val   = max(1, int(n * val_r))
        n_test  = max(1, n - n_train - n_val)

        buckets = {
            "train": videos[:n_train],
            "val":   videos[n_train : n_train + n_val],
            "test":  videos[n_train + n_val :],
        }
        summary[cls] = {sp: len(v) for sp, v in buckets.items()}

        for sp, vids in buckets.items():
            dest = root / sp / cls
            dest.mkdir(parents=True, exist_ok=True)
            for v in vids:
                shutil.copy(v, dest / v.name)

    # Summary table
    print(f"\n{'Class':<20} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>6}")
    print("-" * 46)
    for cls, counts in summary.items():
        total = sum(counts.values())
        print(f"{cls:<20} {counts['train']:>6} {counts['val']:>6} {counts['test']:>6} {total:>6}")
    print("\n Split complete")


split_dataset(CFG["data_root"], CFG["train_ratio"], CFG["val_ratio"])

"""---
## Dataset and DataLoaders
"""

class VideoDataset(Dataset):
    """Dataset for loading video clips with associated labels."""

    def __init__(self, video_paths, label2id):
        """
        Args:
            video_paths: List of pathlib.Path objects to video files.
            label2id: Dictionary mapping class names to integer labels.
        """
        self.video_paths = video_paths
        self.label2id    = label2id

    def __len__(self):
        """Return the number of videos in the dataset."""
        return len(self.video_paths)

    def __getitem__(self, idx):
        """
        Load a video clip and return it with its label.

        Args:
            idx: Index of the video to load.

        Returns:
            tuple: (VideoDecoder object, integer label)
        """
        path  = self.video_paths[idx]
        label = path.parts[-2]          # folder name = class
        try:
            decoder = VideoDecoder(str(path))
        except Exception as e:
            print(f"[WARN] {path.name}: {e}")
            return self.__getitem__((idx + 1) % len(self))
        return decoder, self.label2id[label]


def collate_fn(samples, frames_per_clip, frame_gap, transforms):
    """
    Collate function to process a batch of video samples.

    Args:
        samples: List of (decoder, label) tuples.
        frames_per_clip: Number of frames to extract per clip.
        frame_gap: Number of frames between each extracted frame.
        transforms: torchvision transforms to apply.

    Returns:
        tuple: (stacked video tensors, tensor of labels)
    """
    clips, labels = [], []
    for decoder, lbl in samples:
        try:
            clip = clips_at_random_indices(
                decoder,
                num_clips=1,
                num_frames_per_clip=frames_per_clip,
                num_indices_between_frames=frame_gap,
            ).data
            clips.append(clip)
            labels.append(lbl)
        except Exception as e:
            print(f"[WARN] clip error: {e}")
    if not clips:
        dummy = torch.zeros(1, frames_per_clip, 3, 256, 256, dtype=torch.uint8)
        return dummy, torch.tensor([0])
    videos = torch.cat(clips, dim=0)
    videos = transforms(videos)
    return videos, torch.tensor(labels)


# Load processor to get crop size
print("Loading processor...")
processor = VJEPA2VideoProcessor.from_pretrained(CFG["model_name"])
H = processor.crop_size["height"]
W = processor.crop_size["width"]
print(f"Crop size: {H}×{W}")

# Data augmentation transforms for training
train_tf = v2.Compose([
    v2.RandomResizedCrop((H, W), scale=(0.7, 1.0)),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
])
# Evaluation transform (no augmentation)
eval_tf = v2.Compose([v2.CenterCrop((H, W))])

# Paths and labels
root = pathlib.Path(CFG["data_root"])
train_paths = sorted(root.glob(f"train/**/*.{CFG['video_ext']}"))
val_paths   = sorted(root.glob(f"val/**/*.{CFG['video_ext']}"))
test_paths  = sorted(root.glob(f"test/**/*.{CFG['video_ext']}"))

all_paths    = train_paths + val_paths + test_paths
CLASS_NAMES  = sorted({p.parts[-2] for p in all_paths})
LABEL2ID     = {lbl: i for i, lbl in enumerate(CLASS_NAMES)}
ID2LABEL     = {i: lbl for lbl, i in LABEL2ID.items()}
N_CLASSES    = len(CLASS_NAMES)

_collate = partial(collate_fn,
    frames_per_clip=CFG["frames_per_clip"],
    frame_gap=CFG["frame_gap"])

train_loader = DataLoader(VideoDataset(train_paths, LABEL2ID),
    batch_size=CFG["batch_size"], shuffle=True,
    collate_fn=partial(_collate, transforms=train_tf),
    num_workers=CFG["num_workers"], pin_memory=True)

val_loader = DataLoader(VideoDataset(val_paths, LABEL2ID),
    batch_size=CFG["batch_size"], shuffle=False,
    collate_fn=partial(_collate, transforms=eval_tf),
    num_workers=CFG["num_workers"], pin_memory=True)

test_loader = DataLoader(VideoDataset(test_paths, LABEL2ID),
    batch_size=CFG["batch_size"], shuffle=False,
    collate_fn=partial(_collate, transforms=eval_tf),
    num_workers=CFG["num_workers"], pin_memory=True)

print(f"\n{'='*45}")
print(f"  Classes ({N_CLASSES}): {CLASS_NAMES}")
print(f"  Train : {len(train_paths)} videos")
print(f"  Val   : {len(val_paths)} videos")
print(f"  Test  : {len(test_paths)} videos")
print(f"{'='*45}")

"""---
## Load Model with Classification Head for Our Classes
"""

print(f"Loading model: {CFG['model_name']}")
print(f"Classes: {CLASS_NAMES}")

model = VJEPA2ForVideoClassification.from_pretrained(
    CFG["model_name"],
    torch_dtype=torch.float32,
    label2id=LABEL2ID,
    id2label=ID2LABEL,
    ignore_mismatched_sizes=True,   # reinitializes head with our N classes
).to(DEVICE)

# Freeze backbone -- only train the classification head
if CFG["freeze_backbone"]:
    for param in model.vjepa2.parameters():
        param.requires_grad = False

total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen    = total - trainable

print(f"\n{'='*45}")
print(f"  Total parameters    : {total:>12,}")
print(f"  Backbone (frozen)  : {frozen:>12,}")
print(f"  Head (trainable)   : {trainable:>12,}")
if DEVICE == "cuda":
    print(f"  VRAM used            : {torch.cuda.memory_allocated()/1e9:>11.2f} GB")
    print(f"  VRAM free            : {torch.cuda.mem_get_info()[0]/1e9:>11.2f} GB")
print(f"{'='*45}")

"""---
## Training with Early Stopping
"""

def quick_eval(model, processor, loader, device):
    """
    Quick evaluation on a given DataLoader.

    Args:
        model: The V-JEPA2 model to evaluate.
        processor: The corresponding video processor.
        loader: DataLoader with validation or test data.
        device: 'cuda' or 'cpu'.

    Returns:
        tuple: (accuracy, weighted F1 score)
    """
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for vids, labels in loader:
            inputs = processor(vids, return_tensors="pt").to(device)
            logits = model(**inputs).logits
            preds_all.extend(logits.argmax(-1).cpu().numpy())
            labels_all.extend(labels.numpy())
    acc = accuracy_score(labels_all, preds_all)
    f1  = f1_score(labels_all, preds_all, average="weighted", zero_division=0)
    return acc, f1


# ── Optimizer and Scheduler ─────────────────────────────────────────────────
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer  = torch.optim.AdamW(trainable_params, lr=CFG["lr"],
                                weight_decay=CFG["weight_decay"])
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=CFG["num_epochs"])
writer     = SummaryWriter(CFG["log_dir"])

# ── History for plotting ───────────────────────────────────────────────
history = {"train_loss": [], "val_acc": [], "val_f1": []}

best_val_acc = 0.0
patience_cnt = 0
global_step  = 0


print(f"{'='*55}")
print(f"  TRAINING — {CFG['num_epochs']} max epochs")
print(f"  Effective batch : {CFG['batch_size'] * CFG['accum_steps']}")
print(f"  Initial LR     : {CFG['lr']}")
print(f"  Early stopping : {CFG['patience']} epochs without improvement")
print(f"{'='*55}\n")

for epoch in range(1, CFG["num_epochs"] + 1):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    for step, (vids, labels) in enumerate(train_loader, start=1):
        inputs = processor(vids, return_tensors="pt").to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(**inputs, labels=labels)
        loss    = outputs.loss / CFG["accum_steps"]  # gradient accumulation
        loss.backward()
        running_loss += loss.item() * CFG["accum_steps"]

        if step % CFG["accum_steps"] == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            writer.add_scalar("Train/Loss", running_loss / step, global_step)

    scheduler.step()
    avg_loss = running_loss / len(train_loader)

    # Validation
    val_acc, val_f1 = quick_eval(model, processor, val_loader, DEVICE)
    history["train_loss"].append(avg_loss)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)

    writer.add_scalar("Val/Accuracy", val_acc, epoch)
    writer.add_scalar("Val/F1",       val_f1,  epoch)
    writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

    flag = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_pretrained(CFG["checkpoint_best"])
        processor.save_pretrained(CFG["checkpoint_best"])
        patience_cnt = 0
        flag = "  BEST"
    else:
        patience_cnt += 1

    print(f"Epoch {epoch:>3}/{CFG['num_epochs']} | "
          f"Loss: {avg_loss:.4f} | "
          f"Val Acc: {val_acc:.4f} | "
          f"Val F1: {val_f1:.4f} | "
          f"Best: {best_val_acc:.4f}{flag}")

    if patience_cnt >= CFG["patience"]:
        print(f"\n Early stopping: no improvement for {CFG['patience']} epochs.")
        break

writer.close()
print(f"\n Training complete | Best Val Acc: {best_val_acc:.4f}")
print(f"   Model saved at: {CFG['checkpoint_best']}")

"""---
## Training Curves
"""

epochs_ran = len(history["train_loss"])
xs = range(1, epochs_ran + 1)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Training Curves -- V-JEPA 2", fontsize=13, fontweight="bold")

axes[0].plot(xs, history["train_loss"], "o-", color="#E63946", linewidth=2)
axes[0].set_title("Loss (train)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-entropy loss")
axes[0].grid(alpha=0.3)

axes[1].plot(xs, history["val_acc"], "o-", color="#2A9D8F", linewidth=2)
axes[1].axhline(best_val_acc, color="gray", linestyle="--", alpha=0.5, label=f"Best={best_val_acc:.3f}")
axes[1].set_title("Accuracy (val)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0, 1.05)
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].plot(xs, history["val_f1"], "o-", color="#F4A261", linewidth=2)
axes[2].set_title("F1 Score (val, weighted)")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("F1")
axes[2].set_ylim(0, 1.05)
axes[2].grid(alpha=0.3)

plt.tight_layout()
curves_path = os.path.join(CFG["output_dir"], "training_curves.png")
plt.savefig(curves_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved to: {curves_path}")

"""---
## Complete Evaluation on TEST SET
"""

# Load the BEST saved checkpoint
print(f"Loading best model from: {CFG['checkpoint_best']}")
best_model = VJEPA2ForVideoClassification.from_pretrained(
    CFG["checkpoint_best"],
    torch_dtype=torch.float32
).to(DEVICE).eval()
best_processor = VJEPA2VideoProcessor.from_pretrained(CFG["checkpoint_best"])

# Run inference on entire test set
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for vids, labels in test_loader:
        inputs  = best_processor(vids, return_tensors="pt").to(DEVICE)
        outputs = best_model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        preds   = outputs.logits.argmax(-1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# ── Global Metrics ─────────────────────────────────────────────
acc  = accuracy_score(all_labels, all_preds)
f1   = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
rec  = recall_score(all_labels, all_preds, average="weighted", zero_division=0)

print(f"\n{'='*50}")
print(f"  RESULTS ON TEST SET")
print(f"{'='*50}")
print(f"  Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
print(f"  F1 (wtd)  : {f1:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"{'='*50}")

print(f"\n  Per-class report:")
print(classification_report(all_labels, all_preds,
                             target_names=CLASS_NAMES, zero_division=0))

"""---
## Confusion Matrix and Per-Class Metrics
"""

cm      = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

# ── Figure 1: Confusion Matrices ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f"V-JEPA 2 — Test Set   |   Acc: {acc:.3f}   |   F1: {f1:.3f}",
             fontsize=13, fontweight="bold")

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[0], linewidths=0.5, annot_kws={"size": 11})
axes[0].set_title("Absolute counts", fontsize=11)
axes[0].set_xlabel("Prediction")
axes[0].set_ylabel("True label")
axes[0].tick_params(axis="x", rotation=25)

sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[1], linewidths=0.5, vmin=0, vmax=1,
            annot_kws={"size": 11})
axes[1].set_title("Normalized (per row)", fontsize=11)
axes[1].set_xlabel("Prediction")
axes[1].set_ylabel("True label")
axes[1].tick_params(axis="x", rotation=25)

plt.tight_layout()
cm_path = os.path.join(CFG["output_dir"], "confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.show()

# ── Figure 2: Accuracy per Class ─────────────────────────────────────────
per_class_acc = cm_norm.diagonal()
colors = plt.cm.RdYlGn(per_class_acc)

fig2, ax2 = plt.subplots(figsize=(10, 5))
bars = ax2.bar(CLASS_NAMES, per_class_acc * 100, color=colors, edgecolor="black", linewidth=0.5)
ax2.set_ylim(0, 115)
ax2.set_ylabel("Accuracy (%)", fontsize=11)
ax2.set_title("Accuracy per class -- Test Set", fontsize=12, fontweight="bold")
ax2.axhline(acc * 100, color="navy", linestyle="--", alpha=0.6,
             label=f"Global Acc: {acc*100:.1f}%")
ax2.legend()
ax2.tick_params(axis="x", rotation=20)
for bar, val in zip(bars, per_class_acc):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 2, f"{val*100:.1f}%",
             ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.tight_layout()
acc_path = os.path.join(CFG["output_dir"], "per_class_accuracy.png")
plt.savefig(acc_path, dpi=150, bbox_inches="tight")
plt.show()

# ── Figure 3: Heatmap of Average Probabilities per True Class ───────────
avg_probs = np.zeros((N_CLASSES, N_CLASSES))
for true_cls in range(N_CLASSES):
    idxs = [i for i, l in enumerate(all_labels) if l == true_cls]
    if idxs:
        avg_probs[true_cls] = np.mean([all_probs[i] for i in idxs], axis=0)

fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.heatmap(avg_probs, annot=True, fmt=".2f", cmap="YlOrRd",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax3, vmin=0, vmax=1, linewidths=0.3)
ax3.set_title("Average model probability per true class",
              fontsize=11, fontweight="bold")
ax3.set_xlabel("Predicted class")
ax3.set_ylabel("True class")
ax3.tick_params(axis="x", rotation=25)
plt.tight_layout()
prob_path = os.path.join(CFG["output_dir"], "avg_probabilities.png")
plt.savefig(prob_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"\nSaved:")
print(f"  {cm_path}")
print(f"  {acc_path}")
print(f"  {prob_path}")

"""---
## Save Model and Verify Reload
"""

# The best model was already saved during training.
# This cell verifies that it can be reloaded correctly.

SAVE_PATH = CFG["checkpoint_best"]
print(f"Model saved at: {SAVE_PATH}")
print("Files:", os.listdir(SAVE_PATH))

# ── Also save an explicit final version ───────────────────────────
final_path = os.path.join(CFG["output_dir"], "final_model")
best_model.save_pretrained(final_path)
best_processor.save_pretrained(final_path)
print(f"\nFinal model saved at: {final_path}")

# ── Reload test ───────────────────────────────────────────────────────
print("\nVerifying model reload...")
reload_model = VJEPA2ForVideoClassification.from_pretrained(
    final_path, torch_dtype=torch.float32
).to(DEVICE).eval()
reload_proc  = VJEPA2VideoProcessor.from_pretrained(final_path)

print(f"  Classes in reloaded model: {list(reload_model.config.id2label.values())}")
print(f"  Parameters: {sum(p.numel() for p in reload_model.parameters()):,}")
print("\nModel reloaded successfully")

# Clean up to free VRAM
del reload_model, reload_proc
torch.cuda.empty_cache()

"""---
## Inference on MP4 Video
"""

def infer_video(video_path: str, model, processor, id2label: dict,
                device: str, n_clips: int = 5, frame_gap: int = 3,
                frames_per_clip: int = 16):
    """
    Infer the class of an MP4 video using majority voting
    over multiple randomly sampled clips.

    Args:
        video_path: Path to the MP4 video file.
        model: Loaded VJEPA2 model.
        processor: Corresponding video processor.
        id2label: Dictionary mapping class IDs to label names.
        device: 'cuda' or 'cpu'.
        n_clips: Number of clips to sample (default: 5).
        frame_gap: Frames between each sampled frame (default: 3).
        frames_per_clip: Number of frames per clip (default: 16).

    Returns:
        tuple: (predicted_label, confidence, full_probability_distribution)
    """
    model.eval()
    decoder  = VideoDecoder(video_path)
    all_probs = []

    for _ in range(n_clips):
        clip = clips_at_random_indices(
            decoder,
            num_clips=1,
            num_frames_per_clip=frames_per_clip,
            num_indices_between_frames=frame_gap,
        ).data  # (1, T, C, H, W)

        H = processor.crop_size["height"]
        W = processor.crop_size["width"]
        clip = v2.CenterCrop((H, W))(clip)

        inputs = processor(clip, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        all_probs.append(probs)

    # Average probabilities across all clips
    avg_probs  = np.mean(all_probs, axis=0)
    pred_idx   = avg_probs.argmax()
    pred_label = id2label[pred_idx]
    confidence = avg_probs[pred_idx]

    return pred_label, confidence, avg_probs


def plot_prediction(video_path, pred_label, confidence, avg_probs, class_names):
    """
    Plot inference results for a video.

    Args:
        video_path: Path to the video file (for title).
        pred_label: Predicted class label.
        confidence: Confidence score for the prediction.
        avg_probs: Full probability distribution across all classes.
        class_names: List of all class names.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"Inference: {pathlib.Path(video_path).name}",
                 fontsize=12, fontweight="bold")

    # Confidence bars
    colors = ["#2A9D8F" if c == pred_label else "#ADB5BD" for c in class_names]
    bars = axes[0].barh(class_names, avg_probs * 100, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_xlim(0, 110)
    axes[0].set_xlabel("Confidence (%)")
    axes[0].set_title(f"Prediction: {pred_label.upper()}  ({confidence*100:.1f}%)")
    for bar, val in zip(bars, avg_probs):
        axes[0].text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f"{val*100:.1f}%", va="center", fontsize=9)

    # Pie chart
    wedge_colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    axes[1].pie(avg_probs, labels=class_names, autopct="%1.1f%%",
                colors=wedge_colors, startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    axes[1].set_title("Probability distribution")

    plt.tight_layout()
    plt.show()


# ── Test on all test videos ──────────────────────────────────────────
print("Running inference on test videos...\n")
print(f"{'Video':<35} {'True':<20} {'Prediction':<20} {'Conf':>6} {'OK':>4}")
print("-" * 90)

correct = 0
for path in test_paths[:10]:   # first 10 to avoid taking too long
    true_label = path.parts[-2]
    pred_label, conf, probs = infer_video(
        str(path), best_model, best_processor, ID2LABEL, DEVICE
    )
    ok = "ok" if pred_label == true_label else "bad"
    if pred_label == true_label:
        correct += 1
    print(f"{path.name:<35} {true_label:<20} {pred_label:<20} {conf*100:>5.1f}% {ok}")

n_shown = min(10, len(test_paths))
print(f"\nSample accuracy: {correct}/{n_shown} = {correct/n_shown:.1%}")

# Visualize the first test video
if test_paths:
    first_video = str(test_paths[0])
    pred_label, conf, probs = infer_video(
        first_video, best_model, best_processor, ID2LABEL, DEVICE
    )
    plot_prediction(first_video, pred_label, conf, probs, CLASS_NAMES)

"""---
## Real-Time Inference with Camera

> Requires **display** (does not work on headless servers).
> Press **`q`** in the OpenCV window to exit.
"""

import cv2

def realtime_inference(model, processor, id2label, device, cfg):
    """
    Capture webcam video, accumulate frames in a buffer,
    and classify every INFER_EVERY new frames.

    Args:
        model: Loaded VJEPA2 model.
        processor: Corresponding video processor.
        id2label: Dictionary mapping class IDs to label names.
        device: 'cuda' or 'cpu'.
        cfg: Configuration dictionary with model parameters.
    """
    model.eval()
    H          = processor.crop_size["height"]
    W          = processor.crop_size["width"]
    n_classes  = len(id2label)
    BUFFER_SZ  = cfg["frames_per_clip"]
    INFER_EVERY = 8

    COLORS = [
        (0,255,0), (0,165,255), (255,0,0),
        (0,0,255), (255,255,0), (255,0,255)
    ]

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    buffer      = []
    pred_label  = "Loading..."
    pred_conf   = 0.0
    pred_color  = (180, 180, 180)
    pred_probs  = np.zeros(n_classes)
    frame_count = 0
    fps_time    = time.time()
    display_fps = 0.0

    print("Camera started. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Could not read from camera.")
            break

        frame_count += 1

        # Preprocess frame for buffer
        frame_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (W, H))
        frame_tensor  = torch.from_numpy(frame_resized).permute(2, 0, 1)  # C H W
        buffer.append(frame_tensor)
        if len(buffer) > BUFFER_SZ:
            buffer.pop(0)

        # ── Inference ────────────────────────────────────────────────────
        if len(buffer) == BUFFER_SZ and frame_count % INFER_EVERY == 0:
            clip = torch.stack(buffer).unsqueeze(0)  # 1 T C H W
            with torch.no_grad():
                inputs = processor(clip, return_tensors="pt").to(device)
                logits = model(**inputs).logits
                probs  = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            pred_idx   = probs.argmax()
            pred_label = id2label[pred_idx]
            pred_conf  = float(probs[pred_idx])
            pred_probs = probs
            pred_color = COLORS[pred_idx % len(COLORS)]

        # ── FPS ───────────────────────────────────────────────────────────
        if frame_count % 15 == 0:
            elapsed     = time.time() - fps_time
            display_fps = 15 / elapsed if elapsed > 0 else 0
            fps_time    = time.time()

        # ── Overlay UI ────────────────────────────────────────────────────
        # Semi-transparent background at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (640, 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # Main prediction
        cv2.putText(frame, pred_label.upper(),
                    (12, 38), cv2.FONT_HERSHEY_DUPLEX, 1.1,
                    pred_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Conf: {pred_conf*100:.1f}%  |  FPS: {display_fps:.0f}",
                    (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (220, 220, 220), 1, cv2.LINE_AA)

        # Confidence bar
        bar_w = int(pred_conf * 300)
        cv2.rectangle(frame, (12, 76), (12 + bar_w, 86), pred_color, -1)
        cv2.rectangle(frame, (12, 76), (312, 86), (100, 100, 100), 1)

        # Side panel with per-class confidence bars
        panel_x, panel_y = 450, 100
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (panel_x - 5, panel_y - 10),
                      (635, panel_y + n_classes * 26 + 5), (0,0,0), -1)
        cv2.addWeighted(overlay2, 0.4, frame, 0.6, 0, frame)

        for i, (cls_id, cls_name) in enumerate(id2label.items()):
            prob_i  = float(pred_probs[cls_id]) if len(pred_probs) > cls_id else 0
            bar_len = int(prob_i * 160)
            col     = COLORS[cls_id % len(COLORS)]
            y_i     = panel_y + i * 26
            cv2.rectangle(frame, (panel_x, y_i), (panel_x + bar_len, y_i + 14), col, -1)
            cv2.rectangle(frame, (panel_x, y_i), (panel_x + 160, y_i + 14), (80,80,80), 1)
            cv2.putText(frame, f"{cls_name[:10]}: {prob_i*100:.0f}%",
                        (panel_x, y_i - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (230, 230, 230), 1, cv2.LINE_AA)

        # Buffer indicator
        buf_pct = int(len(buffer) / BUFFER_SZ * 100)
        cv2.putText(frame, f"Buf:{buf_pct}%",
                    (12, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (160,160,160), 1)

        cv2.imshow("V-JEPA 2 -- Real-Time  [q = exit]", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nCamera closed.")


# ── RUN ─────────────────────────────────────────────────────────────
# Use the best model in float16 for faster inference
rt_model = VJEPA2ForVideoClassification.from_pretrained(
    CFG["checkpoint_best"],
    torch_dtype=torch.float16
).to(DEVICE).eval()
rt_processor = VJEPA2VideoProcessor.from_pretrained(CFG["checkpoint_best"])

realtime_inference(rt_model, rt_processor, ID2LABEL, DEVICE, CFG)

"""---
## Resumen de archivos generados
"""

print("\n Archivos generados:\n")
for f in sorted(pathlib.Path(CFG["output_dir"]).rglob("*")):
    if f.is_file():
        size_kb = f.stat().st_size / 1024
        size_str = f"{size_kb/1024:.1f} MB" if size_kb > 1024 else f"{size_kb:.0f} KB"
        print(f"  {str(f):<55} {size_str:>10}")

print("\n Imágenes de evaluación:")
for img_name in ["training_curves.png", "confusion_matrix.png",
                 "per_class_accuracy.png", "avg_probabilities.png"]:
    img_path = os.path.join(CFG["output_dir"], img_name)
    if os.path.exists(img_path):
        print(f"  Mostrando: {img_name}")
        display(IPImage(img_path, width=700))



"""# DEPLOY

import torch
import numpy as np

from torchcodec.decoders import VideoDecoder
from transformers import AutoVideoProcessor, AutoModelForVideoClassification

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and video preprocessor
hf_repo = "facebook/vjepa2-vitg-fpc64-384-ssv2"

model = AutoModelForVideoClassification.from_pretrained(hf_repo).to(device)
processor = AutoVideoProcessor.from_pretrained(hf_repo)

# 1. instalar pyngrok si no lo tienes
#!pip install pyngrok

from transformers import AutoVideoProcessor, VJEPA2Model

m = VJEPA2Model.from_pretrained('facebook/vjepa2-vitg-fpc64-384-ssv2', torch_dtype='float16')

"""