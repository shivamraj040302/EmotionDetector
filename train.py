"""
train.py — Improved EmotionCNN Training Script
Better accuracy + reduced misclassification
"""

import argparse
import os
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

from detect import EmotionCNN, EMOTIONS


# =========================================================
# TRANSFORMS
# =========================================================

TRAIN_TRANSFORM = transforms.Compose([

    transforms.Grayscale(),

    transforms.Resize((64, 64)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(15),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.08, 0.08),
        scale=(0.9, 1.1)
    ),

    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3
    ),

    transforms.ToTensor(),

    transforms.Normalize([0.5], [0.5]),
])


VAL_TRANSFORM = transforms.Compose([

    transforms.Grayscale(),

    transforms.Resize((64, 64)),

    transforms.ToTensor(),

    transforms.Normalize([0.5], [0.5]),
])


# =========================================================
# TRAIN FUNCTION
# =========================================================

def train(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n[Device] {device.upper()}")

    # =====================================================
    # DATASET PATHS
    # =====================================================

    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"\nTrain folder not found: {train_dir}\n"
            f"Expected:\n"
            f"data/train/<emotion folders>"
        )

    # =====================================================
    # DATASETS
    # =====================================================

    train_ds = datasets.ImageFolder(train_dir, TRAIN_TRANSFORM)

    val_ds = datasets.ImageFolder(test_dir, VAL_TRANSFORM)

    print(f"\n[Classes detected]")
    print(train_ds.classes)

    # =====================================================
    # VERIFY CLASSES
    # =====================================================

    for cls in train_ds.classes:
        if cls not in EMOTIONS:
            print(f"WARNING: '{cls}' not in EMOTIONS list")

    # =====================================================
    # CLASS WEIGHTS
    # =====================================================

    targets = train_ds.targets

    class_counts = Counter(targets)

    print("\n[Class Distribution]")

    for idx, count in class_counts.items():
        print(f"{EMOTIONS[idx]} : {count}")

    weights = []
    total_samples = sum(class_counts.values())
    for i in range(len(EMOTIONS)):
        weight = total_samples / (len(EMOTIONS) * class_counts[i])
        weights.append(weight)

    weights = torch.tensor(
        weights,
        dtype=torch.float
    ).to(device)

    print("\n[Class Weights]")
    print(weights)

    # =====================================================
    # DATALOADERS
    # =====================================================

    nw = 0 if os.name == "nt" else min(4, os.cpu_count())

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=nw,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=nw,
        pin_memory=True
    )

    print(f"\n[Data]")
    print(f"Train Images : {len(train_ds)}")
    print(f"Validation Images : {len(val_ds)}")

    # =====================================================
    # MODEL
    # =====================================================

    model = EmotionCNN(
        num_classes=len(EMOTIONS)
    ).to(device)

    # =====================================================
    # LOSS FUNCTION
    # =====================================================

    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=0.05
    )

    # =====================================================
    # OPTIMIZER
    # =====================================================

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    # =====================================================
    # SCHEDULER
    # =====================================================

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # =====================================================
    # TRAINING LOOP
    # =====================================================

    best_val_acc = 0.0

    print(f"\n{'Epoch':>6} {'Loss':>10} {'Train%':>10} {'Val%':>10} {'Time':>8}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):

        model.train()

        running_loss = 0
        correct = 0
        total = 0

        start_time = time.time()

        # =================================================
        # TRAIN
        # =================================================

        for imgs, labels in train_loader:

            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(imgs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

            preds = outputs.argmax(1)

            correct += (preds == labels).sum().item()

            total += imgs.size(0)

        scheduler.step()

        train_acc = correct / total * 100

        epoch_loss = running_loss / total

        # =================================================
        # VALIDATION
        # =================================================

        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for imgs, labels in val_loader:

                imgs = imgs.to(device)

                labels = labels.to(device)

                outputs = model(imgs)

                preds = outputs.argmax(1)

                val_correct += (preds == labels).sum().item()

                val_total += imgs.size(0)

        val_acc = val_correct / val_total * 100

        elapsed = time.time() - start_time

        marker = "✓" if val_acc > best_val_acc else ""

        print(
            f"{epoch:>6} "
            f"{epoch_loss:>10.4f} "
            f"{train_acc:>9.2f}% "
            f"{val_acc:>9.2f}% "
            f"{elapsed:>7.1f}s "
            f"{marker}"
        )

        # =================================================
        # SAVE BEST MODEL
        # =================================================

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                args.save
            )

            print(f"✅ Best model saved : {args.save}")

    # =====================================================
    # FINISHED
    # =====================================================

    print("\n======================================")
    print(f"Training Complete")
    print(f"Best Validation Accuracy : {best_val_acc:.2f}%")
    print(f"Model Saved To : {args.save}")
    print("======================================\n")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train EmotionCNN"
    )

    parser.add_argument(
        "--data_dir",
        default="data",
        help="Dataset root folder"
    )

    parser.add_argument(
        "--epochs",
        default=120,
        type=int,
        help="Training epochs"
    )

    parser.add_argument(
        "--batch",
        default=32,
        type=int,
        help="Batch size"
    )

    parser.add_argument(
        "--lr",
        default=3e-4,
        type=float,
        help="Learning rate"
    )

    parser.add_argument(
        "--save",
        default="emotion_model.pth",
        help="Save model path"
    )

    args = parser.parse_args()

    train(args)