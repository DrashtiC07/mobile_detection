import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader
import os
import random
import numpy as np

# =========================
# 🔹 SEED FOR REPRODUCIBILITY
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# =========================
# 🔹 TRANSFORMS
# =========================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomResizedCrop(260, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
])

val_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.CenterCrop(260),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# =========================
# 🔥 MAIN
# =========================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =========================
    # 🔹 LOAD DATA
    # =========================
    train_data = datasets.ImageFolder("dataset/train", transform=train_transform)
    val_data   = datasets.ImageFolder("dataset/val",   transform=val_transform)

    print(f"\nClasses: {train_data.classes}")
    print(f"Train samples: {len(train_data)} | Val samples: {len(val_data)}")

    train_loader = DataLoader(
        train_data, 
        batch_size=32,          # Increased for B0
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_data, 
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # =========================
    # 🔹 MODEL - EFFICIENTNET-B0 (Faster on CPU)
    # =========================
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = True

    num_classes = len(train_data.classes)
    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(512, num_classes)
    )

    model = model.to(device)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # =========================
    # 🔹 LOSS + OPTIMIZER
    # =========================
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # =========================
    # 🔹 MIXUP
    # =========================
    def mixup_data(x, y, alpha=0.4):
        if alpha > 0:
            lam = torch.distributions.Beta(alpha, alpha).sample().item()
        else:
            lam = 1.0
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(device)
        mixed_x = lam * x + (1 - lam) * x[index]
        return mixed_x, y, y[index], lam

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    # =========================
    # 🔹 TRAINING LOOP
    # =========================
    best_val_acc = 0.0
    os.makedirs("models", exist_ok=True)
    epochs = 30   # Good balance for speed + accuracy

    for epoch in range(epochs):
        # ----- TRAIN -----
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.4)

            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (lam * (predicted == labels_a).float() + 
                       (1 - lam) * (predicted == labels_b).float()).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total

        # ----- VALIDATION -----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:2d}/{epochs} | LR: {current_lr:.6f}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/efficientnet_best.pth")
            print(f"   → Best model saved! Val Acc: {val_acc:.2f}%")

        scheduler.step()

    # Final save
    torch.save(model.state_dict(), "models/efficientnet_final.pth")
    print(f"\n🎉 Training Finished! Best Validation Accuracy: {best_val_acc:.2f}%")