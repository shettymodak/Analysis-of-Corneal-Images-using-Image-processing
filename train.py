import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from sklearn.metrics import classification_report, confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image as PILImage
from config import DATA_ROOT, CLASSES, SEED, IMG_SIZE

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------- 1. Device ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")

# ---------- 2. Transforms ----------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

train_tf = A.Compose([
    A.Rotate(limit=180, p=0.9),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Affine(scale=(0.9, 1.1), translate_percent=0.05, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),
    A.GaussNoise(p=0.3),
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

eval_tf = A.Compose([                # NO augmentation for val/test
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

# ---------- 2b. Wrapper: make albumentations 2.x compatible with ImageFolder ----------
# ImageFolder calls transform(img) positionally and loads PIL images;
# albumentations 2.x requires named args and prefers numpy arrays.
def alb_wrapper(transform):
    def _apply(img):
        if isinstance(img, PILImage.Image):
            img = np.array(img)          # PIL → numpy, stays RGB
        return transform(image=img)["image"]
    return _apply

# ---------- 3. Datasets & loaders ----------
train_ds = datasets.ImageFolder(str(DATA_ROOT / "train"), transform=alb_wrapper(train_tf))
val_ds   = datasets.ImageFolder(str(DATA_ROOT / "val"),   transform=alb_wrapper(eval_tf))
test_ds  = datasets.ImageFolder(str(DATA_ROOT / "test"),  transform=alb_wrapper(eval_tf))

print("Classes (index order):", train_ds.classes)   # alphabetical!
print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,
                          num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False,
                          num_workers=4, pin_memory=True)

# ---------- 4. Model: pretrained ResNet18 ----------
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))  # use ImageFolder's class count/order
model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

EPOCHS = 30
best_val_acc = 0.0

# ---------- 5. Training loop ----------
for epoch in range(EPOCHS):
    # --- train ---
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    scheduler.step()

    # --- validate (REAL images only) ---
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    val_acc = correct / total

    print(f"epoch {epoch:02d} | train_loss {train_loss/len(train_ds):.4f} "
          f"| val_acc {val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pt")
        print(f"   ↳ saved new best (val_acc {val_acc:.3f})")

print(f"\nBest val accuracy: {best_val_acc:.3f}")

# ---------- 6. Final test evaluation (run ONCE, at the end) ----------
model.load_state_dict(torch.load("best_model.pt", weights_only=True))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb.to(device)).argmax(1).cpu()
        y_true.extend(yb.tolist())
        y_pred.extend(preds.tolist())

class_names = test_ds.classes   # alphabetical order matches label indices
print("\n--- TEST SET RESULTS ---")
print(classification_report(y_true, y_pred, target_names=class_names))
print("Confusion matrix (rows = true, cols = predicted):")
print(confusion_matrix(y_true, y_pred))