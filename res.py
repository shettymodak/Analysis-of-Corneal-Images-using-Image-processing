import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score,
                             f1_score)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image as PILImage
from config import DATA_ROOT, CLASSES, SEED, IMG_SIZE

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- transforms (same as eval) ----------
eval_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def alb_wrapper(transform):
    def _apply(img):
        if isinstance(img, PILImage.Image):
            img = np.array(img)
        return transform(image=img)["image"]
    return _apply

# ---------- load model ----------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load("best_model.pt", weights_only=True,
                                 map_location=device))
model = model.to(device)
model.eval()

# ---------- test data ----------
test_ds = datasets.ImageFolder(str(DATA_ROOT / "test"), transform=alb_wrapper(eval_tf))
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
class_names = test_ds.classes

# ---------- collect predictions AND penultimate-layer features ----------
features, y_true, y_pred = [], [], []

def hook(module, inp, out):          # grabs the 512-dim vector before fc
    features.append(inp[0].detach().cpu().numpy())

model.fc.register_forward_hook(hook)

with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb.to(device)).argmax(1).cpu()
        y_pred.extend(preds.tolist())
        y_true.extend(yb.tolist())

features = np.vstack(features)       # shape: (20, 512)

# =========================================================
# 1-5. ALL CLASSIFICATION METRICS
# =========================================================
print("=" * 55)
print("CLASSIFICATION RESULTS (test set, n = %d)" % len(y_true))
print("=" * 55)

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="macro")
rec = recall_score(y_true, y_pred, average="macro")
f1 = f1_score(y_true, y_pred, average="macro")

print(f"Accuracy : {acc:.4f}")
print(f"Precision (macro): {prec:.4f}")
print(f"Recall    (macro): {rec:.4f}")
print(f"F1-score  (macro): {f1:.4f}")

print("\nPer-class report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix (rows = true, cols = predicted):")
print(cm)

# =========================================================
# 6. PCA VISUALIZATION (on CNN features)
# =========================================================
pca = PCA(n_components=2)
pts = pca.fit_transform(features)

colors = {"control": "tab:blue", "recovery": "tab:green", "stress": "tab:red"}
plt.figure(figsize=(8, 6))
for i, cls in enumerate(class_names):
    mask = [j for j, t in enumerate(y_true) if t == i]
    plt.scatter(pts[mask, 0], pts[mask, 1],
                c=colors[cls], label=cls, s=80, edgecolors="k", alpha=0.8)

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.title("PCA of CNN feature space (test set)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("pca_features.png", dpi=200, bbox_inches="tight")
plt.show()

print(f"\nPCA plot saved to pca_features.png")
print(f"PC1+PC2 explain {pca.explained_variance_ratio_.sum():.1%} of variance")