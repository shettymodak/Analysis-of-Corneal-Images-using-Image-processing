import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from sklearn.metrics import classification_report, confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image as PILImage
from pathlib import Path
from config import DATA_ROOT, CLASSES, SEED, IMG_SIZE

torch.manual_seed(SEED)
np.random.seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

# ---------- Transforms (NO augmentation — same as eval in training) ----------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

eval_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

def alb_wrapper(transform):
    def _apply(img):
        if isinstance(img, PILImage.Image):
            img = np.array(img)
        return transform(image=img)["image"]
    return _apply

# ---------- Load the trained model ----------
model = models.resnet18(weights=None)          # architecture only; weights come from file
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load("best_model.pt", weights_only=True, map_location=device))
model = model.to(device)
model.eval()

# ---------- Part 1: Evaluate on the test split ----------
test_ds = datasets.ImageFolder(str(DATA_ROOT / "test"), transform=alb_wrapper(eval_tf))
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

print(f"Classes (index order): {test_ds.classes}")
print(f"Test images: {len(test_ds)}\n")

y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb.to(device)).argmax(1).cpu()
        y_true.extend(yb.tolist())
        y_pred.extend(preds.tolist())

print("--- TEST SET RESULTS ---")
print(classification_report(y_true, y_pred, target_names=test_ds.classes))
print("Confusion matrix (rows = true, cols = predicted):")
print(confusion_matrix(y_true, y_pred))

# ---------- Part 2: Predict a single new image ----------
def predict_image(image_path):
    """Classify one image. Returns (class_name, confidence)."""
    img = PILImage.open(image_path).convert("RGB")
    x = alb_wrapper(eval_tf)(img).unsqueeze(0).to(device)   # add batch dimension

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    conf, idx = probs.max(0)
    return test_ds.classes[idx.item()], conf.item()

# ---------- Part 3: Demo — predict one image per test class folder ----------
print("\n--- SAMPLE PREDICTIONS ---")
for cls in test_ds.classes:
    folder = DATA_ROOT / "test" / cls
    sample = sorted(folder.glob("*"))[0]           # first test image of this class
    pred_cls, conf = predict_image(sample)
    marker = "✓" if pred_cls == cls else "✗ MISMATCH"
    print(f"{sample.name:45s} → predicted: {pred_cls:10s} ({conf:.1%}) {marker}")