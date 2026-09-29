import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from config import DATA_ROOT, CLASSES

train_transform = A.Compose([
    A.Rotate(limit=180, p=0.9),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Affine(scale=(0.9, 1.1), translate_percent=0.05, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),
    A.GaussNoise(p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.15),
])

def preview(cls, n=6):
    src = DATA_ROOT / "train" / cls
    original = sorted(p for p in src.glob("*") if "_aug" not in p.name)[0]
    img = cv2.cvtColor(cv2.imread(str(original)), cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, n, figsize=(18, 3))
    for ax in axes:
        ax.imshow(train_transform(image=img)["image"])
        ax.axis("off")
    fig.suptitle(f"{cls} — augmented variants of {original.stem}")
    plt.show()

for cls in CLASSES:
    preview(cls)