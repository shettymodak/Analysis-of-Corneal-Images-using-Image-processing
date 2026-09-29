import cv2
import albumentations as A
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Fluorescence-safe transforms (joint channel transforms, no hue shifts)
train_transform = A.Compose([
    A.Rotate(limit=180, p=0.9),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Affine(scale=(0.9, 1.1), translate_percent=0.05, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),
    A.GaussNoise(p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.15),
])

DATA_ROOT = Path("/mnt/c/Users/prasun/Downloads/new_split")

# copies per ORIGINAL image, chosen to balance the classes
# control:  51 × 40  ≈ 2040
# stress:   18 × 114 ≈ 2052
# recovery: 12 × 171 ≈ 2052
N_AUG_PER_CLASS = {
    "control": 40,
    "stress": 114,
    "recovery": 171,
}

def augment_class(args):
    cls, n_aug = args
    src_dir = DATA_ROOT / "train" / cls
    count = 0
    for img_path in sorted(src_dir.glob("*")):
        if "_aug" in img_path.stem:      # don't re-augment already-augmented files
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Skipping unreadable: {img_path}")
            continue
        for i in range(n_aug):
            aug = train_transform(image=img)["image"]
            cv2.imwrite(str(src_dir / f"{img_path.stem}_aug{i:03d}.png"), aug)
            count += 1
    return f"{cls}: wrote {count} augmented images"

if __name__ == "__main__":
    with ProcessPoolExecutor() as ex:
        for msg in ex.map(augment_class, list(N_AUG_PER_CLASS.items())):
            print(msg)

# Final verification
print("\nFinal train counts:")
for cls in N_AUG_PER_CLASS:
    n = len(list((DATA_ROOT / "train" / cls).glob("*")))
    print(f"  {cls}: {n}")

