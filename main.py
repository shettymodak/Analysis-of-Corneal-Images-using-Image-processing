import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

CLASSES = ["control", "stress", "recovery"]   # ← match your real folder names
SRC = Path("/mnt/c/Users/prasun/Downloads/dataset")   # WSL-mounted Windows path
DST = Path("/mnt/c/Users/prasun/Downloads/new_split")   # created automatically
SEED = 42

for cls in CLASSES:
    images = sorted((SRC / cls).glob("*"))
    if not images:
        print(f"WARNING: no images found in {SRC / cls} — check the path/name")
        continue    

    labels = [cls] * len(images)

    train_imgs, temp_imgs = train_test_split(
        images, test_size=0.30, stratify=labels, random_state=SEED
    )
    val_imgs, test_imgs = train_test_split(
        temp_imgs, test_size=0.50, stratify=[cls]*len(temp_imgs),
        random_state=SEED
    )

    for split_name, split_imgs in [("train", train_imgs),
                                   ("val", val_imgs),
                                   ("test", test_imgs)]:
        out_dir = DST / split_name / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for img in split_imgs:
            # THE FIX: prefix filename with class → globally unique names
            new_name = f"{cls}_{img.name}"
            shutil.copy2(img, out_dir / new_name)

    print(f"{cls}: {len(train_imgs)} train / {len(val_imgs)} val / {len(test_imgs)} test")