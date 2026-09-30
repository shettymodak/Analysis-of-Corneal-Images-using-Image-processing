"""
Traditional ML pipeline: hand-crafted feature extraction from the 120
ORIGINAL images -> PCA visualization -> SVM / Random Forest / k-NN
classified with stratified 5-fold cross-validation.
Produces: metrics per classifier, aggregated confusion matrix, PCA plot.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from skimage import color, filters, measure, morphology, feature
from skimage.feature import graycomatrix, graycoprops
from scipy import stats as scipy_stats

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

from config import SEED

np.random.seed(SEED)

# =====================================================================
# 1. FEATURE EXTRACTION
# =====================================================================
# Assumes originals are still in class folders:
#   /mnt/c/Users/prasun/Downloads/dataset/{control,stress,recovery}/
ORIGINALS = Path("/mnt/c/Users/prasun/Downloads/dataset")
CLASSES = ["control", "stress", "recovery"]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def extract_features(img_rgb):
    """
    img_rgb: float RGB image, shape (H, W, 3), values in [0, 1]
    Channels: R=red, G=green (tubulin), B=blue (DAPI)
    Returns a 1-D feature vector.
    """
    green = img_rgb[:, :, 1]
    blue  = img_rgb[:, :, 2]
    feats = []

    # ---- A. Intensity statistics (tubulin, green channel) ----
    feats += [green.mean(), green.std(),
              scipy_stats.skew(green.ravel()), scipy_stats.kurtosis(green.ravel()),
              np.percentile(green, 50), np.percentile(green, 90)]

    # Fraction of image occupied by tubulin signal (Otsu threshold)
    g_thr = filters.threshold_otsu(green)
    g_mask = green > g_thr
    feats.append(g_mask.mean())

    # ---- B. Texture (GLCM / Haralick on green channel) ----
    g_u8 = (green * 255).astype(np.uint8)
    glcm = graycomatrix(g_u8, distances=[1, 3], angles=[0, np.pi/2],
                        levels=256, symmetric=True, normed=True)
    for prop in ["contrast", "homogeneity", "energy", "correlation"]:
        feats.append(graycoprops(glcm, prop).mean())

    # ---- C. Tubulin network morphology (skeleton of green mask) ----
    g_clean = morphology.remove_small_objects(
        morphology.opening(g_mask, morphology.disk(2)), max_size=49)
    skeleton = morphology.skeletonize(g_clean)
    feats.append(skeleton.sum())                      # total network length (px)

    # Branch points: skeleton pixels with >2 skeleton neighbours
    neigh = morphology.convex_hull_image  # placeholder to avoid lint; not used
    from scipy.ndimage import convolve
    k = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    neighbour_count = convolve(skeleton.astype(np.uint8), k)
    branch_points = ((skeleton) & (neighbour_count >= 3)).sum()
    feats.append(branch_points)

    # Network fragmentation: number of connected components in green mask
    labeled = measure.label(g_clean)
    feats.append(labeled.max())                       # number of fragments

    # ---- D. Nuclear features (DAPI, blue channel) ----
    b_thr = filters.threshold_otsu(blue)
    b_mask = morphology.remove_small_objects(blue > b_thr, 30)
    b_mask = morphology.binary_opening(b_mask, morphology.disk(2))
    lbl = measure.label(b_mask)
    regions = measure.regionprops(lbl)
    n_nuclei = len(regions)
    feats.append(n_nuclei)
    if n_nuclei > 0:
        areas   = [r.area for r in regions]
        eccens  = [r.eccentricity for r in regions]
        solid   = [r.solidity for r in regions]
        feats += [np.mean(areas), np.std(areas),
                  np.mean(eccens), np.mean(solid)]
    else:
        feats += [0, 0, 0, 0]

    return np.array(feats, dtype=float)


# ---- Load all 120 originals and extract features ----
X, y, names = [], [], []
for label, cls in enumerate(CLASSES):
    folder = ORIGINALS / cls
    files = sorted(f for f in folder.rglob("*")
                   if f.is_file() and f.suffix.lower() in IMG_EXTS)
    for f in files:
        img = plt.imread(f)                            # works for png/tif
        if img.dtype == np.uint8:
            img = img.astype(float) / 255.0
        if img.ndim == 2:                              # grayscale safety
            img = np.stack([img]*3, axis=-1)
        X.append(extract_features(img))
        y.append(label)
        names.append(f"{cls}/{f.name}")

X = np.array(X)
y = np.array(y)
print(f"Extracted {X.shape[1]} features from {X.shape[0]} images")

# =====================================================================
# 2. PCA VISUALIZATION OF HAND-CRAFTED FEATURES
# =====================================================================
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
pts = pca.fit_transform(X_scaled)

colors = ["tab:blue", "tab:green", "tab:red"]
plt.figure(figsize=(8, 6))
for i, cls in enumerate(CLASSES):
    m = y == i
    plt.scatter(pts[m, 0], pts[m, 1], c=colors[i], label=cls,
                s=80, edgecolors="k", alpha=0.8)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.title("PCA of hand-crafted features (all 120 originals)")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig("pca_traditional_features.png", dpi=200, bbox_inches="tight")
plt.show()
print(f"PC1+PC2 explain {pca.explained_variance_ratio_.sum():.1%} of variance\n")

# =====================================================================
# 3. CLASSIFIERS WITH 5-FOLD STRATIFIED CROSS-VALIDATION
#    (every one of the 120 images is used for training AND testing,
#     but never in the same fold -> honest estimate on all data)
# =====================================================================
classifiers = {
    "SVM (RBF)":            make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale", random_state=SEED)),
    "Random Forest":        make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=300, random_state=SEED)),
    "k-NN (k=5)":           make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Logistic Regression":  make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED)),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

results = {}
for name, clf in classifiers.items():
    y_pred = cross_val_predict(clf, X_scaled, y, cv=cv)
    acc  = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y, y_pred, average="macro", zero_division=0)
    cm   = confusion_matrix(y, y_pred)
    results[name] = dict(acc=acc, prec=prec, rec=rec, f1=f1, cm=cm)

    print("=" * 55)
    print(f"{name}  (5-fold CV, n=120)")
    print("=" * 55)
    print(f"Accuracy : {acc:.4f} | Precision: {prec:.4f} | "
          f"Recall: {rec:.4f} | F1: {f1:.4f}")
    print(classification_report(y, y_pred, target_names=CLASSES, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(cm, "\n")

# =====================================================================
# 4. COMPARISON TABLE (traditional ML vs CNN)
# =====================================================================
print("=" * 65)
print("SUMMARY — traditional ML (5-fold CV, n=120) vs CNN (test split, n=20)")
print("=" * 65)
print(f"{'Model':<22}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}")
print("-" * 65)
for name, r in results.items():
    print(f"{name:<22}{r['acc']:>10.4f}{r['prec']:>11.4f}{r['rec']:>9.4f}{r['f1']:>8.4f}")
print("-" * 65)
print(f"{'CNN (ResNet18)':<22}{0.8500:>10.4f}{0.7764:>11.4f}{0.8333:>9.4f}{0.7960:>8.4f}")
print("\nNOTE: CNN numbers are from the single held-out test split (n=20);")
print("traditional numbers are 5-fold CV over all 120 originals.")