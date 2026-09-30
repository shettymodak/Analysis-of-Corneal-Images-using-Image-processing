"""
Nuclear (DAPI) morphometry across control / stress / recovery.
Measures per-nucleus: area, equivalent diameter, eccentricity, solidity.
Aggregates per image, compares across classes, statistical tests + plots.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from skimage import filters, measure, morphology, segmentation
from scipy import ndimage as ndi
from scipy.stats import shapiro, kruskal, mannwhitneyu

ORIGINALS = Path("/mnt/c/Users/prasun/Downloads/dataset")
CLASSES = ["control", "stress", "recovery"]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MIN_NUCLEUS_AREA = 50      # px — tune after looking at your images
EDGE_MARGIN = 5            # ignore nuclei touching image border (they're cut off)


# ---------------------------------------------------------------------
# 1. NUCLEUS SEGMENTATION (blue channel) + watershed to split touching ones
# ---------------------------------------------------------------------
def segment_nuclei(img_rgb):
    """img_rgb: float [0,1], (H,W,3). Returns label image of nuclei."""
    blue = img_rgb[:, :, 2]

    thr = filters.threshold_otsu(blue)
    mask = blue > thr
    mask = morphology.opening(mask, morphology.disk(2))
    mask = morphology.remove_small_objects(mask, max_size=MIN_NUCLEUS_AREA - 1)
    mask = ndi.binary_fill_holes(mask)

    if mask.sum() == 0:
        return np.zeros(blue.shape, dtype=int)

    # distance transform + watershed to separate touching nuclei
    distance = ndi.distance_transform_edt(mask)
    coords = np.column_stack(ndi.center_of_mass(mask, labels=mask))  # fallback seeds
    # smoother seeds: local maxima of distance map
    peaks = morphology.h_maxima(distance, 2)
    markers = measure.label(peaks)
    labels = segmentation.watershed(-distance, markers, mask=mask)
    labels = morphology.remove_small_objects(labels, max_size=MIN_NUCLEUS_AREA - 1)
    return labels


def measure_nuclei(img_rgb):
    """Returns list of dicts, one per valid nucleus."""
    labels = segment_nuclei(img_rgb)
    H, W = labels.shape
    props = measure.regionprops(labels)
    out = []
    for r in props:
        minr, minc, maxr, maxc = r.bbox
        # skip nuclei cut off by the image border
        if minr < EDGE_MARGIN or minc < EDGE_MARGIN or maxr > H - EDGE_MARGIN or maxc > W - EDGE_MARGIN:
            continue
        equiv_diam = 2 * np.sqrt(r.area / np.pi)
        out.append(dict(area=r.area,
                        diameter=equiv_diam,
                        eccentricity=r.eccentricity,
                        solidity=r.solidity))
    return out


# ---------------------------------------------------------------------
# 2. MEASURE ALL 120 IMAGES
# ---------------------------------------------------------------------
rows = []   # one row per NUCLEUS
img_rows = []  # one row per IMAGE (aggregates)

for cls in CLASSES:
    files = sorted(f for f in (ORIGINALS / cls).rglob("*")
                   if f.is_file() and f.suffix.lower() in IMG_EXTS)
    for f in files:
        img = plt.imread(f)
        if img.dtype == np.uint8:
            img = img.astype(float) / 255.0
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        nuclei = measure_nuclei(img)

        for n in nuclei:
            rows.append(dict(cls=cls, image=f.name, **n))

        if nuclei:
            areas = [n["area"] for n in nuclei]
            img_rows.append(dict(cls=cls, image=f.name,
                                 n_nuclei=len(nuclei),
                                 mean_area=np.mean(areas),
                                 mean_diameter=np.mean([n["diameter"] for n in nuclei])))
        else:
            img_rows.append(dict(cls=cls, image=f.name, n_nuclei=0,
                                 mean_area=np.nan, mean_diameter=np.nan))

df_nuc = pd.DataFrame(rows)      # nucleus-level
df_img = pd.DataFrame(img_rows)  # image-level
print(f"Segmented {len(df_nuc)} nuclei across {len(df_img)} images\n")

# ---------------------------------------------------------------------
# 3. PER-CLASS SUMMARY
# ---------------------------------------------------------------------
summary = df_nuc.groupby("cls")["area"].agg(["count", "mean", "std", "median"])
summary["mean_equiv_diam_px"] = df_nuc.groupby("cls")["diameter"].mean()
print("=" * 65)
print("NUCLEAR SIZE BY CLASS (nucleus-level)")
print("=" * 65)
print(summary, "\n")

print("Nuclei per image (mean ± std):")
print(df_img.groupby("cls")["n_nuclei"].agg(["mean", "std"]), "\n")

# ---------------------------------------------------------------------
# 4. STATISTICAL TESTS
# ---------------------------------------------------------------------
groups = [df_nuc.loc[df_nuc.cls == c, "area"].values for c in CLASSES]

# normality per group (decides parametric vs non-parametric wording)
for c, g in zip(CLASSES, groups):
    if len(g) > 3:
        stat, p = shapiro(g)
        print(f"Shapiro-Wilk {c}: p={p:.4f} {'(not normal)' if p < 0.05 else '(~normal)'}")

# omnibus test (non-parametric — safest for skewed size data)
stat, p = kruskal(*groups)
print(f"\nKruskal-Wallis (area across 3 classes): H={stat:.2f}, p={p:.4g}")

# pairwise comparisons
from itertools import combinations
print("\nPairwise Mann-Whitney U (nuclear area):")
for a, b in combinations(CLASSES, 2):
    ga = df_nuc.loc[df_nuc.cls == a, "area"]
    gb = df_nuc.loc[df_nuc.cls == b, "area"]
    u, p = mannwhitneyu(ga, gb, alternative="two-sided")
    diff = gb.mean() - ga.mean()
    print(f"  {a:>8} vs {b:<8}: Δ mean area = {diff:+8.1f} px², p = {p:.4g}")

# ---------------------------------------------------------------------
# 5. PLOTS
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# boxplot of nuclear area by class
data_by_cls = [df_nuc.loc[df_nuc.cls == c, "area"].values for c in CLASSES]
bp = axes[0].boxplot(data_by_cls, tick_labels=CLASSES, patch_artist=True, showfliers=True)
for patch, c in zip(bp["boxes"], ["tab:blue", "tab:green", "tab:red"]):
    patch.set_facecolor(c); patch.set_alpha(0.6)
axes[0].set_ylabel("Nuclear area (px²)")
axes[0].set_title("Nuclear size distribution by condition")

# nuclei count per image by class
count_data = [df_img.loc[df_img.cls == c, "n_nuclei"].values for c in CLASSES]
bp2 = axes[1].boxplot(count_data, tick_labels=CLASSES, patch_artist=True)
for patch, c in zip(bp2["boxes"], ["tab:blue", "tab:green", "tab:red"]):
    patch.set_facecolor(c); patch.set_alpha(0.6)
axes[1].set_ylabel("Nuclei per image")
axes[1].set_title("Nuclear count per field by condition")

plt.tight_layout()
plt.savefig("nuclear_morphometry.png", dpi=200, bbox_inches="tight")
plt.show()

# save raw measurements for the report / appendix
df_nuc.to_csv("nuclear_measurements.csv", index=False)
df_img.to_csv("image_level_summary.csv", index=False)
print("\nSaved: nuclear_morphometry.png, nuclear_measurements.csv, image_level_summary.csv")