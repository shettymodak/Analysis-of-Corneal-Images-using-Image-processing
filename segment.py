import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from pathlib import Path
from scipy import ndimage as ndi

from skimage import filters, measure, morphology, segmentation, color as skcolor

MIN_NUCLEUS_AREA = 50
EDGE_MARGIN = 5

# ---------------------------------------------------------------
# 1. NUCLEI SEGMENTATION (blue/DAPI channel)
# ---------------------------------------------------------------
def segment_nuclei(img_rgb):
    blue = img_rgb[:, :, 2]
    thr = filters.threshold_otsu(blue)
    mask = blue > thr
    mask = morphology.opening(mask, morphology.disk(2))
    mask = morphology.remove_small_objects(mask, max_size=MIN_NUCLEUS_AREA - 1)
    mask = ndi.binary_fill_holes(mask)
    if mask.sum() == 0:
        return np.zeros(blue.shape, dtype=int)
    distance = ndi.distance_transform_edt(mask)
    peaks = morphology.h_maxima(distance, 2)
    markers = measure.label(peaks)
    labels = segmentation.watershed(-distance, markers, mask=mask)
    return morphology.remove_small_objects(labels, max_size=MIN_NUCLEUS_AREA - 1)

# ---------------------------------------------------------------
# 2. TUBULIN SEGMENTATION (green channel)
# ---------------------------------------------------------------
def segment_tubulin(img_rgb):
    green = img_rgb[:, :, 1]
    thr = filters.threshold_otsu(green)
    mask = green > thr
    mask = morphology.opening(mask, morphology.disk(2))
    mask = morphology.remove_small_objects(mask, max_size=49)
    skeleton = morphology.skeletonize(mask)
    return mask, skeleton

def branch_points(skeleton):
    from scipy.ndimage import convolve
    k = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    n = convolve(skeleton.astype(np.uint8), k)
    return (skeleton & (n >= 3))

# ---------------------------------------------------------------
# 3. BUILD THE COLOR OVERLAY
# ---------------------------------------------------------------
def make_overlay(img_rgb):
    """Returns (overlay RGB image, stats dict)."""
    img = np.clip(img_rgb, 0, 1).astype(float)
    h, w = img.shape[:2]

    nuc_labels = segment_nuclei(img_rgb)
    tub_mask, skeleton = segment_tubulin(img_rgb)
    bpts = branch_points(skeleton)

    # Convert to grayscale-intensity base so structures stand out
    base = img_rgb.mean(axis=2)
    base = np.stack([base*0.6]*3, axis=-1)          # dimmed gray background

    overlay = base.copy()

    # -- tubulin mask: yellow tint where signal exists
    overlay[tub_mask] = 0.55*overlay[tub_mask] + 0.45*np.array(to_rgb("yellow"))

    # -- tubulin skeleton: solid red lines (the network structure)
    overlay[skeleton] = to_rgb("red")

    # -- branch points: white dots
    overlay[bpts] = to_rgb("white")

    # -- nuclei: cyan boundary + translucent cyan fill
    nuc_mask = nuc_labels > 0
    overlay[nuc_mask] = 0.6*overlay[nuc_mask] + 0.4*np.array(to_rgb("cyan"))
    boundaries = segmentation.find_boundaries(nuc_labels, mode="outer")
    overlay[boundaries] = to_rgb("deepskyblue")

    # -- stats for the display panel
    n_nuclei = len([r for r in measure.regionprops(nuc_labels)
                    if r.area >= MIN_NUCLEUS_AREA])
    stats = dict(
        n_nuclei=n_nuclei,
        tubulin_coverage=float(tub_mask.mean()),
        network_length_px=int(skeleton.sum()),
        branch_points=int(bpts.sum()),
        fragments=int(measure.label(tub_mask).max()),
    )
    return np.clip(overlay, 0, 1), stats

# ---------------------------------------------------------------
# 4. DISPLAY: side-by-side figure with legend + stats
# ---------------------------------------------------------------
def segment_and_display(image_path, save_name=None):
    img = plt.imread(image_path)
    if img.dtype == np.uint8:
        img = img.astype(float) / 255.0
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)

    overlay, stats = make_overlay(img)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].imshow(img); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(overlay); axes[1].set_title("Segmented structures"); axes[1].axis("off")

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(color="deepskyblue", label="Nuclei (DAPI)"),
               Patch(color="red", label="Tubulin network (skeleton)"),
               Patch(color="yellow", label="Tubulin signal"),
               Patch(color="white", label="Branch points")]
    axes[1].legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.8)

    # Stats text under the figure
    stats_txt = (f"Nuclei: {stats['n_nuclei']}   |   "
                 f"Network length: {stats['network_length_px']} px   |   "
                 f"Branch points: {stats['branch_points']}   |   "
                 f"Coverage: {stats['tubulin_coverage']:.1%}   |   "
                 f"Fragments: {stats['fragments']}")
    fig.suptitle(stats_txt, fontsize=10)
    plt.tight_layout()

    if save_name:
        fig.savefig(save_name, dpi=200, bbox_inches="tight")
        plt.imsave(save_name.replace(".png", "_overlay.png"), overlay)
    plt.show()
    return overlay, stats


# ---------------------------------------------------------------
# DEMO: one image per class
# ---------------------------------------------------------------
if __name__ == "__main__":
    DATA = Path("/mnt/c/Users/prasun/Downloads/dataset")
    for cls in ["control", "stress", "recovery"]:
        sample = sorted((DATA / cls).glob("*"))[0]
        print(f"\n=== {cls}: {sample.name} ===")
        overlay, stats = segment_and_display(sample, f"segmented_{cls}.png")
        print(stats)