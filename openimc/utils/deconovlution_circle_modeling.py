import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
import sys
from pathlib import Path

# ============================================
# Parameters
# ============================================
crater_radius = 0.5   # µm
step = 0.333          # µm
num_rows = 25
num_cols = 25

# CLI overrides
args = sys.argv[1:]
def _get_arg(flag, cast=float, default=None):
    if flag in args:
        try:
            return cast(args[args.index(flag) + 1])
        except Exception:
            return default
    return default

crater_radius = _get_arg('--crater', float, crater_radius) or crater_radius
step = _get_arg('--step', float, step) or step
grid_override = _get_arg('--grid', int, None)
no_plot = ('--no-plot' in args)

no_plot = True

# ============================================
# 1. Raster Setup
# ============================================
central_row = num_rows // 2
central_col = num_cols // 2

shot_centers = []
for r in range(num_rows):
    for c in range(num_cols):
        x = c * step
        y = -r * step
        shot_centers.append((x, y))
shot_centers = np.array(shot_centers, dtype=float)

central_idx = central_row * num_cols + central_col
central_center = shot_centers[central_idx]
effective_shots = shot_centers[:central_idx + 1].copy()

# ============================================
# 2. High-Res Subpixel Grid
# ============================================
grid_res = grid_override if grid_override is not None else 800
xs_rel = np.linspace(-crater_radius, crater_radius, grid_res)
ys_rel = np.linspace(-crater_radius, crater_radius, grid_res)
XX_rel, YY_rel = np.meshgrid(xs_rel, ys_rel)

XX = XX_rel + central_center[0]
YY = YY_rel + central_center[1]
points = np.c_[XX.ravel(), YY.ravel()]

# Mask: Only care about pixels inside the main crater
inside_mask = np.hypot(XX_rel, YY_rel) <= crater_radius
inside_indices = np.where(inside_mask.ravel())[0]
points_inside = points[inside_indices]

# ============================================
# 3. Signature-Based Region Segmentation
# ============================================
# Filter shots close to center
dists_from_center = np.hypot(effective_shots[:, 0] - central_center[0],
                             effective_shots[:, 1] - central_center[1])
rel_shots = effective_shots[dists_from_center <= (2.1 * crater_radius)]

print(f"Calculating overlaps for {len(rel_shots)} intersecting shots...")

# Assign random unique float ID to each shot
rng = np.random.default_rng(42)
shot_signatures = rng.uniform(100, 1000000, size=len(rel_shots))

# Compute overlaps
dists_matrix = np.sqrt(((rel_shots[None, :, :] - points_inside[:, None, :]) ** 2).sum(axis=2))
boolean_overlaps = dists_matrix <= crater_radius

# Pass Map (Integer count)
pass_values_inside = np.sum(boolean_overlaps, axis=1)

# Signature Map (Unique float per region)
signature_values_inside = boolean_overlaps @ shot_signatures

# Reconstruct 2D images
pass_map = np.zeros(grid_res * grid_res, dtype=int)
pass_map[inside_indices] = pass_values_inside
pass_map = pass_map.reshape(grid_res, grid_res)

sig_map = np.zeros(grid_res * grid_res, dtype=float)
sig_map[inside_indices] = signature_values_inside
sig_map = sig_map.reshape(grid_res, grid_res)

# ============================================
# 4. Labeling via Pole of Inaccessibility
# ============================================
region_data = []
min_pixel_area = 0  # Filter dust

unique_sigs = np.unique(sig_map[inside_mask.reshape(grid_res, grid_res)])

print(f"Found {len(unique_sigs)} distinct regions. calculating centers...")

for sig in unique_sigs:
    if sig == 0:
        continue

    # 1. Create binary mask for this specific region
    binary = (sig_map == sig).astype(np.uint8)

    # 2. Label components (handles cases where a region is split)
    labeled = label(binary, connectivity=2)
    props = regionprops(labeled)

    # Determine pass count (grab first valid pixel)
    representative_val = pass_map[binary == 1][0]

    for rp in props:
        if rp.area < min_pixel_area:
            continue

        # --- Pole of inaccessibility ---
        min_row, min_col, max_row, max_col = rp.bbox
        sub_mask = binary[min_row:max_row, min_col:max_col]

        dist_transform = distance_transform_edt(sub_mask)
        max_idx = np.argmax(dist_transform)
        max_pos = np.unravel_index(max_idx, dist_transform.shape)

        cy = max_pos[0] + min_row
        cx = max_pos[1] + min_col

        x_coord = np.interp(cx, np.arange(grid_res), xs_rel)
        y_coord = np.interp(cy, np.arange(grid_res), ys_rel)

        region_data.append({
            'val': representative_val,
            'x': x_coord,
            'y': y_coord,
            'area': float(rp.area)  # subpixel count
        })

print(f"Labeled {len(region_data)} regions.")

# ============================================
# 5. Compute Passes, Contributions, PSF kernel
# ============================================

def save_vector_grouped(vec, fname, group=5):
    """Save 1D array with `group` entries per line."""
    fname = Path(fname)
    try:
        with fname.open("w") as f:
            for i in range(0, len(vec), group):
                line = " ".join(str(v) for v in vec[i:i+group])
                f.write(line + "\n")
        print(f"Saved: {fname}")
    except Exception as e:
        import tempfile
        tmp = Path(tempfile.gettempdir()) / fname.name
        with tmp.open("w") as f:
            for i in range(0, len(vec), group):
                line = " ".join(str(v) for v in vec[i:i+group])
                f.write(line + "\n")
        print(f"[WARN] Could not save to {fname}, wrote to {tmp} instead. Error: {e}")

# 5.1 Passes array (one per region, in region_data order)
passes_array = np.array([rd['val'] for rd in region_data], dtype=int)

# 5.2 Contributions array (area-weighted, normalized)
subpixel_area = (2 * crater_radius / grid_res) ** 2
areas_pixels = np.array([rd['area'] for rd in region_data], dtype=float)
raw_contributions = areas_pixels * subpixel_area
contributions_array = raw_contributions / raw_contributions.sum()

print("\nComputed arrays:")
print(f"  Number of regions:      {len(region_data)}")
print(f"  Passes range:           {passes_array.min()}–{passes_array.max()}")
print(f"  Contributions sum:      {contributions_array.sum():.6f}")

save_vector_grouped(passes_array, "passes_array.txt", group=10)
save_vector_grouped(np.round(contributions_array, 8), "contributions_array.txt", group=10)

# 5.3 PSF kernel construction

def inverse_sigmoid_loss(passes, x0=7.0, I0=1.0):
    """
    Example inverse sigmoidal loss curve (same functional form
    as used in the HR-IMC paper).
    """
    p = np.asarray(passes, dtype=float)
    return I0 - I0 / (1.0 + np.exp(-(p - x0)))

def compute_psf_kernel(region_data, contributions, loss_fn=None):
    """
    Aggregate per-region contributions into a 3x3 PSF kernel.

    Parameters
    ----------
    region_data : list of dict
        Each dict has keys 'x', 'y', 'val'.
    contributions : 1D np.ndarray
        Normalized contribution weight per region (same order as region_data).
    loss_fn : callable or None
        Function mapping an array of pass values -> attenuation factors.
        If None, no attenuation is applied.

    Returns
    -------
    kernel : (3, 3) np.ndarray
        Normalized PSF suitable for Richardson–Lucy deconvolution.
    """
    passes = np.array([rd['val'] for rd in region_data], dtype=float)

    if loss_fn is None:
        attenuation = np.ones_like(passes, dtype=float)
    else:
        attenuation = np.asarray(loss_fn(passes), dtype=float)

    weights = contributions * attenuation

    kernel = np.zeros((3, 3), dtype=float)

    # Partition central pixel into 3x3 bins in normalized coordinates
    # Normalize x,y into [-1, 1] by dividing by crater_radius
    for rd, w in zip(region_data, weights):
        x = rd['x'] / crater_radius
        y = rd['y'] / crater_radius

        # Columns: left, center, right
        if x < -1.0 / 3.0:
            col = 0
        elif x > 1.0 / 3.0:
            col = 2
        else:
            col = 1

        # Rows: top, middle, bottom (y>0 is 'up')
        if y > 1.0 / 3.0:
            row = 0
        elif y < -1.0 / 3.0:
            row = 2
        else:
            row = 1

        kernel[row, col] += w

    # Normalize PSF
    s = kernel.sum()
    if s > 0:
        kernel /= s

    return kernel

# Example PSF: inverse sigmoidal loss (you can swap in any loss_fn, e.g. sinusoidal)
kernel_example = compute_psf_kernel(
    region_data,
    contributions_array,
    loss_fn=lambda p: inverse_sigmoid_loss(p, x0=7.0, I0=1.0)
)

print("\nExample 3x3 PSF kernel (inverse sigmoidal loss, x0=7):")
print(kernel_example)

np.savetxt("psf_kernel_3x3.txt", kernel_example, fmt="%.8f")

# ============================================
# 6. Plotting (unchanged)
# ============================================
if not no_plot:
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

    # Main pixel
    main_circle = Circle((0, 0), crater_radius, fill=False, linewidth=2.0, color='black', zorder=20)
    ax.add_patch(main_circle)

    # Clipping
    clip_path = PathPatch(main_circle.get_path().transformed(main_circle.get_transform()), visible=False)
    ax.add_patch(clip_path)

    # Draw Shots
    for x_s, y_s in rel_shots:
        xr = x_s - central_center[0]
        yr = y_s - central_center[1]
        circ = Circle((xr, yr), crater_radius, fill=False, linewidth=0.8, alpha=0.6, color='#444444')
        circ.set_clip_path(clip_path)
        ax.add_patch(circ)

    # Draw Labels
    for item in region_data:
        ax.text(item['x'], item['y'], str(item['val']),
                fontsize=6, ha='center', va='center',
                color='white', fontweight='bold', zorder=30,
                bbox=dict(boxstyle="circle,pad=0.05", fc="red", ec="none", alpha=0.8))

    ax.set_aspect('equal')
    ax.set_xlim(-crater_radius * 1.05, crater_radius * 1.05)
    ax.set_ylim(-crater_radius * 1.05, crater_radius * 1.05)
    ax.axis('off')
    ax.set_title("Corrected Pass Map (Pole of Inaccessibility)", fontsize=12)
    plt.savefig("pass_map_pole_of_inaccessibility.png", dpi=300, bbox_inches='tight')
