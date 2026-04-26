import os
import numpy as np
import cv2
from glob import glob

# Paths
annot_dir = r"C:\Users\fo37nor\job_work\heat\ground_truth\New folder\all npy"
rgb_dir   = r"C:\Users\fo37nor\job_work\heat\ground_truth\New folder\all ground truth 256"
out_dir   = r"C:\Users\fo37nor\job_work\heat\ground_truth\New folder\visualize_annot"

# Create output directory if it doesn't exist
os.makedirs(out_dir, exist_ok=True)

# Parameters
draw_radius = 3
color_corner = (0, 0, 255)   # Red (BGR)
color_edge   = (255, 0, 0)   # Blue (BGR)
edge_thickness = 2           # Thicker line

# Image extensions to try (in priority order)
IMG_EXTS = [".jpg", ".jpeg", ".png"]

def find_matching_image(rgb_dir, base_name):
    """
    Return (image_path, ext) for the first existing image with given base_name.
    """
    for ext in IMG_EXTS:
        p = os.path.join(rgb_dir, base_name + ext)
        if os.path.exists(p):
            return p, ext
    return None, None

# Process all annotation files
for annot_file in sorted(glob(os.path.join(annot_dir, "*.npy"))):
    data = np.load(annot_file, allow_pickle=True).item()
    if not isinstance(data, dict):
        print(f"❌ Skipping {annot_file} — not a dict")
        continue

    base_name = os.path.splitext(os.path.basename(annot_file))[0]

    image_path, ext = find_matching_image(rgb_dir, base_name)
    if image_path is None:
        print(f"⚠️ Image not found for: {base_name} (tried {', '.join(IMG_EXTS)})")
        continue

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to read image: {image_path}")
        continue

    # Draw edges
    for key, neighbors in data.items():
        x1, y1 = map(int, key)
        for pt in neighbors:
            x2, y2 = map(int, pt)
            cv2.line(image, (x1, y1), (x2, y2), color_edge, edge_thickness)

    # Draw corners
    for key in data.keys():
        x, y = map(int, key)
        cv2.circle(image, (x, y), draw_radius, color_corner, -1)

    out_name = base_name + ext  # save with same extension as input image
    out_path = os.path.join(out_dir, out_name)
    cv2.imwrite(out_path, image)
    print(f"✅ Saved: {out_path}")