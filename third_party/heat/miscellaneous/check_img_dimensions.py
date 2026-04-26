import os
import numpy as np

annot_folder = "/opt/heat/data/outdoor/cities_dataset/annot"
image_size = 512

def is_out_of_bounds(x, y, size):
    return not (0 <= x < size and 0 <= y < size)

for fname in os.listdir(annot_folder):
    if not fname.endswith(".npy"):
        continue

    path = os.path.join(annot_folder, fname)
    try:
        data = np.load(path, allow_pickle=True).item()
    except Exception as e:
        print(f"❌ Failed to load {fname}: {e}")
        continue

    out_of_bounds = []

    for key, connections in data.items():
        if not isinstance(key, tuple):
            continue  # skip if not a coordinate

        x, y = map(float, key)
        if is_out_of_bounds(x, y, image_size):
            out_of_bounds.append(("key", x, y))

        for pt in connections:
            if not isinstance(pt, np.ndarray) or pt.shape != (2,):
                continue
            x2, y2 = map(float, pt)
            if is_out_of_bounds(x2, y2, image_size):
                out_of_bounds.append(("value", x2, y2))

    if out_of_bounds:
        print(f"⚠️ {fname} has out-of-bounds points:")
        for origin, x, y in out_of_bounds:
            print(f"   • {origin} point: ({x}, {y})")
    else:
        print(f"✅ {fname} OK")
