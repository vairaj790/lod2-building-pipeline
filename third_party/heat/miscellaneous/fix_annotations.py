import os
import numpy as np
from tqdm import tqdm

annot_folder = r"C:\Users\fo37nor\job_work\heat\data\outdoor\cities_dataset\New folder"  # Change if needed

def convert_to_dict_format(file_path):
    data = np.load(file_path, allow_pickle=True).item()

    # Format 1: {'corners': ..., 'edges': ...}
    if isinstance(data, dict) and "corners" in data and "edges" in data:
        corners = data["corners"]
        edges = data["edges"]

        corner_map = {i: tuple(np.float64(x) for x in pt) for i, pt in enumerate(corners)}
        coord_dict = {}

        for i in range(len(corners)):
            key = corner_map[i]
            coord_dict[key] = []

        for i, j in edges:
            pt1 = corner_map[i]
            pt2 = corner_map[j]
            coord_dict[pt1].append(np.array(pt2))
            coord_dict[pt2].append(np.array(pt1))

        np.save(file_path, coord_dict, allow_pickle=True)
        print(f"✅ Converted from 'corners+edges': {os.path.basename(file_path)}")
        return

    # Format 2: already a dict of (x, y): [array(...), ...]
    if isinstance(data, dict):
        new_dict = {}
        for k, v_list in data.items():
            if not (isinstance(k, tuple) and len(k) == 2):
                print(f"⚠️ Skipped {os.path.basename(file_path)}: invalid key format.")
                return
            new_key = tuple(np.float64(x) for x in k)
            new_vals = [np.array(v, dtype=np.float64) for v in v_list]
            new_dict[new_key] = new_vals

        np.save(file_path, new_dict, allow_pickle=True)
        print(f"✅ Converted from 'dict of coords': {os.path.basename(file_path)}")
        return

    print(f"⚠️ Skipped {os.path.basename(file_path)}: unsupported format.")

# Run conversion on all .npy files
files = [f for f in os.listdir(annot_folder) if f.endswith(".npy")]

for fname in tqdm(files, desc="🔁 Converting"):
    convert_to_dict_format(os.path.join(annot_folder, fname))
