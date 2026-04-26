import os
import shutil

# === List Paths ===
all_list_path = r"C:\Users\fo37nor\job_work\heat\heat_data\data\outdoor\cities_dataset\all_list.txt"
train_list_path = r"C:\Users\fo37nor\job_work\heat\heat_data\data\outdoor\cities_dataset\train_list.txt"
valid_list_path = r"C:\Users\fo37nor\job_work\heat\heat_data\data\outdoor\cities_dataset\valid_list.txt"

my_list_path = r"C:\Users\fo37nor\job_work\heat\ground_truth\New folder\my_list.txt.txt"

# === Original target folders ===
original_rgb_folder = r"C:\Users\fo37nor\job_work\heat\heat_data\data\outdoor\cities_dataset\rgb"
original_det_folder = r"C:\Users\fo37nor\job_work\heat\heat_data\data\outdoor\det_final"
original_annot_folder = r"C:\Users\fo37nor\job_work\heat\heat_data\data\outdoor\cities_dataset\annot"

# === Source folders with your replacements ===
my_rgb_folder = r"C:\Users\fo37nor\job_work\heat\ground_truth\New folder\all ground truth 256\New folder"
my_det_folder = r"C:\Users\fo37nor\job_work\heat\ground_truth\New folder\det_final"
my_annot_folder = r"C:\Users\fo37nor\job_work\heat\ground_truth\New folder\all npy\New folder"

# === Load all lists ===
with open(all_list_path) as f:
    all_list = [line.strip() for line in f.readlines()]

with open(train_list_path) as f:
    train_list = [line.strip() for line in f.readlines()]

with open(valid_list_path) as f:
    valid_list = [line.strip() for line in f.readlines()]

with open(my_list_path) as f:
    my_list = [line.strip() for line in f.readlines()]

replace_count = len(my_list)

# === Replace from end in all_list and valid_list ===
all_list[-replace_count:] = my_list
valid_list[-replace_count:] = my_list

# === Save updated lists ===
with open(all_list_path, "w") as f:
    f.write("\n".join(all_list))

with open(valid_list_path, "w") as f:
    f.write("\n".join(valid_list))

# === Replace files across all folders ===
for name in my_list:
    # RGB
    src_rgb = os.path.join(my_rgb_folder, name + ".jpg")
    dst_rgb = os.path.join(original_rgb_folder, name + ".jpg")

    # DET_FINAL
    src_det = os.path.join(my_det_folder, name + ".npy")
    dst_det = os.path.join(original_det_folder, name + ".npy")

    # ANNOT
    src_annot = os.path.join(my_annot_folder, name + ".npy")
    dst_annot = os.path.join(original_annot_folder, name + ".npy")

    # === Replace if exists ===
    if os.path.exists(src_rgb):
        shutil.copy2(src_rgb, dst_rgb)
        print(f"✅ Replaced RGB: {dst_rgb}")
    else:
        print(f"⚠️ Missing RGB: {src_rgb}")

    if os.path.exists(src_det):
        shutil.copy2(src_det, dst_det)
        print(f"✅ Replaced DET: {dst_det}")
    else:
        print(f"⚠️ Missing DET: {src_det}")

    if os.path.exists(src_annot):
        shutil.copy2(src_annot, dst_annot)
        print(f"✅ Replaced ANNOT: {dst_annot}")
    else:
        print(f"⚠️ Missing ANNOT: {src_annot}")

print("✅ Done updating lists and replacing files in all folders.")
