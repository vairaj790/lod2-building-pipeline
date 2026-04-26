import os

# === Folder paths inside Docker container ===
npy_folder = "/opt/heat/data/outdoor/det_final"
png_folder = "/opt/heat/data/outdoor/cities_dataset/rgb"

def rename_files_in_folder(folder_path, extension, suffix_to_remove="_out"):
    for filename in os.listdir(folder_path):
        if filename.endswith(extension) and filename.endswith(suffix_to_remove + extension):
            old_path = os.path.join(folder_path, filename)
            new_name = filename.replace(suffix_to_remove + extension, extension)
            new_path = os.path.join(folder_path, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")

# === Rename files ===
rename_files_in_folder(npy_folder, ".npy")
rename_files_in_folder(png_folder, ".png")
