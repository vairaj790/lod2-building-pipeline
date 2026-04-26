import os
import numpy as np

# === Set paths ===
input_folder = "/opt/heat/data/outdoor/cities_dataset/annot"
output_folder = "/opt/heat/data/outdoor/cities_dataset/annot/converted"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# === Process each .npy file ===
for filename in os.listdir(input_folder):
    if filename.endswith(".npy"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            # Load data
            with open(input_path, "rb") as f:
                original_data = np.load(f, allow_pickle=True).item()

            # Convert format
            converted_data = {}
            for key, value_list in original_data.items():
                new_key = (np.float64(key[0]), np.float64(key[1]))
                new_values = [
                    np.array([np.float64(pt[0]), np.float64(pt[1])]) for pt in value_list
                ]
                converted_data[new_key] = new_values

            # Save converted data
            with open(output_path, "wb") as f:
                np.save(f, converted_data)

            print(f"✅ Converted: {filename}")

        except Exception as e:
            print(f"❌ Failed to convert {filename}: {e}")
