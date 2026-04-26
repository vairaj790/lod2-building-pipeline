import os

# Define the folder path
image_folder = r"C:\Users\fo37nor\job_work\heat\data\outdoor\cities_dataset\rgb"

# Get list of image names without extension
image_files = [os.path.splitext(f)[0] for f in os.listdir(image_folder)
               if os.path.isfile(os.path.join(image_folder, f))]

# Sort for consistency
image_files.sort()

# Split into train (2/3) and valid (1/3)
split_index = len(image_files) * 2 // 3
train_list = image_files[:split_index]
valid_list = image_files[split_index:]

# Define output paths
all_list_path = os.path.join(image_folder, "all_list.txt")
train_list_path = os.path.join(image_folder, "train_list.txt")
valid_list_path = os.path.join(image_folder, "valid_list.txt")

# Write the lists
with open(all_list_path, "w") as f:
    f.write("\n".join(image_files))

with open(train_list_path, "w") as f:
    f.write("\n".join(train_list))

with open(valid_list_path, "w") as f:
    f.write("\n".join(valid_list))

print("✅ Lists created successfully.")
