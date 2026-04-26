import os
from PIL import Image

# === Input folder with PNG images ===
input_folder = r"C:\Users\fo37nor\job_work\heat\ground_truth\New folder\all ground truth 256\New folder"  # <- Change this to your folder path

# === Optional: output folder (same as input if left None) ===
output_folder = None  # Or set to another path, e.g. r"C:\path\to\jpg_output"

if output_folder is None:
    output_folder = input_folder

# === Convert each PNG to JPG and remove the PNG ===
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".png"):
        png_path = os.path.join(input_folder, filename)
        jpg_filename = os.path.splitext(filename)[0] + ".jpg"
        jpg_path = os.path.join(output_folder, jpg_filename)

        try:
            with Image.open(png_path) as img:
                rgb_img = img.convert("RGB")  # Convert to RGB (JPG doesn't support alpha)
                rgb_img.save(jpg_path, "JPEG", quality=95)

            os.remove(png_path)  # Delete PNG after successful conversion
            print(f"✅ Converted and deleted: {filename}")

        except Exception as e:
            print(f"❌ Failed to convert {filename}: {e}")

print("🎉 All PNGs processed.")
