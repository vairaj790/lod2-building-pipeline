#!/usr/bin/env python3

import os
from PIL import Image

# === Set your input folder here ===
INPUT_FOLDER = r"C:\Users\fo37nor\job_work\heat\ground_truth\New folder\all ground truth 256"

TARGET_FROM = (512, 512)
TARGET_TO   = (256, 256)

# Process both JPG and PNG
VALID_EXTS = (".jpg", ".jpeg", ".png")

for filename in os.listdir(INPUT_FOLDER):
    if not filename.lower().endswith(VALID_EXTS):
        continue

    path = os.path.join(INPUT_FOLDER, filename)

    try:
        with Image.open(path) as img:
            if img.size == TARGET_FROM:
                print(f"Resizing {filename}...")

                # Resize
                resized_img = img.resize(TARGET_TO, Image.Resampling.LANCZOS)

                # Save back in the same format
                ext = os.path.splitext(filename)[1].lower()

                if ext in (".jpg", ".jpeg"):
                    # JPEG doesn't support alpha; ensure RGB
                    if resized_img.mode not in ("RGB",):
                        resized_img = resized_img.convert("RGB")
                    resized_img.save(path, format="JPEG", quality=95, subsampling=0)

                elif ext == ".png":
                    # PNG can keep alpha; just save as PNG
                    resized_img.save(path, format="PNG", optimize=True)

            else:
                print(f"Skipping {filename} (size: {img.size})")

    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")