import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# === File Paths ===
image_path = r"C:\Users\fo37nor\job_work\heat\data\outdoor\cities_dataset\rgb\building_24427109.jpg"
npy_path = r"C:\Users\fo37nor\job_work\heat\data\outdoor\det_final\building_24427109.npy"

# === Load Image ===
image = Image.open(image_path)

# === Load and Normalize Corner Data ===
corners = np.load(npy_path, allow_pickle=True)

# Unwrap object if needed
if isinstance(corners, np.ndarray) and corners.shape == ():
    corners = corners.item()

# If it's a dict, check for keys like 'corners' or similar
if isinstance(corners, dict):
    # Try known key names or list all keys
    print("Available keys:", list(corners.keys()))
    if 'corners' in corners:
        corners = corners['corners']
    else:
        raise ValueError("Please update the script with the correct key from the dictionary.")

# Convert to NumPy array if it's a list
if isinstance(corners, list):
    corners = np.array(corners)

# === Validate shape ===
if not isinstance(corners, np.ndarray) or corners.ndim != 2 or corners.shape[1] != 2:
    raise ValueError("Corner data is not in expected 2D array format with shape (N, 2).")

# === Visualize ===
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.scatter(corners[:, 0], corners[:, 1], c='red', s=10, label='Corners')
plt.title("Predicted Corners on Image")
plt.axis('off')
plt.legend()
plt.tight_layout()
plt.show()
