import os
import cv2
import numpy as np


# ============================================================
# EMBEDDED PATHS
# ============================================================
INPUT_IMAGE_PATH = r"C:\Users\fo37nor\job_work\heat\image_preprocessing_to_retain_geoinformation\Data created by SAM3\sam3_mask_based_patches\rgb_jpg_512\building_24427109.jpg"
INPUT_NPY_PATH = r"C:\Users\fo37nor\job_work\heat\image_preprocessing_to_retain_geoinformation\Data created by SAM3\sam3_mask_based_patches\rgb_jpg_512\building_24427109.npy"

OUTPUT_IMAGE_PATH = r"C:\Users\fo37nor\job_work\heat\data\outdoor\cities_dataset\building_24427109.jpg"
OUTPUT_NPY_PATH = r"C:\Users\fo37nor\job_work\heat\data\outdoor\cities_dataset\building_24427109.npy"

TARGET_SIZE = 256


def resize_annotation_dict(annotation_dict, scale_x, scale_y):
    resized_dict = {}

    for key, neighbors in annotation_dict.items():
        old_x, old_y = key

        new_key = (
            np.float64(old_x * scale_x),
            np.float64(old_y * scale_y)
        )

        new_neighbors = []
        for neighbor in neighbors:
            nx, ny = neighbor
            resized_neighbor = np.array(
                [np.float64(nx * scale_x), np.float64(ny * scale_y)],
                dtype=np.float64
            )
            new_neighbors.append(resized_neighbor)

        resized_dict[new_key] = new_neighbors

    return resized_dict


def main():
    # ------------------------------------------------------------
    # Load image
    # ------------------------------------------------------------
    if not os.path.exists(INPUT_IMAGE_PATH):
        raise FileNotFoundError(f"Input image not found:\n{INPUT_IMAGE_PATH}")

    image = cv2.imread(INPUT_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image:\n{INPUT_IMAGE_PATH}")

    original_height, original_width = image.shape[:2]

    # ------------------------------------------------------------
    # Validate input image size
    # ------------------------------------------------------------
    if original_width != 512 or original_height != 512:
        raise ValueError(
            f"Expected input image size 512x512, but got {original_width}x{original_height}"
        )

    # ------------------------------------------------------------
    # Load annotation
    # ------------------------------------------------------------
    if not os.path.exists(INPUT_NPY_PATH):
        raise FileNotFoundError(f"Input annotation not found:\n{INPUT_NPY_PATH}")

    annotation = np.load(INPUT_NPY_PATH, allow_pickle=True).item()

    if not isinstance(annotation, dict):
        raise ValueError("Annotation file is not a dictionary.")

    # ------------------------------------------------------------
    # Resize image
    # ------------------------------------------------------------
    resized_image = cv2.resize(
        image,
        (TARGET_SIZE, TARGET_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # ------------------------------------------------------------
    # Resize annotation
    # ------------------------------------------------------------
    scale_x = TARGET_SIZE / original_width
    scale_y = TARGET_SIZE / original_height

    resized_annotation = resize_annotation_dict(annotation, scale_x, scale_y)

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------
    output_image_dir = os.path.dirname(OUTPUT_IMAGE_PATH)
    output_npy_dir = os.path.dirname(OUTPUT_NPY_PATH)

    if output_image_dir:
        os.makedirs(output_image_dir, exist_ok=True)
    if output_npy_dir:
        os.makedirs(output_npy_dir, exist_ok=True)

    image_saved = cv2.imwrite(OUTPUT_IMAGE_PATH, resized_image)
    if not image_saved:
        raise IOError(f"Failed to save resized image:\n{OUTPUT_IMAGE_PATH}")

    np.save(OUTPUT_NPY_PATH, resized_annotation, allow_pickle=True)

    # ------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------
    print("✅ Resizing completed successfully.")
    print(f"Input image:  {INPUT_IMAGE_PATH}")
    print(f"Input npy:    {INPUT_NPY_PATH}")
    print(f"Output image: {OUTPUT_IMAGE_PATH}")
    print(f"Output npy:   {OUTPUT_NPY_PATH}")
    print(f"Original size: {original_width}x{original_height}")
    print(f"New size:      {TARGET_SIZE}x{TARGET_SIZE}")


if __name__ == "__main__":
    main()