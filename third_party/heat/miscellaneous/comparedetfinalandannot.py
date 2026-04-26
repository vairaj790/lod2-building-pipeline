import numpy as np

def load_npy_file(file_path):
    try:
        data = np.load(file_path, allow_pickle=True, encoding='latin1').item()
        print(f"Successfully loaded {file_path}")
        return data
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None

def load_det_final_file(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"Successfully loaded {file_path}")
        return data
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None

def compare_files(det_final_path, annotated_path):
    det_final_data = load_det_final_file(det_final_path)
    annotated_data = load_npy_file(annotated_path)

    if det_final_data is None or annotated_data is None:
        print("Error loading one or both of the files.")
        return

    print("\nDet Final File Content:")
    print(det_final_data)

    print("\nAnnotated File Content:")
    print(annotated_data)

    # Checking the structure and content
    if isinstance(annotated_data, dict):
        annotated_keys = set(annotated_data.keys())
        print(f"\nAnnotated file contains {len(annotated_keys)} keys.")

        for key in annotated_keys:
            print(f"Key: {key}, Connections: {annotated_data[key]}")
    else:
        print("\nAnnotated file is not in the expected dictionary format.")

    if isinstance(det_final_data, np.ndarray):
        print(f"\nDet final file contains {len(det_final_data)} corners.")
        for corner in det_final_data:
            print(f"Corner: {corner}")
    else:
        print("\nDet final file is not in the expected ndarray format.")

# Example usage
det_final_path = r"C:\Users\fo37nor\job_work\heat\demo_data\outdoor\det_final\1548204146.78.npy"
annotated_path = r"C:\Users\fo37nor\job_work\heat\demo_data\outdoor\cities_dataset\annot\1548204146.78.npy"

compare_files(det_final_path, annotated_path)
