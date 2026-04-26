import numpy as np

# Specify the file path
file_path = r"C:\Users\fo37nor\job_work\heat\ground_truth\simple\gabbled\annot\building_1_annotation.npy"

# Load the .npy file with the correct encoding
data = np.load(file_path, allow_pickle=True, encoding='latin1')

# Print the type and structure of the data
#print(type(data))
print(data)

# Print the contents in a more readable format, if it's a dictionary
if isinstance(data, dict):
    for key, value in data.items():
        print(f"Key: {key}, Value: {value}")
