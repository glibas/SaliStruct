import os
import shutil
import pandas as pd

# Paths
csv_file = r"C:\study\PhD\workspace\SaliStruct\data\uicrit\uicrit_public.csv"
source_dir = r"C:\Users\glibb\Downloads\unique_uis\combined"
target_dir = r"C:\study\PhD\workspace\SaliStruct\data\uicrit\screenshots"

# Ensure target directory exists
os.makedirs(target_dir, exist_ok=True)

# Load CSV file
df = pd.read_csv(csv_file)

# Get unique rico_ids
rico_ids = set(df["rico_id"].astype(str))  # Convert to string for filename matching

# Iterate through source directory and copy matching files
for filename in os.listdir(source_dir):
    if filename.endswith(".jpg"):
        file_id = filename.split(".")[0]  # Extract the numeric ID part
        if file_id in rico_ids:
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, filename)
            shutil.copy2(source_path, target_path)
            print(f"Copied: {filename}")

print("Copying complete.")
