import os
import shutil
import glob
from pathlib import Path

# Configuration for the output
OUTPUT_DIR_NAME = "Standardized_Dataset"
CLASSES = ["Fresh", "Rotten"]

def standardize_dataset(raw_data_path, data_folder):
    """
    Restructures the raw dataset into a clean binary format (Fresh vs Rotten).
    renames files to unique IDs to prevent overwriting.
    """
    output_path = os.path.join(data_folder, OUTPUT_DIR_NAME)
    
    print(f"\n--- 🧹 Starting Data Cleaning & Standardization ---")
    print(f"Source: {raw_data_path}")
    print(f"Destination: {output_path}")

    # 1. Create clean directory structure
    if os.path.exists(output_path):
        print(f"⚠️  Cleaning existing standardized folder...")
        shutil.rmtree(output_path)
    
    for c in CLASSES:
        os.makedirs(os.path.join(output_path, c), exist_ok=True)

    # 2. Walk through raw data and move files
    stats = {c: 0 for c in CLASSES}
    
    # Get all subdirectories in the raw data
    subdirs = [d for d in os.listdir(raw_data_path) if os.path.isdir(os.path.join(raw_data_path, d))]
    
    for subdir in subdirs:
        subdir_lower = subdir.lower()
        
        # Determine target class based on folder name
        target_class = None
        if "fresh" in subdir_lower:
            target_class = "Fresh"
        elif "rotten" in subdir_lower:
            target_class = "Rotten"
        
        if not target_class:
            print(f"⚠️  Skipping ambiguous folder: {subdir}")
            continue

        # Process files in this folder
        src_folder = os.path.join(raw_data_path, subdir)
        files = glob.glob(os.path.join(src_folder, "*.*"))
        
        for file_path in files:
            # Check if valid image
            if not file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
                
            # Generate a unique name to prevent collisions (e.g., Fresh_0001.jpg)
            # We ignore the original filename (e.g., 'fresh_orange') to solve the mislabeling confusion.
            stats[target_class] += 1
            new_filename = f"{target_class}_{stats[target_class]:05d}{Path(file_path).suffix}"
            dest_path = os.path.join(output_path, target_class, new_filename)
            
            shutil.copy2(file_path, dest_path)

    print(f"✅ Standardization Complete.")
    print(f"   - Fresh Images: {stats['Fresh']}")
    print(f"   - Rotten Images: {stats['Rotten']}")
    
    return output_path