import os
import glob
import pandas as pd
import sys 

# --- DATASET CONFIGURATION ---
# STRICTLY using the dataset you requested
KAGGLE_SOURCE = "narayanibokde/augmented-dataset-for-fruits-rottenfresh" 

def find_dataset_root(base_search_path):
    """
    Recursively searches for the actual folder containing image classes.
    It looks for any folder that contains subfolders with 'fresh' and 'rotten' 
    in their names (case-insensitive).
    """
    print(f"🔍 Searching for valid data structure inside: {base_search_path}")
    
    for root, dirs, files in os.walk(base_search_path):
        # Convert all directory names to lowercase for checking
        lower_dirs = [d.lower() for d in dirs]
        
        # Check if we see keywords "fresh" and "rotten" in the subdirectories
        has_fresh = any("fresh" in d for d in lower_dirs)
        has_rotten = any("rotten" in d for d in lower_dirs)
        
        if has_fresh and has_rotten:
            print(f"✅ Found dataset root at: {root}")
            return root
            
    # Fallback: Check if the base path itself has images directly (unlikely for this dataset but good safety)
    if len(glob.glob(os.path.join(base_search_path, "*.jpg"))) > 10:
        return base_search_path

    print("❌ Could not find 'Fresh' and 'Rotten' folders anywhere in the download.")
    return None

def download_with_kaggle_api(dataset_name, data_folder):
    """
    Downloads and extracts, then returns the CORRECT inner path where images reside.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("❌ Critical Error: 'kaggle' library not found. Run: pip install kaggle")
        sys.exit(1)
    
    DOWNLOAD_DIR = os.path.join(os.getcwd(), data_folder, dataset_name)
    
    # 1. Check if data exists
    if os.path.exists(DOWNLOAD_DIR):
        print(f"\n--- Checking existing data in {DOWNLOAD_DIR} ---")
        # Try to find the root in the existing folder
        found_root = find_dataset_root(DOWNLOAD_DIR)
        if found_root:
            return found_root
        print("⚠️ Folder exists but seems empty or incorrect. Re-downloading...")

    # 2. Download via Kaggle API
    print(f"\n--- Data Acquisition via Kaggle API ---")
    try:
        api = KaggleApi()
        
        # Secure Auth (No printing of keys)
        if os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'):
            api.set_config_value('username', os.environ.get('KAGGLE_USERNAME'))
            api.set_config_value('key', os.environ.get('KAGGLE_KEY'))
            print(f"🔑 Authenticated using environment variables.")
        else:
            api.authenticate() 
            print(f"🔑 Authenticated using local config.")

        os.makedirs(DOWNLOAD_DIR, exist_ok=True)

        print(f"🌍 Downloading dataset: {KAGGLE_SOURCE}...")
        api.dataset_download_files(dataset=KAGGLE_SOURCE, path=DOWNLOAD_DIR, unzip=True)
        print(f"✅ Download extracted to: {DOWNLOAD_DIR}")
        
        # 3. CRITICAL STEP: Find the actual root again after download
        final_root = find_dataset_root(DOWNLOAD_DIR)
        
        if not final_root:
            raise FileNotFoundError("Downloaded successfully, but valid 'Fresh/Rotten' folders are missing.")
             
        return final_root
        
    except Exception as e:
        print(f"❌ Error during Data Acquisition: {e}")
        sys.exit(1)

def setup_data_acquisition(dataset_name, data_folder):
    return download_with_kaggle_api(dataset_name, data_folder)

def get_data_statistics(data_dir, results_path):
    """
    Scans the verified data_dir and prints stats.
    """
    os.makedirs(results_path, exist_ok=True)
    stats_file_path = os.path.join(results_path, "data_statistics.csv")
    
    # Identify actual class folders (e.g., 'fresh_apples', 'Fresh', etc.)
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    stats = {'Class': [], 'Category': [], 'Count': []}
    total_images = 0
    
    for class_name in classes:
        class_path = os.path.join(data_dir, class_name)
        
        # Recursive glob to catch images
        images = glob.glob(os.path.join(class_path, '**', '*.*'), recursive=True)
        # Filter strictly for images
        images = [i for i in images if i.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        count = len(images)
        
        # Determine category (Fresh/Rotten)
        cat = "Fresh" if "fresh" in class_name.lower() else "Rotten" if "rotten" in class_name.lower() else "Other"
        
        stats['Class'].append(class_name)
        stats['Category'].append(cat)
        stats['Count'].append(count)
        total_images += count
    
    if total_images == 0:
        print("❌ Error: Folders found, but they contain NO images.")
        return None
        
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(stats_file_path, index=False)
    
    print("\n--- Dataset Summary ---")
    print(stats_df.sort_values(by='Category').to_markdown(index=False))
    print(f"\n✅ Total Images: {total_images}")
    print(f"✅ Statistics saved to: {stats_file_path}")
    
    return stats_df