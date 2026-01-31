import os

# Define the root data directory (assuming it's one level up as before)
# If your data folder is inside the project, change '..' to '.'
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), '..', 'data'))

def count_images(directory):
    """Counts png/jpg files in a directory recursively."""
    if not os.path.exists(directory):
        return 0
    
    total = 0
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                total += 1
    return total

def compare_folders(path1, path2, label):
    """Compares file counts between two paths."""
    count1 = count_images(path1)
    count2 = count_images(path2)
    
    print(f"--- Comparison: {label} ---")
    print(f"📂 Path A: .../{os.path.relpath(path1, BASE_DIR)}")
    print(f"   Count: {count1}")
    print(f"📂 Path B: .../{os.path.relpath(path2, BASE_DIR)}")
    print(f"   Count: {count2}")
    
    if count1 == count2 and count1 > 0:
        print(f"✅ MATCH: Both folders have exactly {count1} images.")
    elif count1 == 0 and count2 == 0:
        print("⚠️  BOTH EMPTY: Neither path has images.")
    else:
        print(f"❌ MISMATCH: {count1} vs {count2}")
    print("-" * 40 + "\n")

if __name__ == "__main__":
    print(f"Checking in Root: {BASE_DIR}\n")
    
    # 1. Compare TRAIN folders
    deep_train = os.path.join(BASE_DIR, 'dataset', 'dataset', 'train')
    shallow_train = os.path.join(BASE_DIR, 'dataset', 'train')
    compare_folders(deep_train, shallow_train, "TRAIN SET")

    # 2. Compare TEST folders
    deep_test = os.path.join(BASE_DIR, 'dataset', 'dataset', 'test')
    shallow_test = os.path.join(BASE_DIR, 'dataset', 'test')
    compare_folders(deep_test, shallow_test, "TEST SET")