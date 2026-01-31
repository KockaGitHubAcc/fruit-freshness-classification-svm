import os
import shutil
import cv2
import numpy as np

def get_dataset_stats(data_dir):
    """
    Analyzes dataset structure, dimensions, filename metadata, 
    and captures example paths for the visual report.
    """
    print(f"--- 🔍 Analyzing Dataset at: {data_dir} ---")
    
    if not os.path.exists(data_dir):
        print(f"❌ Error: Directory '{data_dir}' not found.")
        return None

    stats = {
        'total_files': 0,
        'classes': {},
        'dimensions': [],
        'corrupted': 0,
        # 1. COUNTS (For Text Report)
        'augmentations': {
            'rotated': 0,
            'translation': 0,
            'vertical_flip': 0,
            'horizontal_flip': 0,
            'salt_pepper': 0,
            'original': 0
        },
        # 2. PATHS (For Visual Saving)
        'examples': {
            'original': None,
            'rotated': None,
            'translation': None,
            'vertical_flip': None,
            'salt_pepper': None,
            'min_size': {'path': None, 'area': float('inf')},
            'max_size': {'path': None, 'area': 0}
        }
    }

    subsets = ['train', 'test']
    
    for subset in subsets:
        subset_path = os.path.join(data_dir, subset)
        if not os.path.exists(subset_path):
            continue
            
        print(f"Scanning {subset} set...")
        
        for class_name in os.listdir(subset_path):
            class_dir = os.path.join(subset_path, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            if class_name not in stats['classes']:
                stats['classes'][class_name] = 0

            files = os.listdir(class_dir)
            for f in files:
                file_path = os.path.join(class_dir, f)
                
                if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue
                
                # Update basic counts
                stats['classes'][class_name] += 1
                stats['total_files'] += 1
                
                # --- METADATA ANALYSIS (Filename Check) ---
                f_lower = f.lower()
                is_augmented = False
                
                # Check for "rotated"
                if 'rotated' in f_lower:
                    stats['augmentations']['rotated'] += 1
                    if not stats['examples']['rotated']: stats['examples']['rotated'] = file_path
                    is_augmented = True
                
                # Check for "translation"
                if 'translation' in f_lower:
                    stats['augmentations']['translation'] += 1
                    if not stats['examples']['translation']: stats['examples']['translation'] = file_path
                    is_augmented = True
                
                # Check for "vertical_flip"
                if 'vertical_flip' in f_lower:
                    stats['augmentations']['vertical_flip'] += 1
                    if not stats['examples']['vertical_flip']: stats['examples']['vertical_flip'] = file_path
                    is_augmented = True
                
                # Check for "horizontal_flip"
                if 'horizontal_flip' in f_lower:
                    stats['augmentations']['horizontal_flip'] += 1
                    is_augmented = True
                
                # Check for "salt & pepper"
                if 'salt' in f_lower or 'pepper' in f_lower:
                    stats['augmentations']['salt_pepper'] += 1
                    if not stats['examples']['salt_pepper']: stats['examples']['salt_pepper'] = file_path
                    is_augmented = True
                
                # Check for "original"
                if not is_augmented:
                    stats['augmentations']['original'] += 1
                    # Grab a clean original example (ignoring screen shots if possible, or just taking the first clean one)
                    if not stats['examples']['original']: 
                        stats['examples']['original'] = file_path

                # --- DIMENSIONS (Sample every 50th) ---
                if stats['total_files'] % 50 == 0:
                    try:
                        img = cv2.imread(file_path)
                        if img is None:
                            stats['corrupted'] += 1
                        else:
                            h, w, _ = img.shape
                            area = h * w
                            stats['dimensions'].append((h, w))
                            
                            # Track Min/Max for saving
                            if area < stats['examples']['min_size']['area']:
                                stats['examples']['min_size']['area'] = area
                                stats['examples']['min_size']['path'] = file_path
                            
                            if area > stats['examples']['max_size']['area']:
                                stats['examples']['max_size']['area'] = area
                                stats['examples']['max_size']['path'] = file_path
                    except:
                        stats['corrupted'] += 1

    return stats

def save_results(stats, root_results_dir):
    """
    Creates 'analyze' folder inside root_results_dir and saves sample images.
    """
    if not stats: return

    # 1. Create Module Specific Subfolder
    module_dir = os.path.join(root_results_dir, 'analyze')
    if not os.path.exists(module_dir):
        os.makedirs(module_dir)
    
    print(f"\n[Analyze] Saving visual report to: {module_dir}")
    
    # 2. Save Augmentation Examples
    augmentations = ['original', 'rotated', 'translation', 'vertical_flip', 'salt_pepper']
    for aug_type in augmentations:
        src_path = stats['examples'].get(aug_type)
        if src_path:
            ext = os.path.splitext(src_path)[1]
            dst_name = f"example_{aug_type}{ext}"
            shutil.copy(src_path, os.path.join(module_dir, dst_name))

    # 3. Save Size Examples
    min_path = stats['examples']['min_size']['path']
    if min_path:
        shutil.copy(min_path, os.path.join(module_dir, "size_smallest.png"))

    max_path = stats['examples']['max_size']['path']
    if max_path:
        shutil.copy(max_path, os.path.join(module_dir, "size_largest.png"))

def print_report(stats):
    if not stats or stats['total_files'] == 0:
        print("⚠️  No images found.")
        return

    print("\n" + "="*40)
    print("      DATASET ANALYSIS REPORT")
    print("="*40)
    print(f"Total Images: {stats['total_files']}")
    
    print("\n📊 Class Balance:")
    print(f"{'Class Name':<20} | {'Count':<10}")
    print("-" * 33)
    for cls, count in stats['classes'].items():
        print(f"{cls:<20} | {count:<10}")

    print("\n🧪 Augmentation Detection (from filenames):")
    print(f"   - Original (estimated): {stats['augmentations']['original']}")
    print(f"   - Rotated:              {stats['augmentations']['rotated']}")
    print(f"   - Translated:           {stats['augmentations']['translation']}")
    print(f"   - Flipped (Vert):       {stats['augmentations']['vertical_flip']}")
    print(f"   - Flipped (Horiz):      {stats['augmentations']['horizontal_flip']}")
    print(f"   - Salt & Pepper Noise:  {stats['augmentations']['salt_pepper']}")

    # Dimension Analysis
    if stats['dimensions']:
        heights = [d[0] for d in stats['dimensions']]
        widths = [d[1] for d in stats['dimensions']]
        
        min_h, max_h = min(heights), max(heights)
        min_w, max_w = min(widths), max(widths)
        
        print("\n📏 Image Dimensions (Sampled):")
        print(f"   Min: {min_w}x{min_h}")
        print(f"   Max: {max_w}x{max_h}")
        
        if min_h != max_h or min_w != max_w:
            print("\n⚠️  WARNING: Variable sizes detected.")
            print("   >> MUST ADD: Preprocessing (Resize) step.")
        else:
            print("\n✅ Images are uniform.")
    
    print("="*40 + "\n")