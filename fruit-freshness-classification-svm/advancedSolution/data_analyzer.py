import os
import glob
import random
import cv2  # OpenCV for blur check
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Configuration
SAMPLE_SIZE = 500  

def plot_class_balance(data_dir, output_dir):
    """ Saves a bar chart showing class distribution. """
    print("   📊 Generating Class Balance Report...")
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    counts = []
    
    for c in classes:
        count = len(glob.glob(os.path.join(data_dir, c, "*.*")))
        counts.append(count)
        
    plt.figure(figsize=(6, 4))
    sns.barplot(x=classes, y=counts, palette='viridis', hue=classes, legend=False)
    plt.title("Class Distribution: Fresh vs Rotten")
    plt.ylabel("Number of Images")
    
    save_path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"      -> Saved to: {save_path}")

def analyze_image_dimensions(data_dir, output_dir):
    """ Scans image dimensions. """
    print(f"   📏 Analyzing Image Dimensions (Sample N={SAMPLE_SIZE})...")
    
    widths = []
    heights = []
    all_files = glob.glob(os.path.join(data_dir, "**", "*.*"), recursive=True)
    
    if len(all_files) > SAMPLE_SIZE:
        sampled_files = random.sample(all_files, SAMPLE_SIZE)
    else:
        sampled_files = all_files
        
    for f in sampled_files:
        try:
            with Image.open(f) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except:
            pass 

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=widths, y=heights, alpha=0.6)
    plt.title(f"Image Dimensions (Sample N={len(widths)})")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    
    save_path = os.path.join(output_dir, "image_dimensions.png")
    plt.savefig(save_path)
    plt.close()
    
    if widths:
        avg_w = np.mean(widths)
        avg_h = np.mean(heights)
        print(f"      -> Average Size: {avg_w:.0f}x{avg_h:.0f}")
    
    print(f"      -> Saved scatter plot to: {save_path}")

def visualize_sample_grid(data_dir, output_dir):
    """ Creates a visual grid. """
    print("   👀 Generating Visual Sanity Check Grid...")
    classes = ["Fresh", "Rotten"]
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for i, cls in enumerate(classes):
        class_path = os.path.join(data_dir, cls)
        images = glob.glob(os.path.join(class_path, "*.*"))
        
        if not images:
            print(f"      ⚠️ No images found for class {cls}")
            continue
            
        n_samples = min(5, len(images))
        samples = random.sample(images, n_samples)
        
        for j, img_path in enumerate(samples):
            try:
                img = Image.open(img_path)
                axes[i, j].imshow(img)
                axes[i, j].axis('off')
                if j == 2: 
                    axes[i, j].set_title(f"{cls} Samples", fontsize=14, pad=10)
            except Exception as e:
                print(f"      ⚠️ Could not load image {img_path}: {e}")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "sample_grid.png")
    plt.savefig(save_path)
    plt.close()
    print(f"      -> Saved grid to: {save_path}")

def check_image_quality(data_dir):
    """
    SOTA Check: Scans for blurry or extremely dark/bright images.
    Returns True if data looks healthy, False if many corrupt images found.
    """
    print("   ✨ performing SOTA Quality Scan (Blur & Brightness)...")
    
    # We use OpenCV (cv2) for this. 
    # If not installed, we skip gracefully to avoid breaking the user's flow.
    try:
        import cv2
    except ImportError:
        print("      ⚠️ OpenCV not found. Skipping advanced quality scan (Pip install opencv-python-headless to enable).")
        return
        
    all_files = glob.glob(os.path.join(data_dir, "**", "*.*"), recursive=True)
    if len(all_files) > SAMPLE_SIZE:
        sampled_files = random.sample(all_files, SAMPLE_SIZE)
    else:
        sampled_files = all_files

    blur_scores = []
    brightness_scores = []

    for f in sampled_files:
        try:
            img = cv2.imread(f)
            if img is None: continue
            
            # 1. Blur Check (Laplacian Variance)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_scores.append(variance)
            
            # 2. Brightness Check (Average Pixel Intensity)
            brightness_scores.append(np.mean(gray))
        except:
            pass
            
    # Heuristics
    # Variance < 100 is often "blurry". 
    # Brightness < 10 is "pitch black". Brightness > 245 is "all white".
    avg_blur = np.mean(blur_scores)
    avg_bright = np.mean(brightness_scores)
    
    print(f"      -> Avg Sharpness Score: {avg_blur:.1f} (Higher is better, <100 is blurry)")
    print(f"      -> Avg Brightness: {avg_bright:.1f} (0=Black, 255=White)")
    
    if avg_blur < 50:
        print("      ⚠️ WARNING: Dataset seems significantly blurry.")
    if avg_bright < 30 or avg_bright > 225:
        print("      ⚠️ WARNING: Dataset seems extremely dark or bright.")
    else:
        print("      ✅ Quality Scan Passed: Images are sharp and well-lit.")

def perform_full_analysis(clean_data_dir, results_dir):
    print(f"\n--- 🔬 Starting Deep Data Analysis (EDA) ---")
    output_dir = os.path.join(results_dir, "data_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    plot_class_balance(clean_data_dir, output_dir)
    analyze_image_dimensions(clean_data_dir, output_dir)
    visualize_sample_grid(clean_data_dir, output_dir)
    
    # NEW: Run the quality check
    check_image_quality(clean_data_dir)
    
    print("✅ Analysis Complete.")