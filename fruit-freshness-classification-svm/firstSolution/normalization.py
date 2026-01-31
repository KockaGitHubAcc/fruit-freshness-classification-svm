# src/normalization.py
import cv2
import numpy as np
import os

class Normalizer:
    def __init__(self, target_size=(128, 128)):
        """
        :param target_size: Tuple (width, height). Target geometry for all images.
        """
        self.target_size = target_size

    def resize_with_padding(self, image):
        """
        Resizes an image to target_size maintaining aspect ratio 
        by adding black borders (padding) where necessary.
        """
        h, w = image.shape[:2]
        target_w, target_h = self.target_size

        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize the image
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create canvas and center image
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image

        return canvas

    def process_single(self, image_path):
        """
        Loads and standardizes a single image geometry.
        No filters are applied.
        """
        img = cv2.imread(image_path)
        if img is None:
            return None

        # Only Resize + Pad
        img_normalized = self.resize_with_padding(img)

        return img_normalized

def generate_report(stats, root_results_dir):
    """
    Generates a report showing how images are normalized (Geometry check).
    """
    if not stats or 'examples' not in stats:
        print("❌ No stats provided for normalization report.")
        return

    print("\n--- 📏 Generating Normalization Report (Geometry Only) ---")
    
    # 1. Setup Module Directory
    module_dir = os.path.join(root_results_dir, 'normalization')
    if not os.path.exists(module_dir):
        os.makedirs(module_dir)
    print(f"[Normalization] Saving geometry comparisons to: {module_dir}")

    normalizer = Normalizer(target_size=(128, 128))
    
    examples = stats['examples']
    
    for category, data in examples.items():
        image_path = None
        if isinstance(data, dict) and 'path' in data:
            image_path = data['path']
        elif isinstance(data, str):
            image_path = data
            
        if not image_path:
            continue

        # Normalize (Resize/Pad only)
        norm_img = normalizer.process_single(image_path)
        original_img = cv2.imread(image_path)
        
        if norm_img is not None and original_img is not None:
            # Create Display Comparison
            h_final, w_final = norm_img.shape[:2]
            scale_disp = h_final / original_img.shape[0]
            w_disp = int(original_img.shape[1] * scale_disp)
            
            orig_display = cv2.resize(original_img, (w_disp, h_final))
            combined = np.hstack((orig_display, norm_img))
            
            save_name = f"geometry_{category}.jpg"
            save_path = os.path.join(module_dir, save_name)
            cv2.imwrite(save_path, combined)
            
            print(f"   ✅ Normalized: {category:<15} -> Saved: {save_name}")

    print("[Normalization] Report generation complete.")