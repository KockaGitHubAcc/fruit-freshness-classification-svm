import os
import cv2
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import normalization

class FeatureExtractor:
    def __init__(self):
        pass

    def extract_features(self, image):
        """
        Extracts features using Otsu's Adaptive Mask and uses a robust Area feature
        instead of fragile Hu Moments.
        """
        # --- 1. GENERATE SMART MASK (ADAPTIVE) ---
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        
        # Apply Otsu's method on Saturation to find the optimal threshold for THIS specific image.
        _, smart_mask = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # --- Clean up the mask slightly (Robust Morphology) ---
        kernel = np.ones((5, 5), np.uint8) 
        smart_mask = cv2.morphologyEx(smart_mask, cv2.MORPH_CLOSE, kernel)
        smart_mask = cv2.morphologyEx(smart_mask, cv2.MORPH_OPEN, kernel)
        
        # --- 2. COLOR (Masked Histogram) ---
        hist = cv2.calcHist([hsv], [0, 1, 2], smart_mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        hist_features = hist.flatten()

        # --- 3. SHAPE (CRITICAL CHANGE: Area and First Hu Moment) ---
        moments = cv2.moments(smart_mask)
        hu_moments = cv2.HuMoments(moments).flatten()

        # Use Area (m00) as the primary stable shape feature
        area = moments['m00']
        if area < 1e-6:
            area = 1e-6 
            
        # Use only the Area and the log-transformed first Hu Moment (I1)
        shape_features = np.array([
            area,
            -np.sign(hu_moments[0]) * np.log10(np.abs(hu_moments[0]) + 1e-10)
        ])
        
        # --- 4. TEXTURE ---
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_val, std_dev = cv2.meanStdDev(gray, mask=smart_mask)
        texture_features = np.array([mean_val[0][0], std_dev[0][0]])

        # --- COMBINE ALL ---
        # FINAL VECTOR: [Color_512, Shape_2, Tex_2]
        global_features = np.hstack([hist_features, shape_features, texture_features])
        
        return global_features

    def create_histogram_plot(self, image, title_prefix=""):
        # Visualization logic (ensures the debug plot also uses Otsu's method)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 10))
        
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title(f"{title_prefix} - Input")
        ax1.axis('off')

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        
        _, smart_mask = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8) 
        smart_mask = cv2.morphologyEx(smart_mask, cv2.MORPH_CLOSE, kernel)
        smart_mask = cv2.morphologyEx(smart_mask, cv2.MORPH_OPEN, kernel)

        ax2.set_title("Color Dist (Otsu-Masked)")
        colors = ('orange', 'purple', 'cyan')
        for i, (color) in enumerate(colors):
            hist = cv2.calcHist([hsv], [i], smart_mask, [256], [0, 256])
            ax2.plot(hist, color=color)
        ax2.set_xlim([0, 256])

        ax3.imshow(smart_mask, cmap='gray')
        ax3.set_title("Otsu Mask (Adaptive to Light)")
        ax3.axis('off')

        plt.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        plot_img = buf.reshape((h, w, 4))
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
        plt.close(fig)
        return plot_img

def generate_report(stats, root_results_dir):
    print("\n--- 🧠 Phase 3: Feature Extraction Report (Area-Based Build) ---")
    
    module_dir = os.path.join(root_results_dir, 'process')
    if not os.path.exists(module_dir):
        os.makedirs(module_dir)

    normalizer = normalization.Normalizer(target_size=(128, 128))
    extractor = FeatureExtractor()
    
    sample_path = stats['examples'].get('original')
    if not sample_path:
        sample_path = next((v for v in stats['examples'].values() if isinstance(v, str)), None)
    
    if not sample_path: return

    class_dir = os.path.dirname(sample_path) 
    train_dir = os.path.dirname(class_dir) 
    
    if os.path.exists(train_dir):
        classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        
        for cls_name in classes:
            cls_path = os.path.join(train_dir, cls_name)
            images = [f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.png'))]
            
            if not images: continue
            
            img_name = random.choice(images)
            full_path = os.path.join(cls_path, img_name)
            
            norm_img = normalizer.process_single(full_path)
            if norm_img is None: continue
            
            report_img = extractor.create_histogram_plot(norm_img, title_prefix=cls_name)
            
            save_name = f"analysis_{cls_name}.jpg"
            save_path = os.path.join(module_dir, save_name)
            cv2.imwrite(save_path, report_img)
            print(f"   ✅ Processed: {cls_name:<15} -> Saved: {save_name}")

    print("[Process] Report generation complete.")