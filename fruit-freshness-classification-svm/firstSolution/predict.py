import os
import random
import numpy as np
import joblib
import cv2
import csv
import matplotlib
try:
    matplotlib.use('TkAgg') # Try interactive mode first
except:
    matplotlib.use('Agg')   # Fallback to headless
import matplotlib.pyplot as plt

# Import local modules
import normalization
import process

# --- CONFIGURATION ---
# We calculate paths dynamically so it works on any computer
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# If predict.py is inside firstSolution, we usually look in the current dir
# But based on your structure, let's keep it robust:
PROJECT_ROOT = CURRENT_SCRIPT_DIR 

DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_DIR = os.path.join(RESULTS_DIR, "model")
PREDICT_DIR = os.path.join(RESULTS_DIR, "predict")
NUM_SAMPLES = 10  # How many random images to test

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_all_test_images(data_dir):
    """
    Crawls the test directory and lists all images with their actual labels.
    """
    test_path = os.path.join(data_dir, "test")
    image_list = []
    
    if not os.path.exists(test_path):
        # Fallback: check if we need to go up one level
        test_path = os.path.join(os.path.dirname(PROJECT_ROOT), "data", "dataset", "test")
        if not os.path.exists(test_path):
            # Fallback 2: Check current dir
            test_path = os.path.join("dataset", "test")

    if not os.path.exists(test_path):
        print(f"❌ Error: Test path not found. Checked: {test_path}")
        return []

    for label in os.listdir(test_path):
        label_dir = os.path.join(test_path, label)
        if not os.path.isdir(label_dir):
            continue
            
        for file in os.listdir(label_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(label_dir, file)
                image_list.append((full_path, label))
                
    return image_list

def run_random_predictions():
    print("--- 🔮 Phase 5: Random Inference Test (With Scaler) ---")
    
    # 1. Setup
    ensure_dir(PREDICT_DIR)
    
    # 2. Load Model, Encoder AND SCALER
    try:
        model_path = os.path.join(MODEL_DIR, "best_fruit_classifier.pkl")
        le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl") # <--- CRITICAL NEW FILE
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
             print(f"❌ Error: Files missing in {MODEL_DIR}")
             print("   Did you run Phase 4 (train.py)?")
             return

        model = joblib.load(model_path)
        le = joblib.load(le_path)
        scaler = joblib.load(scaler_path) # <--- Load the scaler
        
        print("[System] Model, Encoder, and Scaler loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model files: {e}")
        return

    # 3. Get Random Samples
    all_images = get_all_test_images(DATA_DIR)
    
    # Try looking in 'data/dataset' if 'dataset' failed
    if not all_images:
         all_images = get_all_test_images(os.path.join(os.path.dirname(PROJECT_ROOT), "data", "dataset"))

    if not all_images:
        print("⚠️ Warning: No images found to test.")
        return

    if len(all_images) < NUM_SAMPLES:
        selected_samples = all_images
    else:
        selected_samples = random.sample(all_images, NUM_SAMPLES)
        print(f"[System] Selected {NUM_SAMPLES} random images for testing.\n")

    # Tools
    normalizer = normalization.Normalizer(target_size=(128, 128))
    extractor = process.FeatureExtractor()
    
    # Results Log
    results_log = []
    correct_count = 0

    # 4. Loop through samples
    for i, (img_path, actual_label) in enumerate(selected_samples):
        filename = os.path.basename(img_path)
        
        # A. Process
        norm_img = normalizer.process_single(img_path)
        if norm_img is None:
            continue
            
        features = extractor.extract_features(norm_img)
        features = features.reshape(1, -1) 
        
        # B. SCALE (Crucial Step!)
        # We must apply the exact same math we used during training
        features = scaler.transform(features) 
        
        # C. Predict
        pred_idx = model.predict(features)[0]
        predicted_label = le.inverse_transform([pred_idx])[0]
        confidence = model.predict_proba(features)[0][pred_idx] * 100
        
        # D. Check Status
        is_correct = (predicted_label == actual_label)
        status_icon = "✅" if is_correct else "❌"
        if is_correct: correct_count += 1
        
        print(f"Sample {i+1}: {filename}")
        print(f"   Actual: {actual_label} | Predicted: {predicted_label} ({confidence:.1f}%) {status_icon}")

        # E. Save Result Image
        plt.figure(figsize=(5, 5))
        plt.imshow(cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB))
        
        title_color = 'green' if is_correct else 'red'
        plt.title(f"Act: {actual_label}\nPred: {predicted_label} ({confidence:.1f}%)", 
                  color=title_color, fontsize=12, fontweight='bold')
        plt.axis('off')
        
        save_name = f"result_{i+1}_{'CORRECT' if is_correct else 'WRONG'}.png"
        save_path = os.path.join(PREDICT_DIR, save_name)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        # F. Add to log
        results_log.append([filename, actual_label, predicted_label, f"{confidence:.2f}%", "Correct" if is_correct else "Wrong"])

    # 5. Save CSV Report
    csv_path = os.path.join(PREDICT_DIR, "prediction_report.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Actual Label", "Predicted Label", "Confidence", "Status"])
        writer.writerows(results_log)

    print(f"\n--- 📝 Summary ---")
    print(f"Total Tested: {len(selected_samples)}")
    print(f"Accuracy on Random Batch: {correct_count}/{len(selected_samples)}")
    print(f"📂 Visual results saved to: {PREDICT_DIR}")

if __name__ == "__main__":
    run_random_predictions()