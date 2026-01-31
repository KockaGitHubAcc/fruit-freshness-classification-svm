import os
import requests
import numpy as np
import joblib
import cv2
import matplotlib
# Force "Headless" mode (No popup windows)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# Import your local pipeline modules
import normalization
import process

# --- CONFIGURATION ---
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

MODEL_PATH = os.path.join(RESULTS_DIR, 'model', 'best_fruit_classifier.pkl')
ENCODER_PATH = os.path.join(RESULTS_DIR, 'model', 'label_encoder.pkl')
SCALER_PATH = os.path.join(RESULTS_DIR, 'model', 'scaler.pkl')

TEMP_IMG_NAME = "temp_internet_download.png" # Use PNG for potential transparency fix
SAVE_FILE_NAME = "internet_result.png"

def download_and_read_image_robust(url, save_path):
    """Downloads image data robustly, handling corruption and transparency, and saves it to a file."""
    print(f"[Network] Downloading image from web...")
    try:
        # 1. Download the raw data into memory
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # 2. Convert raw data to a numpy array (for OpenCV)
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)

        # 3. Decode the array into an OpenCV image object
        img = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            raise ValueError("OpenCV failed to decode the image data.")

        # 4. Handle transparency (if 4 channels)
        if len(img.shape) == 3 and img.shape[2] == 4:
            # Convert BGRA to BGR, dropping the Alpha channel
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # 5. Save the cleaned BGR image to a temporary file
        cv2.imwrite(save_path, img)
        
        print(f"   ✅ Downloaded and cleaned image saved to: {save_path}")
        return img
        
    except Exception as e:
        print(f"   ❌ Download or decoding failed: {e}")
        return None

def predict_external_image(image_source):
    print(f"\n--- 🌍 INTERNET FRUIT DETECTOR ---")

    # 1. Load Brains
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print(f"❌ CRITICAL ERROR: Model files missing in {RESULTS_DIR}")
        print(f"   👉 STEP 1: Delete 'results/model' folder.")
        print(f"   👉 STEP 2: Run 'main.py' to retrain.")
        return

    print(f"[System] Loading AI Brain...")
    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)

    # 2. Get Image (Robust Download & Save)
    img_original = download_and_read_image_robust(image_source, TEMP_IMG_NAME)

    if img_original is None:
        return

    # 3. Pipeline: Process the saved file path
    print("[Pipeline] Processing...")
    normalizer = normalization.Normalizer(target_size=(128, 128))
    
    # Process the image from the file, using the original normalization function
    norm_img = normalizer.process_single(TEMP_IMG_NAME) 
    
    if norm_img is None:
        print("❌ Error: Could not process image.")
        return

    extractor = process.FeatureExtractor()
    features = extractor.extract_features(norm_img).reshape(1, -1)

    # 4. SCALE (Crucial!)
    features = scaler.transform(features)

    # 5. Predict
    probabilities = model.predict_proba(features)[0]
    prediction_index = model.predict(features)[0]
    result_label = le.inverse_transform([prediction_index])[0]
    confidence = probabilities[prediction_index] * 100

    # 6. Results
    print(f"\n🍎 VERDICT: {result_label.upper()}")
    print(f"📊 Confidence: {confidence:.2f}%")
    
    # 7. Visualization (MERGED DEBUG + RESULT)
    
    # Generate the diagnostic plot from process.py
    diagnostic_plot_bgr = extractor.create_histogram_plot(norm_img, title_prefix="AI Input (Normalized)")
    
    # Create the final result image with the diagnostic plot
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2]) # 1/3 for original, 2/3 for diagnosis

    # Top Panel: Original Image and Prediction
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    
    title_color = 'green' if result_label in ['freshapples', 'freshbanana', 'freshoranges'] else 'red'
    ax1.set_title(f"Original Image | Prediction: {result_label.upper()} ({confidence:.1f}%)", 
                  fontsize=16, color=title_color)
    ax1.axis('off')

    # Bottom Panel: Diagnostic Mask
    ax2 = fig.add_subplot(gs[1, 0])
    # The diagnostic plot is already a BGR image, we need to show it:
    ax2.imshow(cv2.cvtColor(diagnostic_plot_bgr, cv2.COLOR_BGR2RGB))
    ax2.axis('off')

    save_path = os.path.join(RESULTS_DIR, SAVE_FILE_NAME)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() # Close memory

    print(f"\n🖼️  Result and Diagnostic saved to: {save_path}")
    print(f"    (Inspect the bottom of the image to see the mask!)")

    # Cleanup
    if os.path.exists(TEMP_IMG_NAME):
        os.remove(TEMP_IMG_NAME)


if __name__ == "__main__":
    # Test with the problematic dark image
    url = "https://media.istockphoto.com/id/1208618501/photo/one-rotten-and-uneatable-apple.jpg?s=612x612&w=0&k=20&c=uqcqPJV5Enz62Y6tx9eVpJUYqifCR5h_pzj48H7MB7E="
    
    predict_external_image(url)