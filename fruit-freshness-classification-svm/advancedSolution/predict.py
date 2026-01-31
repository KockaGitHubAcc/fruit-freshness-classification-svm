import tensorflow as tf
import numpy as np
import os
import argparse
import sys

# --- CONFIG ---
IMG_SIZE = (224, 224)
CLASSES = ["Fresh", "Rotten"]

def predict_with_tta(model, img_path):
    """
    Test Time Augmentation (TTA):
    Predicts on the image 3 times (original + flips) and averages the result.
    """
    if not os.path.exists(img_path):
        print(f"❌ Error: Image not found at {img_path}")
        return None, None

    # 1. Load and Preprocess
    try:
        img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
        img_tensor = tf.expand_dims(img_array, 0) # (1, 224, 224, 3)
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None, None

    # 2. Create Variations (TTA)
    img_original = img_tensor
    img_flip_lr = tf.image.flip_left_right(img_tensor)
    img_flip_ud = tf.image.flip_up_down(img_tensor)

    # 3. Predict on all 3
    # verbose=0 suppresses progress bar
    pred_1 = model.predict(img_original, verbose=0)
    pred_2 = model.predict(img_flip_lr, verbose=0)
    pred_3 = model.predict(img_flip_ud, verbose=0)

    # 4. Average Results (Ensemble in Time)
    avg_pred = (pred_1 + pred_2 + pred_3) / 3.0
    
    score = avg_pred[0]
    result_index = np.argmax(score)
    result_class = CLASSES[result_index]
    confidence = np.max(score)

    return result_class, confidence

def main():
    # Allow running without arguments if hardcoded
    import sys
    if len(sys.argv) > 1 and sys.argv[1].endswith(('.jpg', '.png', '.jpeg')):
        img_path = sys.argv[1]
    else:
        # DEFAULT PATH: Change this to test a specific file easily
        img_path = "test_image.jpg"

    model_path = "results/checkpoints/best_model.keras"

    print(f"🔄 Loading Model...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    print(f"🧪 Analyzing: {img_path} ...")
    label, conf = predict_with_tta(model, img_path)
    
    if label:
        print("\n" + "="*30)
        print(f"🍎 PREDICTION: {label.upper()}")
        print(f"📊 CONFIDENCE: {conf:.2%}")
        print("="*30 + "\n")

if __name__ == "__main__":
    main()