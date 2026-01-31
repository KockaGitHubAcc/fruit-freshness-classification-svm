import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIGURATION ---
DATA_DIR = "data/Standardized_Dataset"
MODEL_PATH = "results/checkpoints/best_model.keras"
RESULTS_DIR = "results/evaluation"
IMG_SIZE = (224, 224)
BATCH_SIZE = 256 

def evaluate_performance():
    print(f"🚀 Starting FIXED Model Evaluation...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        return

    print(f"🔄 Loading Model: {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"🔹 Loading Validation Data (Correctly Mixed)...")
    
    # CRITICAL FIX: We MUST use shuffle=True + seed=42 to match the Training Split.
    # This ensures we get a mix of Fresh and Rotten images, not just the last folder.
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,             # Must match the seed used in training!
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True         # <--- Changed to True to fix the "0.00 Recall" bug
    )

    print("📥 Extracting Images & Labels to RAM (ensuring alignment)...")
    # We extract the data to numpy arrays. This is the only way to guarantee
    # that y_true and y_pred align perfectly when shuffle=True.
    val_images = []
    val_labels = []
    
    for images, labels in val_ds:
        val_images.append(images.numpy())
        val_labels.append(labels.numpy())
        
    # Concatenate batches into one giant array
    X_val = np.concatenate(val_images)
    y_val_true = np.concatenate(val_labels)
    
    # Convert One-Hot to Index (0 or 1)
    y_true_indices = np.argmax(y_val_true, axis=1)

    print(f"🧪 Running Predictions on {len(X_val)} images...")
    y_pred_probs = model.predict(X_val, verbose=1)
    y_pred_indices = np.argmax(y_pred_probs, axis=1)
    
    class_names = val_ds.class_names # Should be ['Fresh', 'Rotten']

    print("\n--- 📊 CLASSIFICATION REPORT ---")
    print(classification_report(y_true_indices, y_pred_indices, target_names=class_names))

    print("\n--- 🧩 CONFUSION MATRIX ---")
    cm = confusion_matrix(y_true_indices, y_pred_indices)
    print(cm)
    
    # Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Validation Set)')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    print(f"✅ Matrix saved to {RESULTS_DIR}")

if __name__ == "__main__":
    evaluate_performance()