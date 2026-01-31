import tensorflow as tf
import os

# --- ⚙️ CONFIGURATION ---
# UPDATED: 256 is appropriate for 2x A100 (80GB). 
# If you get OOM (Out of Memory), lower to 128.
BATCH_SIZE = 256  
IMG_SIZE = (224, 224)
SEED = 42

def create_data_generators(data_dir):
    """
    Creates high-performance tf.data Datasets (SOTA standard).
    """
    print(f"\n--- ⚙️ Setting up High-Performance Data Pipeline (tf.data) ---")
    print(f"📂 Loading data from: {data_dir}")

    # 1. Training Dataset
    print(f"🔹 Creating Training Set (Batch Size: {BATCH_SIZE})...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True
    )

    # 2. Validation Dataset (Initial 20% split)
    print(f"🔹 Creating Validation & Test Sets (from 20% split)...")
    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False # Important: Don't shuffle yet to ensure stable split
    )

    # 3. Split Validation into Val (10%) and Test (10%)
    val_batches = tf.data.experimental.cardinality(val_test_ds)
    test_ds = val_test_ds.take(val_batches // 2)
    val_ds = val_test_ds.skip(val_batches // 2)

    print(f"   -> Split: Train (80%) | Val (~10%) | Test (~10%)")

    # 4. Performance Optimization
    AUTOTUNE = tf.data.AUTOTUNE
    
    # Cache and Prefetch are critical for keeping the GPU fed
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print("✅ Data Pipeline Ready (One-Hot Encoded, Mixed Precision Compatible)")
    return train_ds, val_ds, test_ds