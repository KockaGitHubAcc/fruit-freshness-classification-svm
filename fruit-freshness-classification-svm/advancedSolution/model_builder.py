import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras import mixed_precision

# --- 🚀 A100 OPTIMIZATION: MIXED PRECISION ---
# This drastically speeds up training and reduces VRAM usage
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Configuration
IMG_SIZE = (224, 224)
NUM_CLASSES = 2 # <--- FIXED: You only have 'Fresh' and 'Rotten' folders

def get_strategy():
    """Detects hardware and returns the distribution strategy."""
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Device:', tpu.master())
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("✅ TPU Strategy Activated")
    except ValueError:
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) > 1:
            strategy = tf.distribute.MirroredStrategy()
            print(f"✅ Multi-GPU Strategy Activated: {len(gpus)} GPUs (Active)")
        else:
            strategy = tf.distribute.get_strategy()
            print("✅ Single GPU/CPU Strategy Activated")
    return strategy

def build_sota_model():
    """
    Builds ConvNeXtLarge with Internal GPU Augmentation & SOTA Regularization.
    """
    # 1. Input Layer
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # 2. GPU-Accelerated Data Augmentation
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.25)(x)
    x = layers.RandomZoom(0.2)(x)
    x = layers.RandomContrast(0.2)(x)

    # 3. Base Model (ConvNeXt Large)
    base_model = tf.keras.applications.ConvNeXtLarge(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        pooling=None 
    )

    # Freeze Base Model initially
    base_model.trainable = False

    # Pass inputs through base model
    x = base_model(x, training=False) # Important: keep BatchNormalization in inference mode when frozen

    # 4. SOTA Classification Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x) 
    
    x = layers.Dense(512, activation='gelu', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # Output Layer
    # dtype='float32' is REQUIRED for Mixed Precision output stability
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Stable_ConvNeXt_Large")
    return model