import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.losses import CategoricalCrossentropy

# --- CONSTANTS ---
EPOCHS_PHASE_1 = 2   # Warmup (train head only)
EPOCHS_PHASE_2 = 15  # Fine-tuning (train deep layers)

def train_sota_model(strategy, model, train_gen, val_gen, results_dir):
    """
    Executes a 2-Phase SOTA Training Loop.
    Features: Label Smoothing, AdamW, Cosine Decay, and Robust Unfreezing.
    """
    
    # Directories
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    log_dir = os.path.join(results_dir, "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)

    with strategy.scope():
        # --- 🔒 PHASE 1: WARMUP (Frozen Base) ---
        print("\n--- 🔒 PHASE 1: Warmup (Frozen Base) ---")
        
        # Optimizer: AdamW (Standard for modern ConvNets)
        optimizer_p1 = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
        
        # LOSS FUNCTION: SOTA UPGRADE
        # Label Smoothing=0.1 prevents the model from becoming overconfident (Overfitting)
        loss_fn = CategoricalCrossentropy(label_smoothing=0.1)
        
        model.compile(
            optimizer=optimizer_p1,
            loss=loss_fn, 
            metrics=['accuracy', 'auc']
        )

        callbacks_p1 = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
        ]

        history_warmup = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS_PHASE_1,
            callbacks=callbacks_p1,
            verbose=1
        )

        # --- 🔓 PHASE 2: FINE-TUNING (Unfrozen) ---
        print("\n--- 🔓 PHASE 2: Fine-Tuning (Unfreezing Top Layers) ---")
        
        # 1. ROBUST LAYER SEARCH
        # We find the ConvNeXt layer dynamically by name. 
        # This prevents crashing if augmentation layers shift indices.
        base_model = None
        for layer in model.layers:
            # Check for 'convnext' or 'model' (generic name)
            if "convnext" in layer.name.lower() or "model" in layer.name.lower():
                # Ensure it's not the input or augmentation layer
                if len(layer.trainable_weights) > 0 or hasattr(layer, 'layers'):
                    base_model = layer
                    break
        
        # Fallback safety
        if base_model is None:
            print("⚠️ WARNING: Could not find base model by name. Unfreezing layer[-5].")
            base_model = model.layers[-5]
        else:
            print(f"   -> Found Base Model: '{base_model.name}'. Unfreezing...")

        base_model.trainable = True
        
        # Fine-tune only the top layers
        # We access the *inner* layers of the ConvNeXt model
        if hasattr(base_model, 'layers'):
            fine_tune_at = int(len(base_model.layers) * 0.7)
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
            print(f"   -> Froze first {fine_tune_at} layers of ConvNeXt.")
            
        # 2. Re-compile with LOW Learning Rate (Cosine Decay)
        # Calculates decay steps automatically based on new BATCH_SIZE
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-5, 
            decay_steps=len(train_gen) * EPOCHS_PHASE_2,
            alpha=0.1 
        )
        
        optimizer_p2 = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-5)
        
        model.compile(
            optimizer=optimizer_p2,
            loss=loss_fn, # Keep Label Smoothing
            metrics=['accuracy', 'auc']
        )
        
        # 3. SOTA Callbacks
        callbacks_p2 = [
            ModelCheckpoint(
                os.path.join(checkpoint_dir, "best_model.keras"),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            # Patience 5 gives the model time to adjust to unfreezing
            EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
            TensorBoard(log_dir=log_dir)
        ]

        history_finetune = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS_PHASE_2,
            initial_epoch=len(history_warmup.history['loss']),
            callbacks=callbacks_p2,
            verbose=1
        )
        
    print("✅ Training Complete.")
    return history_finetune