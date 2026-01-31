import os
import sys
import random
import numpy as np

# --- 1. CRITICAL STABILITY FIXES (Must be first) ---
# Disables XLA to prevent "Illegal Address" errors on A100s
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Reduces log clutter

import tensorflow as tf
from dotenv import load_dotenv

# Import our custom modules
import data_handler 
import data_cleaner
import data_analyzer      # <--- NEW: Deep EDA
import data_preprocessing # <--- NEW: tf.data + One-Hot
import model_builder      # <--- NEW: GPU Augmentation + Dropout
import trainer            # <--- NEW: Cosine Decay + Categorical Loss

# --- 2. CONFIGURATION ---
DATASET_NAME = "Fresh_Rotten_Fruits_Dataset"
DATA_FOLDER = "data"                         
RESULTS_DIR = "results" 
ENV_FILE = ".env"
SEED = 42

def set_global_seed(seed_value):
    """Ensures reproducibility across CPU and GPU."""
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

def setup_project_directories():
    """Creates the necessary folder structure."""
    os.makedirs(DATA_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Create subfolders for organization
    for folder in ["logs", "checkpoints", "predictions", "data_analysis"]:
        os.makedirs(os.path.join(RESULTS_DIR, folder), exist_ok=True)

def main():
    print("🚀 Starting STABLE SOTA Training (ConvNeXt Large + Dual A100s)...")
    
    # 1. Setup
    set_global_seed(SEED)
    if os.path.exists(ENV_FILE): load_dotenv(ENV_FILE)
    setup_project_directories()

    # 2. Data Acquisition
    # Downloads dataset if not found
    raw_data_dir = data_handler.setup_data_acquisition(DATASET_NAME, DATA_FOLDER)
    
    if raw_data_dir:
        # 3. Data Cleaning
        # Standardizes folder structure (train/validation)
        clean_data_dir = data_cleaner.standardize_dataset(raw_data_dir, DATA_FOLDER)
        
        # 4. Deep Data Analysis (EDA)
        # Checks for image sizes, blurriness, and class balance
        data_analyzer.perform_full_analysis(clean_data_dir, RESULTS_DIR)
        
        # 5. Data Pipeline (tf.data)
        # Loads images, resizes to 224x224, applies One-Hot Encoding
        train_ds, val_ds, test_ds = data_preprocessing.create_data_generators(clean_data_dir)
        
        # 6. Hardware Strategy
        # Detects Dual A100s
        strategy = model_builder.get_strategy()
        
        # 7. Build Model
        # Instantiates ConvNeXt Large with GPU Augmentation & Dropout
        with strategy.scope():
            model = model_builder.build_sota_model()
            
        # 8. Train
        # Runs Phase 1 (Warmup) and Phase 2 (Fine-Tuning with Cosine Decay)
        trainer.train_sota_model(strategy, model, train_ds, val_ds, RESULTS_DIR)
        
    else:
        print("❌ Pipeline failed: Data could not be acquired.")

if __name__ == '__main__':
    main()