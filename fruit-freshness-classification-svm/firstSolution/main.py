import sys
import os
import analyze 
import normalization 
import process
import train
import predict  # <--- NEW IMPORT for Phase 5

# --- PATH CONFIGURATION ---
# We calculate paths dynamically so it works on any computer
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'dataset')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

def setup_directories():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

def run():
    setup_directories()
    print("==========================================")
    print("🍏 ITC6109A1 Fruit Classification System 🍌")
    print("==========================================\n")
    
    # --- PHASE 1: ANALYSIS ---
    # Checks how many images you have and if the classes are balanced
    stats = analyze.get_dataset_stats(DATA_PATH)
    analyze.save_results(stats, RESULTS_DIR)
    
    if not stats or stats['total_files'] == 0:
        print("❌ Critical Error: No data found. Stopping.")
        return

    # --- PHASE 2: NORMALIZATION ---
    # Generates a sample image to show how the "robot eye" sees the fruit (resized/blurred)
    normalization.generate_report(stats, RESULTS_DIR)

    # --- PHASE 3: PROCESSING ---
    # Generates histograms to show the color distribution (the "features")
    process.generate_report(stats, RESULTS_DIR)

    # --- PHASE 4: TRAINING ---
    # The heavy lifting: Trains the SVM, validates it with the seed, and saves the model
    train.train_and_evaluate(DATA_PATH, RESULTS_DIR)

    # --- PHASE 5: PREDICTION (The Fun Part) ---
    # Picks 10 random images from the test set and classifies them using the saved model
    # Note: We temporarily update predict's paths to match main's configuration
    print("\n[System] Initializing Inference Engine...")
    
    # We override the paths in predict.py to ensure they match exactly what main.py sees
    predict.DATA_DIR = DATA_PATH
    predict.RESULTS_DIR = RESULTS_DIR
    predict.MODEL_DIR = os.path.join(RESULTS_DIR, 'model')
    predict.PREDICT_DIR = os.path.join(RESULTS_DIR, 'predict')
    
    predict.run_random_predictions()
    
    print("\n==========================================")
    print("✅ SYSTEM EXECUTION COMPLETE")
    print("==========================================")

if __name__ == "__main__":
    run()