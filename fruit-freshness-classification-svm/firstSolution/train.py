import os
import numpy as np
import time
import joblib

# Plotting (Headless mode for saving files)
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# Machine Learning Imports
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay,
    precision_recall_fscore_support
)

# Import our local modules
import normalization
import process

# --- 🔒 REPRODUCIBILITY SETUP ---
SEED = 42
np.random.seed(SEED)

def load_dataset(data_dir, subset):
    """
    Loads images, normalizes them, and extracts the new extended features.
    """
    print(f"   Loading {subset} data...")
    path = os.path.join(data_dir, subset)
    
    data = []
    labels = []
    
    normalizer = normalization.Normalizer(target_size=(128, 128))
    extractor = process.FeatureExtractor()
    
    if not os.path.exists(path):
        print(f"❌ Error: Path not found {path}")
        return [], []

    for label in os.listdir(path):
        label_dir = os.path.join(path, label)
        if not os.path.isdir(label_dir):
            continue
            
        for file in os.listdir(label_dir):
            if not file.endswith(('.jpg', '.png', '.jpeg')):
                continue
                
            image_path = os.path.join(label_dir, file)
            
            # 1. Normalize
            norm_img = normalizer.process_single(image_path)
            if norm_img is None: continue
            
            # 2. Extract Features (Color + Shape + Texture)
            features = extractor.extract_features(norm_img)
            
            # Safety Check: If an image is broken and returns NaNs, skip it
            if np.any(np.isnan(features)):
                print(f"⚠️ Warning: Skipping {file} (Found NaN features)")
                continue

            data.append(features)
            labels.append(label)
            
    return np.array(data), np.array(labels)

def train_and_evaluate(data_path, root_results_dir):
    print(f"\n--- 🧠 Phase 4: Model Training (With Scaling & Shape Features) ---")
    
    model_dir = os.path.join(root_results_dir, 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    log_file_path = os.path.join(model_dir, "tuning_results.txt")

    # 1. Load Data
    print("[System] Reading dataset...")
    X_train_full, y_train_full = load_dataset(data_path, 'train')
    X_test, y_test = load_dataset(data_path, 'test')
    
    if len(X_train_full) == 0:
        print("❌ Error: No training data found.")
        return

    # 2. Split Data (80% Train, 20% Eval)
    print(f"[System] Splitting Data...")
    X_train, X_eval, y_train, y_eval = train_test_split(
        X_train_full, y_train_full, 
        test_size=0.20, 
        random_state=SEED, 
        stratify=y_train_full
    )

    # 3. SCALE THE DATA (CRITICAL STEP!)
    # We fit the scaler on TRAIN data, then apply it to EVAL and TEST.
    print(f"[System] Applying StandardScaler...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_eval = scaler.transform(X_eval)
    X_test = scaler.transform(X_test)

    # 4. Encode Labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_eval_enc = le.transform(y_eval)
    y_test_enc = le.transform(y_test)

    # 5. Hyperparameter Grid
    param_grid = {
        'kernel': ['linear', 'rbf'],
        'C': [1, 10, 100],
        'gamma': ['scale']
    }
    
    print(f"\n[Tuning] Starting Grid Search on {len(ParameterGrid(param_grid))} combinations...")
    
    best_score = 0
    best_model = None
    best_params = None
    
    with open(log_file_path, "w") as f:
        f.write(f"HYPERPARAMETER TUNING LOG (Seed={SEED})\n")
        f.write("=========================================\n\n")

        for params in ParameterGrid(param_grid):
            print(f"   Testing: {params} ...", end=" ")
            
            model = SVC(**params, probability=True, class_weight='balanced', random_state=SEED)
            model.fit(X_train, y_train_enc)
            
            predictions = model.predict(X_eval)
            score = accuracy_score(y_eval_enc, predictions)
            
            print(f"Accuracy: {score*100:.2f}%")
            
            f.write(f"Parameters: {params}\n")
            f.write(f"Accuracy (Eval Set): {score*100:.2f}%\n")
            
            if score > best_score:
                best_score = score
                best_model = model
                best_params = params
        
        f.write(f"\nWINNER: {best_params}\n")
        f.write(f"BEST ACCURACY: {best_score*100:.2f}%\n")

    print(f"\n✅ Tuning Complete. Best Validation Accuracy: {best_score*100:.2f}%")
    print(f"   Best Parameters: {best_params}")

    # 6. Final Test
    print("\n[Evaluation] 🔓 Unlocking Test Set for Final Validation...")
    final_predictions = best_model.predict(X_test)
    
    acc = accuracy_score(y_test_enc, final_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_enc, final_predictions, average='weighted'
    )
    
    print(f"\n🏆 FINAL TEST RESULTS")
    print(f"=====================================")
    print(f"   Accuracy:  {acc * 100:.2f}%")
    print(f"   Precision: {precision * 100:.2f}%")
    print(f"   Recall:    {recall * 100:.2f}%")
    print(f"   F1-Score:  {f1 * 100:.2f}%")
    print(f"=====================================")

    # 7. Confusion Matrix
    cm = confusion_matrix(y_test_enc, final_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)
    plt.title(f"Confusion Matrix (Acc: {acc*100:.1f}%)")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "confusion_matrix.png"))
    plt.close()

    # 8. SAVE EVERYTHING (Model + Encoder + SCALER)
    joblib.dump(best_model, os.path.join(model_dir, "best_fruit_classifier.pkl"))
    joblib.dump(le, os.path.join(model_dir, "label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    
    print(f"\n💾 System saved (Model, Encoder, and Scaler) to: {model_dir}")