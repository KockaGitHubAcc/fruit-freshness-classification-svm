import numpy as np
import matplotlib.pyplot as plt

from ImageLoading import load_dataset
from FeatureExtraction import extract_hsv_histograms

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from flaml import AutoML
import shap



# ============================================================
# LOAD DATA
# ============================================================
ROOT_DIR = "/Users/Dimitrije/Documents/MachineVisionProject/Images"

X_rgb, X_gray, y, fruits = load_dataset(ROOT_DIR)
X = extract_hsv_histograms(X_rgb)



# ============================================================
# PCA SETTINGS
# ============================================================
USE_PCA = True
PCA_COMPONENTS = 50



# ============================================================
# TRAIN / TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test, fruits_train, fruits_test = train_test_split(
    X, y, fruits, test_size=0.20, random_state=42, stratify=y
)



# ============================================================
# SCALING
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# ============================================================
# PCA (OPTIONAL)
# ============================================================
if USE_PCA:
    pca = PCA(n_components=PCA_COMPONENTS)
    X_train_scaled = pca.fit_transform(X_train_scaled)
    X_test_scaled = pca.transform(X_test_scaled)
    print(f"\nPCA enabled → reduced features to {PCA_COMPONENTS}\n")



# ============================================================
# MODEL EVALUATION FUNCTION
# ============================================================
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    print(f"\n=== {name.upper()} RESULTS ===")

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    print(f"Accuracy: {acc}")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))

    return pred, acc, model



# ============================================================
# ROC CURVE FUNCTION
# ============================================================
def plot_roc(name, model, X_test, y_test):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        try:
            probs = model.decision_function(X_test)
        except:
            print(f"Cannot compute ROC for {name}")
            return

    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")



# ============================================================
# PER-FRUIT CONFUSION MATRIX
# ============================================================
def per_fruit_confusion(name, pred, y_test, fruits_test):
    print(f"\n=== PER-FRUIT ERROR ANALYSIS ({name}) ===")

    unique_fruits = np.unique(fruits_test)
    for fruit in unique_fruits:
        idx = np.where(fruits_test == fruit)[0]
        if len(idx) < 5:
            continue

        yt = y_test[idx]
        yp = pred[idx]
        cm = confusion_matrix(yt, yp)

        print(f"\n---- {fruit.upper()} ----")
        print(cm)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")



# ============================================================
# GRID SEARCH DEFINITIONS
# ============================================================
param_svm = {
    "C": [1, 10, 50],
    "gamma": ["scale", 0.01, 0.001],
    "kernel": ["rbf"]
}

param_knn = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"]
}

param_rf = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}



# ============================================================
# GRID SEARCH TRAINING
# ============================================================
print("\n=== GridSearchCV: SVM ===")
svm_gs = GridSearchCV(SVC(probability=True), param_svm, cv=3, n_jobs=-1)
svm_pred, svm_acc, svm_model = evaluate_model(
    "SVM (GridSearch)", svm_gs,
    X_train_scaled, y_train, X_test_scaled, y_test
)
print("Best SVM params:", svm_gs.best_params_)



print("\n=== GridSearchCV: KNN ===")
knn_gs = GridSearchCV(KNeighborsClassifier(), param_knn, cv=3, n_jobs=-1)
knn_pred, knn_acc, knn_model = evaluate_model(
    "KNN (GridSearch)", knn_gs,
    X_train_scaled, y_train, X_test_scaled, y_test
)
print("Best KNN params:", knn_gs.best_params_)



print("\n=== GridSearchCV: Random Forest ===")
rf_gs = GridSearchCV(RandomForestClassifier(), param_rf, cv=3, n_jobs=-1)
rf_pred, rf_acc, rf_model = evaluate_model(
    "Random Forest (GridSearch)", rf_gs,
    X_train, y_train, X_test, y_test
)
print("Best RF params:", rf_gs.best_params_)



# ============================================================
# FLAML AutoML
# ============================================================
print("\n=== RUNNING FLAML ===")
automl = AutoML()
automl.fit(
    X_train, y_train,
    task="classification",
    time_budget=120,
    metric="accuracy"
)

flaml_pred = automl.predict(X_test)
flaml_acc = accuracy_score(y_test, flaml_pred)

print("\n=== FLAML RESULTS ===")
print("Accuracy:", flaml_acc)
print(confusion_matrix(y_test, flaml_pred))
print(classification_report(y_test, flaml_pred))



# ============================================================
# ROC CURVES
# ============================================================
plt.figure(figsize=(8, 6))
plot_roc("SVM", svm_model, X_test_scaled, y_test)
plot_roc("KNN", knn_model, X_test_scaled, y_test)
plot_roc("Random Forest", rf_model, X_test, y_test)
plot_roc("FLAML", automl, X_test, y_test)

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()



# ============================================================
# PER-FRUIT PERFORMANCE
# ============================================================
print("\n\n===== PER-FRUIT PERFORMANCE =====")
per_fruit_confusion("SVM", svm_pred, y_test, fruits_test)
per_fruit_confusion("KNN", knn_pred, y_test, fruits_test)
per_fruit_confusion("Random Forest", rf_pred, y_test, fruits_test)
per_fruit_confusion("FLAML", flaml_pred, y_test, fruits_test)



# ============================================================
# METRIC COMPARISON TABLE
# ============================================================
def avg_scores(y_true, y_pred):
    return (
        precision_score(y_true, y_pred, average="binary"),
        recall_score(y_true, y_pred, average="binary"),
        f1_score(y_true, y_pred, average="binary")
    )

svm_prec, svm_rec, svm_f1 = avg_scores(y_test, svm_pred)
knn_prec, knn_rec, knn_f1 = avg_scores(y_test, knn_pred)
rf_prec, rf_rec, rf_f1   = avg_scores(y_test, rf_pred)
fl_prec, fl_rec, fl_f1   = avg_scores(y_test, flaml_pred)

print("\n\n=== DETAILED MODEL METRIC COMPARISON ===")
print(f"{'MODEL':15}  {'ACC':6} {'PREC':6} {'RECALL':6} {'F1':6}")
print("-" * 50)
print(f"{'SVM':15}  {svm_acc:.3f} {svm_prec:.3f} {svm_rec:.3f} {svm_f1:.3f}")
print(f"{'KNN':15}  {knn_acc:.3f} {knn_prec:.3f} {knn_rec:.3f} {knn_f1:.3f}")
print(f"{'RandomForest':15}  {rf_acc:.3f} {rf_prec:.3f} {rf_rec:.3f} {rf_f1:.3f}")
print(f"{'FLAML':15}  {flaml_acc:.3f} {fl_prec:.3f} {fl_rec:.3f} {fl_f1:.3f}")



# ============================================================
# SHAP MODEL EXPLANATION (FIXED FINAL VERSION)
# ============================================================
import shap
print("\n\n=== SHAP MODEL EXPLANATIONS ===")

# -----------------------------
# SHAP for Random Forest (GS)
# -----------------------------
print("\n--- SHAP for Random Forest (GS) ---")

# RF WAS TRAINED ON RAW HISTOGRAM FEATURES → use X_test (NOT PCA!)
X_test_rf = X_test

try:
    explainer_rf = shap.TreeExplainer(rf_model.best_estimator_)
    shap_values_rf = explainer_rf.shap_values(X_test_rf)

    shap.summary_plot(
        shap_values_rf[1],  # class 1 (rotten)
        X_test_rf,
        show=False
    )
    plt.title("SHAP Summary Plot - Random Forest (Class: Rotten)")
    plt.show()

except Exception as e:
    print("SHAP RF error:", e)


# -----------------------------
# SHAP for FLAML Best Model
# -----------------------------
print("\n--- SHAP for FLAML Best Model ---")

best_model = automl.model
is_tree_model = (
    "xgboost" in str(type(best_model)).lower()
    or "lightgbm" in str(type(best_model)).lower()
    or "forest" in str(type(best_model)).lower()
)

if not is_tree_model:
    print("FLAML best model is NOT tree-based → SHAP not available.")
else:
    try:
        explainer_fl = shap.TreeExplainer(best_model)
        shap_values_fl = explainer_fl.shap_values(X_test)

        shap.summary_plot(
            shap_values_fl[1],
            X_test,
            show=False
        )
        plt.title("SHAP Summary Plot — FLAML Best Model")
        plt.show()

    except Exception as e:
        print("SHAP FLAML error:", e)
