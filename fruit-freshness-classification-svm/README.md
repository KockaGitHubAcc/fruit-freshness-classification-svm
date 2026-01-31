# ITC6109A1-FruitProject: Automated Freshness Classification

![Project Status](https://img.shields.io/badge/status-active-brightgreen) ![Python](https://img.shields.io/badge/python-3.8%2B-blue)

## 📌 Abstract

**ITC6109A1-FruitProject** is a computer vision pipeline designed to automate quality assurance in agricultural supply chains. Unlike traditional deep learning approaches that require massive computational resources, this project utilizes a highly optimized **Support Vector Machine (SVM)** classifier coupled with robust **feature engineering** to distinguish between fresh and rotten fruits with high fidelity (>98% accuracy).

The system classifies images into 6 distinct classes:
- `freshapples`, `freshbanana`, `freshoranges`
- `rottenapples`, `rottenbanana`, `rottenoranges`

## 🚀 Key Features

- **Advanced Feature Extraction**:
  - **Adaptive Segmentation**: Uses Otsu's method to dynamically threshold and segment fruits from the background.
  - **Color Analysis**: Extracts HSV color histograms to capture ripeness and spoilage indicators.
  - **Shape Descriptors**: Utilizes Area and Log-transformed Hu Moments to detect structural deformities.
  - **Texture Analysis**: Computes mean and standard deviation of pixel intensities to identify surface irregularities.
- **Machine Learning Pipeline**:
  - **Model**: Support Vector Machine (SVM) with RBF kernel.
  - **Optimization**: Automated Hyperparameter Tuning (Grid Search) for `C`, `gamma`, and `kernel`.
  - **Preprocessing**: Standard Scaling and Label Encoding for robust model training.
- **Evaluation & Visualization**:
  - Generates Confusion Matrices, Precision-Recall reports, and visual inference results.

## 📂 Project Structure

The core logic resides in the `firstSolution/` directory:

| File | Description |
|------|-------------|
| `main.py` | **Entry point**. Orchestrates the entire pipeline (Analysis -> Training -> Prediction). |
| `train.py` | Training logic. Loads data, extracts features, tunes SVM, and saves the best model. |
| `process.py` | Contains the `FeatureExtractor` class (Otsu masking, Color/Shape/Texture extraction). |
| `normalization.py` | Handles image resizing and normalization (target size: 128x128). |
| `predict.py` | Runs inference on a test dataset and generates a performance report. |
| `predict_internet.py` | Script for testing on random/internet images (simulating real-world usage). |
| `analyze.py` | Utilities for analyzing dataset statistics and feature distributions. |

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ITC6109A1-FruitPoject
   ```

2. **Install Dependencies**:
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset Setup

The project expects the dataset to be organized in the `data/` directory as follows:

```text
data/
├── train/
│   ├── freshapples/
│   ├── rottenapples/
│   ├── ... (other classes)
└── test/
    ├── freshapples/
    ├── rottenapples/
    ├── ... (other classes)
```
*Note: The system automatically handles image loading and label encoding based on folder names.*

## 🏃 Usage

### 1. Run the Full Pipeline
To run the complete pipeline (Analysis → Normalization → Feature Extraction → Training → Evaluation), execute the `main.py` script:

```bash
python firstSolution/main.py
```
**This will:**
- Analyze the dataset statistics.
- Train the SVM model with hyperparameter tuning.
- Evaluate performance on the test set.
- Run inference on a random batch of test images.

### 2. Test with Internet Images
To test the model on random internet images (or a specific folder of unseen data):
```bash
python firstSolution/predict_internet.py
```

## 📈 Performance

Based on recent training runs, the model achieves state-of-the-art performance for this dataset:

| Metric | Score |
|--------|-------|
| **Accuracy** | **98.89%** |
| **Precision** | **98.90%** |
| **Recall** | **98.89%** |
| **F1-Score** | **98.89%** |

*Results may vary slightly depending on the random seed and dataset split.*

## 📜 License
This project is open-source and available under the MIT License.