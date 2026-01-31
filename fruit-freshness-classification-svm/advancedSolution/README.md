# SOTA Fruit Quality Detection System (Advanced Solution)

This directory contains a state-of-the-art (SOTA) deep learning solution for classifying fruits as **Fresh** or **Rotten**. It leverages modern computer vision techniques, including **ConvNeXt Large**, **Mixed Precision Training**, and a high-performance `tf.data` pipeline optimized for NVIDIA A100 GPUs.

## 🚀 Key Features

*   **Model Architecture**: Uses `ConvNeXt Large` (ImageNet weights) as the backbone, known for its Transformer-like performance with ConvNet efficiency.
*   **Training Strategy**:
    *   **Mixed Precision (`float16`)**: Drastically reduces memory usage and speeds up training on Tensor Cores.
    *   **2-Phase Training**: 
        1.  **Warmup**: Trains only the classification head with the backbone frozen.
        2.  **Fine-Tuning**: Unfreezes the top layers of the backbone with a lower learning rate and **Cosine Decay**.
    *   **Optimizer**: Uses **AdamW** (Adam with Weight Decay) for better generalization.
    *   **Loss Function**: **Categorical Crossentropy with Label Smoothing (0.1)** to prevent overfitting and overconfidence.
*   **Data Pipeline**:
    *   Rigorous **80% Train / 10% Validation / 10% Test** split.
    *   **Data Cleaning**: Standardizes filenames and removes corrupt images.
    *   **Deep EDA**: Analyzes class balance, image dimensions, and performs SOTA quality checks (blur/brightness).
    *   **Performance**: Uses `tf.data` with `cache()` and `prefetch(AUTOTUNE)` to keep GPUs fully utilized.
*   **Inference**:
    *   **Test Time Augmentation (TTA)**: Averages predictions from original and flipped versions of the image for higher accuracy.
    *   **FastAPI Service**: A production-ready API for real-time inference.

## 📂 Project Structure

*   **`main.py`**: The entry point. Orchestrates the entire pipeline from data download to training.
*   **`model_builder.py`**: Defines the ConvNeXt Large model with internal GPU augmentation layers.
*   **`data_preprocessing.py`**: Builds the `tf.data` pipeline and handles the 80/10/10 split.
*   **`trainer.py`**: Implements the custom 2-phase training loop with callbacks (EarlyStopping, ModelCheckpoint).
*   **`evaluate.py`**: Generates confusion matrices and classification reports on the held-out Test set.
*   **`api.py`**: A FastAPI server to serve the model.
*   **`predict.py`**: A CLI tool for predicting on single images using TTA.
*   **`checkGPU.py`**: Diagnostic tool to verify GPU health and XLA compatibility.

## 🛠️ Usage

### 1. Training the Model
Run the main script to start the full pipeline. This handles data downloading (via Kaggle API), cleaning, and training.

```bash
python main.py
```

*Note: Ensure you have your Kaggle credentials set up (`KAGGLE_USERNAME` and `KAGGLE_KEY`) or in `~/.kaggle/kaggle.json` if the dataset needs to be downloaded.*

### 2. Evaluation
After training, evaluate the model on the held-out Test set to get unbiased metrics.

```bash
python evaluate.py
```

### 3. Inference (Single Image)
Run predictions on a local image file using Test Time Augmentation.

```bash
python predict.py path/to/image.jpg
```

### 4. Start the API
Launch the FastAPI server for production-style inference.

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
Then visit `http://localhost:8000/docs` to test the API via Swagger UI.

## 📊 Data Split Strategy
To ensure rigorous evaluation, the dataset is split as follows:
*   **Train (80%)**: Used for learning weights.
*   **Validation (10%)**: Used for hyperparameter tuning and Early Stopping during training.
*   **Test (10%)**: Completely held-out until the end. Used only by `evaluate.py` to report final performance.

## 🔧 Requirements
*   Python 3.8+
*   TensorFlow 2.10+
*   NVIDIA GPU (Recommended: A100 or V100 for Mixed Precision benefits)
*   Dependencies listed in `../requirements.txt`
