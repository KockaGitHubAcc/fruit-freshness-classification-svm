# Fruit Freshness Classification (SVM + FastAPI)

End-to-end computer vision system for **fresh vs rotten fruit classification** combining classical machine learning (SVM + feature engineering) with a deployable **FastAPI inference service**.

The project demonstrates a full applied ML workflow:

- dataset bootstrapping via web scraping  
- feature-engineered SVM training  
- evaluation and visualization  
- real-time inference via REST API  

**Performance:** ~**98.89% accuracy** on the project test split.

---

## Problem Statement

Manual fruit quality inspection is slow, subjective, and expensive.  
This project automates freshness classification from images.

Supported classes:

- freshapples  
- freshbanana  
- freshoranges  
- rottenapples  
- rottenbanana  
- rottenoranges  

Final predictions are simplified to **Fresh vs Rotten**.

---

## System Overview

This repository contains three progressively more advanced implementations.

---

### 1. `firstSolution/` — Feature-Engineered SVM Pipeline

Classical computer vision + machine learning approach:

- Otsu adaptive segmentation  
- HSV color histograms  
- Shape descriptors (Area + Hu Moments)  
- Texture statistics  
- Support Vector Machine (RBF kernel)  
- Grid Search hyperparameter tuning  
- Standard scaling + label encoding  
- Confusion matrix + classification report  
- Inference on unseen images  

This pipeline produces the reported **98.89% accuracy**.

---

### 2. `cockaSolution/` — Prototype + Dataset Bootstrapping

Early experimental phase where I:

- implemented web scraping to collect raw fruit images  
- validated dataset organization and labeling strategy  
- tested preprocessing approaches  

This prototype was used to bootstrap the dataset before building the final SVM pipeline.

> The full dataset is intentionally not included to keep the repository lightweight and avoid redistribution issues.

---

### 3. `advancedSolution/` — FastAPI Real-Time Inference Service

Deployment-oriented implementation exposing the trained model via **FastAPI**.

Capabilities:

- accepts a public image URL of any fruit  
- downloads and preprocesses the image automatically  
- runs model inference  
- returns:
  - Fresh / Rotten prediction  
  - confidence score  
  - inference time  

This demonstrates how the model can be integrated into production systems.

---

## Project Structure

```text
fruit-freshness-classification-svm/
├── firstSolution/        # SVM + feature engineering pipeline
├── cockaSolution/       # Web scraping + early prototype
├── advancedSolution/    # FastAPI inference service
├── results/             # Evaluation outputs
├── assets/              # Screenshots (confusion matrix, predictions)
├── requirements.txt
└── README.md
