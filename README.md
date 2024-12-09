# Credit Risk Anomaly Detection System
---

## Overview
This project implements an end-to-end pipeline for unsupervised anomaly detection in credit risk datasets. The system includes multiple unsupervised learning algorithms

---

## Features
- **Algorithms**:
  - Isolation Forest
  - Local Outlier Factor (LOF)
  - Principal Component Analysis (PCA)
  - K-Means Clustering
  - Ensemble Models: Voting and Stacking Ensembles ** (Yet to implement)
- **Evaluation**:
  - Anomaly score computation for datasets
- **Outputs**:
  - Anomaly scores saved in CSV format

---
## Setup Instructions

### 1. Prerequisites
Python: 3.8
Numpy: 1.24.4
Pandas: 2.0.3
scikit-learn: 1.3.2
Matplotlib: 3.7.5
Seaborn: 0.13.2

---
## Project Structure
.
├── Dataset/                     # Credt Card Dataset
├── Model/                       # Model Saved Directory
├── visualization/               # Visualization
├── Train/                       # Unsupervised Training Algorithm with model evaluation
├── load_and_preprocess_data.py  # Script to load and preprocess the datan
├── save_and_load_model.py       # Helper functions for saving the best model and loading the model
├── main.py                      # Run the pipeline step by step

## Observation
Mean Anomaly Score for Isolation Forest: -0.1804
Mean Anomaly Score for Local Outlier Factor: -0.9800
Mean Anomaly Score for PCA: 0.8812
Mean Anomaly Score for K-Means: 4.4026