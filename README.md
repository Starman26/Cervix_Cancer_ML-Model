# Cervix_Cancer_ML-Model
Welcome to my **Cervical Cancer Prediction** project! This repository contains a machine learning implementation designed to assist in predicting cervical cancer based on medical data. By leveraging advanced algorithms, this project aims to provide a data-driven approach to identifying potential cases, contributing to early detection and improving patient outcomes.

## Objective

Cervical cancer remains one of the leading causes of mortality among women worldwide. Early and accurate prediction is vital for timely interventions and treatment. This project explores various machine learning models to classify and predict cervical cancer risk factors, using publicly available datasets.

## Features

- **Data Visualization**: 
  - `matplotlib` and `seaborn` for creating insightful visualizations of the dataset.
- **Data Preprocessing**: 
  - `StandardScaler` for feature scaling.
  - `SMOTE` for handling imbalanced datasets.
  - `train_test_split` for splitting data into training and testing sets.
- **Dimensionality Reduction**: 
  - `PCA` for reducing dataset dimensionality.
- **Model Training and Evaluation**:
  - `sklearn` models such as `LogisticRegression` for classification.
  - Evaluation metrics including `accuracy_score`, `classification_report`, `precision_recall_curve`, and `roc_auc_score`.
  - Confusion matrices displayed with `ConfusionMatrixDisplay`.
  - Cross-validation using `KFold` and `cross_val_score`.
  - Hyperparameter tuning with `RandomizedSearchCV`.
- **Deep Learning**:
  - `PyTorch` for building, training, and evaluating neural networks.
  - `torch.nn` and `torch.optim` for model construction and optimization.
  - `DataLoader` and `TensorDataset` for handling data in neural networks.
- **Hyperparameter Optimization**:
  - `Optuna` for advanced hyperparameter tuning.
