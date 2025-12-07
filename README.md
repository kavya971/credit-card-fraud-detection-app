Credit Card Fraud Detection

This project implements a machine learning pipeline to detect fraudulent credit card transactions. It includes data preprocessing, feature engineering, model training, evaluation, and deployment-ready components. The goal is to build an efficient and reliable fraud detection system using real-world transactional data.

Table of Contents

Introduction

Dataset

Methodology

Model Training

Evaluation

Results

Installation

Usage

Technologies Used

Future Improvements

1.Introduction

Credit card fraud is a major challenge for financial institutions, costing billions every year. This project applies machine learning techniques to classify transactions as fraudulent or legitimate. It aims to achieve high accuracy while ensuring minimal false positives to avoid customer inconvenience.

2.Dataset

The project uses a standard credit card fraud detection dataset containing anonymized transaction features, including:

Numerical features transformed using PCA

Time and Amount features

Highly imbalanced classes (fraud vs. non‑fraud)

Class imbalance is handled using resampling methods to improve model robustness.

3.Methodology

Performed exploratory data analysis to understand distributions and correlations.

Cleaned and preprocessed data (scaling, handling imbalance).

Applied SMOTE/undersampling techniques to balance the dataset.

Trained multiple machine learning models and compared performance.

Selected the best-performing model and saved the pipeline for reuse.

4.Model Training

The following algorithms were evaluated:

Logistic Regression

XGBoost

Gradient Boosting

Decision Tree

Hyperparameter tuning was performed using GridSearchCV/RandomizedSearchCV.

5.Evaluation

Metrics used in evaluation:

Accuracy

Precision

Recall

F1 Score

ROC‑AUC Score

Confusion Matrix

Due to class imbalance, more focus was given to recall and ROC‑AUC.

6.Results

The final model achieved strong performance on the test set with:

High recall (detecting most fraud cases)

Reduced false positives

Stable performance across multiple validation sets

Exact numerical results can be viewed in the evaluation outputs or reports folder.

7.Installation

Clone the repository:

git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection


Install dependencies:

pip install -r requirements.txt

Usage

To train the model:

python src/train_model.py


To evaluate the model:

python src/evaluate.py


To load and use the saved model pipeline:

import joblib
model = joblib.load('models/fraud_detection_pipeline.pkl')
prediction = model.predict(input_data)

8.Technologies Used

Python

NumPy

Pandas

Scikit-Learn

Matplotlib and Seaborn

Imbalanced‑Learn

9.Future Improvements

Deployment as a REST API or web application

Integration with streaming data for real-time fraud detection

Experimentation with neural network-based models

Enhanced feature engineering and anomaly detection
