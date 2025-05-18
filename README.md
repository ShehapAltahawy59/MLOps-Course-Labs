Sure! Here's a README file tailored for your bank churn prediction ML project using the provided code:

---

# Bank Consumer Churn Prediction

This project implements a machine learning pipeline to predict customer churn in a bank dataset. It includes data preprocessing, model training, evaluation, and experiment tracking using **MLflow**.

---

## Project Overview

The goal is to predict whether a customer will exit the bank based on various features such as credit score, geography, age, balance, and others. The dataset is rebalanced to address class imbalance before model training.

---

## Features

* Data balancing via downsampling of majority class
* Preprocessing including scaling numerical features and one-hot encoding categorical features
* Training multiple classification models:

  * Logistic Regression
  * Support Vector Machine (SVM)
  * Random Forest Classifier
  * Gradient Boosting Classifier
* Evaluation metrics:

  * Accuracy
  * Precision
  * Recall
  * F1-score
* Confusion matrix visualization for each model
* MLflow integration for:

  * Experiment tracking
  * Parameter, metric, and tag logging
  * Model artifact logging
  * Dataset and plot artifact logging

---

## Requirements

* Python 3.7+
* Libraries:

  * pandas
  * matplotlib
  * scikit-learn
  * mlflow

Install dependencies via pip:

```bash
pip install pandas matplotlib scikit-learn mlflow
```

---

## Dataset

The dataset used is `Churn_Modelling.csv`, which should be placed inside a folder named `dataset` in the root directory.

---

## Usage

Run the main script:

```bash
python your_script_name.py
```

This will:

* Load and preprocess data
* Train each model in separate MLflow runs
* Log parameters, metrics, tags, and artifacts (models, confusion matrices, dataset)
* Display confusion matrices for each model

---

## MLflow Tracking Server

Ensure MLflow tracking server is running locally:

```bash
mlflow ui
```

Access the MLflow UI at [http://localhost:5000](http://localhost:5000) to view experiments and models.

---

## File Structure

```
.
├── dataset
│   └── Churn_Modelling.csv
├── your_script_name.py
├── confusion_matrix_LR.png
├── confusion_matrix_SVC.png
├── confusion_matrix_RF.png
├── confusion_matrix_GBOOST.png
└── col_transform.txt
```

---

Model Selection
After training and evaluating multiple models (Logistic Regression, SVM, Random Forest, Gradient Boosting), the XGBoost model was chosen as the final model due to its superior performance on both accuracy and F1-score, as illustrated in the performance comparison plot below:


The XGBoost model achieved the highest balance between precision and recall, making it the most effective choice for predicting customer churn.
[models_compare.png]
