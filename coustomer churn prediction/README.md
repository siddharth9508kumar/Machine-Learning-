# Customer Churn Prediction Model

A machine learning model that predicts customer churn for a telecom company using the Telco Customer Churn dataset and a Random Forest classifier.

---

## Overview

Customer churn — when customers stop using a service — is a critical business problem. This project builds a binary classification model to identify customers likely to churn, enabling proactive retention strategies.

---

## Dataset

**Source:** Telco Customer Churn dataset (`Telco-Customer-Churn.csv`)

- **Rows:** 7,043 customers
- **Columns:** 21 features
- **Target:** `Churn` (Yes / No)
- **Class distribution:** ~73.5% No Churn, ~26.5% Churn

### Key Features

| Category | Features |
|---|---|
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` |
| Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Charges | `MonthlyCharges`, `TotalCharges` |

---

## Project Structure

```
├── customer_churn_prediction.ipynb          # Main model script
├── Telco-Customer-Churn.csv                 # Dataset
└── README.md
```

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## How It Works

### 1. Data Loading & Exploration
- Loads the CSV and inspects the first few rows
- Checks for null values and basic statistics

### 2. Preprocessing
- Converts `TotalCharges` to numeric (coercing errors)
- Fills missing `TotalCharges` values with the median
- Encodes all categorical columns using `LabelEncoder`

### 3. Train/Test Split
- Features: all columns except `customerID` and `Churn`
- Target: `Churn`
- Split: 80% training / 20% testing (`random_state=0`)

### 4. Feature Scaling
- Applies `StandardScaler` to normalize features

### 5. Model Training
- Trains a `RandomForestClassifier` with default hyperparameters

### 6. Evaluation
- Reports accuracy score
- Displays a confusion matrix with "No Churn" / "Churn" labels

---

## Results

| Metric | Value |
|---|---|
| Accuracy | **78%** |

The confusion matrix provides a breakdown of true positives, true negatives, false positives, and false negatives for deeper evaluation.

---

## Usage

Run the script end-to-end:

```bash
python customer_churn_prediction.ipynb
```

Or open it in a Jupyter/Colab notebook and run cells sequentially.

---

## Potential Improvements

- **Handle class imbalance** — use SMOTE or `class_weight='balanced'` to improve recall on the churn (minority) class
- **Hyperparameter tuning** — use `GridSearchCV` or `RandomizedSearchCV` on the Random Forest
- **Try other models** — XGBoost, Logistic Regression, or a stacking ensemble
- **Feature importance analysis** — identify which features drive churn most
- **Additional metrics** — evaluate with precision, recall, F1-score, and ROC-AUC alongside accuracy
