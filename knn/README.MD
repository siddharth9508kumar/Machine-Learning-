# Machine Learning Lab — KNN & Naive Bayes

A hands-on implementation of two classical supervised learning algorithms: **K-Nearest Neighbors (KNN)** and **Gaussian Naive Bayes**, applied to a diabetes prediction dataset.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Experiments](#experiments)
  - [Q1 — KNN from Scratch (Toy Dataset)](#q1--knn-from-scratch-toy-dataset)
  - [Q2 — KNN on Diabetes Dataset](#q2--knn-on-diabetes-dataset)
  - [Q3 — Naive Bayes on Diabetes Dataset](#q3--naive-bayes-on-diabetes-dataset)
- [Results Summary](#results-summary)
- [Dataset](#dataset)
- [Notes & Warnings](#notes--warnings)

---

## Overview

This lab covers:

1. Manual Euclidean distance calculation and KNN prediction on a small toy dataset.
2. KNN classification on the [Pima Indians Diabetes dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) with full evaluation.
3. Gaussian Naive Bayes classification on the same dataset for comparison.

---

## Requirements

```bash
pip install numpy pandas scikit-learn
```

| Package      | Purpose                        |
|-------------|-------------------------------|
| `numpy`     | Array operations               |
| `pandas`    | Data loading and manipulation  |
| `scikit-learn` | KNN, Naive Bayes, metrics   |

---

## Project Structure

```
.
├── diabetes.csv        # Kaggle diabetes dataset
├── q1_knn_toy.py       # KNN on toy dataset + Euclidean distance
├── q2_knn_diabetes.py  # KNN on diabetes dataset with evaluation
├── q3_naive_bayes.py   # Gaussian Naive Bayes with evaluation
└── README.md
```

---

## Experiments

### Q1 — KNN from Scratch (Toy Dataset)

A minimal example using 5 labeled 2D points to illustrate KNN prediction and Euclidean distance.

**Dataset:**

| Point | X1 | X2 | Class |
|-------|----|----|-------|
| P1    | 2  | 4  | A     |
| P2    | 4  | 2  | A     |
| P3    | 4  | 4  | B     |
| P4    | 6  | 2  | B     |
| P5    | 6  | 4  | B     |

**Prediction for [5, 3] with k=3:** `B`

**Euclidean distance between P1 and P2:** `2.828`

**Key code:**

```python
from sklearn.neighbors import KNeighborsClassifier
import math

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x, y)
print(knn.predict([[5, 3]]))  # ['B']

def euclidean_distance(x1, x2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))
```

---

### Q2 — KNN on Diabetes Dataset

Trains a KNN classifier (`k=10`) on the full diabetes dataset and evaluates it on training data.

**Prediction for sample `[1, 85, 66, 29, 0, 26.6, 0.351, 31]`:** `0` (non-diabetic)

**Evaluation Results:**

| Metric    | Class 0 | Class 1 | Overall |
|-----------|---------|---------|---------|
| Precision | 0.78    | 0.76    | —       |
| Recall    | 0.91    | 0.53    | —       |
| F1-Score  | 0.84    | 0.63    | —       |
| Accuracy  | —       | —       | **77.7%** |

**Confusion Matrix:**

```
[[454  46]
 [125 143]]
```

---

### Q3 — Naive Bayes on Diabetes Dataset

Trains a Gaussian Naive Bayes classifier on the same dataset.

**Prediction for sample `[6, 148, 72, 35, 0, 33.6, 0.627, 50]`:** `1` (diabetic)

**Evaluation Results:**

| Metric    | Class 0 | Class 1 | Overall |
|-----------|---------|---------|---------|
| Precision | 0.80    | 0.68    | —       |
| Recall    | 0.84    | 0.62    | —       |
| F1-Score  | 0.82    | 0.64    | —       |
| Accuracy  | —       | —       | **76.3%** |

**Confusion Matrix:**

```
[[421  79]
 [103 165]]
```

---

## Results Summary

| Model         | Accuracy | Notes                              |
|--------------|----------|------------------------------------|
| KNN (k=10)   | 77.7%    | Better recall on majority class    |
| Naive Bayes  | 76.3%    | Better precision on majority class |

Both models perform similarly overall. KNN achieves slightly higher accuracy, while Naive Bayes shows better balance between precision and recall for the positive (diabetic) class.

---

## Dataset

- **Source:** [Kaggle — Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **File:** `diabetes.csv`
- **Rows:** 768
- **Features:** 8 numeric medical features (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
- **Target:** `Outcome` — 0 (non-diabetic) or 1 (diabetic)

---

## Notes & Warnings

- **Training-set evaluation:** All models are evaluated on the same data they were trained on, which inflates accuracy. Use `train_test_split` for unbiased results in production.
- **Feature name warning:** Passing a plain Python list to `predict()` after fitting on a DataFrame triggers a scikit-learn warning. Wrap predictions in a DataFrame to suppress it:
  ```python
  import pandas as pd
  knn.predict(pd.DataFrame([[1, 85, 66, 29, 0, 26.6, 0.351, 31]], columns=x.columns))
  ```
- **Class imbalance:** The dataset has 500 class-0 vs 268 class-1 samples, which contributes to lower recall on the diabetic class across both models.
