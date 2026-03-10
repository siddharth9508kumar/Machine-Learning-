# 🚢 TitanicSurvivor

A machine learning project that predicts passenger survival on the RMS Titanic using a Random Forest classifier.

---

## 📌 Overview

TitanicSurvivor trains on historical Titanic passenger data to predict who survived the disaster.
It handles missing data, encodes categorical features, and evaluates model performance —
making it a great end-to-end example of a binary classification pipeline.

---

## 📊 Model Details

| Property           | Value                          |
|--------------------|--------------------------------|
| Algorithm          | Random Forest Classifier       |
| Estimators         | 100                            |
| Validation Split   | 80/20                          |
| Validation Accuracy| ~82%                           |
| Target Variable    | `Survived` (0 = No, 1 = Yes)   |

---

## 🧬 Features Used

- `Pclass` — Ticket class (1st, 2nd, 3rd)
- `Sex` — Gender (one-hot encoded)
- `Age` — Age in years (median-imputed)
- `SibSp` — Siblings/spouses aboard
- `Parch` — Parents/children aboard
- `Fare` — Ticket fare

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas scikit-learn
```

### Run
```bash
python titanic.py
```

> Place `titanic.csv` in the project root before running.
> where to 'download the dataset:https://www.kaggle.com/datasets/heptapod/titanic'

---

## ⚙️ Pipeline

1. Load `titanic.csv`
2. Impute missing `Age` with median; fill missing `Embarked` with `'S'`
3. One-hot encode `Sex`
4. Split data 80/20 (train/validation)
5. Train `RandomForestClassifier(n_estimators=100)`
6. Report validation accuracy

---

## 💡 Key Insight

> **Women and children in 1st class had the highest survival rates.**
> Gender was the strongest predictive feature, followed by passenger class and fare.

---

## 🔧 Possible Improvements

- Extract titles from `Name` (Mr., Mrs., Miss.)
- Engineer `FamilySize = SibSp + Parch + 1`
- Use `Cabin` deck letter as a feature
- Tune with `GridSearchCV`
- Add `classification_report` for precision/recall/F1
- Generate a Kaggle submission file

---
