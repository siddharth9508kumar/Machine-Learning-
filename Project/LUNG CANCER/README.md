# 🫁 Lung Cancer Risk Factor Analysis — Exploratory Data Analysis (EDA)

## Overview

This project performs an **Exploratory Data Analysis (EDA)** on a lung cancer dataset (`lung_cancer_examples.csv`) to uncover patterns, distributions, and correlations among key risk factors that may contribute to lung cancer outcomes.

---

## Dataset Description

| Column    | Description                                      |
|-----------|--------------------------------------------------|
| `Name`    | First name of the patient                        |
| `Surname` | Last name of the patient                         |
| `Age`     | Age of the patient (years)                       |
| `Smokes`  | Smoking intensity/frequency score                |
| `AreaQ`   | Area quality index (environmental risk factor)   |
| `Alkhol`  | Alcohol consumption score                        |
| `Result`  | Lung cancer diagnosis (0 = No, 1 = Yes)          |

- **Total Records:** 59 patients  
- **Missing Values:** None (0% missing across all columns)

---

## Statistical Summary

| Feature  | Mean  | Std Dev | Min | Max |
|----------|-------|---------|-----|-----|
| Age      | 42.63 | 16.24   | 18  | 77  |
| Smokes   | 15.07 | 7.98    | 0   | 34  |
| AreaQ    | 5.20  | 2.46    | 1   | 10  |
| Alkhol   | 3.24  | 2.38    | 0   | 8   |
| Result   | 0.47  | 0.50    | 0   | 1   |

> Roughly **47.5%** of patients in this dataset are diagnosed with lung cancer (`Result = 1`), indicating a near-balanced class distribution.

---

## Analysis Steps

### 1. Data Loading & Inspection
- Loaded dataset using `pandas`
- Previewed the first 5 rows with `df.head()`
- Confirmed column names and data types

### 2. Missing Value Analysis
- Computed missing value counts and percentages
- **Result:** No missing values found in any column

### 3. Descriptive Statistics
- Used `df.describe()` to compute mean, standard deviation, min/max, and quartile values for all numerical features

### 4. Target Variable Distribution (`Result`)
- Plotted a **bar chart** of lung cancer outcomes (0 vs 1)
- Reveals near-equal class split — good for balanced modeling

### 5. Age Distribution
- Plotted a **histogram** of patient age (20 bins)
- Age ranges from 18 to 77, with a mean around 42–43

### 6. Age vs. Result Boxplot
- Used a **Seaborn boxplot** to compare age distributions across diagnosed (`1`) and non-diagnosed (`0`) groups
- Helps identify if older patients skew toward positive diagnoses

### 7. Correlation Heatmap
- Computed correlation matrix for all numerical features: `Age`, `Smokes`, `AreaQ`, `Alkhol`, `Result`
- Visualized using a **Seaborn heatmap** with `coolwarm` colormap
- Highlights which risk factors are most strongly correlated with lung cancer outcome

---

## Key Observations

- **Smoking** is expected to show a positive correlation with `Result` — a primary risk factor for lung cancer
- **Alcohol consumption** and **area quality** may contribute as secondary environmental/lifestyle factors
- **Age** distribution is broad; older patients may show higher diagnosis rates
- The dataset is small (59 records), which limits the generalizability of findings but is suitable for exploratory analysis

---

## Technologies Used

| Library        | Purpose                                 |
|----------------|-----------------------------------------|
| `pandas`       | Data loading, inspection, statistics    |
| `matplotlib`   | Bar charts, histograms                  |
| `seaborn`      | Boxplots, heatmaps                      |
| `numpy`        | Numerical operations, feature selection |

---

## How to Run

```bash
# 1. Install dependencies
pip install pandas matplotlib seaborn numpy

# 2. Place the dataset in the working directory
#    File: lung_cancer_examples.csv

# 3. Run the analysis notebook or script
jupyter notebook lung_cancer_eda.ipynb
# or
python lung_cancer_eda.py
```

---

## File Structure

```
lung-cancer-eda/
│
├── lung_cancer_examples.csv   # Raw dataset
├── LUNG_CANCER.ipynb      # Jupyter Notebook with full analysis
└── README.md                  # Project documentation (this file)
```

---

## Future Work

- Apply machine learning models (Logistic Regression, Random Forest) for prediction
- Explore feature importance to identify top risk factors
- Handle class imbalance if using a larger dataset
- Incorporate additional features (e.g., family history, gender, occupation)

---

## License

This project is for educational and research purposes only. Patient data has been anonymized.
