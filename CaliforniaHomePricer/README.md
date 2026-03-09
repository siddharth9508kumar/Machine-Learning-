# 🏡 CaliforniaHomePricer

A machine learning project that predicts California housing prices using Linear Regression. Built with scikit-learn and pandas on the classic California Housing dataset.

---

## 📌 Overview

Using census-derived features like income, population, and location, CaliforniaHomePricer trains a Linear Regression model to estimate median home values across California districts. It serves as a clean, practical introduction to regression-based ML workflows.

---

## 🗃️ Dataset

- **Source:** `sklearn.datasets.fetch_california_housing` (built-in)
- **Samples:** 20,640 districts
- **Target:** Median house value (in $100,000s)
- **Features:** 8

| Feature     | Description                              |
|-------------|------------------------------------------|
| MedInc      | Median income in block group             |
| HouseAge    | Median house age in block group          |
| AveRooms    | Average number of rooms per household    |
| AveBedrms   | Average number of bedrooms per household |
| Population  | Block group population                   |
| AveOccup    | Average number of household members      |
| Latitude    | Block group latitude                     |
| Longitude   | Block group longitude                    |

---

## 🗂️ Project Structure

```
CaliforniaHomePricer/
│
├── main.py            # Main script
└── README.md          # Project documentation
```

---

## ⚙️ Requirements

- Python 3.7+
- scikit-learn
- pandas

Install dependencies:

```bash
pip install scikit-learn pandas
```

---

## 🚀 Usage

Run the model:

```bash
python main.py
```

**Expected Output:**
```
   MedInc  HouseAge  AveRooms  ...  Longitude  MedianPrice
0  8.3252      41.0  6.984127  ... -122.23        4.526
1  8.3014      21.0  6.238137  ... -122.22        3.585
...

Total records: 20640

Mean Absolute Error: $52,000
```

> MAE represents the average dollar difference between predicted and actual home prices.

---

## 🧠 How It Works

1. **Load Data** — Fetches the California Housing dataset from scikit-learn
2. **Build DataFrame** — Converts to a pandas DataFrame with named columns
3. **Preview** — Prints first 5 rows and total record count
4. **Split** — 80% training / 20% test set (`random_state=42`)
5. **Train** — Fits a `LinearRegression` model on training data
6. **Evaluate** — Computes Mean Absolute Error (MAE) on test predictions

---

## 📊 Results

| Metric                  | Value              |
|-------------------------|--------------------|
| Model                   | Linear Regression  |
| Training Set            | 80% (16,512 rows)  |
| Test Set                | 20% (4,128 rows)   |
| Mean Absolute Error     | ~$50,000 – $55,000 |

> Linear Regression serves as a strong baseline. MAE can be reduced further with more advanced models.

---

## 📖 Code

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pandas as pd

# Load California housing data
housing_data = fetch_california_housing()
df = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
df['MedianPrice'] = housing_data.target

# Preview the dataset
print(df.head())
print(f"\nTotal records: {len(df)}")

# Prepare features and target
X = df.drop('MedianPrice', axis=1)
y = df['MedianPrice']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Evaluate
y_pred = regressor.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nMean Absolute Error: ${mae * 100000:.0f}")
```

---

## 💡 Next Steps

- Visualize predicted vs. actual prices with matplotlib
- Try advanced regressors: Random Forest, Gradient Boosting, XGBoost
- Add feature correlation heatmap to identify key price drivers
- Normalize features with `StandardScaler` to potentially improve accuracy
- Perform cross-validation for a more robust error estimate

---
