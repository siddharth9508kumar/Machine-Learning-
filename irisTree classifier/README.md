# 🌸 IrisTree Classifier

A beginner-friendly machine learning project that classifies Iris flowers into three species using a Decision Tree algorithm. Built with scikit-learn.

---

## 📌 Overview

The Iris dataset is one of the most well-known datasets in machine learning. This project trains a Decision Tree classifier to distinguish between three species of Iris flowers — *Setosa*, *Versicolor*, and *Virginica* — based on four physical measurements.

---

### 🌼 Dataset

- **Source:** `sklearn.datasets.load_iris` (built-in)
- **Samples:** 150
- **Features:** 4
  - Sepal length (cm)
  - Sepal width (cm)
  - Petal length (cm)
  - Petal width (cm)
- **Classes:** 3
  - Setosa
  - Versicolor
  - Virginica

---

#### 🗂️ Project Structure

```
IrisTree-Classifier/
│
├── main.py            # Main script
└── README.md          # Project documentation
```

---

##### ⚙️ Requirements

- Python 3.7+
- scikit-learn

Install dependencies:

```bash
pip install scikit-learn
```

---

###### 🚀 Usage

Run the classifier:

```bash
python iris_tree.py
```

**Expected Output:**
```
Accuracy: 0.9667
Predicted Classes: [1 0 2 1 1 0 ...]
Actual Classes:    [1 0 2 1 1 0 ...]
Class Names: ['versicolor', 'setosa', 'virginica', ...]
```

---

####### 🧠 How It Works

1. **Load Data** — Fetches the Iris dataset from scikit-learn
2. **Split** — Divides data into 80% training and 20% test sets (`random_state=42` for reproducibility)
3. **Train** — Fits a `DecisionTreeClassifier` on the training data
4. **Predict** — Runs predictions on the test set
5. **Evaluate** — Computes accuracy score and displays results

---

####### 📊 Results

| Metric   | Value         |
|----------|---------------|
| Accuracy | ~93% – 100%   |
| Model    | Decision Tree |
| Test Set | 20% (30 samples) |

> Accuracy may vary slightly without `random_state` set.

---

## 📖 Code

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

d = load_iris()
x_tr, x_te, y_tr, y_te = train_test_split(
    d.data, d.target, test_size=0.2, random_state=42
)

tree = DecisionTreeClassifier(random_state=42)
tree.fit(x_tr, y_tr)
y_pred = tree.predict(x_te)

accuracy = accuracy_score(y_te, y_pred)
print("Accuracy:", accuracy)
print("Predicted Classes:", y_pred)
print("Actual Classes:   ", y_te)
print("Class Names:", [d.target_names[i] for i in y_pred])
```

---

## 💡 Next Steps

- Visualize the decision tree with `sklearn.tree.plot_tree`
- Compare with other classifiers (Random Forest, KNN, SVM)
- Add cross-validation for more robust evaluation
- Build a simple CLI to predict species from user input

---


