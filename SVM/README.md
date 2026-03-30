# Support Vector Machine (SVM) — Implementations

A collection of SVM implementations in Python using scikit-learn, covering linearly separable data, non-linearly separable data, randomly generated classification data, and a real-world Kaggle dataset.

---

## Project Structure

| Notebook Section | Description |
|---|---|
| Linear SVM | Manual linearly separable data with decision boundary plot |
| RBF SVM | Circular non-linearly separable data using `make_circles` |
| Random Data SVM | Classification on generated data with accuracy metrics |
| Kaggle Dataset SVM | Student placement prediction on a real CSV dataset |

---

## Requirements

```bash
pip install numpy matplotlib scikit-learn pandas
```

---

## Implementations

### 1. Linear SVM on Manually Created Data

Uses a small hand-crafted dataset of 6 points split into two classes. Trains a linear kernel SVM and plots the decision boundary.

```python
x = np.array([[1,2],[2,3],[3,3],[6,5],[7,8],[8,8]])
y = np.array([0,0,0,1,1,1])
model = svm.SVC(kernel='linear')
```

**Output:** Scatter plot with the linear decision boundary drawn in red.

---

### 2. RBF SVM on Non-Linearly Separable Data

Generates concentric circle data using `make_circles` (200 samples, noise=0.1, factor=0.5) and trains an RBF kernel SVM, which maps data into higher dimensions to find a non-linear decision boundary.

```python
x, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)
model = SVC(kernel='rbf')
```

**Output:** Scatter plot visualising the two circular classes.

---

### 3. SVM on Randomly Generated Data

Generates a 300-sample, 2-feature classification dataset using `make_classification`, performs a 70/30 train-test split, and evaluates the linear SVM.

**Results:**

| Metric | Score |
|---|---|
| Accuracy | 95.56% |
| Precision (class 0) | 0.96 |
| Precision (class 1) | 0.95 |
| F1-score (macro avg) | 0.96 |

---

### 4. SVM on Kaggle Dataset (Student Placement)

Reads a CSV file (`test.csv`) containing student records with categorical features. Encodes categorical columns, scales features with `StandardScaler`, and trains an RBF SVM to predict placement status.

**Preprocessing steps:**
- Label encoding for `Gender`, `Degree`, `Branch`, `Placement_Status`
- Feature scaling with `StandardScaler`
- 70/30 train-test split

**Results:**

| Metric | Score |
|---|---|
| Accuracy | 95.6% |
| Precision (class 0) | 0.96 |
| Precision (class 1) | 0.95 |
| F1-score (weighted avg) | 0.96 |

> **Note:** Place your `test.csv` dataset in the same directory as the notebook before running.

---

## Key Concepts

- **Linear kernel** — best for linearly separable data; draws a straight hyperplane.
- **RBF kernel** — maps data to higher dimensions to separate non-linear patterns (default in scikit-learn's SVC).
- **Decision boundary** — the hyperplane that maximises the margin between classes.
- **Support vectors** — the data points closest to the decision boundary that define the margin.

---

## Usage

Run the notebook cells in order. Each section is self-contained. For the Kaggle section, ensure `test.csv` is available in the working directory.

---

- [scikit-learn SVM documentation](https://scikit-learn.org/stable/modules/svm.html)
- [make_circles](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html)
- [make_classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)
