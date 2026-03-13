# 🔢 DigitSense

A machine learning project that recognizes handwritten digits (0–9) using a Random Forest classifier. Built with scikit-learn on the classic Digits dataset.

---

## 📌 Overview

DigitSense trains a Random Forest model to identify handwritten digits from 8×8 pixel grayscale images. It demonstrates a full image classification pipeline — from raw pixel data to high-accuracy predictions — and includes an optional visualization of sample digits.

---

## 🗃️ Dataset

- **Source:** `sklearn.datasets.load_digits` (built-in)
- **Samples:** 1,797 images
- **Classes:** 10 (digits 0–9)
- **Image Size:** 8×8 pixels (64 features per sample)

---

## 🗂️ Project Structure

```
DigitSense/
│
├── main.py            # Main script
└── README.md          # Project documentation
```

---

## ⚙️ Requirements

- Python 3.7+
- scikit-learn
- matplotlib *(optional, for digit visualization)*

Install dependencies:

```bash
pip install scikit-learn matplotlib
```

---

## 🚀 Usage

Run the classifier:

```bash
python main.py
```

**Expected Output:**
```
Digit Recognition Accuracy: 97.2%
```

To also **visualize sample digits**, uncomment the display block in the script:

```python
# Display sample digits
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
for idx, ax in enumerate(axes.flatten()):
    ax.imshow(digit_data.images[idx], cmap='gray')
    ax.set_title(f"Digit: {digit_data.target[idx]}")
    ax.axis('off')
plt.tight_layout()
plt.show()
```

This renders an 8-panel grid showing sample digits with their labels.

---

## 🧠 How It Works

1. **Load Data** — Fetches the Digits dataset (1,797 labeled 8×8 images)
2. **Split** — 80% training / 20% test set (`random_state=42`)
3. **Train** — Fits a `RandomForestClassifier` with 100 decision trees
4. **Evaluate** — Scores accuracy directly on the test set

---

## 📊 Results

| Metric           | Value                     |
|------------------|---------------------------|
| Model            | Random Forest             |
| Trees            | 100 (`n_estimators=100`)  |
| Training Set     | 80% (1,437 samples)       |
| Test Set         | 20% (360 samples)         |
| Accuracy         | ~97% – 98%                |

> Random Forest achieves near-human accuracy on this dataset due to its ensemble of decorrelated decision trees.

---

## 📖 Code

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the digits dataset
digit_data = load_digits()

# Prepare and split data
X_train, X_test, y_train, y_test = train_test_split(
    digit_data.data, digit_data.target, test_size=0.2, random_state=42)

# Train Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Calculate and display accuracy
accuracy = rf_model.score(X_test, y_test)
print(f"\nDigit Recognition Accuracy: {accuracy * 100:.1f}%")
```

---

## 💡 Next Steps

- Uncomment the visualization block to inspect sample images
- Add a confusion matrix to see which digits get misclassified
- Try a CNN (Convolutional Neural Network) with TensorFlow/Keras for even higher accuracy
- Test on real handwritten input using OpenCV
- Tune `n_estimators` and `max_depth` to optimize performance

---
