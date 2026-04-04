# Machine Learning Projects 🤖

> A hands-on collection of ML algorithms and real-world projects built while learning core concepts in data science and machine learning.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)

---

## 📁 Repository Structure

```
Machine-Learning-projects/
│
├── Linear regression/          # Simple & multiple linear regression
├── multiple linear regression/ # Multi-feature regression models
├── polynomial regression/      # Non-linear curve fitting
├── LOGISTIC REGRESSION/        # Binary classification
├── SVM/                        # Support Vector Machine classifier
├── knn/                        # K-Nearest Neighbors
└── coustomer churn prediction/ # End-to-end ML project
    └── Project/
```

---

## 🚀 Projects & Results

### 🔴 Customer Churn Prediction *(Main Project)*
Predict whether a telecom customer will leave based on usage patterns and account info.

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~80% | ~78% | ~76% | ~77% |
| Random Forest | ~85% | ~83% | ~81% | ~82% |
| SVM | ~82% | ~80% | ~79% | ~79% |

> **Dataset:** Telco Customer Churn — [Kaggle Link](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
> **Features used:** tenure, contract type, monthly charges, internet service, etc.  
> **Target:** `Churn` (Yes/No)

---

### 📈 Regression Models

| Notebook | Algorithm | Dataset | Key Metric |
|----------|-----------|---------|------------|
| Linear Regression | Simple Linear | Custom / Boston | R² Score |
| Multiple Linear Regression | Multi-feature | House prices | MSE, R² |
| Polynomial Regression | Degree-n fitting | Salary data | R² Score |

---

### 🎯 Classification Models

| Notebook | Algorithm | Dataset | Accuracy |
|----------|-----------|---------|----------|
| Logistic Regression | Binary classification | Iris / Custom | ~85% |
| SVM | Kernel SVM | Classification dataset | ~87% |
| KNN | Distance-based | Custom | ~82% |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| NumPy | Numerical computing |
| Pandas | Data manipulation |
| Matplotlib / Seaborn | Visualization |
| Scikit-learn | ML models & evaluation |
| Jupyter Notebook | Interactive development |

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/siddharth9508kumar/Machine-Learning-.git
cd Machine-Learning-

# 2. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# 3. Launch Jupyter
jupyter notebook
```

Open any `.ipynb` file and run cells top to bottom.

---

## 📚 Concepts Covered

- Data Cleaning & Preprocessing (missing values, encoding, scaling)
- Exploratory Data Analysis (EDA) with visualizations
- Supervised Learning: Regression & Classification
- Model Evaluation: accuracy, precision, recall, F1, confusion matrix
- Feature Engineering & Feature Selection
- Train/Test Split & Cross Validation

---

## 🗺️ Roadmap

- [x] Linear & Polynomial Regression
- [x] Logistic Regression, SVM, KNN
- [x] Customer Churn end-to-end project
- [ ] XGBoost & ensemble methods
- [ ] Deep Learning with PyTorch
- [ ] Deploy churn model on Streamlit Cloud

---

## 👤 Author

**Siddharth Kumar** — CSE (AI/ML), 2nd Year  
📧 siddharth9508.kumar@gmail.com  
🐙 [github.com/siddharth9508kumar](https://github.com/siddharth9508kumar)

---

*Built with curiosity and lots of coffee ☕ — learning ML one dataset at a time.*
