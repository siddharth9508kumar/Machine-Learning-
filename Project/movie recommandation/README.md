# 🎬 Movie Review Sentiment Analysis

A lightweight NLP pipeline that classifies movie reviews as **positive** or **negative** using a Naive Bayes classifier trained on bag-of-words features.

---

## Overview

This project demonstrates a foundational text classification workflow using `scikit-learn`. It vectorizes raw review text into word-count features and trains a Multinomial Naive Bayes model to predict sentiment.

---

## How It Works

```
Raw Text → CountVectorizer → Feature Matrix → MultinomialNB → Sentiment Label
```

1. **Text Vectorization** — `CountVectorizer` converts each review into a sparse vector of word counts (bag-of-words).
2. **Model Training** — `MultinomialNB` learns word-to-sentiment associations from labeled examples.
3. **Prediction** — Unseen reviews are transformed and classified as `1` (Positive) or `0` (Negative).

---

## Dataset

The model is trained on 10 hand-labeled movie reviews:

| Review | Sentiment |
|--------|-----------|
| "Absolutely loved this film! Outstanding performance." | ✅ Positive |
| "Complete waste of time. Terrible acting." | ❌ Negative |
| "A masterpiece of modern cinema!" | ✅ Positive |
| "Dull and uninspiring storyline." | ❌ Negative |
| ... | ... |

> Labels: `1` = Positive, `0` = Negative

---

## Sample Output

```
Sentiment Analysis Results:
----------------------------------------
"What an amazing experience!"
  → Predicted: Positive

"Worst movie I have ever seen"
  → Predicted: Negative

"Decent entertainment overall"
  → Predicted: Positive
```

---

## Requirements

```
Python 3.7+
scikit-learn
```

Install dependencies:

```bash
pip install scikit-learn
```

---

## Usage

```bash
python sentiment_analysis.py
```

To classify your own reviews, modify the `test_reviews` list in the script:

```python
test_reviews = [
    "Your review here",
    "Another review here"
]
```

---

## Project Structure

```
movie-sentiment/
│
├── main.py                 # Main script
└── README.md               # This file
```

---

## Limitations & Future Improvements

| Limitation | Suggested Improvement |
|---|---|
| Only 10 training samples | Use a large dataset (e.g., IMDb 50K reviews) |
| Raw word counts | Switch to TF-IDF weighting |
| No neutral/mixed sentiment | Add multi-class classification |
| No model persistence | Save model with `joblib` for reuse |
| No evaluation metrics | Add train/test split + accuracy report |

---
