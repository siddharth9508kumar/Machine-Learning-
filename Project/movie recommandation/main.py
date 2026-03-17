#Movie Review Sentiment Analysis

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample movie reviews dataset
movie_reviews = [
    "Absolutely loved this film! Outstanding performance.",
    "Complete waste of time. Terrible acting.",
    "A masterpiece of modern cinema!",
    "Dull and uninspiring storyline.",
    "One of the best movies this decade!",
    "Poor direction and weak script.",
    "Incredible visuals and compelling story.",
    "Could not stay awake through this.",
    "A must-watch for everyone!",
    "Disappointing and forgettable."
]

# Sentiment labels: 1 = positive, 0 = negative
sentiments = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

# Transform text to feature vectors
text_vectorizer = CountVectorizer()
X_vectors = text_vectorizer.fit_transform(movie_reviews)

# Build and train the sentiment classifier
sentiment_clf = MultinomialNB()
sentiment_clf.fit(X_vectors, sentiments)

# Test with brand new reviews
test_reviews = [
    "What an amazing experience!",
    "Worst movie I have ever seen",
    "Decent entertainment overall"
]
test_vectors = text_vectorizer.transform(test_reviews)
predictions = sentiment_clf.predict(test_vectors)

# Display predictions
print("Sentiment Analysis Results:")
print("-" * 40)
for review, prediction in zip(test_reviews, predictions):
    mood = "Positive       " if prediction == 1 else "Negative     "
    print(f'"{review}"')
    print(f"  → Predicted: {mood}\n")