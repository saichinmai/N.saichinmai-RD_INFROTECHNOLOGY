import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv("fake_news.csv")

# Features and labels
X = data["text"]
y = data["label"]

# Convert text into vectors
vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()

model.fit(X_vectorized, y)

# Save model
pickle.dump(
    model,
    open("news_model.pkl", "wb")
)

pickle.dump(
    vectorizer,
    open("vectorizer.pkl", "wb")
)

print("Model trained successfully!")