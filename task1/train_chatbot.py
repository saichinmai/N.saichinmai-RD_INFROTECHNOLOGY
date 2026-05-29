import json
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load intents
with open("intents.json", "r") as file:
    data = json.load(file)

texts = []
labels = []

# Collect patterns and tags
for intent in data["intents"]:
    tag = intent["tag"]

    for pattern in intent["patterns"]:
        texts.append(pattern.lower())
        labels.append(tag)

# Convert text into vectors
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(texts)

# Train model
model = MultinomialNB()
model.fit(X, labels)

# Save model
pickle.dump(model, open("chatbot_model.pkl", "wb"))
pickle.dump(vectorizer, open("words.pkl", "wb"))

print("Chatbot training completed successfully!")