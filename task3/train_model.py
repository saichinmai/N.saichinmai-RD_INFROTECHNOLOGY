import pandas as pd
import pickle

from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = pd.read_csv("medical_data.csv")

# Input features
X = data.drop("disease", axis=1)

# Output
y = data["disease"]

# Train model
model = DecisionTreeClassifier()

model.fit(X, y)

# Save model
pickle.dump(model, open("disease_model.pkl", "wb"))

print("Medical diagnosis model trained successfully!")