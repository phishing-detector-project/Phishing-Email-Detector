import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load the new dataset
data = pd.read_csv("phishing_email.csv")

# Remove rows with missing email text or labels
data = data.dropna(subset=["text_combined", "label"])

# Separate email text and labels
X = data["text_combined"].astype(str)
y = data["label"].astype(int)

# Split dataset into training and testing data
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Convert email text into numerical features
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

print("\nModel training completed!")
print("Accuracy:", accuracy_score(y_test, predictions) * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save the model and vectorizer
with open("phishing_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel and vectorizer saved successfully!")