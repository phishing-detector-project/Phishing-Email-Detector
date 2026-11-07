import pickle

# Load trained model and vectorizer
with open("phishing_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("Email Spam / Phishing Detector")
print("--------------------------------")

email = input("Enter email text: ")

# Convert email into numerical features
email_vector = vectorizer.transform([email])

# Predict email category
prediction = model.predict(email_vector)[0]

if prediction == 1:
    print("\nResult: SPAM / SUSPICIOUS EMAIL")
else:
    print("\nResult: LEGITIMATE EMAIL")