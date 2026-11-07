from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained ML model
with open("phishing_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load TF-IDF vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def home():

    result = ""
    confidence = 0
    email = ""
    status = ""

    if request.method == "POST":

        email = request.form.get("email", "").strip()

        if not email:
            result = "Please enter email content."
            status = "error"

        else:
            # Convert email into numerical features
            email_vector = vectorizer.transform([email])

            # Predict email category
            prediction = model.predict(email_vector)[0]

            # Calculate actual model probability
            probabilities = model.predict_proba(email_vector)[0]
            confidence = round(max(probabilities) * 100, 2)

            if prediction == 1:
                result = "SPAM / SUSPICIOUS EMAIL"
                status = "danger"
            else:
                result = "LEGITIMATE EMAIL"
                status = "safe"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        email=email,
        status=status
    )


if __name__ == "__main__":
    app.run(debug=False)