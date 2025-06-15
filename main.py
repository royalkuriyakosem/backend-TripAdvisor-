from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  # Allow CORS for all origins

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route("/predict", methods=["POST"])
def predict_sentiment():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "Text is required"}), 400

    # Vectorize and predict
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    confidence = model.predict_proba(text_vec)[0][prediction]
    sentiment = "Positive" if prediction == 1 else "Negative"

    return jsonify({
        "sentiment": sentiment,
        "confidence": round(float(confidence), 4)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
