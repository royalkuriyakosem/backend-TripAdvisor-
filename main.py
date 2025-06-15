from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib

# Initialize app
app = FastAPI()

# CORS setup (adjust for your frontend URL)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, use ["https://your-frontend.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and vectorizer
try:
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except Exception as e:
    model = None
    vectorizer = None
    print("Model loading failed:", e)

# Request schema
class Review(BaseModel):
    text: str

# Health check
@app.get("/")
def read_root():
    return {"message": "API is running"}

# Predict route
@app.post("/predict")
def predict_sentiment(review: Review):
    if model is None or vectorizer is None:
        return {"error": "Model or vectorizer not loaded."}

    if not review.text.strip():
        return {"error": "Text cannot be empty."}

    try:
        text_vec = vectorizer.transform([review.text])
        prediction = model.predict(text_vec)[0]
        confidence = model.predict_proba(text_vec)[0][prediction]
        sentiment = "Positive" if prediction == 1 else "Negative"

        return {
            "sentiment": sentiment,
            "confidence": round(float(confidence), 4)
        }
    except Exception as e:
        return {"error": str(e)}

# For local dev only
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
