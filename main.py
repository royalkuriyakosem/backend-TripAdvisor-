from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# Initialize FastAPI app (only once)
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Define input format
class Review(BaseModel):
    text: str

@app.post("/predict")
async def predict_sentiment(review: Review):
    # Vectorize the input text
    text_vec = vectorizer.transform([review.text])
    
    # Predict sentiment and probability
    prediction = model.predict(text_vec)[0]
    probabilities = model.predict_proba(text_vec)[0]
    
    # Map prediction to sentiment
    sentiment = "Positive" if prediction == 1 else "Negative"
    # Confidence is the max probability for the predicted class
    confidence = float(probabilities[prediction])
    
    return {"sentiment": sentiment, "confidence": confidence}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)