from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os

app = FastAPI()

# Check if model files exist before loading
if not os.path.exists("D:\\modelss\\spam_model.h5") or not os.path.exists("D:\\modelss\\Spam_tokenizer2.pkl"):
    raise FileNotFoundError("Spam model or tokenizer file not found.")

if not os.path.exists("D:\\modelss\\Content_model.h5") or not os.path.exists("D:\\modelss\\content_tokenizer2.pkl"):
    raise FileNotFoundError("Content model or tokenizer file not found.")

# Load models and tokenizers
spam_model = tf.keras.models.load_model("D:\\modelss\\spam_model.h5")
with open('D:\\modelss\\Spam_tokenizer2.pkl', 'rb') as f:
    spam_tokenizer = pickle.load(f)

content_model = tf.keras.models.load_model("D:\\modelss\\Content_model.h5")
with open('D:\\modelss\\content_tokenizer2.pkl', 'rb') as f:
    content_tokenizer = pickle.load(f)


def preprocess_text(text: str, tokenizer, max_len: int = 100) -> np.ndarray:
    sequence = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=max_len, padding='post')

class TextInput(BaseModel):
    spam_text: str
    content_text: str


@app.get("/")
def read_root():
    return {"message": "API is working. Go to /docs to test the endpoints."}


@app.post("/predict/")
async def predict(input_data: TextInput):
    try:
        spam_text = input_data.spam_text
        content_text = input_data.content_text

        # Spam Prediction
        spam_processed = preprocess_text(spam_text, spam_tokenizer)
        spam_prediction = spam_model.predict(spam_processed)[0][0].item()
        spam_prediction = 1 if spam_prediction >= 0.5 else 0
        spam_label = "Fake" if spam_prediction >= 0.5 else "Real"

        # Content Prediction
        content_processed = preprocess_text(content_text, content_tokenizer)
        content_prediction = content_model.predict(content_processed)[0][0].item()
        content_prediction = 1 if content_prediction >= 0.5 else 0
        content_label = "Bad review" if content_prediction >= 0.5 else "Good review"

        return {
            "spam_prediction": spam_prediction,
            "spam_label": spam_label,
            "content_prediction": content_prediction,
            "content_label": content_label
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
