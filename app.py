import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_SSR_MODE"] = "False"
 
# Patch spaces reloader bug before anything loads
import unittest.mock as mock
import sys
 
# Block the broken spaces reloading module entirely
sys.modules["spaces.reloading"] = mock.MagicMock()
sys.modules["spaces.reloading.server"] = mock.MagicMock()

import gradio as gr
import re
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model_dense  = load_model("dense_model.h5")
model_lstm   = load_model("lstm_model.h5")
model_bilstm = load_model("bilstm_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict(review):
    if not review.strip():
        return "⚠️ Please enter a review!", "", ""

    cleaned = clean_text(review)
    seq     = tokenizer.texts_to_sequences([cleaned])
    padded  = pad_sequences(seq, maxlen=200)

    p1 = float(model_dense.predict(padded)[0][0])
    p2 = float(model_lstm.predict(padded)[0][0])
    p3 = float(model_bilstm.predict(padded)[0][0])

    avg = (p1 + p2 + p3) / 3

    sentiment  = "🟢 Positive 😊" if avg > 0.5 else "🔴 Negative 😞"
    confidence = f"{round(avg * 100, 2)}%" if avg > 0.5 else f"{round((1 - avg) * 100, 2)}%"

    breakdown = (
        f"Dense   → {'Positive' if p1 > 0.5 else 'Negative'} ({round(p1*100,1)}%)\n"
        f"LSTM    → {'Positive' if p2 > 0.5 else 'Negative'} ({round(p2*100,1)}%)\n"
        f"BiLSTM  → {'Positive' if p3 > 0.5 else 'Negative'} ({round(p3*100,1)}%)\n"
        f"--------------------\n"
        f"Ensemble → {sentiment} ({confidence})"
    )

    return sentiment, confidence, breakdown

with gr.Blocks() as demo:

    gr.Markdown("# 🎬 IMDb Movie Sentiment Analysis")


    with gr.Row():
        with gr.Column():
            review_input = gr.Textbox(
                lines=5,
                placeholder="Enter your movie review here...",
                label="🎥 Movie Review"
            )
            predict_btn = gr.Button("🔍 Predict Sentiment")

        with gr.Column():
            sentiment_out  = gr.Textbox(label="Final Sentiment")
            confidence_out = gr.Textbox(label="Confidence Score")
            breakdown_out  = gr.Textbox(label="Model-wise Breakdown", lines=6)

    predict_btn.click(
        fn=predict,
        inputs=review_input,
        outputs=[sentiment_out, confidence_out, breakdown_out]
    )

    gr.Markdown("---")

demo.launch(share=False, show_error=True, ssr_mode = False)