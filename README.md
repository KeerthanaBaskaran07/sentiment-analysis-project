# 🎬 IMDb Movie Sentiment Analysis (Ensemble Model)

This project performs sentiment analysis on movie reviews using deep learning models and an ensemble approach.

---

## 🔍 Models Used

- Dense Neural Network (Global Average Pooling)
- LSTM (Long Short-Term Memory)
- Bidirectional LSTM (BiLSTM)

---

## 🧠 Ensemble Approach ⭐

Instead of relying on a single model, this project combines predictions from all three models:

- Each model generates a probability score
- Final prediction is calculated using **average probability**
- Improves robustness and stability of predictions

---

## 📊 Results

- Individual models achieved ~86–88% accuracy
- Ensemble provides more stable predictions across diverse inputs
- Handles ambiguous sentences better than single models

---

## 🚀 Deployment

- Deployed using Hugging Face Spaces
- Built with Gradio UI
- Real-time prediction with model-wise breakdown

---

## 💡 Features

- ✅ Real-time sentiment prediction  
- ✅ Confidence score display  
- ✅ Model-wise prediction breakdown  
- ✅ Ensemble-based final decision  

---

## 🧪 Example Output
