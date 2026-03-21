# NeuroVoice (EAMVCS)
Emotion-Aware Multilingual Voice Cloning System

## 🚀 Overview
NeuroVoice is a professional-grade voice cloning system that preserves the emotional state of the speaker across 12 languages.

### Key Features:
* **Emotion CNN**: 4-class ResNet classifier (Neutral, Happy, Sad, Angry) with real-world noise robustness.
* **Emotion-Speaker Fusion**: Dynamic injection of emotion embeddings into speaker identity space.
* **Prosody Control**: Automated speed and pitch adjustments based on detected emotion.
* **Active Learning**: Built-in feedback loop to improve accuracy over time.

---

## 📂 Project Structure
* `Backend/`: Consolidated API, TTS, and Machine Learning logic.
* `Frontend/`: Modern UI using responsive design.
* `emotion_training/`: Robust ResNet training pipeline with noise augmentation.
* `Data/`: Reference audio datasets (RAVDESS).
* `user_feedback_data/`: Stores user-corrected labels for retraining.

---

## 🛠️ Setup & Usage

### 1. Environment Setup
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start the Backend
```bash
python -m Backend.app
```
*API will run on: http://localhost:8000*

### 3. Start the Frontend
Simply open `Frontend/index.html` in any modern browser.

---

## 📈 Improving Accuracy
To retrain the model with new noise-robust features:
1. Navigate to `emotion_training/`
2. Run `python train.py`
3. The best model will be saved as `emotion_model.pth` and automatically used by the Backend.

