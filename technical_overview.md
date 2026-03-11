# 🎙️ NeuroVoice: Technical Project Overview

## **1. Project Mission**
NeuroVoice is an **Emotion-Aware Multilingual Text-to-Speech (TTS)** system. Unlike standard TTS that sounds robotic and flat, NeuroVoice uses Deep Learning to detect the emotional state of a reference voice and "transfer" that emotion into a synthesized clone.

## **2. The Machine Learning Model (The Brain)**
We developed a custom **Convolutional Neural Network (CNN)** specifically for Audio Emotion Recognition.
*   **Architecture**: 4-layer 2D CNN with Batch Normalization (for stability) and Dropout (to prevent overfitting).
*   **Input**: Log-Mel Spectrograms (visual representations of sound frequencies over time).
*   **Dataset**: Trained on the **RAVDESS** dataset (24 professional actors) combined with **Real-time User Feedback data**.
*   **Training**: The model underwent **50 Epochs** of training (~16 hours total), using **Data Augmentation** techniques like pitch shifting and noise injection to make it robust to different microphones.

## **3. How it Works (The Pipeline)**
1.  **Emotion Extraction**: The CNN analyzes the upload and generates a **128-dimensional Emotion Embedding**. This is a mathematical "fingerprint" of the user's mood.
2.  **VAD Analysis**: The system maps the voice onto three metrics:
    *   **Valence**: Positive vs. Negative vibes.
    *   **Arousal**: Energy and Volume levels.
    *   **Dominance**: Authority and Control.
3.  **Embedding Fusion**: We "mix" the emotion embedding with the **Speaker Identity Embedding** using a weighted fusion layer.
4.  **Prosody Transformation**: Based on the detected emotion, we dynamically adjust the **Speed, Pitch, and Pauses** of the voice to match human speech patterns.

## **4. Innovation: Active Learning**
Our project features a **Human-in-the-Loop Feedback System**. When a user corrects a misclassified emotion on the frontend, the system saves that audio for **Retraining**. This allows the AI to learn from its mistakes and adapt to individual users in real-time.

## **5. Technology Stack**
*   **Backend**: Python, FastAPI, PyTorch (Deep Learning), Torchaudio.
*   **Frontend**: HTML5, Vanilla CSS (Glassmorphism), JavaScript (Async API interaction).
*   **TTS Engine**: YourTTS Multilingual Backbone.
