"""
extract.py

This module extracts:
1. Speaker embeddings (voice identity representation)
2. Emotion embeddings (emotional tone representation)

Models are loaded once at startup for performance optimization.
Embeddings are saved as .npy files for reuse or analysis.
"""

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from speechbrain.inference.interfaces import foreign_class

# -------------------------------------------------------------------
# Model Initialization (Loaded Once for Efficiency)
# -------------------------------------------------------------------

# Speaker embedding model (Resemblyzer)
speaker_encoder = VoiceEncoder()

# Emotion classification model (SpeechBrain)
emotion_classifier = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier",
)


def extract_speaker_embedding(wav_path: str):
    """
    Extract speaker embedding from an audio file.

    Speaker embedding represents the unique voice characteristics
    of a person.

    Args:
        wav_path (str): Path to cleaned WAV file.

    Returns:
        numpy.ndarray: 256-dimensional speaker embedding vector.
    """

    wav = preprocess_wav(wav_path)
    embedding = speaker_encoder.embed_utterance(wav)

    # Save embedding to file
    np.save("speaker_embedding.npy", embedding)

    return embedding


def extract_emotion_embedding(wav_path: str):
    """
    Extract emotion embedding from an audio file.

    Emotion embedding captures emotional characteristics such as:
    happy, sad, angry, neutral, etc.

    Args:
        wav_path (str): Path to cleaned WAV file.

    Returns:
        dict: Contains:
              - detected emotion label
              - emotion embedding vector (as list)
    """

    out_prob, score, index, text_lab = emotion_classifier.classify_file(wav_path)
    emotion_embedding = out_prob.detach().cpu().numpy()

    # Save embedding to file
    np.save("emotion_embedding.npy", emotion_embedding)

    return {
        "emotion": text_lab,
        "embedding": emotion_embedding.tolist()
    }
