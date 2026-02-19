import numpy as np
import torch
import librosa
from resemblyzer import VoiceEncoder, preprocess_wav
from speechbrain.inference.interfaces import foreign_class

def extract_speaker_embedding(wav_path):
    wav = preprocess_wav(wav_path)
    encoder = VoiceEncoder()
    embedding = encoder.embed_utterance(wav)
    np.save("speaker_embedding.npy", embedding)
    print("Speaker embedding saved.")
    return embedding

def extract_emotion_embedding(wav_path):
    classifier = foreign_class(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier",
    )
    
    out_prob, score, index, text_lab = classifier.classify_file(wav_path)
    
    emotion_embedding = out_prob.detach().numpy()
    np.save("emotion_embedding.npy", emotion_embedding)
    
    print("Emotion detected:", text_lab)
    return emotion_embedding

if __name__ == "__main__":
    path = input("Enter cleaned wav path: ")
    extract_speaker_embedding(path)
    extract_emotion_embedding(path)

#this is a test