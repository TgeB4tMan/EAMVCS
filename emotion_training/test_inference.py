import os
import sys

# Add Backend to path so we can test the detector
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Backend'))
from emotion_detector import predict_emotion

# The RAVDESS dataset contains numbers corresponding to:
# 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
# We mapped: Neutral (1,2), Happy (3), Sad (4), Angry (5).

test_files = [
    # Neutral
    (r"C:\Users\Baevin\Desktop\MiniProject\EAMVCS\Data\archive\audio_speech_actors_01-24\Actor_01\03-01-01-01-01-01-01.wav", "Neutral"),
    # Happy
    (r"C:\Users\Baevin\Desktop\MiniProject\EAMVCS\Data\archive\audio_speech_actors_01-24\Actor_01\03-01-03-01-01-01-01.wav", "Happy"),
    # Sad
    (r"C:\Users\Baevin\Desktop\MiniProject\EAMVCS\Data\archive\audio_speech_actors_01-24\Actor_01\03-01-04-01-01-01-01.wav", "Sad"),
    # Angry
    (r"C:\Users\Baevin\Desktop\MiniProject\EAMVCS\Data\archive\audio_speech_actors_01-24\Actor_01\03-01-05-01-01-01-01.wav", "Angry")
]

print("Running Inference Test on 4 Distinct Emotional Wav Files...\n")

for filepath, expected in test_files:
    if os.path.exists(filepath):
        res = predict_emotion(filepath)
        print(f"File: {os.path.basename(filepath)}")
        print(f"Expected: {expected}")
        print(f"Predicted: {res['emotion']} (Confidence: {res['confidence']:.2f}%)\n")
    else:
        print(f"File not found: {filepath}")
