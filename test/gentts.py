import numpy as np
from TTS.api import TTS
import torch

def generate_speech(text, reference_audio):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    tts.tts_to_file(
        text=text,
        speaker_wav=reference_audio,
        file_path="generated_output.wav"
    )

    print("Speech generated as generated_output.wav")

if __name__ == "__main__":
    text = input("Enter new dialogue: ")
    ref_audio = input("Enter cleaned wav path: ")
    generate_speech(text, ref_audio)
#hmmm