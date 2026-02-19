import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
from pydub import AudioSegment
import os

def convert_to_wav(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_path, format="wav")

def clean_audio(input_wav, output_wav):
    y, sr = librosa.load(input_wav, sr=16000)
    
    # Noise reduction
    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    
    # Normalize
    normalized_audio = librosa.util.normalize(reduced_noise)
    
    sf.write(output_wav, normalized_audio, sr)

def preprocess(input_file):
    temp_wav = "temp.wav"
    final_wav = "cleaned_audio.wav"
    
    convert_to_wav(input_file, temp_wav)
    clean_audio(temp_wav, final_wav)
    
    os.remove(temp_wav)
    print("Preprocessing complete. Output:", final_wav)
    return final_wav

if __name__ == "__main__":
    file_path = input("Enter audio file path: ")
    preprocess(file_path)
