"""
preprocessing.py

This module handles all audio preprocessing steps required before
feature extraction and TTS generation.

Responsibilities:
- Convert input audio (mp3/wav/other formats) to 16kHz mono WAV
- Apply noise reduction
- Normalize waveform amplitude

Output:
- Cleaned WAV file ready for embedding extraction and TTS cloning
"""

import os
import librosa
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment

# Target sampling rate required by most speech models
TARGET_SR = 16000


def preprocess(input_path: str, output_path: str = "cleaned_audio.wav") -> str:
    """
    Preprocess an input audio file.

    Steps performed:
    1. Convert audio to WAV format
    2. Resample to 16kHz
    3. Convert to mono channel
    4. Apply noise reduction
    5. Normalize amplitude

    Args:
        input_path (str): Path to the input audio file.
        output_path (str): Path where cleaned audio will be saved.

    Returns:
        str: Path to the cleaned WAV file.
    """

    # Convert input audio to 16kHz mono WAV
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(TARGET_SR).set_channels(1)
    audio.export("temp.wav", format="wav")

    # Load waveform using librosa
    y, sr = librosa.load("temp.wav", sr=TARGET_SR)

    # Reduce background noise
    y = nr.reduce_noise(y=y, sr=sr)

    # Normalize waveform amplitude
    y = librosa.util.normalize(y)

    # Save cleaned audio
    sf.write(output_path, y, sr)

    # Remove temporary file
    os.remove("temp.wav")

    return output_path
