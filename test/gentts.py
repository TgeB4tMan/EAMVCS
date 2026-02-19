"""
gentts.py

FastAPI backend for Emotion-Aware Multimodal Voice Cloning System (EAMVCS).

Pipeline:
1. Receive audio + text from user
2. Preprocess audio
3. Extract speaker embedding
4. Extract emotion embedding
5. Generate new speech using XTTS voice cloning
6. Return detected emotion + generated audio

This module acts as the main controller of the system.
"""

import os
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from TTS.api import TTS

from preprocessing import preprocess
from extract import extract_speaker_embedding, extract_emotion_embedding

# -------------------------------------------------------------------
# FastAPI Application Initialization
# -------------------------------------------------------------------

app = FastAPI(
    title="Emotion Aware Multimodal Voice Cloning System",
    description="Backend service for cloning voice and preserving emotional tone."
)

# -------------------------------------------------------------------
# Model Initialization (Loaded Once for Performance)
# -------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

# XTTS v2 multilingual voice cloning model
tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2"
).to(device)


# -------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------

@app.post("/clone-voice/")
async def clone_voice(
    file: UploadFile = File(...),
    text: str = Form(...)
):
    """
    Clone a voice with preserved emotion.

    Inputs:
        file: Audio file containing reference voice
        text: New dialogue to synthesize

    Returns:
        JSON response containing:
        - detected emotion
        - generated audio filename
    """

    # Save uploaded file locally
    input_path = f"input_{file.filename}"
    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())

    # Step 1: Audio Preprocessing
    cleaned_audio = preprocess(input_path)

    # Step 2: Extract Speaker & Emotion Embeddings
    extract_speaker_embedding(cleaned_audio)
    emotion_data = extract_emotion_embedding(cleaned_audio)

    # Step 3: Generate Cloned Speech
    output_file = "generated_output.wav"

    tts.tts_to_file(
        text=text,
        speaker_wav=cleaned_audio,
        file_path=output_file
    )

    # Clean up uploaded file
    os.remove(input_path)

    return {
        "detected_emotion": emotion_data["emotion"],
        "audio_file": output_file
    }


@app.get("/download/")
def download_audio():
    """
    Download the generated cloned audio file.
    """
    return FileResponse("generated_output.wav", media_type="audio/wav")