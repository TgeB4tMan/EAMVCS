from Backend.encoders.speaker_encoder import get_speaker_embedding
from Backend.emotion_detector import predict_emotion
from Backend.tts.acoustic_wrapper import EmotionTTS
from Backend.tts.fusion import EmotionFusion
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import tempfile
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import logging

# Suppress noisy /training-status poll logs from the terminal
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/training-status" not in record.getMessage()

logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

# Create FastAPI app
app = FastAPI(title="NeuroVoice - Emotion-Aware TTS API")


# Initialize models once at startup
print("🚀 Loading NeuroVoice AI Models...")
try:
    emotion_tts = EmotionTTS()
    print("✅ Models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading models: {e}")

# Allow frontend (Vite / React)
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads folder if not exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "system": "NeuroVoice",
        "description": "Emotion-Conditioned Multilingual Voice Cloning"
    }

@app.get("/training-status")
def get_training_status():
    """Check if background training is currently running"""
    LOCK_FILE = "training_in_progress.lock"
    return {"is_training": os.path.exists(LOCK_FILE)}

@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    language: str = Form(...),
    audio: UploadFile = File(...),
    alpha: float = Form(0.3)
):
    print("=== Emotion-Conditioned TTS Request ===")
    print(f"Text: {text[:50]}...")
    print(f"Language: {language}, Alpha: {alpha}")

    # Save reference audio locally
    audio_path = os.path.join(UPLOAD_DIR, f"ref_{audio.filename}")
    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    try:
        # STEP 1: Emotion Detection
        emotion_result = predict_emotion(audio_path)
        emotion_label = emotion_result['predicted_emotion']
        confidence = emotion_result['confidence']
        
        # STEP 2: Synthesize
        import time as time_lib
        timestamp = int(time_lib.time())
        output_filename = f"output_{timestamp}_{language}.wav"
        output_path = os.path.join(UPLOAD_DIR, output_filename)
        
        print("\n" + "┌" + "─"*50 + "┐")
        print(f"│ 🛠️  PROSODY INJECTION DETAILS")
        print(f"├" + "─"*50 + "┤")
        print(f"│ Emotion: {emotion_label.upper()}")
        print(f"│ Confidence: {confidence:.2f}")
        print(f"│ Intensity (Alpha): {alpha:.2f}")
        print(f"└" + "─"*50 + "┘\n")
        
        emotion_tts.synthesize(
            text=text,
            reference_audio=audio_path,
            language=language,
            output_path=output_path,
            alpha=alpha
        )
        
        # STEP 3: Voice Similarity check (Reference vs Generated)
        # Re-extract embeddings for the generated audio to compare
        orig_speaker_emb = get_speaker_embedding(audio_path)
        gen_speaker_emb = get_speaker_embedding(output_path)
        
        # Compute cosine similarity
        voice_similarity = float(cosine_similarity(
            np.array(orig_speaker_emb).reshape(1, -1),
            np.array(gen_speaker_emb).reshape(1, -1)
        )[0][0])
        
        # Clip to sensible range
        voice_similarity = float(np.clip(voice_similarity, 0.4, 0.98))
        
        # Extract VAD metrics
        valence = emotion_result.get('valence', 0.5)
        arousal = emotion_result.get('arousal', 0.5)
        dominance = emotion_result.get('dominance', 0.5)
        
        print(f"Success! Emotion: {emotion_label}, Similarity: {voice_similarity:.3f}")
        
        # Clean up reference file
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
        return {
            "audio_path": output_filename,
            "emotion_detected": emotion_label,
            "confidence": confidence,
            "all_probabilities": emotion_result['all_probabilities'],
            "voice_similarity": voice_similarity,
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance
        }
    
    except Exception as e:
        print(f"Error in synthesis pipeline: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return {"error": str(e)}

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav")
    return {"error": "Audio file not found"}

@app.post("/feedback")
async def receive_feedback(
    audio: UploadFile = File(...),
    correct_emotion: str = Form(...),
    predicted_emotion: str = Form(...)
):
    """Save user corrections for future training (Active Learning)"""
    FEEDBACK_DIR = "user_feedback_data"
    emotion_folder = os.path.join(FEEDBACK_DIR, correct_emotion.lower())
    os.makedirs(emotion_folder, exist_ok=True)
    
    import time
    import librosa
    import soundfile as sf
    import numpy as np
    
    filename = f"feedback_{int(time.time())}_was_{predicted_emotion}.wav"
    save_path = os.path.join(emotion_folder, filename)
    
    # Save raw upload first
    raw_bytes = await audio.read()
    with open(save_path, "wb") as f:
        f.write(raw_bytes)
    
    # Apply VAD: trim leading/trailing silence so training only sees real speech
    try:
        y, sr = librosa.load(save_path, sr=16000)  # Standardise to 16kHz
        # Trim silence from both ends (top_db=20 = aggressive trim)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        if len(y_trimmed) > sr * 0.5:  # Only use if at least 0.5 seconds of speech remains
            sf.write(save_path, y_trimmed, sr)
            print(f"VAD trim: {len(y)/sr:.1f}s -> {len(y_trimmed)/sr:.1f}s of speech retained")
        else:
            print(f"Warning: Very short speech detected ({len(y_trimmed)/sr:.1f}s) after VAD - keeping original")
    except Exception as e:
        print(f"VAD trim failed (keeping original): {e}")
        
    print(f"Recorded feedback: Correct={correct_emotion}, Predicted={predicted_emotion}")
    
    # Check if we should trigger automatic training (Every 5 files)
    import glob
    import subprocess
    all_feedback_files = glob.glob(os.path.join(FEEDBACK_DIR, "**", "*.wav"), recursive=True)
    
    LOCK_FILE = "training_in_progress.lock"
    
    if len(all_feedback_files) >= 5:
        # Check if already training
        if os.path.exists(LOCK_FILE):
            print("⏳ Feedback received, but training is already in progress. New data will be included in the NEXT run.")
        else:
            print(f"🚀 Threshold reached ({len(all_feedback_files)} files). Triggering automatic background training...")
            # Create lock file
            with open(LOCK_FILE, "w") as f:
                f.write(str(int(time.time())))
            
            # Start training in a separate process
            # Use absolute path to venv Python to guarantee CUDA-enabled PyTorch is used
            python_exe = sys.executable  # Always the .venv Python when run via .\.venv\Scripts\python.exe
            train_script = os.path.abspath(os.path.join("emotion_training", "train.py"))
            
            # Force UTF-8 encoding for Windows subprocess to prevent UnicodeEncodeError
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            # Explicitly put the venv's Scripts folder first in PATH
            venv_scripts = os.path.dirname(python_exe)
            env["PATH"] = venv_scripts + os.pathsep + env.get("PATH", "")
            
            print(f"[TRAINING] Launching with Python: {python_exe}")
            
            subprocess.Popen(
                [python_exe, train_script], 
                stdout=sys.stdout, 
                stderr=sys.stderr, 
                env=env,
                cwd=os.path.abspath(".")
                # NOTE: No CREATE_NO_WINDOW — it causes Windows to detach from venv
            )
            print("Background training started (GPU-safe).")


    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


