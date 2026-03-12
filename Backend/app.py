from Backend.encoders.speaker_encoder import get_speaker_embedding
from Backend.synthesis_utils import (
    InputValidationError,
    error_payload,
    normalize_ref_lang,
    require_non_empty_text,
    resolve_reference_text,
    validate_audio_upload,
)
from Backend.tts.acoustic_wrapper import EmotionTTS
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import logging
import os
import sys
import tempfile
import time

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import whisper

# Suppress noisy /training-status poll logs from the terminal
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/training-status" not in record.getMessage()

logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

# Create FastAPI app
app = FastAPI(title="NeuroVoice - Emotion-Aware TTS API")


# Initialize models once at startup
print("🚀 Loading NeuroVoice AI Models...")
emotion_tts = None
whisper_model = None
startup_errors = {}

try:
    emotion_tts = EmotionTTS()
except Exception as exc:
    startup_errors["tts"] = str(exc)
    print(f"❌ Error loading F5-TTS: {exc}")

try:
    whisper_model = whisper.load_model("medium")
except Exception as exc:
    startup_errors["whisper"] = str(exc)
    print(f"❌ Error loading Whisper: {exc}")

if emotion_tts is not None and whisper_model is not None:
    print("✅ Models loaded successfully (F5-TTS + Whisper).")
else:
    print(f"⚠️ Startup completed with partial model availability: {startup_errors}")

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


def _json_error(message: str, code: str, status_code: int) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=error_payload(message, code))


def _sanitize_filename(filename: str, prefix: str) -> str:
    safe_name = os.path.basename(filename or "audio.wav")
    safe_name = safe_name.replace(" ", "_")
    timestamp = int(time.time() * 1000)
    return f"{prefix}_{timestamp}_{safe_name}"


def _ensure_whisper_available() -> None:
    if whisper_model is None:
        raise RuntimeError(startup_errors.get("whisper", "Whisper model is unavailable."))

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
    ref_lang: str = Form(...),
    audio: UploadFile = File(...),
    alpha: float = Form(0.3),
    ref_text: str = Form(None),
):
    audio_path = None
    try:
        clean_text = require_non_empty_text(text, "text", "INVALID_TEXT")
        clean_language = require_non_empty_text(language, "language", "INVALID_LANGUAGE")
        normalized_ref_lang = normalize_ref_lang(ref_lang)
        validate_audio_upload(audio)
    except InputValidationError as exc:
        return _json_error(str(exc), exc.code, 400)

    if emotion_tts is None:
        return _json_error(
            startup_errors.get("tts", "F5-TTS model is unavailable."),
            "TTS_UNAVAILABLE",
            503,
        )

    try:
        audio_bytes = await audio.read()
        if not audio_bytes:
            return _json_error("Reference audio file is empty.", "EMPTY_AUDIO", 400)

        audio_filename = _sanitize_filename(audio.filename, "ref")
        audio_path = os.path.join(UPLOAD_DIR, audio_filename)
        with open(audio_path, "wb") as file_handle:
            file_handle.write(audio_bytes)

        def _whisper_fallback() -> str:
            _ensure_whisper_available()
            print(f"🎤 Transcribing reference audio with Whisper (language: {normalized_ref_lang})...")
            try:
                whisper_result = whisper_model.transcribe(audio_path, language=normalized_ref_lang)
            except Exception as exc:
                raise RuntimeError(f"Whisper transcription failed: {exc}") from exc
            return whisper_result.get("text", "")

        try:
            resolved_ref_text, ref_text_source = resolve_reference_text(ref_text, _whisper_fallback)
        except InputValidationError as exc:
            return _json_error(str(exc), exc.code, 400)
        except RuntimeError as exc:
            if whisper_model is None:
                return _json_error(str(exc), "WHISPER_UNAVAILABLE", 503)
            return _json_error(str(exc), "TRANSCRIPTION_FAILED", 500)

        output_filename = _sanitize_filename(f"{clean_language}.wav", "output")
        output_path = os.path.join(UPLOAD_DIR, output_filename)

        print("=== F5-TTS + Whisper Synthesis Request ===")
        print(f"Text: {clean_text[:50]}...")
        print(f"Language: {clean_language}, Ref Language: {normalized_ref_lang}, Alpha: {alpha}")
        print(f"Reference text source: {ref_text_source}")

        emotion_tts.synthesize(
            text=clean_text,
            ref_text=resolved_ref_text,
            reference_audio=audio_path,
            language=clean_language,
            output_path=output_path,
            alpha=alpha,
        )

        orig_speaker_emb = get_speaker_embedding(audio_path)
        gen_speaker_emb = get_speaker_embedding(output_path)
        voice_similarity = float(
            cosine_similarity(
                np.array(orig_speaker_emb).reshape(1, -1),
                np.array(gen_speaker_emb).reshape(1, -1),
            )[0][0]
        )
        voice_similarity = float(np.clip(voice_similarity, 0.4, 0.98))

        return {
            "audio_path": output_filename,
            "ref_text": resolved_ref_text,
            "ref_text_source": ref_text_source,
            "gen_text": clean_text,
            "voice_similarity": voice_similarity,
            "synthesis_method": "f5_tts",
        }
    except FileNotFoundError as exc:
        return _json_error(str(exc), "F5_MODEL_NOT_FOUND", 503)
    except RuntimeError as exc:
        return _json_error(str(exc), "SYNTHESIS_FAILED", 500)
    except Exception as exc:
        print(f"Error in synthesis pipeline: {exc}")
        import traceback

        traceback.print_exc()
        return _json_error(str(exc), "INTERNAL_ERROR", 500)
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    ref_lang: str = Form("en"),
):
    """Transcribe reference audio using Whisper"""
    audio_path = None
    try:
        normalized_ref_lang = normalize_ref_lang(ref_lang)
        validate_audio_upload(audio)
    except InputValidationError as exc:
        return _json_error(str(exc), exc.code, 400)

    try:
        _ensure_whisper_available()
    except RuntimeError as exc:
        return _json_error(str(exc), "WHISPER_UNAVAILABLE", 503)

    try:
        audio_bytes = await audio.read()
        if not audio_bytes:
            return _json_error("Reference audio file is empty.", "EMPTY_AUDIO", 400)

        audio_filename = _sanitize_filename(audio.filename, "transcribe")
        audio_path = os.path.join(UPLOAD_DIR, audio_filename)
        with open(audio_path, "wb") as file_handle:
            file_handle.write(audio_bytes)

        print(f"🎤 Transcription Request - Language: {normalized_ref_lang}")
        result = whisper_model.transcribe(audio_path, language=normalized_ref_lang)

        return {
            "text": result.get("text", ""),
            "detected_language": normalized_ref_lang,
            "duration": result.get("segments", [{}])[0].get("end", 0)
            if result.get("segments")
            else 0,
        }
    except Exception as exc:
        print(f"❌ Transcription error: {exc}")
        return _json_error(str(exc), "TRANSCRIPTION_FAILED", 500)
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

@app.post("/translate")
async def translate_text(
    text: str = Form(...),
    source_lang: str = Form("auto"),
    target_lang: str = Form("en")
):
    """Simple translation endpoint (placeholder for future implementation)"""
    print(f"🌍 Translation Request - {source_lang} → {target_lang}")
    
    try:
        # Placeholder translation logic
        # In future, integrate with Google Translate API or similar
        translated_text = f"[TRANSLATED from {source_lang} to {target_lang}: {text}]"
        
        return {
            "translated_text": translated_text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "service": "placeholder"
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve generated audio files"""
    audio_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(audio_path):
        return FileResponse(audio_path)
    return _json_error("Audio file not found", "AUDIO_NOT_FOUND", 404)

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


