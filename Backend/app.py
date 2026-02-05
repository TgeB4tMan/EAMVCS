from speaker import get_speaker_embedding
from emotion import get_emotion_embedding
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os

# Create FastAPI app
app = FastAPI()

# Allow frontend (Vite / React)
origins = [
    "http://localhost:5173",
]

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

# Test endpoint
@app.get("/")
def read_root():
    return {"message": "Backend is running"}
@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    language: str = Form(...),
    audio: UploadFile = File(...)
):
    # Step 1: Log inputs
    print("Text received:", text)
    print("Language received:", language)
    print("Audio received:", audio.filename)

    # Step 2: Save audio
    audio_path = os.path.join(UPLOAD_DIR, audio.filename)
    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    print("Audio saved at:", audio_path)

    # STEP 3: Speaker embedding
    speaker_embedding = get_speaker_embedding(audio_path)

    # STEP 3: Emotion embedding
    emotion_embedding = get_emotion_embedding(audio_path)

    print("Speaker embedding length:", len(speaker_embedding))
    print("Emotion embedding length:", len(emotion_embedding))

    # Step 4: Return response
    return {
        "status": "success",
        "message": "Audio processed and embeddings extracted",
        "language": language,
        "audio_file": audio.filename
    }

