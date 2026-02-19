import torch
from TTS.api import TTS

# Importing our modules
from preprocessing import preprocess
from extract import extract_speaker_embedding, extract_emotion_embedding


def run_pipeline(input_audio_path, new_text):
    print("\n--- STEP 1: PREPROCESSING ---")
    cleaned_audio_path = preprocess(input_audio_path)

    print("\n--- STEP 2: EXTRACTING EMBEDDINGS ---")
    speaker_embedding = extract_speaker_embedding(cleaned_audio_path)
    emotion_embedding = extract_emotion_embedding(cleaned_audio_path)

    print("\n--- STEP 3: GENERATING SPEECH ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    tts.tts_to_file(
        text=new_text,
        speaker_wav=cleaned_audio_path,
        file_path="generated_output.wav"
    )

    print("\nPipeline complete.")
    print("Generated file: generated_output.wav")


if __name__ == "__main__":
    input_audio = input("Enter original audio file path (mp3/wav): ")
    text = input("Enter new dialogue text: ")
    run_pipeline(input_audio, text)