import torch
import numpy as np
from TTS.api import TTS

# Updated imports for consolidated structure
from Backend.encoders.speaker_encoder import get_speaker_embedding
from Backend.emotion_detector import predict_emotion
from Backend.tts.fusion import EmotionFusion
from Backend.tts.prosody import emotion_to_prosody, apply_emotional_pauses, post_process_audio
from Backend.tts.multilingual import text_to_phonemes


class EmotionTTS:
    def __init__(self, model_name="tts_models/multilingual/multi-dataset/xtts_v2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load YourTTS backbone
        self.tts = TTS(model_name=model_name, gpu=torch.cuda.is_available())
        
        # Load fusion module
        self.fusion = EmotionFusion().to(self.device)

    def synthesize(
        self,
        text,
        reference_audio,
        language="en",
        output_path="output.wav",
        alpha=0.3
    ):
        # --- Speaker embedding (256-d) ---
        speaker_emb = get_speaker_embedding(reference_audio)
        speaker_emb = torch.tensor(speaker_emb, dtype=torch.float32).to(self.device).unsqueeze(0)

        # --- Use trained ML model for emotion detection ---
        emotion_result = predict_emotion(reference_audio)
        
        if not emotion_result or 'predicted_emotion' not in emotion_result:
            print(f"Warning: Could not detect emotion, using neutral")
            emotion_name = "neutral"
            confidence = 0.6
            adjusted_emb = speaker_emb
        else:
            emotion_name = emotion_result['predicted_emotion']
            confidence = emotion_result['confidence'] / 100.0
            
            # Get real 128-dim emotion embedding
            emotion_emb = torch.tensor(emotion_result['embedding'], dtype=torch.float32).to(self.device).unsqueeze(0)
            
            # IDENTITY PROTECTION: Scale down emotion fusion significantly
            # Even at high alpha, we don't want to lose the teacher's voice
            subtle_alpha = alpha * 0.4  # Reduce the warping effect on identity
            adjusted_emb = self.fusion(speaker_emb, emotion_emb, subtle_alpha)

        # --- High Fidelity Check: Pure Clone Mode ---
        # If alpha is very low, we skip all prosody warping for absolute identity accuracy
        if alpha < 0.1:
            print("🚀 PURE CLONE MODE: Bypassing emotional prosody for 100% identity accuracy.")
            prosody = {"speed": 1.0, "pitch": 1.0, "volume": 1.0}
            processed_text = text
            phoneme_text = text # Keep text original to prevent phoneme-based accent shift
        else:
            # --- Apply emotion to prosody (RE-ENABLED) ---
            prosody = emotion_to_prosody(emotion_name, confidence, alpha)
            processed_text = apply_emotional_pauses(text, emotion_name, alpha)
            phoneme_text = text_to_phonemes(processed_text, language)
        
        print(f"\n{'='*60}")
        print(f"🔊 PROSODY DEBUG:")
        print(f"   Speed (length scale) = {prosody['speed']:.3f}")
        print(f"   Pitch factor         = {prosody['pitch']:.3f}")
        print(f"   Volume factor        = {prosody['volume']:.3f}")
        print(f"   Original text: '{text[:60]}...'")
        print(f"   Processed text: '{processed_text[:60]}...'")
        print(f"   Phoneme text: '{phoneme_text[:60]}...'")
        print(f"{'='*60}\n")

        # --- Generate speech with prosody control ---
        # XTTS-v2 handles speaker_wav directly and is much more accurate for identity.
        # We pass the reference_audio (cleaned) and the calculated language.
        
        try:
            # Note: XTTS-v2 speed control is via the 'speed' parameter
            # It also supports emotion conditioning if we pass 'emotion' (though our prosody layer is more fine-tuned)
            self.tts.tts_to_file(
                text=phoneme_text,
                speaker_wav=reference_audio,
                language=calc_lang,
                file_path=output_path,
                speed=prosody['speed']
            )
        except Exception as e:
            print(f"XTTS Synthesis failed: {e}. Trying fallback with speaker embedding.")
            # Fallback if specific XTTS parameters fail or if using a different model
            self.tts.tts_to_file(
                text=phoneme_text,
                speaker_wav=reference_audio,
                speaker_embedding=adjusted_emb.squeeze(0).detach().cpu().numpy() if 'adjusted_emb' in locals() else None,
                language=calc_lang,
                file_path=output_path,
                speed=prosody['speed']
            )

        # --- Post-processing: Apply Pitch and Volume shifts ---
        post_process_audio(output_path, prosody['pitch'], prosody['volume'])

        print(f"Generated audio saved to {output_path}")

