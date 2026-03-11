import torch
import numpy as np
import os
import subprocess
import tempfile

# Updated imports for consolidated structure
from Backend.emotion_detector import predict_emotion
from Backend.tts.fusion import EmotionFusion
from Backend.tts.prosody import emotion_to_prosody, apply_emotional_pauses, post_process_audio
from Backend.tts.multilingual import text_to_phonemes


class EmotionTTS:
    def __init__(self, model_path="model_1200000.safetensors"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        print(f" F5-TTS initialized with model: {model_path}")

    def synthesize(
        self,
        text,
        ref_text,
        reference_audio,
        language="en",
        output_path="output.wav",
        alpha=0.3
    ):
        # --- Emotion Detection Only ---
        emotion_result = predict_emotion(reference_audio)
        
        if emotion_result and 'predicted_emotion' in emotion_result:
            emotion_name = emotion_result['predicted_emotion']
            confidence = emotion_result['confidence'] / 100.0
            print(f" Detected Emotion: {emotion_name} ({confidence:.1%})")
        else:
            print(f" Could not detect emotion")
            emotion_name = "neutral"
            confidence = 0.0

        # --- F5-TTS Synthesis Only ---
        print(f" F5-TTS synthesis with exact parameters from Colab test")
        print(f"   Reference Text: '{ref_text}'")
        print(f"   Generation Text: '{text}'")
        print(f"   Language: {language}")
        print(f"   Alpha: {alpha}")
        
        try:
            # F5-TTS command with exact parameters
            cmd = [
                "python", "-m", "f5_tts.infer_cli",
                "--model_path", self.model_path,
                "--ref_audio", reference_audio,
                "--ref_text", ref_text,
                "--gen_text", text,
                "--output_path", output_path,
                "--cfg_strength", "3.5",  # High similarity
                "--nfe_step", "64",      # Smooth, high-quality
                "--speed", "0.80",        # Natural teacher-like pace
                "--remove_silence"
            ]
            
            print(f" Running F5-TTS command:")
            print(f"   {' '.join(cmd)}")
            
            # Run F5-TTS inference
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            if result.returncode == 0:
                print(f" F5-TTS synthesis successful!")
                print(f" Output saved to: {output_path}")
                
                # Check if file was created
                if os.path.exists(output_path):
                    # --- Return Results ---
                    result = {
                        "emotion": emotion_name,
                        "confidence": confidence * 100,
                        "output_path": output_path,
                        "synthesis_method": "f5_tts",
                        "parameters": {
                            "cfg_strength": 3.5,
                            "nfe_step": 64,
                            "speed": 0.80,
                            "remove_silence": True
                        }
                    }
                    
                    print(f" Generated: {output_path}")
                    print(f" Emotion: {emotion_name} ({confidence:.1%})")
                    print(f" Method: F5-TTS")
                    
                    return result
                else:
                    print(f" F5-TTS output file not found at: {output_path}")
                    return {"success": False, "error": "Output file not created"}
            else:
                print(f" F5-TTS synthesis failed!")
                print(f"Error: {result.stderr}")
                return {"success": False, "error": result.stderr}
                
        except Exception as e:
            print(f" Exception in F5-TTS synthesis: {e}")
            return {"success": False, "error": str(e)}
