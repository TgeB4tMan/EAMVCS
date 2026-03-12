import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

import torch

from Backend.emotion_detector import predict_emotion


class EmotionTTS:
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.project_root = Path(__file__).resolve().parents[2]
        self.model_path = self._resolve_model_path(model_path)
        print(f"F5-TTS initialized with model: {self.model_path}")

    def _resolve_model_path(self, model_path: str = None) -> str:
        configured_path = model_path or os.getenv("F5_MODEL_PATH") or "model_1200000.safetensors"
        candidate = Path(configured_path)
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        candidate = candidate.resolve()

        if not candidate.exists():
            raise FileNotFoundError(
                "F5 model file not found. Set F5_MODEL_PATH or place model at "
                f"{candidate}."
            )

        return str(candidate)

    def _build_f5_command(
        self,
        text: str,
        ref_text: str,
        reference_audio: str,
        output_path: str,
    ) -> list:
        return [
            sys.executable,
            "-m",
            "f5_tts.infer_cli",
            "--model_path",
            self.model_path,
            "--ref_audio",
            reference_audio,
            "--ref_text",
            ref_text,
            "--gen_text",
            text,
            "--output_path",
            output_path,
            "--cfg_strength",
            "3.5",
            "--nfe_step",
            "64",
            "--speed",
            "0.80",
            "--remove_silence",
        ]

    def synthesize(
        self,
        text: str,
        ref_text: str,
        reference_audio: str,
        language: str = "en",
        output_path: str = "output.wav",
        alpha: float = 0.3,
    ) -> Dict:
        clean_text = (text or "").strip()
        clean_ref_text = (ref_text or "").strip()
        if not clean_text:
            raise ValueError("Generation text must be non-empty.")
        if not clean_ref_text:
            raise ValueError("Reference text must be non-empty.")
        if not os.path.exists(reference_audio):
            raise FileNotFoundError(f"Reference audio not found: {reference_audio}")

        output_file = Path(output_path).resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        emotion_result = predict_emotion(reference_audio) or {}
        emotion_name = emotion_result.get("predicted_emotion", "neutral")
        confidence = float(emotion_result.get("confidence", 0.0))

        cmd = self._build_f5_command(
            text=clean_text,
            ref_text=clean_ref_text,
            reference_audio=reference_audio,
            output_path=str(output_file),
        )

        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
        )

        if process.returncode != 0:
            raise RuntimeError(
                "F5-TTS synthesis failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDERR: {process.stderr.strip()}\n"
                f"STDOUT: {process.stdout.strip()}"
            )

        if not output_file.exists():
            raise RuntimeError(f"F5-TTS did not create output file: {output_file}")

        if output_file.stat().st_size <= 0:
            raise RuntimeError(f"F5-TTS created empty output file: {output_file}")

        return {
            "emotion": emotion_name,
            "confidence": confidence,
            "output_path": str(output_file),
            "synthesis_method": "f5_tts",
            "language": language,
            "alpha": float(alpha),
            "parameters": {
                "cfg_strength": 3.5,
                "nfe_step": 64,
                "speed": 0.80,
                "remove_silence": True,
            },
        }
