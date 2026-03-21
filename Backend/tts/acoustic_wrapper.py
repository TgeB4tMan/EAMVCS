import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
import torch

from Backend.emotion_detector import predict_emotion


class EmotionTTS:
    DEFAULT_MODEL_DIR = "models/indicf5"

    def __init__(self, model_dir: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.project_root = Path(__file__).resolve().parents[2]
        self._configure_hf_environment()
        self.model_dir = self._resolve_model_dir(model_dir)
        self.ckpt_file, self.vocab_file = self._validate_model_dir(self.model_dir)
        self.model = self._load_model()
        print(f"IndicF5 initialized on {self.device} using {self.model_dir}")

    def _configure_hf_environment(self) -> None:
        hf_home = self.project_root / ".hf_cache"
        hf_home.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(hf_home))

        token = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
            or os.getenv("HUGGINGFACE_HUB_TOKEN")
        )
        if token:
            os.environ.setdefault("HF_TOKEN", token)
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)

    def _resolve_model_dir(self, model_dir: Optional[str]) -> Path:
        configured_path = model_dir or os.getenv("INDICF5_MODEL_DIR") or self.DEFAULT_MODEL_DIR
        candidate = Path(configured_path)
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        return candidate.resolve()

    def _validate_model_dir(self, model_dir: Path) -> tuple[Path, Path]:
        if not model_dir.exists():
            raise FileNotFoundError(
                f"IndicF5 model directory not found: {model_dir}. "
                "Set INDICF5_MODEL_DIR or place files under models/indicf5."
            )
        if not model_dir.is_dir():
            raise FileNotFoundError(f"IndicF5 model path is not a directory: {model_dir}")

        ckpt_file = model_dir / "model.safetensors"
        vocab_file = model_dir / "checkpoints" / "vocab.txt"

        missing = []
        if not ckpt_file.exists():
            missing.append(str(ckpt_file))
        if not vocab_file.exists():
            missing.append(str(vocab_file))

        if missing:
            raise FileNotFoundError(
                "IndicF5 local files are incomplete. Missing: "
                + ", ".join(missing)
            )

        return ckpt_file, vocab_file

    def _load_model(self):
        try:
            from f5_tts.infer.utils_infer import load_model, load_vocoder
            from f5_tts.model import DiT
            from safetensors.torch import load_file
        except Exception as exc:
            raise RuntimeError(
                "f5-tts is required for IndicF5 runtime. Install the project requirements first."
            ) from exc

        try:
            vocoder = load_vocoder(
                vocoder_name="vocos",
                device=self.device,
                hf_cache_dir=os.environ.get("HF_HOME"),
            )
            model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
            ema_model = load_model(
                DiT,
                model_cfg,
                mel_spec_type="vocos",
                vocab_file=str(self.vocab_file),
                device=self.device,
            )

            raw_state_dict = load_file(str(self.ckpt_file), device=self.device)
            cleaned_state_dict = {}
            for key, value in raw_state_dict.items():
                normalized_key = key
                if normalized_key.startswith("ema_model."):
                    normalized_key = normalized_key[len("ema_model.") :]
                if normalized_key.startswith("_orig_mod."):
                    normalized_key = normalized_key[len("_orig_mod.") :]
                if normalized_key.startswith("vocoder."):
                    continue
                cleaned_state_dict[normalized_key] = value

            missing_keys, unexpected_keys = ema_model.load_state_dict(cleaned_state_dict, strict=False)
            remaining_missing = [key for key in missing_keys if "mel_spec.mel_stft" not in key]
            if remaining_missing or unexpected_keys:
                raise RuntimeError(
                    "Checkpoint structure still does not match IndicF5 runtime. "
                    f"Missing keys: {remaining_missing[:10]}. Unexpected keys: {unexpected_keys[:10]}."
                )

            return {"ema_model": ema_model, "vocoder": vocoder}
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize IndicF5 through the supported F5-TTS API. "
                f"Checkpoint: {self.ckpt_file}. Error: {exc}"
            ) from exc

    @staticmethod
    def _write_waveform(output_file: Path, waveform: Any, sample_rate: int) -> None:
        waveform_np = np.asarray(waveform, dtype=np.float32)
        waveform_np = np.nan_to_num(waveform_np)
        if waveform_np.ndim > 1:
            waveform_np = np.squeeze(waveform_np)
        waveform_np = np.clip(waveform_np, -1.0, 1.0)
        sf.write(str(output_file), waveform_np, sample_rate, subtype="PCM_16")

    def synthesize(
        self,
        text: str,
        ref_text: str,
        reference_audio: str,
        language: str = "hi",
        output_path: str = "output.wav",
        alpha: float = 0.3,
        ref_lang: Optional[str] = None,
    ) -> Dict[str, Any]:
        del alpha
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

        target_lang = (language or "").strip().lower() or "hi"
        source_lang = (ref_lang or target_lang).strip().lower()

        emotion_result = predict_emotion(reference_audio) or {}
        emotion_name = emotion_result.get("predicted_emotion", "neutral")
        confidence = float(emotion_result.get("confidence", 0.0))
        emotion_probabilities = emotion_result.get("all_probabilities") or {}
        valence = emotion_result.get("valence")
        arousal = emotion_result.get("arousal")
        dominance = emotion_result.get("dominance")

        try:
            from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

            prepared_ref_audio, prepared_ref_text = preprocess_ref_audio_text(
                reference_audio,
                clean_ref_text,
                device=self.device,
            )
            waveform, sample_rate, _ = infer_process(
                prepared_ref_audio,
                prepared_ref_text,
                clean_text,
                self.model["ema_model"],
                self.model["vocoder"],
                mel_spec_type="vocos",
                speed=1.0,
                device=self.device,
            )
        except Exception as exc:
            raise RuntimeError(f"IndicF5 inference failed: {exc}") from exc

        self._write_waveform(output_file, waveform, int(sample_rate))

        if not output_file.exists() or output_file.stat().st_size <= 0:
            raise RuntimeError(f"IndicF5 did not create a valid output file: {output_file}")

        return {
            "emotion": emotion_name,
            "confidence": confidence,
            "emotion_probabilities": emotion_probabilities,
            "valence": float(valence) if valence is not None else None,
            "arousal": float(arousal) if arousal is not None else None,
            "dominance": float(dominance) if dominance is not None else None,
            "output_path": str(output_file),
            "synthesis_method": "indicf5",
            "engine": "indicf5",
            "language": target_lang,
            "ref_lang": source_lang,
            "device": self.device,
        }
