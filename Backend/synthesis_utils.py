import math
import os
import re
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio


REF_LANG_ALIASES = {
    "en": "en",
    "english": "en",
    "ml": "ml",
    "malayalam": "ml",
}

MAX_AUDIO_UPLOAD_BYTES = 10 * 1024 * 1024
MIN_AUDIO_DURATION_SECONDS = 3.0
MAX_AUDIO_DURATION_SECONDS = 30.0
TARGET_AUDIO_SAMPLE_RATE = 16000
MAX_TEXT_LENGTH = 500

V2_SUPPORTED_LANGUAGES = {
    "as": "Assamese",
    "bn": "Bengali",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "or": "Odia",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
}

V2_LANGUAGE_ALIASES = {
    **{code: code for code in V2_SUPPORTED_LANGUAGES},
    "assamese": "as",
    "bengali": "bn",
    "gujarati": "gu",
    "hindi": "hi",
    "kannada": "kn",
    "malayalam": "ml",
    "marathi": "mr",
    "odia": "or",
    "oriya": "or",
    "punjabi": "pa",
    "tamil": "ta",
    "telugu": "te",
}

V2_REF_LANGUAGE_ALIASES = {
    **V2_LANGUAGE_ALIASES,
    "en": "en",
    "english": "en",
}

TEXT_LANG_ALIASES = {
    "auto": "auto",
    "en": "en",
    "english": "en",
    "target": "target",
    "native": "target",
    "indic": "target",
}

# Unicode blocks used for lightweight script validation
LANGUAGE_SCRIPT_RANGES = {
    "as": ((0x0980, 0x09FF),),  # Bengali/Assamese
    "bn": ((0x0980, 0x09FF),),  # Bengali
    "gu": ((0x0A80, 0x0AFF),),  # Gujarati
    "hi": ((0x0900, 0x097F),),  # Devanagari
    "kn": ((0x0C80, 0x0CFF),),  # Kannada
    "ml": ((0x0D00, 0x0D7F),),  # Malayalam
    "mr": ((0x0900, 0x097F),),  # Devanagari
    "or": ((0x0B00, 0x0B7F),),  # Odia
    "pa": ((0x0A00, 0x0A7F),),  # Gurmukhi
    "ta": ((0x0B80, 0x0BFF),),  # Tamil
    "te": ((0x0C00, 0x0C7F),),  # Telugu
}

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".webm", ".m4a", ".ogg", ".flac", ".aac"}


class InputValidationError(ValueError):
    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.code = code


def error_payload(message: str, code: str) -> dict:
    return {"error": message, "code": code}


def normalize_ref_lang(ref_lang: Optional[str]) -> str:
    if ref_lang is None:
        raise InputValidationError(
            "Reference language is required.",
            "INVALID_REF_LANG",
        )

    normalized = ref_lang.strip().lower()
    if normalized in REF_LANG_ALIASES:
        return REF_LANG_ALIASES[normalized]

    raise InputValidationError(
        "Unsupported reference language. Use 'en' or 'ml'.",
        "INVALID_REF_LANG",
    )


def normalize_v2_language(value: Optional[str], field_name: str) -> str:
    if value is None:
        code = "INVALID_TARGET_LANG" if field_name == "target_lang" else "INVALID_REF_LANG"
        raise InputValidationError(f"{field_name} is required.", code)

    normalized = value.strip().lower()
    language = V2_LANGUAGE_ALIASES.get(normalized)
    if language:
        return language

    codes = ", ".join(sorted(V2_SUPPORTED_LANGUAGES.keys()))
    code = "INVALID_TARGET_LANG" if field_name == "target_lang" else "INVALID_REF_LANG"
    raise InputValidationError(
        f"Unsupported {field_name}. Use one of: {codes}.",
        code,
    )


def normalize_v2_ref_language(value: Optional[str]) -> str:
    if value is None:
        raise InputValidationError("ref_lang is required.", "INVALID_REF_LANG")

    normalized = value.strip().lower()
    language = V2_REF_LANGUAGE_ALIASES.get(normalized)
    if language:
        return language

    codes = ", ".join(sorted(set(V2_SUPPORTED_LANGUAGES.keys()) | {"en"}))
    raise InputValidationError(
        f"Unsupported ref_lang. Use one of: {codes}.",
        "INVALID_REF_LANG",
    )


def get_supported_v2_languages() -> Dict[str, str]:
    return dict(V2_SUPPORTED_LANGUAGES)


def normalize_text_lang(value: Optional[str]) -> str:
    if value is None:
        return "auto"

    normalized = value.strip().lower()
    if not normalized:
        return "auto"

    resolved = TEXT_LANG_ALIASES.get(normalized)
    if resolved:
        return resolved

    raise InputValidationError(
        "Unsupported text_lang. Use one of: auto, en, target.",
        "INVALID_TEXT_LANG",
    )


def is_latin_heavy_text(text: str, threshold: float = 0.6) -> bool:
    total_letters = 0
    latin_letters = 0

    for char in text:
        if not unicodedata.category(char).startswith("L"):
            continue
        total_letters += 1
        if "LATIN" in unicodedata.name(char, ""):
            latin_letters += 1

    if total_letters == 0:
        return False
    return (latin_letters / total_letters) >= threshold


def resolve_text_lang(text: str, text_lang: Optional[str]) -> str:
    normalized = normalize_text_lang(text_lang)
    if normalized != "auto":
        return normalized
    return "en" if is_latin_heavy_text(text) else "target"


def require_non_empty_text(value: Optional[str], field_name: str, code: str) -> str:
    if value is None or not value.strip():
        raise InputValidationError(f"{field_name} must be non-empty.", code)
    return value.strip()


def normalize_text_input(
    value: Optional[str],
    field_name: str,
    code: str,
    max_length: int = MAX_TEXT_LENGTH,
) -> str:
    text = require_non_empty_text(value, field_name=field_name, code=code)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([!?])\1{1,}", r"\1", text)
    text = re.sub(r"\.{3,}", "...", text)

    if len(text) > max_length:
        raise InputValidationError(
            f"{field_name} exceeds max length of {max_length} characters.",
            code,
        )
    return text


def _in_script_ranges(char: str, ranges: Tuple[Tuple[int, int], ...]) -> bool:
    codepoint = ord(char)
    for start, end in ranges:
        if start <= codepoint <= end:
            return True
    return False


def is_text_script_compatible(text: str, language: str) -> bool:
    ranges = LANGUAGE_SCRIPT_RANGES.get(language)
    if not ranges:
        return True

    letters_total = 0
    letters_in_target_script = 0
    for char in text:
        category = unicodedata.category(char)
        if not category.startswith("L"):
            continue
        letters_total += 1
        if _in_script_ranges(char, ranges):
            letters_in_target_script += 1

    # If there are no letter codepoints, keep it permissive and let model handle it.
    if letters_total == 0:
        return True

    return letters_total == letters_in_target_script


def validate_text_script(
    text: str,
    language: str,
    field_name: str,
    code: str,
) -> None:
    if not is_text_script_compatible(text, language):
        raise InputValidationError(
            f"{field_name} script does not match language '{language}'.",
            code,
        )


def normalize_alpha(value: Optional[float], minimum: float = 0.0, maximum: float = 1.0) -> float:
    if value is None:
        return 0.3
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise InputValidationError("alpha must be a valid number.", "INVALID_ALPHA") from exc

    if not math.isfinite(parsed):
        raise InputValidationError("alpha must be a finite number.", "INVALID_ALPHA")

    return float(max(minimum, min(maximum, parsed)))


def validate_audio_upload(audio_upload) -> None:
    if audio_upload is None:
        raise InputValidationError("Reference audio file is required.", "MISSING_AUDIO")

    filename = getattr(audio_upload, "filename", None)
    if not filename or not filename.strip():
        raise InputValidationError("Reference audio filename is missing.", "MISSING_AUDIO")


def _load_audio_with_fallback(temp_audio_path: str) -> Tuple[np.ndarray, int]:
    try:
        waveform, sample_rate = torchaudio.load(temp_audio_path)
    except Exception:
        try:
            waveform_np, sample_rate = librosa.load(temp_audio_path, sr=None, mono=False)
        except Exception as exc:
            raise InputValidationError(
                "Unable to decode reference audio.",
                "AUDIO_DECODE_FAILED",
            ) from exc

        if isinstance(waveform_np, np.ndarray) and waveform_np.ndim == 1:
            waveform = torch.from_numpy(waveform_np).unsqueeze(0)
        else:
            waveform = torch.from_numpy(waveform_np)

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != TARGET_AUDIO_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate, TARGET_AUDIO_SAMPLE_RATE)
        waveform = resampler(waveform)
        sample_rate = TARGET_AUDIO_SAMPLE_RATE

    audio_array = waveform.squeeze(0).detach().cpu().numpy().astype(np.float32)
    audio_array = np.nan_to_num(audio_array)
    return audio_array, sample_rate


def _validate_audio_content_type(filename: str, content_type: Optional[str]) -> None:
    suffix = Path(filename or "").suffix.lower()
    if suffix and suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise InputValidationError(
            f"Unsupported audio extension '{suffix}'. Allowed: {sorted(ALLOWED_AUDIO_EXTENSIONS)}",
            "INVALID_AUDIO_TYPE",
        )

    normalized_content_type = (content_type or "").strip().lower()
    if normalized_content_type and not normalized_content_type.startswith("audio/"):
        # Keep this permissive when extension is known audio.
        if suffix not in ALLOWED_AUDIO_EXTENSIONS:
            raise InputValidationError(
                "Unsupported audio content type.",
                "INVALID_AUDIO_TYPE",
            )


def preprocess_reference_audio(
    audio_bytes: bytes,
    filename: str,
    content_type: Optional[str],
    output_dir: str,
    prefix: str = "refv2",
) -> Dict[str, Any]:
    if not audio_bytes:
        raise InputValidationError("Reference audio file is empty.", "EMPTY_AUDIO")

    if len(audio_bytes) > MAX_AUDIO_UPLOAD_BYTES:
        raise InputValidationError(
            f"Reference audio exceeds {MAX_AUDIO_UPLOAD_BYTES // (1024 * 1024)}MB limit.",
            "AUDIO_TOO_LARGE",
        )

    _validate_audio_content_type(filename=filename, content_type=content_type)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename or "reference.wav").suffix or ".wav"

    temp_input_path = None
    output_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(output_root)) as temp_input:
            temp_input.write(audio_bytes)
            temp_input_path = temp_input.name

        waveform, sample_rate = _load_audio_with_fallback(temp_input_path)
        original_duration = float(len(waveform) / sample_rate) if sample_rate else 0.0

        trimmed_waveform, _ = librosa.effects.trim(waveform, top_db=25)
        trimmed_silence = False
        min_post_trim_samples = int(0.3 * sample_rate)
        if trimmed_waveform.size >= min_post_trim_samples:
            waveform = trimmed_waveform
            trimmed_silence = True

        duration_seconds = float(len(waveform) / sample_rate) if sample_rate else 0.0
        if duration_seconds < MIN_AUDIO_DURATION_SECONDS or duration_seconds > MAX_AUDIO_DURATION_SECONDS:
            raise InputValidationError(
                (
                    "Reference audio duration must be between "
                    f"{MIN_AUDIO_DURATION_SECONDS:.0f}s and {MAX_AUDIO_DURATION_SECONDS:.0f}s "
                    f"(received {duration_seconds:.2f}s)."
                ),
                "AUDIO_DURATION_OUT_OF_RANGE",
            )

        safe_base = Path(filename or "reference").stem.replace(" ", "_")
        output_name = f"{prefix}_{int(time.time() * 1000)}_{safe_base}.wav"
        output_path = output_root / output_name
        sf.write(str(output_path), waveform, sample_rate, subtype="PCM_16")

        return {
            "audio_path": str(output_path),
            "filename": output_name,
            "sample_rate": sample_rate,
            "channels": 1,
            "duration_sec": duration_seconds,
            "original_duration_sec": original_duration,
            "trimmed_silence": trimmed_silence,
            "input_size_bytes": len(audio_bytes),
        }
    except InputValidationError:
        if output_path and output_path.exists():
            output_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        if output_path and output_path.exists():
            output_path.unlink(missing_ok=True)
        raise InputValidationError(
            f"Failed to preprocess reference audio: {exc}",
            "AUDIO_PREPROCESS_FAILED",
        ) from exc
    finally:
        if temp_input_path and os.path.exists(temp_input_path):
            os.remove(temp_input_path)


def resolve_reference_text(
    manual_ref_text: Optional[str],
    fallback_transcriber: Callable[[], str],
) -> Tuple[str, str]:
    cleaned_manual = (manual_ref_text or "").strip()
    if cleaned_manual:
        return cleaned_manual, "manual"

    fallback_text = fallback_transcriber()
    resolved = require_non_empty_text(
        fallback_text,
        field_name="reference transcription",
        code="EMPTY_REF_TEXT",
    )
    return resolved, "whisper"
