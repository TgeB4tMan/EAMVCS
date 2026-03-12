from typing import Callable, Optional, Tuple


REF_LANG_ALIASES = {
    "en": "en",
    "english": "en",
    "ml": "ml",
    "malayalam": "ml",
}


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


def require_non_empty_text(value: Optional[str], field_name: str, code: str) -> str:
    if value is None or not value.strip():
        raise InputValidationError(f"{field_name} must be non-empty.", code)
    return value.strip()


def validate_audio_upload(audio_upload) -> None:
    if audio_upload is None:
        raise InputValidationError("Reference audio file is required.", "MISSING_AUDIO")

    filename = getattr(audio_upload, "filename", None)
    if not filename or not filename.strip():
        raise InputValidationError("Reference audio filename is missing.", "MISSING_AUDIO")


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
