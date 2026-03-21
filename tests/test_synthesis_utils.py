import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from Backend.synthesis_utils import (
    MAX_AUDIO_UPLOAD_BYTES,
    InputValidationError,
    is_text_script_compatible,
    is_latin_heavy_text,
    normalize_alpha,
    normalize_ref_lang,
    normalize_text_lang,
    normalize_text_input,
    normalize_v2_language,
    normalize_v2_ref_language,
    preprocess_reference_audio,
    require_non_empty_text,
    resolve_text_lang,
    resolve_reference_text,
    validate_audio_upload,
    validate_text_script,
)


class _DummyUpload:
    def __init__(self, filename):
        self.filename = filename


class SynthesisUtilsTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_normalize_ref_lang_accepts_codes_and_names(self):
        self.assertEqual(normalize_ref_lang("en"), "en")
        self.assertEqual(normalize_ref_lang("EN"), "en")
        self.assertEqual(normalize_ref_lang("english"), "en")
        self.assertEqual(normalize_ref_lang("Malayalam"), "ml")
        self.assertEqual(normalize_ref_lang("ml"), "ml")

    def test_normalize_ref_lang_rejects_invalid_value(self):
        with self.assertRaises(InputValidationError) as ctx:
            normalize_ref_lang("fr")
        self.assertEqual(ctx.exception.code, "INVALID_REF_LANG")

    def test_require_non_empty_text(self):
        self.assertEqual(
            require_non_empty_text(" hello ", "text", "INVALID_TEXT"),
            "hello",
        )
        with self.assertRaises(InputValidationError) as ctx:
            require_non_empty_text("   ", "text", "INVALID_TEXT")
        self.assertEqual(ctx.exception.code, "INVALID_TEXT")

    def test_validate_audio_upload(self):
        validate_audio_upload(_DummyUpload("sample.wav"))
        with self.assertRaises(InputValidationError) as ctx:
            validate_audio_upload(_DummyUpload(" "))
        self.assertEqual(ctx.exception.code, "MISSING_AUDIO")

    def test_resolve_reference_text_prefers_manual(self):
        text, source = resolve_reference_text(
            " Manual transcript ",
            fallback_transcriber=lambda: "should_not_run",
        )
        self.assertEqual(text, "Manual transcript")
        self.assertEqual(source, "manual")

    def test_resolve_reference_text_uses_fallback(self):
        text, source = resolve_reference_text(
            "",
            fallback_transcriber=lambda: " from_whisper ",
        )
        self.assertEqual(text, "from_whisper")
        self.assertEqual(source, "whisper")

    def test_resolve_reference_text_raises_on_empty_fallback(self):
        with self.assertRaises(InputValidationError) as ctx:
            resolve_reference_text("", fallback_transcriber=lambda: " ")
        self.assertEqual(ctx.exception.code, "EMPTY_REF_TEXT")

    def test_normalize_v2_language_accepts_codes_and_names(self):
        self.assertEqual(normalize_v2_language("hi", "target_lang"), "hi")
        self.assertEqual(normalize_v2_language("Hindi", "target_lang"), "hi")
        self.assertEqual(normalize_v2_language("Malayalam", "ref_lang"), "ml")

    def test_normalize_v2_language_rejects_invalid_value(self):
        with self.assertRaises(InputValidationError) as ctx:
            normalize_v2_language("en", "target_lang")
        self.assertEqual(ctx.exception.code, "INVALID_TARGET_LANG")

    def test_normalize_v2_ref_language_accepts_en_and_indic_codes(self):
        self.assertEqual(normalize_v2_ref_language("en"), "en")
        self.assertEqual(normalize_v2_ref_language("English"), "en")
        self.assertEqual(normalize_v2_ref_language("hi"), "hi")

    def test_normalize_v2_ref_language_rejects_invalid_value(self):
        with self.assertRaises(InputValidationError) as ctx:
            normalize_v2_ref_language("fr")
        self.assertEqual(ctx.exception.code, "INVALID_REF_LANG")

    def test_normalize_text_lang_accepts_supported_values(self):
        self.assertEqual(normalize_text_lang(None), "auto")
        self.assertEqual(normalize_text_lang(""), "auto")
        self.assertEqual(normalize_text_lang("auto"), "auto")
        self.assertEqual(normalize_text_lang("EN"), "en")
        self.assertEqual(normalize_text_lang("english"), "en")
        self.assertEqual(normalize_text_lang("target"), "target")
        self.assertEqual(normalize_text_lang("native"), "target")

    def test_normalize_text_lang_rejects_invalid_value(self):
        with self.assertRaises(InputValidationError) as ctx:
            normalize_text_lang("spanish")
        self.assertEqual(ctx.exception.code, "INVALID_TEXT_LANG")

    def test_resolve_text_lang_auto_detects_latin_heavy_input(self):
        self.assertTrue(is_latin_heavy_text("hello world"))
        self.assertFalse(is_latin_heavy_text("नमस्ते दुनिया"))
        self.assertEqual(resolve_text_lang("hello world", "auto"), "en")
        self.assertEqual(resolve_text_lang("नमस्ते दुनिया", "auto"), "target")
        self.assertEqual(resolve_text_lang("hello world", "target"), "target")

    def test_normalize_text_input_and_script_validation(self):
        text = normalize_text_input("  नमस्ते   दुनिया  ", "text", "INVALID_TEXT")
        self.assertEqual(text, "नमस्ते दुनिया")
        self.assertTrue(is_text_script_compatible(text, "hi"))
        self.assertFalse(is_text_script_compatible("hello world", "hi"))
        with self.assertRaises(InputValidationError) as ctx:
            validate_text_script("hello world", "hi", field_name="text", code="INVALID_TEXT_SCRIPT")
        self.assertEqual(ctx.exception.code, "INVALID_TEXT_SCRIPT")

    def test_normalize_alpha_clamps_to_range(self):
        self.assertEqual(normalize_alpha(-1.0), 0.0)
        self.assertEqual(normalize_alpha(2.0), 1.0)
        self.assertAlmostEqual(normalize_alpha(0.37), 0.37)

    def test_preprocess_reference_audio_rejects_large_file(self):
        payload = b"0" * (MAX_AUDIO_UPLOAD_BYTES + 1)
        with self.assertRaises(InputValidationError) as ctx:
            preprocess_reference_audio(
                audio_bytes=payload,
                filename="sample.wav",
                content_type="audio/wav",
                output_dir=self.tmpdir.name,
            )
        self.assertEqual(ctx.exception.code, "AUDIO_TOO_LARGE")

    def test_preprocess_reference_audio_success_for_valid_clip(self):
        sample_rate = 16000
        seconds = 4
        t = np.linspace(0, seconds, sample_rate * seconds, endpoint=False)
        waveform = 0.2 * np.sin(2 * np.pi * 220 * t)

        buffer = io.BytesIO()
        sf.write(buffer, waveform, sample_rate, format="WAV")
        audio_bytes = buffer.getvalue()

        info = preprocess_reference_audio(
            audio_bytes=audio_bytes,
            filename="voice.wav",
            content_type="audio/wav",
            output_dir=self.tmpdir.name,
        )

        self.assertIn("audio_path", info)
        self.assertGreater(info["duration_sec"], 3.0)
        self.assertEqual(info["sample_rate"], 16000)
        self.assertTrue(Path(info["audio_path"]).exists())


if __name__ == "__main__":
    unittest.main()
