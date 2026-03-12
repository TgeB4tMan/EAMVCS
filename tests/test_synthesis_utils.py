import unittest

from Backend.synthesis_utils import (
    InputValidationError,
    normalize_ref_lang,
    require_non_empty_text,
    resolve_reference_text,
    validate_audio_upload,
)


class _DummyUpload:
    def __init__(self, filename):
        self.filename = filename


class SynthesisUtilsTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
