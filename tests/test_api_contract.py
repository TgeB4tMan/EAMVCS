import io
import math
import unittest
import wave
from pathlib import Path
from unittest import mock

IMPORT_ERROR = None
try:
    from fastapi.testclient import TestClient
    import Backend.app as app_module
except Exception as exc:  # pragma: no cover - environment dependent
    IMPORT_ERROR = exc
    TestClient = None
    app_module = None


@unittest.skipIf(TestClient is None, f"fastapi runtime unavailable: {IMPORT_ERROR}")
class ApiContractTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app_module.app)
        self.uploads_dir = Path(app_module.UPLOAD_DIR)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

        self.tts_mock = mock.Mock()
        self.whisper_mock = mock.Mock()
        self.translation_mock = mock.Mock()
        app_module.emotion_tts = self.tts_mock
        app_module.whisper_model = self.whisper_mock
        app_module.translation_service = self.translation_mock
        app_module.startup_errors = {}

        self.whisper_mock.transcribe.return_value = {"text": "hello reference"}
        self.translation_mock.translate_english_to_target.return_value = "नमस्ते दुनिया"
        self.embedding_patch = mock.patch.object(
            app_module,
            "get_speaker_embedding",
            side_effect=[list(range(256)), list(range(256))],
        )
        self.embedding_patch.start()

        def _synthesize_side_effect(**kwargs):
            Path(kwargs["output_path"]).write_bytes(b"RIFF")
            return {
                "synthesis_method": "indicf5",
                "output_path": kwargs["output_path"],
                "emotion": "happy",
                "confidence": 91.2,
                "emotion_probabilities": {
                    "neutral": 1.2,
                    "happy": 91.2,
                    "sad": 2.1,
                    "angry": 5.5,
                },
                "valence": 0.8,
                "arousal": 0.7,
                "dominance": 0.6,
            }

        self.tts_mock.synthesize.side_effect = _synthesize_side_effect

    def tearDown(self):
        self.embedding_patch.stop()
        for wav_file in self.uploads_dir.glob("output_*.wav"):
            wav_file.unlink(missing_ok=True)
        for wav_file in self.uploads_dir.glob("output_v2_*.wav"):
            wav_file.unlink(missing_ok=True)
        for wav_file in self.uploads_dir.glob("refv2_*.wav"):
            wav_file.unlink(missing_ok=True)

    def _generate_wav_bytes(self, seconds=4, sample_rate=16000, freq=220.0):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            total_samples = int(seconds * sample_rate)
            frames = bytearray()
            for idx in range(total_samples):
                sample = int(0.2 * 32767 * math.sin(2 * math.pi * freq * idx / sample_rate))
                frames.extend(sample.to_bytes(2, byteorder="little", signed=True))
            wav_file.writeframes(bytes(frames))
        return buffer.getvalue()

    def _post_synthesize(self, extra_data=None):
        data = {
            "text": "sample text",
            "language": "en",
            "ref_lang": "en",
            "alpha": "0.3",
        }
        if extra_data:
            data.update(extra_data)
        files = {"audio": ("reference.wav", b"RIFFdata", "audio/wav")}
        return self.client.post("/synthesize", data=data, files=files)

    def _post_synthesize_v2(self, extra_data=None, audio_bytes=None, filename="reference.wav", content_type="audio/wav"):
        data = {
            "text": "नमस्ते दुनिया",
            "target_lang": "hi",
            "ref_lang": "hi",
            "alpha": "0.3",
        }
        if extra_data:
            data.update(extra_data)
        files = {
            "audio": (
                filename,
                audio_bytes if audio_bytes is not None else self._generate_wav_bytes(),
                content_type,
            )
        }
        return self.client.post("/synthesize-v2", data=data, files=files)

    def test_synthesize_uses_manual_ref_text_when_provided(self):
        response = self._post_synthesize({"ref_text": "manual transcript"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["ref_text_source"], "manual")
        self.whisper_mock.transcribe.assert_not_called()

    def test_synthesize_uses_whisper_fallback_without_ref_text(self):
        response = self._post_synthesize()
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["ref_text_source"], "whisper")
        self.whisper_mock.transcribe.assert_called_once()

    def test_synthesize_invalid_language_returns_400(self):
        response = self._post_synthesize({"ref_lang": "fr"})
        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertEqual(payload["code"], "INVALID_REF_LANG")

    def test_synthesize_empty_text_returns_400(self):
        response = self._post_synthesize({"text": "   "})
        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertEqual(payload["code"], "INVALID_TEXT")

    def test_synthesize_returns_503_when_tts_unavailable(self):
        app_module.emotion_tts = None
        app_module.startup_errors["tts"] = "model missing"
        response = self._post_synthesize()
        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertEqual(payload["code"], "TTS_UNAVAILABLE")

    def test_audio_endpoint_serves_generated_file(self):
        wav_path = self.uploads_dir / "output_test.wav"
        wav_path.write_bytes(b"RIFF")
        response = self.client.get("/audio/output_test.wav")
        self.assertEqual(response.status_code, 200)
        self.assertGreater(len(response.content), 0)
        wav_path.unlink(missing_ok=True)

    def test_synthesize_v2_uses_manual_ref_text_when_provided(self):
        response = self._post_synthesize_v2({"ref_text": "नमस्ते"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["engine"], "indicf5")
        self.assertEqual(payload["ref_text_source"], "manual")
        self.assertEqual(payload["emotion"], "happy")
        self.assertAlmostEqual(payload["emotion_confidence"], 91.2)
        self.assertEqual(payload["emotion_profile"]["valence"], 0.8)
        self.assertEqual(payload["diagnostics"]["emotion"]["predicted"], "happy")
        self.whisper_mock.transcribe.assert_not_called()

    def test_synthesize_v2_uses_whisper_fallback_without_ref_text(self):
        self.whisper_mock.transcribe.return_value = {"text": "नमस्ते"}
        response = self._post_synthesize_v2()
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["ref_text_source"], "whisper")
        self.whisper_mock.transcribe.assert_called_once()

    def test_synthesize_v2_accepts_english_ref_language(self):
        self.whisper_mock.transcribe.return_value = {"text": "hello reference"}
        response = self._post_synthesize_v2({"ref_lang": "en", "ref_text": "hello reference"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["language"]["reference"], "en")

    def test_synthesize_v2_translates_english_input_in_auto_mode(self):
        response = self._post_synthesize_v2({"text": "hello how are you", "target_lang": "hi"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["translation_applied"])
        self.assertEqual(payload["text_lang_resolved"], "en")
        self.assertEqual(payload["tts_text"], "नमस्ते दुनिया")
        self.translation_mock.translate_english_to_target.assert_called_once_with(
            "hello how are you",
            "hi",
        )

    def test_synthesize_v2_returns_503_when_translation_unavailable(self):
        app_module.translation_service = None
        app_module.startup_errors["translation"] = "translation model missing"
        response = self._post_synthesize_v2({"text": "hello world"})
        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertEqual(payload["code"], "TRANSLATION_UNAVAILABLE")

    def test_synthesize_v2_returns_500_when_translation_fails(self):
        self.translation_mock.translate_english_to_target.side_effect = RuntimeError("translator boom")
        response = self._post_synthesize_v2({"text": "hello world"})
        self.assertEqual(response.status_code, 500)
        payload = response.json()
        self.assertEqual(payload["code"], "TRANSLATION_FAILED")

    def test_synthesize_v2_rejects_invalid_script_for_target_language(self):
        response = self._post_synthesize_v2(
            {"text": "hello world", "target_lang": "hi", "text_lang": "target"}
        )
        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertEqual(payload["code"], "INVALID_TEXT_SCRIPT")

    def test_synthesize_v2_target_text_lang_bypasses_translation(self):
        response = self._post_synthesize_v2({"text": "नमस्ते दुनिया", "text_lang": "target"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["translation_applied"])
        self.assertEqual(payload["text_lang_resolved"], "target")
        self.translation_mock.translate_english_to_target.assert_not_called()

    def test_synthesize_v2_rejects_oversized_audio(self):
        huge_audio = b"0" * (10 * 1024 * 1024 + 1)
        response = self._post_synthesize_v2(audio_bytes=huge_audio)
        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertEqual(payload["code"], "AUDIO_TOO_LARGE")

    def test_synthesize_v2_returns_503_when_tts_unavailable(self):
        app_module.emotion_tts = None
        app_module.startup_errors["tts"] = "model missing"
        response = self._post_synthesize_v2()
        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertEqual(payload["code"], "TTS_UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
