import os
import unittest
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
        app_module.emotion_tts = self.tts_mock
        app_module.whisper_model = self.whisper_mock
        app_module.startup_errors = {}

        self.whisper_mock.transcribe.return_value = {"text": "hello reference"}
        self.embedding_patch = mock.patch.object(
            app_module,
            "get_speaker_embedding",
            side_effect=[list(range(256)), list(range(256))],
        )
        self.embedding_patch.start()

        def _synthesize_side_effect(**kwargs):
            Path(kwargs["output_path"]).write_bytes(b"RIFF")
            return {"synthesis_method": "f5_tts"}

        self.tts_mock.synthesize.side_effect = _synthesize_side_effect

    def tearDown(self):
        self.embedding_patch.stop()
        for wav_file in self.uploads_dir.glob("output_*.wav"):
            wav_file.unlink(missing_ok=True)

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


if __name__ == "__main__":
    unittest.main()
