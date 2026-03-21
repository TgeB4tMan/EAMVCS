import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


class _StubF5TTS:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def infer(self, ref_file, ref_text, gen_text, speed=1.0):
        del ref_file, ref_text, gen_text, speed
        return np.zeros(2400, dtype=np.float32), 24000, None


class _FailingF5TTS:
    def __init__(self, **kwargs):
        del kwargs

    def infer(self, **kwargs):
        del kwargs
        raise RuntimeError("model boom")


class AcousticWrapperTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.model_dir = Path(self.tmpdir.name) / "indicf5"
        (self.model_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.model_dir / "model.safetensors").write_bytes(b"weights")
        (self.model_dir / "checkpoints" / "vocab.txt").write_text("abc", encoding="utf-8")
        self.reference_audio = Path(self.tmpdir.name) / "reference.wav"
        self.reference_audio.write_bytes(b"audio")
        self.output_path = Path(self.tmpdir.name) / "output.wav"

    def tearDown(self):
        self.tmpdir.cleanup()

    def _load_wrapper_module(self):
        emotion_stub = types.SimpleNamespace(
            predict_emotion=lambda _: {
                "predicted_emotion": "neutral",
                "confidence": 77.0,
                "all_probabilities": {"neutral": 77.0},
                "valence": 0.5,
                "arousal": 0.4,
                "dominance": 0.5,
            }
        )

        with mock.patch.dict(sys.modules, {"Backend.emotion_detector": emotion_stub}):
            sys.modules.pop("Backend.tts.acoustic_wrapper", None)
            module = importlib.import_module("Backend.tts.acoustic_wrapper")

        return module

    def test_init_loads_local_indicf5_bundle(self):
        module = self._load_wrapper_module()
        with mock.patch("f5_tts.api.F5TTS", side_effect=_StubF5TTS) as f5tts:
            wrapper = module.EmotionTTS(model_dir=str(self.model_dir))

        self.assertEqual(wrapper.model_dir, self.model_dir.resolve())
        self.assertEqual(wrapper.ckpt_file, (self.model_dir / "model.safetensors").resolve())
        self.assertEqual(wrapper.vocab_file, (self.model_dir / "checkpoints" / "vocab.txt").resolve())
        f5tts.assert_called_once()

    def test_init_fails_for_missing_required_model_files(self):
        module = self._load_wrapper_module()
        broken_dir = Path(self.tmpdir.name) / "broken_model"
        broken_dir.mkdir(parents=True, exist_ok=True)
        (broken_dir / "model.safetensors").write_bytes(b"weights")

        with self.assertRaises(FileNotFoundError) as ctx:
            module.EmotionTTS(model_dir=str(broken_dir))
        self.assertIn("Missing", str(ctx.exception))

    def test_synthesize_success(self):
        module = self._load_wrapper_module()
        with mock.patch("f5_tts.api.F5TTS", side_effect=_StubF5TTS):
            wrapper = module.EmotionTTS(model_dir=str(self.model_dir))

        result = wrapper.synthesize(
            text="namaste duniya",
            ref_text="namaste",
            reference_audio=str(self.reference_audio),
            language="hi",
            output_path=str(self.output_path),
            alpha=0.5,
            ref_lang="hi",
        )

        self.assertEqual(result["synthesis_method"], "indicf5")
        self.assertEqual(result["engine"], "indicf5")
        self.assertEqual(Path(result["output_path"]).resolve(), self.output_path.resolve())
        self.assertIn("emotion_probabilities", result)
        self.assertIn("valence", result)
        self.assertIn("arousal", result)
        self.assertIn("dominance", result)
        self.assertTrue(self.output_path.exists())

    def test_synthesize_surfaces_model_error(self):
        module = self._load_wrapper_module()
        with mock.patch("f5_tts.api.F5TTS", side_effect=_FailingF5TTS):
            wrapper = module.EmotionTTS(model_dir=str(self.model_dir))

        with self.assertRaises(RuntimeError) as ctx:
            wrapper.synthesize(
                text="namaste",
                ref_text="namaste",
                reference_audio=str(self.reference_audio),
                output_path=str(self.output_path),
            )
        self.assertIn("IndicF5 inference failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
