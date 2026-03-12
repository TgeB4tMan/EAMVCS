import importlib
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


def _load_wrapper_module():
    torch_stub = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        device=lambda name: name,
    )
    emotion_stub = types.SimpleNamespace(
        predict_emotion=lambda _: {"predicted_emotion": "neutral", "confidence": 77.0}
    )

    with mock.patch.dict(
        sys.modules,
        {"torch": torch_stub, "Backend.emotion_detector": emotion_stub},
    ):
        sys.modules.pop("Backend.tts.acoustic_wrapper", None)
        return importlib.import_module("Backend.tts.acoustic_wrapper")


class AcousticWrapperTests(unittest.TestCase):
    def setUp(self):
        self.module = _load_wrapper_module()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.model_path = Path(self.tmpdir.name) / "model_1200000.safetensors"
        self.model_path.write_bytes(b"model")
        self.reference_audio = Path(self.tmpdir.name) / "reference.wav"
        self.reference_audio.write_bytes(b"audio")
        self.output_path = Path(self.tmpdir.name) / "output.wav"
        self.wrapper = self.module.EmotionTTS(model_path=str(self.model_path))

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_build_command_uses_current_python(self):
        cmd = self.wrapper._build_f5_command(
            text="hello",
            ref_text="hello there",
            reference_audio=str(self.reference_audio),
            output_path=str(self.output_path),
        )
        self.assertEqual(cmd[0], sys.executable)
        self.assertIn("--cfg_strength", cmd)
        self.assertIn("--nfe_step", cmd)
        self.assertIn("--speed", cmd)
        self.assertIn("--remove_silence", cmd)

    def test_synthesize_surfaces_subprocess_error(self):
        failure = subprocess.CompletedProcess(
            args=["cmd"],
            returncode=1,
            stdout="stdout text",
            stderr="stderr text",
        )
        with mock.patch.object(self.module.subprocess, "run", return_value=failure):
            with self.assertRaises(RuntimeError) as ctx:
                self.wrapper.synthesize(
                    text="hello",
                    ref_text="hello ref",
                    reference_audio=str(self.reference_audio),
                    output_path=str(self.output_path),
                )

        message = str(ctx.exception)
        self.assertIn("F5-TTS synthesis failed", message)
        self.assertIn("stderr text", message)
        self.assertIn("Command:", message)

    def test_synthesize_fails_when_output_not_created(self):
        success = subprocess.CompletedProcess(
            args=["cmd"],
            returncode=0,
            stdout="ok",
            stderr="",
        )
        with mock.patch.object(self.module.subprocess, "run", return_value=success):
            with self.assertRaises(RuntimeError) as ctx:
                self.wrapper.synthesize(
                    text="hello",
                    ref_text="hello ref",
                    reference_audio=str(self.reference_audio),
                    output_path=str(self.output_path),
                )
        self.assertIn("did not create output file", str(ctx.exception))

    def test_synthesize_fails_when_output_empty(self):
        success = subprocess.CompletedProcess(
            args=["cmd"],
            returncode=0,
            stdout="ok",
            stderr="",
        )

        def _side_effect(*args, **kwargs):
            self.output_path.write_bytes(b"")
            return success

        with mock.patch.object(self.module.subprocess, "run", side_effect=_side_effect):
            with self.assertRaises(RuntimeError) as ctx:
                self.wrapper.synthesize(
                    text="hello",
                    ref_text="hello ref",
                    reference_audio=str(self.reference_audio),
                    output_path=str(self.output_path),
                )
        self.assertIn("empty output file", str(ctx.exception))

    def test_synthesize_success(self):
        success = subprocess.CompletedProcess(
            args=["cmd"],
            returncode=0,
            stdout="ok",
            stderr="",
        )

        def _side_effect(*args, **kwargs):
            self.output_path.write_bytes(b"RIFF")
            return success

        with mock.patch.object(self.module.subprocess, "run", side_effect=_side_effect):
            result = self.wrapper.synthesize(
                text="hello",
                ref_text="hello ref",
                reference_audio=str(self.reference_audio),
                output_path=str(self.output_path),
            )

        self.assertEqual(result["synthesis_method"], "f5_tts")
        self.assertEqual(Path(result["output_path"]).resolve(), self.output_path.resolve())


if __name__ == "__main__":
    unittest.main()
