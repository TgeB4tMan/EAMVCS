import os
import unittest
from pathlib import Path

from Backend.tts.acoustic_wrapper import EmotionTTS


def _has_local_indicf5_artifacts() -> bool:
    model_dir = Path(os.getenv("INDICF5_MODEL_DIR", "models/indicf5")).resolve()
    if not model_dir.exists() or not model_dir.is_dir():
        return False
    if not (model_dir / "config.json").exists():
        return False
    if not any(model_dir.glob("*.safetensors")) and not any(model_dir.glob("*.bin")):
        return False
    return True


@unittest.skipUnless(_has_local_indicf5_artifacts(), "Local IndicF5 model files are not available")
class IndicF5SmokeTests(unittest.TestCase):
    def test_can_initialize_indicf5_model(self):
        wrapper = EmotionTTS()
        self.assertIsNotNone(wrapper.model)


if __name__ == "__main__":
    unittest.main()
