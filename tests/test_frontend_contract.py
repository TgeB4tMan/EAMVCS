import unittest
from pathlib import Path


class FrontendContractTests(unittest.TestCase):
    def setUp(self):
        self.script = Path("Frontend/script.js").read_text(encoding="utf-8")
        self.html = Path("Frontend/index.html").read_text(encoding="utf-8")

    def test_generate_formdata_contains_required_fields(self):
        self.assertIn('formData.append("text"', self.script)
        self.assertIn('formData.append("language"', self.script)
        self.assertIn('formData.append("ref_lang"', self.script)
        self.assertIn('formData.append("audio"', self.script)
        self.assertIn('formData.append("ref_text"', self.script)

    def test_generate_uses_json_then_audio_fetch(self):
        self.assertIn("await response.json()", self.script)
        self.assertIn("/audio/${encodeURIComponent(result.audio_path)}", self.script)
        self.assertIn("generatedAudio = await audioResponse.blob()", self.script)

    def test_reference_language_and_optional_text_inputs_exist(self):
        self.assertIn('id="refLangSelect"', self.html)
        self.assertIn('id="refTextInput"', self.html)


if __name__ == "__main__":
    unittest.main()
