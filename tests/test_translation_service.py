import unittest

from Backend.translation_service import TARGET_LANG_TO_NLLB, TranslationService


class TranslationServiceContractTests(unittest.TestCase):
    def test_nllb_mapping_covers_all_supported_target_languages(self):
        expected_codes = {"as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"}
        self.assertEqual(set(TARGET_LANG_TO_NLLB.keys()), expected_codes)
        for code, nllb_tag in TARGET_LANG_TO_NLLB.items():
            self.assertIsInstance(nllb_tag, str)
            self.assertIn("_", nllb_tag, msg=f"Invalid NLLB tag for {code}: {nllb_tag}")

    def test_offline_mode_requires_local_translation_model_directory(self):
        with self.assertRaises(RuntimeError) as ctx:
            TranslationService(
                model_dir="models/this_translation_model_does_not_exist",
                local_files_only=True,
            )
        self.assertIn("offline mode", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
