def text_to_phonemes(text, language="en"):
    """Convert text to phonemes for multilingual TTS."""
    try:
        from phonemizer.backend import EspeakBackend
        from phonemizer.phonemize import phonemize
        
        # Language mapping for espeak
        language_map = {
            "en": "en-us",  # English US
            "es": "es",     # Spanish
            "fr": "fr-fr",   # French
            "de": "de",      # German
            "it": "it",      # Italian
            "pt": "pt-br",   # Portuguese Brazil
            "ja": "ja",      # Japanese
            "ko": "ko",      # Korean
            "zh": "zh",      # Chinese
            "hi": "hi",      # Hindi
            "ar": "ar",      # Arabic
            "ru": "ru",      # Russian
        }
        
        # Get espeak language code
        espeak_lang = language_map.get(language, "en-us")
        
        import os
        # Set espeak path for Windows if available
        espeak_path = r"C:\Program Files (x86)\eSpeak\command_line\espeak.exe"
        if os.path.exists(espeak_path):
            os.environ['PHONEMIZER_ESPEAK_EXECUTABLE'] = espeak_path

        backend = EspeakBackend(language=espeak_lang)
        phonemes = backend.phonemize([text])[0]

        return phonemes
        
    except ImportError:
        print("Warning: phonemizer not installed. Using original text.")
        print("Install with: pip install phonemizer")
        return text
    except Exception as e:
        print(f"Phoneme conversion failed: {e}")
        return text

def get_supported_languages():
    """Get list of supported languages."""
    return {
        "en": "English (US)",
        "es": "Spanish", 
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese (Brazil)",
        "ja": "Japanese",
        "ko": "Korean",
        "zh": "Chinese",
        "hi": "Hindi",
        "ar": "Arabic",
        "ru": "Russian"
    }
