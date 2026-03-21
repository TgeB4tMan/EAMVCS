import os
from pathlib import Path
from typing import Dict, Optional

import torch


DEFAULT_TRANSLATION_MODEL_ID = "facebook/nllb-200-distilled-600M"
DEFAULT_TRANSLATION_MODEL_DIR = "models/nllb-200-distilled-600M"
DEFAULT_SOURCE_LANG = "eng_Latn"

# Mapping from app language codes to NLLB-200 tags.
TARGET_LANG_TO_NLLB: Dict[str, str] = {
    "as": "asm_Beng",
    "bn": "ben_Beng",
    "gu": "guj_Gujr",
    "hi": "hin_Deva",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "or": "ory_Orya",
    "pa": "pan_Guru",
    "ta": "tam_Taml",
    "te": "tel_Telu",
}


class TranslationError(RuntimeError):
    pass


class TranslationService:
    def __init__(
        self,
        model_id: Optional[str] = None,
        model_dir: Optional[str] = None,
        local_files_only: Optional[bool] = None,
        source_lang: str = DEFAULT_SOURCE_LANG,
    ):
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "transformers seq2seq translation runtime is unavailable."
            ) from exc

        self.source_lang = source_lang
        self.project_root = Path(__file__).resolve().parents[1]
        self.model_id = model_id or os.getenv("TRANSLATION_MODEL_ID", DEFAULT_TRANSLATION_MODEL_ID)
        self.local_files_only = self._resolve_local_files_only(local_files_only)
        self.model_dir = model_dir or os.getenv("TRANSLATION_MODEL_DIR") or DEFAULT_TRANSLATION_MODEL_DIR
        self.model_source = self._resolve_model_source()

        device_override = (os.getenv("TRANSLATION_DEVICE") or "").strip().lower()
        if device_override in {"cpu", "cuda"}:
            self.device = torch.device(device_override)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_source,
            local_files_only=self.local_files_only,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_source,
            local_files_only=self.local_files_only,
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_local_files_only(local_files_only: Optional[bool]) -> bool:
        if local_files_only is not None:
            return bool(local_files_only)
        env_value = os.getenv("TRANSLATION_LOCAL_FILES_ONLY", "1").strip().lower()
        return env_value not in {"0", "false", "no"}

    def _resolve_model_source(self) -> str:
        if self.model_dir:
            candidate = Path(self.model_dir).expanduser()
            if not candidate.is_absolute():
                candidate = self.project_root / candidate
            if candidate.exists():
                return str(candidate.resolve())
            cached_snapshot = self._find_cached_model_snapshot()
            if cached_snapshot:
                return cached_snapshot
            if self.local_files_only:
                raise RuntimeError(
                    "Translation model directory not found for offline mode: "
                    f"{candidate}. Download NLLB locally or set TRANSLATION_MODEL_DIR."
                )
        return self.model_id

    def _find_cached_model_snapshot(self) -> Optional[str]:
        # Allow loading from any previously downloaded HF snapshot in offline mode.
        model_dir_name = f"models--{self.model_id.replace('/', '--')}"
        cache_roots = []

        hf_home = os.getenv("HF_HOME")
        if hf_home:
            cache_roots.append(Path(hf_home).expanduser() / "hub")

        hub_cache = os.getenv("HUGGINGFACE_HUB_CACHE")
        if hub_cache:
            cache_roots.append(Path(hub_cache).expanduser())

        cache_roots.append(self.project_root / ".hf_cache" / "hub")
        cache_roots.append(Path.home() / ".cache" / "huggingface" / "hub")

        for root in cache_roots:
            repo_dir = root / model_dir_name
            snapshots_dir = repo_dir / "snapshots"
            if not snapshots_dir.exists():
                continue

            snapshot_dirs = [path for path in snapshots_dir.iterdir() if path.is_dir()]
            snapshot_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)

            for snapshot in snapshot_dirs:
                config_file = snapshot / "config.json"
                if config_file.exists():
                    return str(snapshot.resolve())

        return None

    def _resolve_target_lang_tag(self, target_lang: str) -> str:
        tag = TARGET_LANG_TO_NLLB.get(target_lang)
        if not tag:
            raise TranslationError(f"Unsupported target language for translation: {target_lang}")
        return tag

    def translate_english_to_target(self, text: str, target_lang: str) -> str:
        clean_text = (text or "").strip()
        if not clean_text:
            raise TranslationError("Input text for translation is empty.")

        target_tag = self._resolve_target_lang_tag(target_lang)
        self.tokenizer.src_lang = self.source_lang
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(target_tag)
        if forced_bos_token_id is None or forced_bos_token_id < 0:
            raise TranslationError(f"Unable to resolve translation target tag: {target_tag}")

        tokenized = self.tokenizer(
            clean_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        tokenized = {key: value.to(self.device) for key, value in tokenized.items()}

        with torch.no_grad():
            generated = self.model.generate(
                **tokenized,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=512,
            )
        translated = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        if not translated:
            raise TranslationError("Translation output is empty.")
        return translated
