"""Unit tests for i18n.translations — uk/en parity and t() behavior."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from i18n.translations import TRANSLATIONS, t

PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def _placeholders(value) -> set:
    if isinstance(value, str):
        return set(PLACEHOLDER_RE.findall(value))
    return set()


class TestTranslationParity:
    def test_both_languages_present(self):
        assert set(TRANSLATIONS.keys()) == {"uk", "en"}

    def test_key_sets_match(self):
        """Every key must exist in both uk and en sections."""
        uk_keys = set(TRANSLATIONS["uk"])
        en_keys = set(TRANSLATIONS["en"])
        assert uk_keys - en_keys == set(), f"keys missing in en: {sorted(uk_keys - en_keys)}"
        assert en_keys - uk_keys == set(), f"keys missing in uk: {sorted(en_keys - uk_keys)}"

    def test_format_placeholders_match(self):
        """{placeholders} must be identical across languages for each key."""
        for key, uk_val in TRANSLATIONS["uk"].items():
            en_val = TRANSLATIONS["en"][key]
            assert _placeholders(uk_val) == _placeholders(en_val), f"placeholder mismatch for key '{key}'"

    def test_value_types_match(self):
        """Value types (str vs list) must be identical across languages."""
        for key, uk_val in TRANSLATIONS["uk"].items():
            en_val = TRANSLATIONS["en"][key]
            assert isinstance(en_val, type(uk_val)), f"type mismatch for key '{key}'"

    def test_no_empty_values(self):
        for lang, table in TRANSLATIONS.items():
            for key, val in table.items():
                assert val, f"empty translation for '{key}' in '{lang}'"


class TestTFunction:
    def test_known_key_returns_translation(self):
        assert t("tagline") == TRANSLATIONS["uk"]["tagline"]

    def test_unknown_key_falls_back_to_key(self):
        assert t("__no_such_key__") == "__no_such_key__"

    def test_interpolation(self):
        result = t("rehab_saved", session_id=42)
        assert "42" in result

    def test_missing_format_arg_returns_raw_string(self):
        # Missing kwargs must not raise — the raw template is returned.
        result = t("rehab_saved")
        assert isinstance(result, str)
