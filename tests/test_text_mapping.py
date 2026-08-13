from __future__ import annotations

from cwr_eg.contracts import CharacterInterval
from cwr_eg.text_mapping import TextNormalizer, mapping_coverage


def test_combining_character_maps_back_conservatively() -> None:
    normalized = TextNormalizer("NFC").normalize("e\u0301x")
    assert normalized.normalized_text == "éx"
    assert normalized.normalized_span_to_raw(0, 1) == CharacterInterval(0, 2)
    assert normalized.normalized_span_to_raw(1, 2) == CharacterInterval(2, 3)


def test_nfkc_expansion_maps_to_one_raw_code_point() -> None:
    normalized = TextNormalizer("NFKC").normalize("ﬀa")
    assert normalized.normalized_text == "ffa"
    assert normalized.normalized_span_to_raw(0, 1) == CharacterInterval(0, 1)
    assert normalized.normalized_span_to_raw(1, 2) == CharacterInterval(0, 1)
    assert normalized.normalized_span_to_raw(2, 3) == CharacterInterval(1, 2)


def test_token_mapping_supports_emoji_and_special_tokens() -> None:
    normalized = TextNormalizer("NFC").normalize("中😀文")
    offsets = [(0, 1), (1, 2), (0, 0), (2, 3)]
    mapped = normalized.map_token_offsets(offsets)
    assert mapped == (
        CharacterInterval(0, 1),
        CharacterInterval(1, 2),
        None,
        CharacterInterval(2, 3),
    )
    assert mapping_coverage(offsets, mapped) == 1.0
