"""Unit tests for the ASR verification rule (no network; pure functions)."""

from wyoming_nanowakeword.asr_verify import passes_prompted, passes_unbiased

KEYWORDS = ["agat", "agad", "agac"]


def _verbose_json(words: list[tuple[str, float]]) -> dict:
    return {
        "segments": [
            {
                "words": [
                    {"word": text, "probability": prob} for text, prob in words
                ]
            }
        ]
    }


def test_unbiased_accepts_wake_word_anywhere() -> None:
    assert passes_unbiased("Agata, włącz światło.", KEYWORDS)
    assert passes_unbiased("no to agatka przyszła", KEYWORDS)


def test_unbiased_rejects_lookalikes() -> None:
    # Real transcripts of mined false positives: literal substring match
    # only, so lookalike words within edit distance 2 do not pass.
    assert not passes_unbiased("i edukacji, ale nie stać", KEYWORDS)
    assert not passes_unbiased("stare akta sprawy", KEYWORDS)
    assert not passes_unbiased("teoria Hertzberga", KEYWORDS)
    assert not passes_unbiased("", KEYWORDS)


def test_prompted_accepts_confident_word() -> None:
    result = _verbose_json([(" Ag", 0.95), ("ata", 0.99), (" słucham", 0.9)])
    passed, detail = passes_prompted(result, KEYWORDS, min_prob=0.68)
    assert passed
    assert "agata" in detail


def test_prompted_rejects_low_confidence_hallucination() -> None:
    # Prompt-biased whisper hallucinates the wake word into unclear audio,
    # but with low token probability.
    result = _verbose_json([(" Ag", 0.51), ("ata", 0.60)])
    passed, _ = passes_prompted(result, KEYWORDS, min_prob=0.68)
    assert not passed


def test_prompted_rejects_when_word_absent() -> None:
    result = _verbose_json([(" dzień", 0.99), (" dobry", 0.99)])
    passed, detail = passes_prompted(result, KEYWORDS, min_prob=0.68)
    assert not passed
    assert "not in prompted transcript" in detail


def test_prompted_merges_subword_pieces() -> None:
    # whisper.cpp splits words into pieces; only the first piece starts
    # with a space. "Agatka" split three ways must still match.
    result = _verbose_json([(" A", 0.9), ("gat", 0.92), ("ka", 0.94)])
    passed, detail = passes_prompted(result, KEYWORDS, min_prob=0.68)
    assert passed
    assert "agatka" in detail


def test_prompted_fuzzy_matches_inflected_form() -> None:
    # Prompted pass allows edit distance <= 2 for inflections the substring
    # misses ("agacie" contains 'agac' though; use a distance-2 form).
    result = _verbose_json([(" agta", 0.9)])
    passed, _ = passes_prompted(result, ["agata"], min_prob=0.68)
    assert passed
