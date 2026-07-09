"""ASR (whisper) verification of candidate wake word detections.

A sensitive wake word model behind an ASR check gives both high recall and
near-zero false accepts: the model proposes, whisper disposes. The candidate
audio is transcribed twice by a whisper.cpp-compatible ``/inference`` server:

1. **Unbiased pass** (no prompt): the transcript must contain one of the
   configured keyword substrings. A prompted decode readily hallucinates the
   wake word into lookalike audio ("Herzberga", "akta"), an unprompted one
   does not — this pass is the false-accept gate.
2. **Prompted pass** (optional, when ``verify_asr_prompt`` is set): decoding
   biased toward the wake word must find a matching word with mean token
   probability >= ``verify_asr_min_prob``. This recovers hard genuine
   pronunciations (diminutives, distorted voices) that gate recall.

Word timestamps from whisper.cpp are NOT used: they are unreliable on
mostly-silent segments (a 0.5 s word can be reported spanning 2+ seconds).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import aiohttp

if TYPE_CHECKING:
    from .settings import ServerSettings

_LOGGER = logging.getLogger(__name__)


def _keywords(settings: ServerSettings) -> list[str]:
    return [
        keyword.strip().lower()
        for keyword in settings.verify_asr_keyword.split(",")
        if keyword.strip()
    ]


def _edit_distance(a: str, b: str) -> int:
    distances = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        previous, distances[0] = distances[0], i
        for j, char_b in enumerate(b, 1):
            previous, distances[j] = distances[j], min(
                distances[j] + 1,
                distances[j - 1] + 1,
                previous + (char_a != char_b),
            )
    return distances[-1]


def _normalize(word: str) -> str:
    return "".join(char for char in word.strip().lower() if char.isalpha())


def passes_unbiased(text: str, keywords: list[str]) -> bool:
    """The unprompted transcript must literally contain a keyword.

    Substring-only on purpose: fuzzy matching here admits real lookalike
    words (Polish "akta" is edit distance 2 from "agata").
    """

    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def passes_prompted(
    result: dict[str, Any], keywords: list[str], min_prob: float
) -> tuple[bool, str]:
    """Find the wake word in a prompted verbose_json decode and gate on
    its mean token probability. Returns (passed, detail-for-logs)."""

    words: list[dict[str, Any]] = [
        word
        for segment in result.get("segments", [])
        for word in segment.get("words", [])
    ]
    # whisper.cpp splits words into sub-tokens (" Ag" + "ata"); merge pieces
    # that do not start a new word.
    merged: list[dict[str, Any]] = []
    for word in words:
        piece = {
            "text": _normalize(str(word.get("word", ""))),
            "probs": [float(word.get("probability", 0.0))],
        }
        if merged and not str(word.get("word", "")).startswith(" "):
            merged[-1]["text"] += piece["text"]
            merged[-1]["probs"] += piece["probs"]
        else:
            merged.append(piece)

    for candidate in merged:
        text = candidate["text"]
        is_match = any(keyword in text for keyword in keywords) or (
            len(text) >= 3
            and any(_edit_distance(text, keyword) <= 2 for keyword in keywords)
        )
        if is_match:
            probability = sum(candidate["probs"]) / len(candidate["probs"])
            return (
                probability >= min_prob,
                f"word={text!r} prob={probability:.2f}",
            )
    return False, "wake word not in prompted transcript"


async def _transcribe(
    session: aiohttp.ClientSession,
    settings: ServerSettings,
    wav_bytes: bytes,
    prompt: str | None,
) -> dict[str, Any]:
    form = aiohttp.FormData()
    form.add_field("file", wav_bytes, filename="candidate.wav")
    form.add_field("response_format", "verbose_json")
    if settings.verify_asr_language:
        form.add_field("language", settings.verify_asr_language)
    if prompt:
        form.add_field("prompt", prompt)

    async with session.post(
        settings.verify_asr_url,
        data=form,
        timeout=aiohttp.ClientTimeout(total=settings.verify_asr_timeout),
    ) as response:
        if response.status >= 400:
            raise RuntimeError(
                f"ASR verifier returned {response.status}: "
                f"{(await response.text())[:200]}"
            )
        return await response.json(content_type=None)


async def verify_wake_word(
    session: aiohttp.ClientSession,
    settings: ServerSettings,
    wav_bytes: bytes,
) -> tuple[bool | None, str]:
    """Run the dual-pass check. Returns (verdict, detail).

    verdict is None when the ASR server could not be reached or answered
    with an error — the caller decides via ``verify_fail_open``.
    """

    keywords = _keywords(settings)
    if not keywords:
        return None, "verify_asr_keyword is empty"

    try:
        unbiased = await _transcribe(session, settings, wav_bytes, None)
        text = str(unbiased.get("text", ""))
        if not passes_unbiased(text, keywords):
            return False, f"unbiased transcript {text.strip()!r}"

        if not settings.verify_asr_prompt:
            return True, f"unbiased transcript {text.strip()!r}"

        prompted = await _transcribe(
            session, settings, wav_bytes, settings.verify_asr_prompt
        )
        passed, detail = passes_prompted(
            prompted, keywords, settings.verify_asr_min_prob
        )
        return passed, detail
    except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as err:
        return None, str(err)
