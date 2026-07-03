"""Score fusion for ensemble wake words.

Shared by the Wyoming handler (live detection) and the HTTP API's recording
test endpoint so both always agree on what a detection would be.
"""

from __future__ import annotations

from typing import Any

from .models import ModelEntry


def score_from_result(result: Any, model_id: str) -> float:
    """Extract a float score from a NanoWakeWord predict() result."""

    if hasattr(result, "get"):
        score = result.get(model_id, None)
        if score is not None:
            return float(score)

    if hasattr(result, "score"):
        return float(result.score)

    if isinstance(result, dict):
        return float(result.get(model_id, 0.0))

    return 0.0


def fuse_scores(
    entry: ModelEntry, scores: dict[str, float], default_threshold: float
) -> float:
    """Combine member scores into one score for an ensemble wake word."""

    if entry.fusion == "weighted_average":
        total_weight = sum(max(member.weight, 0.0) for member in entry.members)
        if total_weight <= 0:
            return 0.0

        return sum(
            scores[member.model] * max(member.weight, 0.0) for member in entry.members
        ) / total_weight

    if entry.fusion == "all":
        return min(scores.values()) if scores else 0.0

    return _primary_and_verifier_score(entry, scores, default_threshold)


def _primary_and_verifier_score(
    entry: ModelEntry, scores: dict[str, float], default_threshold: float
) -> float:
    primary = next(
        (member for member in entry.members if member.role.lower() == "primary"),
        entry.members[0],
    )
    verifiers = [
        member
        for member in entry.members
        if member.model != primary.model
        and member.role.lower() in {"verifier", "confirm", "member"}
    ]

    primary_score = scores.get(primary.model, 0.0)
    primary_threshold = (
        primary.threshold if primary.threshold is not None else default_threshold
    )
    if primary_score <= primary_threshold:
        return 0.0

    for verifier in verifiers:
        verifier_score = scores.get(verifier.model, 0.0)
        verifier_threshold = (
            verifier.threshold if verifier.threshold is not None else default_threshold
        )
        if verifier_score <= verifier_threshold:
            return 0.0

    return primary_score
