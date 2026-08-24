from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

_IDENTITY_RE = re.compile(
    r"\b(?:women?|men|female|male|girl|boy|gay|lesbian|bisexual|trans(?:gender)?|queer|"
    r"muslim|christian|jewish|hindu|religion|faith|black|white|asian|latino|latina|"
    r"ethnic|race|immigrant|foreigner|refugee|disabled|disability)\b",
    re.IGNORECASE,
)
_PROFANITY_RE = re.compile(
    r"\b(?:damn|hell|crap|shit|bullshit|bastard|idiot|moron|asshole|jackass)\b",
    re.IGNORECASE,
)
_NEGATION_COUNTER_RE = re.compile(
    r"\b(?:not|never|isn['’]?t|aren['’]?t|don['’]?t|stop|oppose|condemn|against|"
    r"shouldn['’]?t|cannot|can['’]?t)\b",
    re.IGNORECASE,
)
_QUOTED_REPORTED_RE = re.compile(
    r"(?:[\"“”'].+[\"“”']|\b(?:said|says|quoted|reported|called|wrote|tweeted)\b)",
    re.IGNORECASE,
)
_OBFUSCATION_RE = re.compile(
    r"(?:[A-Za-z][.*_\-]{1,3}[A-Za-z]|[A-Za-z]+[013457][A-Za-z0-9]*|[^\w\s]{4,})"
)


def vote_summary(row: Mapping[str, Any]) -> dict[str, float | int | bool]:
    votes = sorted(
        (
            int(row["hate_speech_count"]),
            int(row["offensive_language_count"]),
            int(row["neither_count"]),
        ),
        reverse=True,
    )
    total = int(row["count"])
    return {
        "unanimous": votes[0] == total,
        "vote_margin": votes[0] - votes[1],
        "top_vote_fraction": votes[0] / total,
    }


def matches_slice(row: Mapping[str, Any], specification: Mapping[str, Any]) -> bool:
    allowed_labels = specification.get("labels")
    if allowed_labels is not None and int(row["class"]) not in set(allowed_labels):
        return False

    predicate = specification["predicate"]
    text = str(row.get("tweet", ""))
    votes = vote_summary(row)
    if predicate == "always":
        return True
    if predicate == "unanimous":
        return bool(votes["unanimous"])
    if predicate == "non_unanimous":
        return not bool(votes["unanimous"])
    if predicate == "vote_margin_lte":
        return int(votes["vote_margin"]) <= int(specification["value"])
    if predicate == "top_vote_fraction_lt":
        return float(votes["top_vote_fraction"]) < float(specification["value"])
    if predicate == "identity_mention_v1":
        return bool(_IDENTITY_RE.search(text))
    if predicate == "profanity_v1":
        return bool(_PROFANITY_RE.search(text))
    if predicate == "token_count_lte":
        return len(text.split()) <= int(specification["value"])
    if predicate == "negation_or_counterspeech_v1":
        return bool(_NEGATION_COUNTER_RE.search(text) and _IDENTITY_RE.search(text))
    if predicate == "quoted_or_reported_v1":
        return bool(_QUOTED_REPORTED_RE.search(text))
    if predicate == "obfuscation_v1":
        return bool(_OBFUSCATION_RE.search(text))
    raise ValueError(f"unknown slice predicate: {predicate}")


def slice_metric_report(
    rows: Sequence[Mapping[str, Any]],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    configuration: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Evaluate every predeclared ambiguity/hard-negative slice."""
    if not (len(rows) == len(y_true) == len(y_pred)):
        raise ValueError("slice rows and predictions must align")
    from .metrics import classification_metrics

    output: dict[str, dict[str, Any]] = {}
    for category in ("ambiguity_slices", "hard_negative_slices"):
        output[category] = {}
        for name, specification in configuration[category].items():
            indices = [
                index for index, row in enumerate(rows) if matches_slice(row, specification)
            ]
            output[category][name] = {
                "support": len(indices),
                "metrics": (
                    classification_metrics(
                        [int(y_true[index]) for index in indices],
                        [int(y_pred[index]) for index in indices],
                    )
                    if indices
                    else None
                ),
            }
    return output
