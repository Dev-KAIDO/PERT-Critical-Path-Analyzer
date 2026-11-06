"""
Validation helpers for review actions (activity IDs, durations).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Optional

_ACTIVITY_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,40}$")


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    value: Any = None
    message: str = ""


def validate_activity_id(text: str) -> ValidationResult:
    """Validate an activity ID entered for a CORRECT decision."""
    text = text.strip()
    if not text:
        return ValidationResult(ok=False, value=None, message="Activity ID cannot be empty")
    if not _ACTIVITY_ID_RE.match(text):
        return ValidationResult(
            ok=False,
            value=None,
            message="Invalid identifier (use letters, digits, -, _, 1\u201340 chars)",
        )
    return ValidationResult(ok=True, value=text)


def is_duplicate_activity_id(
    text: str,
    session: Any,
    exclude_node_id: Optional[str] = None,
) -> Optional[str]:
    """
    Check whether *text* collides with any already-resolved activity ID.

    Returns the geometric_node_id of the conflicting activity or None.
    """
    from pert_analyzer.pipeline.human_review import ReviewStatus

    review_session = getattr(session, "review_session", None) if not hasattr(session, "activities") else session
    if review_session is None:
        return None

    for act in getattr(review_session, "activities", []) or []:
        node_id = getattr(act, "geometric_node_id", None)
        if node_id and node_id == exclude_node_id:
            continue
        status = getattr(act, "status", None)
        resolved = getattr(act, "resolved_activity_id", None)
        if status == ReviewStatus.CORRECTED and act.corrected_activity_id == text:
            return node_id
        if status == ReviewStatus.ACCEPTED and act.current_activity_id == text:
            return node_id
    return None


def validate_duration(text: str) -> ValidationResult:
    """
    Validate a duration value entered for a CORRECT decision.

    Rejects NaN, infinity, negative values, and zero (CPM requires
    positive durations; dummy activities are not entered through this
    review UI).
    """
    text = text.strip()
    if not text:
        return ValidationResult(ok=False, value=None, message="Duration cannot be empty")

    try:
        value = float(text)
    except ValueError:
        return ValidationResult(ok=False, value=None, message="Not a valid number")

    if math.isnan(value):
        return ValidationResult(ok=False, value=None, message="Duration cannot be NaN")
    if math.isinf(value):
        return ValidationResult(ok=False, value=None, message="Duration cannot be infinite")
    if value < 0:
        return ValidationResult(ok=False, value=None, message="Duration must be positive")
    if value == 0:
        return ValidationResult(ok=False, value=None, message="Duration cannot be zero (use LEAVE UNRESOLVED to keep as-is)")

    return ValidationResult(ok=True, value=value)


def _duration_component(label: str, text: str) -> ValidationResult:
    """One non-empty positive number component of a three-point entry."""
    text = text.strip()
    if not text:
        return ValidationResult(
            ok=False, value=None, message=f"{label} cannot be empty"
        )
    try:
        value = float(text)
    except ValueError:
        return ValidationResult(
            ok=False, value=None, message=f"{label} is not a valid number"
        )
    if math.isnan(value):
        return ValidationResult(
            ok=False, value=None, message=f"{label} cannot be NaN"
        )
    if math.isinf(value):
        return ValidationResult(
            ok=False, value=None, message=f"{label} cannot be infinite"
        )
    if value <= 0:
        return ValidationResult(
            ok=False, value=None, message=f"{label} must be positive"
        )
    return ValidationResult(ok=True, value=value)


def validate_duration_triplet(
    optimistic_text: str,
    most_likely_text: str,
    pessimistic_text: str,
) -> ValidationResult:
    """
    Validate a corrected three-point estimate (O <= M <= P) for the
    three-point duration editor (A-15).

    Returns a ValidationResult whose value is the (o, m, p) float tuple;
    the pipeline re-checks the same ordering when the decision is applied.
    """
    for label, text in (
        ("Optimistic (O)", optimistic_text),
        ("Most Likely (M)", most_likely_text),
        ("Pessimistic (P)", pessimistic_text),
    ):
        result = _duration_component(label, text)
        if not result.ok:
            return result
    o = float(optimistic_text)
    m = float(most_likely_text)
    p = float(pessimistic_text)
    if o > m:
        return ValidationResult(
            ok=False,
            value=None,
            message=f"Optimistic (O) {o:g} exceeds Most Likely (M) {m:g}",
        )
    if p < m:
        return ValidationResult(
            ok=False,
            value=None,
            message=f"Pessimistic (P) {p:g} is below Most Likely (M) {m:g}",
        )
    return ValidationResult(ok=True, value=(o, m, p))


def geometric_node_label(geometric_node_id: Any, semantic_id: Optional[str]) -> str:
    """Human-friendly label for a geometric node."""
    g = str(geometric_node_id) if geometric_node_id is not None else "?"
    s = str(semantic_id) if semantic_id else ""
    if s and s != g:
        return f"{g} \u2192 {s}"
    return g
