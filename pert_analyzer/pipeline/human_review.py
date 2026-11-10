"""
Human review data models.

Defines structured review issues, correction actions, the review
session (activities / dependencies / durations under review), the
deterministic application of review decisions into a validated
graph candidate, and the CPM gate.

Core principle: the CV pipeline never invents uncertain dependencies.
STRONG evidence is accepted automatically; WEAK/AMBIGUOUS evidence is
forwarded to human review; INVALID/FALSE evidence is rejected.  Review
decisions are stored as a separate decision layer and NEVER overwrite
the original raw OCR/CV evidence.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


def _utc_now() -> str:
    """Current UTC timestamp as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# Legacy review types (kept for backward compatibility)
# =============================================================================


class ReviewIssueType(Enum):
    """Types of review issues that can arise during reconstruction."""

    UNCERTAIN_DURATION = "UNCERTAIN_DURATION"
    UNCERTAIN_ACTIVITY_ID = "UNCERTAIN_ACTIVITY_ID"
    UNCERTAIN_ACTIVITY_LABEL = "UNCERTAIN_ACTIVITY_LABEL"
    AMBIGUOUS_ARROW_TARGET = "AMBIGUOUS_ARROW_TARGET"
    AMBIGUOUS_ARROW_SOURCE = "AMBIGUOUS_ARROW_SOURCE"
    DUPLICATE_ACTIVITY_ID = "DUPLICATE_ACTIVITY_ID"
    MISSING_DURATION = "MISSING_DURATION"
    ISOLATED_NODE = "ISOLATED_NODE"
    LOW_OCR_CONFIDENCE = "LOW_OCR_CONFIDENCE"
    LOW_SHAPE_CONFIDENCE = "LOW_SHAPE_CONFIDENCE"
    ORPHAN_ARROW = "ORPHAN_ARROW"
    DIAGRAM_TYPE_UNCERTAIN = "DIAGRAM_TYPE_UNCERTAIN"
    DISCONNECTED_GRAPH = "DISCONNECTED_GRAPH"
    CYCLE_DETECTED = "CYCLE_DETECTED"
    MISSING_DEPENDENCY = "MISSING_DEPENDENCY"
    FALSE_DETECTION = "FALSE_DETECTION"


@dataclass
class CandidateValue:
    """A candidate value with confidence for a review item."""

    value: Any
    confidence: float = 0.0
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "confidence": round(self.confidence, 3),
            "source": self.source,
        }


@dataclass
class ReviewIssue:
    """
    A single review issue requiring human attention.

    Captures what is uncertain, which elements are involved,
    what alternatives exist, and suggested resolution.
    """

    issue_type: ReviewIssueType
    description: str
    element_id: str = ""
    element_type: str = ""
    candidates: List[CandidateValue] = field(default_factory=list)
    suggested_resolution: str = ""
    severity: str = "WARNING"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.issue_type.value,
            "description": self.description,
            "element_id": self.element_id,
            "element_type": self.element_type,
            "candidates": [c.to_dict() for c in self.candidates],
            "suggested_resolution": self.suggested_resolution,
            "severity": self.severity,
        }


@dataclass
class CorrectionAction:
    """
    A correction to be applied to the reconstructed diagram.

    Represents a single change such as updating an activity ID,
    duration, arrow source/target, or removing a false detection.
    """

    action_type: str
    target_id: str = ""
    field_name: str = ""
    old_value: Any = None
    new_value: Any = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "target_id": self.target_id,
            "field_name": self.field_name,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "reason": self.reason,
        }


@dataclass
class HumanReviewItem:
    """
    A structured review item for a single element.

    Contains the detected value, alternatives, and the
    correction that was applied (if any).
    """

    element_id: str
    element_type: str
    field_name: str
    detected_value: Any
    detected_confidence: float = 0.0
    alternatives: List[CandidateValue] = field(default_factory=list)
    correction: Optional[CorrectionAction] = None
    is_resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "element_id": self.element_id,
            "element_type": self.element_type,
            "field_name": self.field_name,
            "detected_value": self.detected_value,
            "detected_confidence": round(self.detected_confidence, 3),
            "alternatives": [a.to_dict() for a in self.alternatives],
            "is_resolved": self.is_resolved,
        }
        if self.correction:
            result["correction"] = self.correction.to_dict()
        return result


@dataclass
class HumanReviewResult:
    """
    Complete result of human review processing.

    Contains all review issues found, corrections applied,
    and the final state after review.
    """

    issues: List[ReviewIssue] = field(default_factory=list)
    items: List[HumanReviewItem] = field(default_factory=list)
    corrections: List[CorrectionAction] = field(default_factory=list)
    is_fully_resolved: bool = False

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    @property
    def correction_count(self) -> int:
        return len(self.corrections)

    @property
    def unresolved_count(self) -> int:
        return sum(1 for item in self.items if not item.is_resolved)

    def add_issue(self, issue: ReviewIssue) -> None:
        """Add a review issue."""
        self.issues.append(issue)

    def add_correction(self, correction: CorrectionAction) -> None:
        """Record a correction action."""
        self.corrections.append(correction)

    def get_issues_by_type(
        self, issue_type: ReviewIssueType
    ) -> List[ReviewIssue]:
        """Get all issues of a specific type."""
        return [i for i in self.issues if i.issue_type == issue_type]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_count": self.issue_count,
            "correction_count": self.correction_count,
            "unresolved_count": self.unresolved_count,
            "is_fully_resolved": self.is_fully_resolved,
            "issues": [i.to_dict() for i in self.issues],
            "corrections": [c.to_dict() for c in self.corrections],
        }


# =============================================================================
# Review model
# =============================================================================


class ReviewStatus(Enum):
    """State of a single review item."""

    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    CORRECTED = "CORRECTED"


class DependencyAction(Enum):
    """Actions a reviewer can take on a dependency review item."""

    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    REVERSE = "REVERSE"
    CHANGE_SOURCE = "CHANGE_SOURCE"
    CHANGE_TARGET = "CHANGE_TARGET"


class DurationStatus(Enum):
    """Confidence state of a duration."""

    CONFIRMED = "CONFIRMED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    CORRECTED = "CORRECTED"


class ApprovalState(Enum):
    """Overall approval state of a review session."""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class GraphStatus(Enum):
    """Validation status of a graph model."""

    VALID = "VALID"
    INVALID = "INVALID"


class CpmGateStatus(Enum):
    """Whether CPM is allowed to run on a candidate graph."""

    RUNNABLE = "RUNNABLE"
    BLOCKED_REVIEW = "BLOCKED_REVIEW"


@dataclass
class ReviewEvidence:
    """A reference to the original evidence supporting a review item."""

    source: str
    reference_ids: List[str] = field(default_factory=list)
    description: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "reference_ids": list(self.reference_ids),
            "description": self.description,
            "confidence": round(self.confidence, 3),
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class ActivityReview:
    """
    Review of an activity's semantic identity.

    The geometric node always stays valid regardless of OCR quality;
    only the proposed semantic ID is under review.
    """

    geometric_node_id: str
    current_activity_id: Optional[str] = None
    proposed_activity_id: Optional[str] = None
    alternatives: List[CandidateValue] = field(default_factory=list)
    confidence: float = 0.0
    evidence: List[ReviewEvidence] = field(default_factory=list)
    reason: str = ""
    status: ReviewStatus = ReviewStatus.PENDING
    provenance: str = ""
    user_decision: Optional[str] = None
    corrected_activity_id: Optional[str] = None
    comment: str = ""
    timestamp: str = ""

    # Raw OCR semantic id is preserved (never overwritten; used for audit only)
    raw_label: str = ""
    raw_semantic_id: Optional[str] = None

    @property
    def current_value(self) -> Optional[str]:
        return self.current_activity_id

    @property
    def resolved_activity_id(self) -> Optional[str]:
        if self.status == ReviewStatus.CORRECTED and self.corrected_activity_id:
            return self.corrected_activity_id
        if self.status == ReviewStatus.ACCEPTED:
            return self.current_activity_id
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "geometric_node_id": self.geometric_node_id,
            "current_activity_id": self.current_activity_id,
            "proposed_activity_id": self.proposed_activity_id,
            "alternatives": [a.to_dict() for a in self.alternatives],
            "confidence": round(self.confidence, 3),
            "evidence": [e.to_dict() for e in self.evidence],
            "reason": self.reason,
            "status": self.status.value,
            "provenance": self.provenance,
            "user_decision": self.user_decision,
            "corrected_activity_id": self.corrected_activity_id,
            "comment": self.comment,
            "timestamp": self.timestamp,
            "raw_label": self.raw_label,
            "raw_semantic_id": self.raw_semantic_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActivityReview":
        return cls(
            geometric_node_id=data["geometric_node_id"],
            current_activity_id=data.get("current_activity_id"),
            proposed_activity_id=data.get("proposed_activity_id"),
            alternatives=[
                CandidateValue(**alt) for alt in data.get("alternatives", [])
            ],
            confidence=data.get("confidence", 0.0),
            evidence=[ReviewEvidence(**e) for e in data.get("evidence", [])],
            reason=data.get("reason", ""),
            status=ReviewStatus[data["status"]],
            provenance=data.get("provenance", ""),
            user_decision=data.get("user_decision"),
            corrected_activity_id=data.get("corrected_activity_id"),
            comment=data.get("comment", ""),
            timestamp=data.get("timestamp", ""),
            raw_label=data.get("raw_label", ""),
            raw_semantic_id=data.get("raw_semantic_id"),
        )


@dataclass
class DependencyReview:
    """
    Review of a possibly-ambiguous dependency (arrow).

    Stores the geometric source/target nodes, the proposed direction,
    an alternative direction, confidence, evidence factors, the reason
    for review, and the reviewer action.
    """

    arrow_id: Optional[str] = None
    source_node_id: Optional[str] = None
    target_node_id: Optional[str] = None
    current_source_id: Optional[str] = None
    current_target_id: Optional[str] = None
    proposed_direction: Optional[str] = None
    alternative_direction: Optional[str] = None
    confidence: float = 0.0
    evidence: List[ReviewEvidence] = field(default_factory=list)
    reason: str = ""
    status: ReviewStatus = ReviewStatus.PENDING
    provenance: str = ""
    action: Optional[DependencyAction] = None
    user_decision: Optional[str] = None
    corrected_source_id: Optional[str] = None
    corrected_target_id: Optional[str] = None
    comment: str = ""
    timestamp: str = ""

    @property
    def current_direction(self) -> Optional[str]:
        if self.current_source_id is not None and self.current_target_id is not None:
            return f"{self.current_source_id}->{self.current_target_id}"
        return self.proposed_direction

    @property
    def is_resolved(self) -> bool:
        return self.status != ReviewStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arrow_id": self.arrow_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "current_source_id": self.current_source_id,
            "current_target_id": self.current_target_id,
            "proposed_direction": self.proposed_direction,
            "alternative_direction": self.alternative_direction,
            "confidence": round(self.confidence, 3),
            "evidence": [e.to_dict() for e in self.evidence],
            "reason": self.reason,
            "status": self.status.value,
            "provenance": self.provenance,
            "action": self.action.value if self.action else None,
            "user_decision": self.user_decision,
            "corrected_source_id": self.corrected_source_id,
            "corrected_target_id": self.corrected_target_id,
            "comment": self.comment,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DependencyReview":
        return cls(
            arrow_id=data.get("arrow_id"),
            source_node_id=data.get("source_node_id"),
            target_node_id=data.get("target_node_id"),
            current_source_id=data.get("current_source_id"),
            current_target_id=data.get("current_target_id"),
            proposed_direction=data.get("proposed_direction"),
            alternative_direction=data.get("alternative_direction"),
            confidence=data.get("confidence", 0.0),
            evidence=[ReviewEvidence(**e) for e in data.get("evidence", [])],
            reason=data.get("reason", ""),
            status=ReviewStatus[data["status"]],
            provenance=data.get("provenance", ""),
            action=(
                DependencyAction[data["action"]] if data.get("action") else None
            ),
            user_decision=data.get("user_decision"),
            corrected_source_id=data.get("corrected_source_id"),
            corrected_target_id=data.get("corrected_target_id"),
            comment=data.get("comment", ""),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class DurationReview:
    """
    Review of an activity duration.

    A missing/invalid duration (e.g. OCR produced 0) is represented as
    REVIEW_REQUIRED and must be corrected by the reviewer before CPM.
    The system never guesses a duration.
    """

    geometric_node_id: Optional[str] = None
    activity_id: Optional[str] = None
    current_duration: Optional[float] = None
    proposed_duration: Optional[float] = None
    duration_status: DurationStatus = DurationStatus.REVIEW_REQUIRED
    confidence: float = 0.0
    evidence: List[ReviewEvidence] = field(default_factory=list)
    reason: str = ""
    status: ReviewStatus = ReviewStatus.PENDING
    corrected_duration: Optional[float] = None
    # A-15: corrected three-point extremes (m lives in corrected_duration).
    corrected_optimistic: Optional[float] = None
    corrected_pessimistic: Optional[float] = None
    user_decision: Optional[str] = None
    comment: str = ""
    timestamp: str = ""
    provenance: str = ""

    @property
    def resolved_duration(self) -> Optional[float]:
        if self.status == ReviewStatus.CORRECTED and self.corrected_duration is not None:
            return self.corrected_duration
        if self.status == ReviewStatus.ACCEPTED and self.current_duration is not None:
            return self.current_duration
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "geometric_node_id": self.geometric_node_id,
            "activity_id": self.activity_id,
            "current_duration": self.current_duration,
            "proposed_duration": self.proposed_duration,
            "duration_status": self.duration_status.value,
            "confidence": round(self.confidence, 3),
            "evidence": [e.to_dict() for e in self.evidence],
            "reason": self.reason,
            "status": self.status.value,
            "corrected_duration": self.corrected_duration,
            "corrected_optimistic": self.corrected_optimistic,
            "corrected_pessimistic": self.corrected_pessimistic,
            "user_decision": self.user_decision,
            "comment": self.comment,
            "timestamp": self.timestamp,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DurationReview":
        return cls(
            geometric_node_id=data.get("geometric_node_id"),
            activity_id=data.get("activity_id"),
            current_duration=data.get("current_duration"),
            proposed_duration=data.get("proposed_duration"),
            duration_status=DurationStatus[data.get("duration_status", "REVIEW_REQUIRED")],
            confidence=data.get("confidence", 0.0),
            evidence=[ReviewEvidence(**e) for e in data.get("evidence", [])],
            reason=data.get("reason", ""),
            status=ReviewStatus[data["status"]],
            corrected_duration=data.get("corrected_duration"),
            corrected_optimistic=data.get("corrected_optimistic"),
            corrected_pessimistic=data.get("corrected_pessimistic"),
            user_decision=data.get("user_decision"),
            comment=data.get("comment", ""),
            timestamp=data.get("timestamp", ""),
            provenance=data.get("provenance", ""),
        )


# =============================================================================
# Audit trail
# =============================================================================


@dataclass
class ReviewDecision:
    """
    An immutable audit-trail entry for a single user decision.

    Preserves the original value, the corrected value, the decision,
    reason/comment, timestamp, and a reference to the underlying evidence.
    """

    item_type: str
    item_id: str
    decision: str
    original_value: Any = None
    corrected_value: Any = None
    reason: str = ""
    comment: str = ""
    timestamp: str = ""
    evidence_reference: str = ""
    provenance: str = "human_review"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_type": self.item_type,
            "item_id": self.item_id,
            "decision": self.decision,
            "original_value": self.original_value,
            "corrected_value": self.corrected_value,
            "reason": self.reason,
            "comment": self.comment,
            "timestamp": self.timestamp,
            "evidence_reference": self.evidence_reference,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewDecision":
        return cls(
            item_type=data["item_type"],
            item_id=data["item_id"],
            decision=data["decision"],
            original_value=data.get("original_value"),
            corrected_value=data.get("corrected_value"),
            reason=data.get("reason", ""),
            comment=data.get("comment", ""),
            timestamp=data.get("timestamp", ""),
            evidence_reference=data.get("evidence_reference", ""),
            provenance=data.get("provenance", "human_review"),
        )


# =============================================================================
# Review session
# =============================================================================


@dataclass
class ReviewSession:
    """
    A complete human-review session for one source image.

    The original reconstruction is referenced but never mutated; all
    review decisions are stored in the separate decisions layer.
    """

    source_image_id: str
    created_at: str = ""
    version: str = "1.0"
    reconstruction: Any = None  # immutable reference (ReconstructedDiagram)
    activities: List[ActivityReview] = field(default_factory=list)
    dependencies: List[DependencyReview] = field(default_factory=list)
    durations: List[DurationReview] = field(default_factory=list)
    ambiguities: List[Any] = field(default_factory=list)
    decisions: List[ReviewDecision] = field(default_factory=list)
    approval_state: ApprovalState = ApprovalState.PENDING
    approved_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = _utc_now()

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get_activity_review(self, geometric_node_id: str) -> Optional[ActivityReview]:
        for item in self.activities:
            if item.geometric_node_id == geometric_node_id:
                return item
        return None

    def get_dependency_review(self, arrow_id: str) -> Optional[DependencyReview]:
        for item in self.dependencies:
            if item.arrow_id == arrow_id:
                return item
        return None

    def get_dependency_review_by_arrow(
        self, arrow_id: str
    ) -> Optional[DependencyReview]:
        return self.get_dependency_review(arrow_id)

    def get_duration_review(self, geometric_node_id: str) -> Optional[DurationReview]:
        for item in self.durations:
            if item.geometric_node_id == geometric_node_id:
                return item
        return None

    # ------------------------------------------------------------------
    # Reviewer actions
    # ------------------------------------------------------------------

    def _record(
        self,
        item_type: str,
        item_id: str,
        decision: str,
        original_value: Any,
        corrected_value: Any,
        reason: str,
        comment: str,
        evidence_reference: str,
    ) -> ReviewDecision:
        entry = ReviewDecision(
            item_type=item_type,
            item_id=item_id,
            decision=decision,
            original_value=original_value,
            corrected_value=corrected_value,
            reason=reason,
            comment=comment,
            timestamp=_utc_now(),
            evidence_reference=evidence_reference,
        )
        self.decisions.append(entry)
        return entry

    def decide_activity(
        self,
        geometric_node_id: str,
        decision: str,
        corrected_id: Optional[str] = None,
        reason: str = "",
        comment: str = "",
        return_item: bool = False,
    ) -> Any:
        """
        Decide on an activity review item.

        decision: "ACCEPT", "CORRECT", "REJECT", "LEAVE_UNRESOLVED".
        """
        item = self.get_activity_review(geometric_node_id)
        if item is None:
            raise KeyError(
                f"No activity review for geometric node '{geometric_node_id}'"
            )
        if item.status != ReviewStatus.PENDING:
            return (False, item) if return_item else False

        ts = _utc_now()
        item.timestamp = ts
        item.comment = comment
        item.user_decision = decision

        if decision == "ACCEPT":
            item.status = ReviewStatus.ACCEPTED
            self._record(
                "activity", geometric_node_id, "ACCEPT",
                item.current_activity_id, item.current_activity_id,
                reason, comment, item.geometric_node_id,
            )
        elif decision == "CORRECT":
            if not corrected_id:
                raise ValueError("corrected_id is required for CORRECT decision")
            item.corrected_activity_id = corrected_id
            item.status = ReviewStatus.CORRECTED
            self._record(
                "activity", geometric_node_id, "CORRECT_ID",
                item.current_activity_id, corrected_id,
                reason, comment, item.geometric_node_id,
            )
        elif decision == "REJECT":
            item.status = ReviewStatus.REJECTED
            self._record(
                "activity", geometric_node_id, "REJECT",
                item.current_activity_id, None,
                reason, comment, item.geometric_node_id,
            )
        elif decision == "LEAVE_UNRESOLVED":
            # Explicit non-decision: item stays PENDING, nothing recorded.
            pass
        else:
            raise ValueError(f"Unknown activity decision: {decision}")

        return (True, item) if return_item else True

    def decide_dependency(
        self,
        arrow_id: str,
        action: str,
        corrected_source_id: Optional[str] = None,
        corrected_target_id: Optional[str] = None,
        reason: str = "",
        comment: str = "",
        return_item: bool = False,
    ) -> Any:
        """
        Decide on a dependency review item.

        action: ACCEPT, REJECT, REVERSE, CHANGE_SOURCE, CHANGE_TARGET.
        """
        item = self.get_dependency_review(arrow_id)
        if item is None:
            raise KeyError(f"No dependency review for arrow '{arrow_id}'")
        if item.status != ReviewStatus.PENDING:
            return (False, item) if return_item else False

        action_enum = DependencyAction(action)
        ts = _utc_now()
        item.timestamp = ts
        item.comment = comment
        item.action = action_enum
        item.user_decision = action

        if action_enum == DependencyAction.ACCEPT:
            item.status = ReviewStatus.ACCEPTED
            self._record(
                "dependency", arrow_id, "ACCEPT",
                item.current_direction, item.current_direction,
                reason, comment, item.arrow_id or item.source_node_id or "",
            )
        elif action_enum == DependencyAction.REJECT:
            item.status = ReviewStatus.REJECTED
            self._record(
                "dependency", arrow_id, "REJECT",
                item.current_direction, None,
                reason, comment, item.arrow_id or "",
            )
        elif action_enum == DependencyAction.REVERSE:
            s, t = item.current_source_id, item.current_target_id
            item.corrected_source_id = t
            item.corrected_target_id = s
            item.status = ReviewStatus.CORRECTED
            self._record(
                "dependency", arrow_id, "REVERSE",
                item.current_direction,
                f"{t}->{s}" if s is not None and t is not None else None,
                reason, comment, item.arrow_id or "",
            )
        elif action_enum == DependencyAction.CHANGE_SOURCE:
            if not corrected_source_id:
                raise ValueError(
                    "corrected_source_id is required for CHANGE_SOURCE"
                )
            item.corrected_source_id = corrected_source_id
            item.corrected_target_id = item.current_target_id
            item.status = ReviewStatus.CORRECTED
            self._record(
                "dependency", arrow_id, "CHANGE_SOURCE",
                item.current_direction,
                f"{corrected_source_id}->{item.current_target_id}",
                reason, comment, item.arrow_id or "",
            )
        elif action_enum == DependencyAction.CHANGE_TARGET:
            if not corrected_target_id:
                raise ValueError(
                    "corrected_target_id is required for CHANGE_TARGET"
                )
            item.corrected_source_id = item.current_source_id
            item.corrected_target_id = corrected_target_id
            item.status = ReviewStatus.CORRECTED
            self._record(
                "dependency", arrow_id, "CHANGE_TARGET",
                item.current_direction,
                f"{item.current_source_id}->{corrected_target_id}",
                reason, comment, item.arrow_id or "",
            )

        return (True, item) if return_item else True

    def decide_duration(
        self,
        geometric_node_id: str,
        decision: str,
        corrected_duration: Optional[float] = None,
        corrected_optimistic: Optional[float] = None,
        corrected_pessimistic: Optional[float] = None,
        reason: str = "",
        comment: str = "",
        return_item: bool = False,
    ) -> Any:
        """
        Decide on a duration review item.

        decision: "ACCEPT" (confirm current), "CORRECT" (set corrected_duration),
        "LEAVE_UNRESOLVED".
        """
        item = self.get_duration_review(geometric_node_id)
        if item is None:
            raise KeyError(
                f"No duration review for geometric node '{geometric_node_id}'"
            )
        if item.status != ReviewStatus.PENDING:
            return (False, item) if return_item else False

        ts = _utc_now()
        item.timestamp = ts
        item.comment = comment
        item.user_decision = decision

        if decision == "ACCEPT":
            item.status = ReviewStatus.ACCEPTED
            item.duration_status = DurationStatus.CONFIRMED
            self._record(
                "duration", geometric_node_id, "ACCEPT",
                item.current_duration, item.current_duration,
                reason, comment, item.geometric_node_id or "",
            )
        elif decision == "CORRECT":
            if corrected_duration is None:
                raise ValueError(
                    "corrected_duration is required for CORRECT decision"
                )
            item.corrected_duration = float(corrected_duration)
            m = item.corrected_duration
            # A-15: optional corrected O/P extremes ride the same decision;
            # PERT validation requires 0 <= O <= M <= P on the graph side.
            if corrected_optimistic is not None:
                o = float(corrected_optimistic)
                if o < 0:
                    raise ValueError(
                        "corrected_optimistic must be non-negative"
                    )
                if o > m:
                    raise ValueError(
                        f"corrected_optimistic ({o}) exceeds corrected "
                        f"duration ({m}); O must be <= M"
                    )
                item.corrected_optimistic = o
            if corrected_pessimistic is not None:
                p = float(corrected_pessimistic)
                if p < 0:
                    raise ValueError(
                        "corrected_pessimistic must be non-negative"
                    )
                if p < m:
                    raise ValueError(
                        f"corrected_pessimistic ({p}) is below the corrected "
                        f"duration ({m}); P must be >= M"
                    )
                item.corrected_pessimistic = p
            item.status = ReviewStatus.CORRECTED
            item.duration_status = DurationStatus.CORRECTED
            self._record(
                "duration", geometric_node_id, "CORRECT_DURATION",
                item.current_duration, item.corrected_duration,
                reason, comment, item.geometric_node_id or "",
            )
        elif decision == "LEAVE_UNRESOLVED":
            pass
        else:
            raise ValueError(f"Unknown duration decision: {decision}")

        return (True, item) if return_item else True

    # ------------------------------------------------------------------
    # Session-level actions
    # ------------------------------------------------------------------

    def approve(self, comment: str = "") -> None:
        self.approval_state = ApprovalState.APPROVED
        self.approved_at = _utc_now()
        self.metadata["approval_comment"] = comment

    def mark_rejected(self, comment: str = "") -> None:
        self.approval_state = ApprovalState.REJECTED
        self.metadata["approval_comment"] = comment

    @property
    def pending_activity_count(self) -> int:
        return sum(1 for a in self.activities if a.status == ReviewStatus.PENDING)

    @property
    def pending_dependency_count(self) -> int:
        return sum(1 for d in self.dependencies if d.status == ReviewStatus.PENDING)

    @property
    def pending_duration_count(self) -> int:
        return sum(1 for d in self.durations if d.status == ReviewStatus.PENDING)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_image_id": self.source_image_id,
            "created_at": self.created_at,
            "version": self.version,
            "activities": [a.to_dict() for a in self.activities],
            "dependencies": [d.to_dict() for d in self.dependencies],
            "durations": [d.to_dict() for d in self.durations],
            "ambiguities": [_ambiguity_to_dict(a) for a in self.ambiguities],
            "decisions": [d.to_dict() for d in self.decisions],
            "approval_state": self.approval_state.value,
            "approved_at": self.approved_at,
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        reconstruction: Any = None,
    ) -> "ReviewSession":
        session = cls(
            source_image_id=data.get("source_image_id", ""),
            created_at=data.get("created_at", ""),
            version=data.get("version", "1.0"),
            reconstruction=reconstruction,
            activities=[
                ActivityReview.from_dict(a) for a in data.get("activities", [])
            ],
            dependencies=[
                DependencyReview.from_dict(d) for d in data.get("dependencies", [])
            ],
            durations=[
                DurationReview.from_dict(d) for d in data.get("durations", [])
            ],
            ambiguities=[_ambiguity_from_dict(a) for a in data.get("ambiguities", [])],
            decisions=[
                ReviewDecision.from_dict(d) for d in data.get("decisions", [])
            ],
            approval_state=ApprovalState[data.get("approval_state", "PENDING")],
            approved_at=data.get("approved_at"),
            metadata=copy.deepcopy(data.get("metadata", {})),
        )
        return session


def _ambiguity_to_dict(amb: Any) -> Dict[str, Any]:
    """Best-effort serialization of a reconstruction AmbiguityIssue."""
    if hasattr(amb, "to_dict"):
        return amb.to_dict()
    ambiguity_type = getattr(amb, "ambiguity_type", None)
    severity = getattr(amb, "severity", None)
    return {
        "ambiguity_type": getattr(ambiguity_type, "value", str(ambiguity_type)),
        "description": getattr(amb, "description", ""),
        "involved_ids": list(getattr(amb, "involved_ids", []) or []),
        "alternatives": list(getattr(amb, "alternatives", []) or []),
        "suggested_resolution": getattr(amb, "suggested_resolution", ""),
        "severity": getattr(severity, "value", str(severity)),
        "metadata": copy.deepcopy(getattr(amb, "metadata", {}) or {}),
    }


def _ambiguity_from_dict(data: Dict[str, Any]) -> Any:
    """Rehydrate an AmbiguityIssue from the serialized form (if available)."""
    try:
        from pert_analyzer.cv.reconstruction_models import (
            AmbiguityIssue,
            AmbiguityType,
            ValidationSeverity,
        )
    except ImportError:  # pragma: no cover - defensive
        return data
    if isinstance(data, dict) and "ambiguity_type" in data:
        atype = data.get("ambiguity_type")
        try:
            atype = AmbiguityType[atype] if isinstance(atype, str) else atype
        except KeyError:
            pass
        severity = data.get("severity")
        try:
            severity = (
                ValidationSeverity[severity]
                if isinstance(severity, str) else severity
            )
        except KeyError:
            pass
        return AmbiguityIssue(
            ambiguity_type=atype,
            description=data.get("description", ""),
            involved_ids=list(data.get("involved_ids", [])),
            alternatives=list(data.get("alternatives", [])),
            suggested_resolution=data.get("suggested_resolution", ""),
            severity=severity,
            metadata=copy.deepcopy(data.get("metadata", {})),
        )
    return data


# =============================================================================
# Graph validation
# =============================================================================


@dataclass
class GraphValidationIssue:
    """A single structured validation finding."""

    code: str
    message: str
    elements: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "elements": list(self.elements),
        }


@dataclass
class GraphValidationResult:
    """Result of validating a graph structure."""

    status: GraphStatus = GraphStatus.VALID
    errors: List[GraphValidationIssue] = field(default_factory=list)
    warnings: List[GraphValidationIssue] = field(default_factory=list)
    component_count: int = 1
    is_acyclic: bool = True

    @property
    def is_valid(self) -> bool:
        return self.status == GraphStatus.VALID

    @property
    def error_count(self) -> int:
        return len(self.errors)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "is_valid": self.is_valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [e.to_dict() for e in self.warnings],
            "component_count": self.component_count,
            "is_acyclic": self.is_acyclic,
        }


def validate_graph_structure(
    activity_ids: List[str],
    durations: Dict[str, float],
    edges: List[tuple],
    is_dummy: Optional[set] = None,
) -> GraphValidationResult:
    """
    Validate an AON graph structure.

    Checks: empty graph, missing IDs, missing durations, unknown
    dependency references, self-loops, duplicate edges, disconnected
    components, cycles, and invalid AON semantics (isolated activities).

    Never silently repairs the graph; it returns structured errors.
    """
    errors: List[GraphValidationIssue] = []
    warnings: List[GraphValidationIssue] = []
    dummy = is_dummy or set()

    if not activity_ids:
        errors.append(GraphValidationIssue(
            "empty_graph", "Graph has no activities", [],
        ))

    id_set = set(activity_ids)
    for aid in activity_ids:
        if not aid or not str(aid).strip():
            errors.append(GraphValidationIssue(
                "missing_activity_id", "Activity has an empty ID", [str(aid)],
            ))

    for aid in activity_ids:
        dur = durations.get(aid, 0.0) or 0.0
        if dur <= 0 and aid not in dummy:
            errors.append(GraphValidationIssue(
                "missing_duration",
                f"Activity '{aid}' has invalid duration {dur:g}; "
                "CPM is blocked until a valid duration is supplied",
                [aid],
            ))

    seen_edges: set = set()
    for s, t in edges:
        if s not in id_set:
            errors.append(GraphValidationIssue(
                "unknown_dependency_source",
                f"Dependency source '{s}' does not reference an activity",
                [str(s)],
            ))
        if t not in id_set:
            errors.append(GraphValidationIssue(
                "unknown_dependency_target",
                f"Dependency target '{t}' does not reference an activity",
                [str(t)],
            ))
        if s == t:
            errors.append(GraphValidationIssue(
                "self_loop", f"Self-loop dependency {s}->{s}", [str(s)],
            ))
        if (s, t) in seen_edges:
            errors.append(GraphValidationIssue(
                "duplicate_edge",
                f"Duplicate dependency {s}->{t}",
                [f"{s}->{t}"],
            ))
        seen_edges.add((s, t))

    # Cycle detection (Kahn's algorithm).
    adj: Dict[str, List[str]] = {a: [] for a in activity_ids}
    indeg: Dict[str, int] = {a: 0 for a in activity_ids}
    for s, t in edges:
        if s in adj and t in adj:
            adj[s].append(t)
            indeg[t] += 1
    queue = [a for a in activity_ids if indeg.get(a, 0) == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for nxt in adj[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    acyclic = visited == len(activity_ids)
    if not acyclic:
        errors.append(GraphValidationIssue(
            "cycle", "Graph contains a cycle; CPM cannot run on cyclic graphs",
            [],
        ))

    # Disconnected components (union-find) including isolated activities.
    parent = {a: a for a in activity_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    incident = {a: False for a in activity_ids}
    for s, t in edges:
        if s in id_set and t in id_set:
            union(s, t)
            incident[s] = True
            incident[t] = True
    components = {find(a) for a in activity_ids}
    component_count = len(components)
    if component_count > 1:
        errors.append(GraphValidationIssue(
            "disconnected_components",
            f"Graph has {component_count} disconnected components",
            sorted(components),
        ))

    for aid in activity_ids:
        if not incident.get(aid):
            errors.append(GraphValidationIssue(
                "isolated_activity",
                f"Activity '{aid}' is not connected to any dependency "
                "(invalid AON semantics)",
                [aid],
            ))

    result = GraphValidationResult(
        status=GraphStatus.INVALID if errors else GraphStatus.VALID,
        errors=errors,
        warnings=warnings,
        component_count=component_count,
        is_acyclic=acyclic,
    )
    return result


def validate_graph_model(graph: Any) -> GraphValidationResult:
    """Validate an existing GraphModel instance."""
    if graph is None:
        return GraphValidationResult(
            status=GraphStatus.INVALID,
            errors=[GraphValidationIssue(
                "empty_graph", "No graph was produced", [],
            )],
            is_acyclic=True,
            component_count=0,
        )
    activity_ids = list(graph.activities.keys())
    durations = {
        aid: float(act.duration) for aid, act in graph.activities.items()
    }
    edges = [(dep.source, dep.target) for dep in graph.dependencies]
    return validate_graph_structure(
        activity_ids=activity_ids,
        durations=durations,
        edges=edges,
        is_dummy={aid for aid, a in graph.activities.items() if a.is_dummy},
    )


# =============================================================================
# Reviewed graph candidate
# =============================================================================


@dataclass
class ReviewedGraphCandidate:
    """
    Output of applying review decisions.

    Contains the (reviewed) GraphModel, validation result, CPM result
    (only when the gate is open), and the review session provenance.
    """

    review_session: ReviewSession
    graph: Optional[Any] = None
    validation: Optional[GraphValidationResult] = None
    cpm: Optional[Any] = None
    cpm_gate: CpmGateStatus = CpmGateStatus.BLOCKED_REVIEW
    pure_critical_paths: List[List[str]] = field(default_factory=list)
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = _utc_now()

    @property
    def cpm_project_duration(self) -> Optional[float]:
        if self.cpm is not None:
            return float(self.cpm.project_duration)
        return None

    @property
    def critical_path_count(self) -> Optional[int]:
        if self.cpm is not None:
            return len(self.pure_critical_paths)
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpm_gate": self.cpm_gate.value,
            "cpm_project_duration": self.cpm_project_duration,
            "critical_path_count": self.critical_path_count,
            "cpm_critical_paths": [list(p) for p in self.pure_critical_paths],
            "validation": self.validation.to_dict() if self.validation else None,
            "graph_activity_count": (
                len(self.graph.activities) if self.graph is not None else 0
            ),
            "graph_dependency_count": (
                len(self.graph.dependencies) if self.graph is not None else 0
            ),
            "created_at": self.created_at,
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        review_session: ReviewSession,
    ) -> "ReviewedGraphCandidate":
        return cls(
            review_session=review_session,
            graph=None,
            validation=None,
            cpm=None,
            cpm_gate=CpmGateStatus[data.get("cpm_gate", "BLOCKED_REVIEW")],
            created_at=data.get("created_at", ""),
            metadata=copy.deepcopy(data.get("metadata", {})),
        )


# =============================================================================
# Review summary
# =============================================================================


@dataclass
class ReviewSummary:
    """Human-readable counts over a review session / candidate."""

    total_activities: int = 0
    confirmed_activities: int = 0
    activity_reviews_pending: int = 0
    total_dependencies: int = 0
    accepted_dependencies: int = 0
    rejected_dependencies: int = 0
    pending_dependency_reviews: int = 0
    duration_reviews_pending: int = 0
    duration_corrected: int = 0
    graph_validation_status: str = GraphStatus.INVALID.value
    cpm_eligibility: str = CpmGateStatus.BLOCKED_REVIEW.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_activities": self.total_activities,
            "confirmed_activities": self.confirmed_activities,
            "activity_reviews_pending": self.activity_reviews_pending,
            "total_dependencies": self.total_dependencies,
            "accepted_dependencies": self.accepted_dependencies,
            "rejected_dependencies": self.rejected_dependencies,
            "pending_dependency_reviews": self.pending_dependency_reviews,
            "duration_reviews_pending": self.duration_reviews_pending,
            "duration_corrected": self.duration_corrected,
            "graph_validation_status": self.graph_validation_status,
            "cpm_eligibility": self.cpm_eligibility,
        }

    def __str__(self) -> str:
        lines = [
            f"Activities:",
            f"    {self.total_activities}",
            f"    {self.confirmed_activities} confirmed",
            f"    {self.activity_reviews_pending} pending review",
            f"Dependencies:",
            f"    {self.accepted_dependencies} accepted",
            f"    {self.pending_dependency_reviews} pending review",
            f"    {self.rejected_dependencies} rejected",
            f"Duration:",
            f"    {self.duration_reviews_pending} pending",
            f"    {self.duration_corrected} corrected",
            f"Graph:",
            f"    {self.graph_validation_status}",
            f"CPM:",
            f"    {self.cpm_eligibility}",
        ]
        return "\n".join(lines)


def build_review_summary(candidate: ReviewedGraphCandidate) -> ReviewSummary:
    """Build the review summary for a reviewed candidate."""
    session = candidate.review_session
    total_activities = (
        len(session.reconstruction.activities)
        if session.reconstruction is not None
        else 0
    )
    activity_pending = session.pending_activity_count
    confirmed = total_activities - activity_pending

    # Accepted / rejected dependency counts come from decisions + final graph.
    rejected = sum(
        1 for d in session.decisions
        if d.item_type == "dependency" and d.decision == "REJECT"
    )
    accepted = candidate.graph.dependency_count if candidate.graph is not None else 0
    pending_dep = session.pending_dependency_count

    # Duration reviews.
    duration_pending = session.pending_duration_count
    duration_corrected = sum(
        1 for d in session.durations
        if d.status == ReviewStatus.CORRECTED
    )

    return ReviewSummary(
        total_activities=total_activities,
        confirmed_activities=confirmed,
        activity_reviews_pending=activity_pending,
        total_dependencies=accepted + pending_dep,
        accepted_dependencies=accepted,
        rejected_dependencies=rejected,
        pending_dependency_reviews=pending_dep,
        duration_reviews_pending=duration_pending,
        duration_corrected=duration_corrected,
        graph_validation_status=(
            candidate.validation.status.value
            if candidate.validation is not None else GraphStatus.INVALID.value
        ),
        cpm_eligibility=candidate.cpm_gate.value,
    )


# =============================================================================
# Review session construction
# =============================================================================


def _calibrate_activity_review(
    activity: Any, reasons: List[str]
) -> Dict[str, Any]:
    """Classify review priority without auto-accepting any activity."""
    status = getattr(getattr(activity, "semantic_status", None), "value", "")
    activity_id = getattr(activity, "activity_id", "") or ""
    confidence = float(getattr(activity, "confidence", 0.0) or 0.0)
    if activity_id.startswith("INFERRED_") or status == "review_required":
        tier = "BLOCKING_REVIEW"
    elif confidence < 0.5:
        tier = "HIGH_PRIORITY_REVIEW"
    elif confidence < 0.7:
        tier = "STANDARD_REVIEW"
    else:
        tier = "LOW_RISK_REVIEW"
    return {
        "calibration": "phase15.5",
        "tier": tier,
        "confidence": round(confidence, 3),
        "auto_accept": False,
        "reason_codes": list(reasons),
    }


def _calibrate_duration_review(
    duration: float, reasons: List[str], confidence: float
) -> Dict[str, Any]:
    """Classify duration review urgency while keeping CPM blocking intact."""
    if duration <= 0:
        tier = "BLOCKING_REVIEW"
    elif confidence < 0.4:
        tier = "HIGH_PRIORITY_REVIEW"
    else:
        tier = "STANDARD_REVIEW"
    return {
        "calibration": "phase15.5",
        "tier": tier,
        "confidence": round(confidence, 3),
        "auto_accept": False,
        "cpm_blocked_until_resolved": duration <= 0,
        "reason_codes": list(reasons),
    }


def _calibrate_dependency_review(confidence: float) -> Dict[str, Any]:
    """Triage arrow reviews by confidence; never auto-accept a relation."""
    if confidence < 0.4:
        tier = "HIGH_RISK_REVIEW"
    elif confidence < 0.75:
        tier = "STANDARD_REVIEW"
    else:
        tier = "STRONG_EVIDENCE_REVIEW"
    return {
        "calibration": "phase15.5",
        "tier": tier,
        "confidence": round(confidence, 3),
        "auto_accept": False,
        "relationship_gate_unchanged": True,
    }


def build_review_session(
    pipeline_result: Any,
    source_image_id: Optional[str] = None,
    provenance: str = "cv_pipeline",
) -> ReviewSession:
    """
    Build a ReviewSession from a pipeline result.

    Scans the reconstruction for:
    - activities that need review (missing/uncertain/duplicate/low-
      confidence semantic IDs)
    - dependency review candidates (REVIEW_REQUIRED arrows)
    - durations that are missing or low-confidence (REVIEW_REQUIRED)

    The raw OCR/CV evidence is copied into each review item and is
    never modified.
    """
    reconstruction = getattr(pipeline_result, "_reconstruction", None)
    if reconstruction is None:
        reconstruction = pipeline_result

    image_id = source_image_id or getattr(
        getattr(pipeline_result, "image_metadata", None), "image_id", ""
    ) or "unknown_image"

    session = ReviewSession(
        source_image_id=image_id,
        reconstruction=reconstruction,
        ambiguities=list(getattr(reconstruction, "ambiguities", [])),
        metadata={"provenance": provenance},
    )

    if reconstruction is None:
        return session

    activities = list(getattr(reconstruction, "activities", []) or [])
    node_to_activity: Dict[str, str] = {}
    for act in activities:
        node = getattr(act, "source_node_id", None) or getattr(
            act, "geometric_node_id", None
        )
        if node:
            node_to_activity[node] = act.activity_id

    seen_ids: Dict[str, str] = {}
    for act in activities:
        node = getattr(act, "source_node_id", None) or getattr(
            act, "geometric_node_id", None
        )
        if not node:
            node = act.activity_id
        reasons = []
        raw_id = getattr(act, "semantic_activity_id", None)
        raw_label = getattr(act, "label", "") or act.activity_id

        if act.activity_id in seen_ids:
            reasons.append("duplicate semantic ID in reconstruction")
        seen_ids[act.activity_id] = node

        status = getattr(act, "status", None)
        if status is not None and getattr(status, "value", status) == "CONFIRMED":
            if getattr(act, "needs_review", False):
                reasons.append("flagged for review despite confirmed status")
            if not reasons:
                continue
        else:
            if getattr(act, "needs_review", False):
                reasons.append("semantic identity needs review")
            if not getattr(act, "activity_id", "").startswith("INFERRED_"):
                if getattr(act, "semantic_status", None) is not None and getattr(
                    act.semantic_status, "value",
                    act.semantic_status,
                ) == "REVIEW_REQUIRED":
                    reasons.append("uncertain OCR semantic ID")
            if act.activity_id.startswith("INFERRED_"):
                reasons.append("missing OCR ID (spatially inferred)")

        if (getattr(act, "confidence", 1.0) or 1.0) < 0.5 and "low confidence" not in reasons:
            reasons.append("low confidence semantic ID")

        if not reasons:
            continue

        calibration = _calibrate_activity_review(act, reasons)
        evidence = [
            ReviewEvidence(
                source=e.source_phase,
                reference_ids=list(getattr(e, "source_ids", []) or []),
                description=getattr(e, "description", ""),
                confidence=getattr(e, "confidence_contribution", 0.0) or 0.0,
                metadata={"review_calibration": calibration},
            )
            for e in getattr(act, "evidence", []) or []
        ]

        session.activities.append(ActivityReview(
            geometric_node_id=node,
            current_activity_id=act.activity_id,
            proposed_activity_id=raw_id or act.activity_id,
            confidence=getattr(act, "confidence", 0.0) or 0.0,
            evidence=evidence,
            reason="; ".join(reasons),
            status=ReviewStatus.PENDING,
            provenance=provenance,
            raw_label=raw_label,
            raw_semantic_id=raw_id,
        ))

    # Durations: missing (<=0) or low-confidence duration evidence.
    for act in activities:
        node = getattr(act, "source_node_id", None) or getattr(
            act, "geometric_node_id", None
        )
        if not node:
            node = act.activity_id
        if getattr(act, "is_dummy", False):
            continue
        duration = float(getattr(act, "duration", 0.0) or 0.0)
        reasons = []
        if duration <= 0:
            reasons.append(
                "missing/invalid duration (OCR produced zero or none); "
                "CPM is blocked until corrected"
            )
        evidence_confs = [
            float(e.confidence_contribution or 0.0)
            for e in getattr(act, "evidence", []) or []
            if getattr(e, "source_phase", "") in (
                "region_ocr_duration", "semantic_resolution_duration",
            )
        ]
        if evidence_confs and max(evidence_confs) < 0.4:
            reasons.append("low-confidence OCR duration")
        if not reasons:
            continue
        duration_confidence = max(evidence_confs, default=0.0)
        calibration = _calibrate_duration_review(
            duration, reasons, duration_confidence
        )
        session.durations.append(DurationReview(
            geometric_node_id=node,
            activity_id=act.activity_id,
            current_duration=duration,
            duration_status=DurationStatus.REVIEW_REQUIRED,
            # A16-A: surface the measured OCR confidence instead of the
            # default 0.0, so the review center shows a real trust signal.
            confidence=duration_confidence,
            evidence=[
                ReviewEvidence(
                    source=e.source_phase,
                    reference_ids=list(getattr(e, "source_ids", []) or []),
                    description=getattr(e, "description", ""),
                    confidence=getattr(e, "confidence_contribution", 0.0) or 0.0,
                    metadata={"review_calibration": calibration},
                )
                for e in getattr(act, "evidence", []) or []
            ],
            reason="; ".join(reasons),
            status=ReviewStatus.PENDING,
            provenance=provenance,
        ))

    # Dependency review candidates from the validated-dependency report.
    report = None
    metadata = getattr(reconstruction, "metadata", {}) or {}
    if isinstance(metadata, dict):
        report = metadata.get("dependency_validation")
    if report is not None:
        for vdep in list(getattr(report, "review_candidates", []) or []):
            src_node = getattr(vdep, "source_id", None) or getattr(
                vdep, "source_node_id", None
            )
            tgt_node = getattr(vdep, "target_id", None) or getattr(
                vdep, "target_node_id", None
            )
            arrow_id = getattr(vdep, "arrow_id", None)
            confidence = float(getattr(vdep, "confidence_score", 0.0) or 0.0)
            calibration = _calibrate_dependency_review(confidence)
            cur_src = node_to_activity.get(src_node) if src_node else None
            cur_tgt = node_to_activity.get(tgt_node) if tgt_node else None
            evidence = [
                ReviewEvidence(
                    source="validated_dependency",
                    reference_ids=[arrow_id] if arrow_id else [],
                    description=getattr(vdep, "reason", "")
                    or "ambiguous arrow pair",
                    confidence=confidence,
                    metadata={
                        "review_calibration": calibration,
                        "source_boundary": getattr(
                            getattr(vdep, "evidence", None),
                            "source_boundary_intersection", None,
                        ),
                        "direction_consistency": getattr(
                            getattr(vdep, "evidence", None),
                            "direction_consistency", None,
                        ),
                        "angular_consistency": getattr(
                            getattr(vdep, "evidence", None),
                            "angular_consistency", None,
                        ),
                    },
                )
            ]
            session.dependencies.append(DependencyReview(
                arrow_id=arrow_id,
                source_node_id=src_node,
                target_node_id=tgt_node,
                current_source_id=cur_src,
                current_target_id=cur_tgt,
                proposed_direction=(
                    f"{cur_src}->{cur_tgt}"
                    if cur_src and cur_tgt
                    else (f"{src_node}->{tgt_node}" if src_node and tgt_node else None)
                ),
                alternative_direction=(
                    f"{cur_tgt}->{cur_src}"
                    if cur_src and cur_tgt
                    else None
                ),
                confidence=confidence,
                evidence=evidence,
                reason="ambiguous or uncertain arrow (REVIEW_REQUIRED)",
                status=ReviewStatus.PENDING,
                provenance=provenance,
            ))

    calibration_items = (
        list(session.activities) + list(session.durations) + list(session.dependencies)
    )
    tier_counts: Dict[str, int] = {}
    for item in calibration_items:
        if isinstance(item, ActivityReview):
            if item.current_activity_id and item.current_activity_id.startswith("INFERRED_"):
                fallback_tier = "BLOCKING_REVIEW"
            elif item.confidence < 0.5:
                fallback_tier = "HIGH_PRIORITY_REVIEW"
            elif item.confidence < 0.7:
                fallback_tier = "STANDARD_REVIEW"
            else:
                fallback_tier = "LOW_RISK_REVIEW"
        elif isinstance(item, DurationReview):
            fallback_tier = (
                "BLOCKING_REVIEW"
                if (item.current_duration or 0.0) <= 0
                else "STANDARD_REVIEW"
            )
        else:
            fallback_tier = _calibrate_dependency_review(
                float(getattr(item, "confidence", 0.0) or 0.0)
            )["tier"]
        item_tier = None
        for evidence in getattr(item, "evidence", []) or []:
            calibration = getattr(evidence, "metadata", {}).get(
                "review_calibration", {}
            )
            tier = calibration.get("tier")
            if tier:
                item_tier = tier
                break
        tier = item_tier or fallback_tier
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    session.metadata["review_calibration"] = {
        "policy": "phase15.5",
        "auto_accept_enabled": False,
        "relationship_gate_unchanged": True,
        "tier_counts": tier_counts,
    }
    return session


# =============================================================================
# Apply review decisions
# =============================================================================


def _resolve_final_activities(
    reconstruction: Any,
    session: ReviewSession,
) -> tuple:
    """
    Resolve the final activity IDs / durations after applying review
    decisions. Returns (node_id->final_id, final_activities, errors).
    """
    final_by_node: Dict[str, str] = {}
    activities_out: List[tuple] = []  # (node, final_id, duration, is_dummy, act)
    removed_nodes: set = set()
    errors: List[GraphValidationIssue] = []
    raw_to_final: Dict[str, str] = {}

    for act in list(getattr(reconstruction, "activities", []) or []):
        node = getattr(act, "source_node_id", None) or getattr(
            act, "geometric_node_id", None
        )
        if not node:
            node = act.activity_id

        arev = session.get_activity_review(node)
        if arev is not None and arev.status == ReviewStatus.REJECTED:
            removed_nodes.add(node)
            continue

        final_id = act.activity_id
        if arev is not None:
            if arev.status == ReviewStatus.CORRECTED and arev.corrected_activity_id:
                final_id = arev.corrected_activity_id
            elif arev.status == ReviewStatus.ACCEPTED:
                final_id = arev.current_activity_id or act.activity_id
            # PENDING / LEAVE_UNRESOLVED keep the original id.

        duration = float(act.duration or 0.0)
        drev = session.get_duration_review(node)
        if drev is not None and drev.status == ReviewStatus.CORRECTED:
            if drev.corrected_duration is not None:
                duration = float(drev.corrected_duration)
        elif drev is not None and drev.status == ReviewStatus.ACCEPTED:
            if drev.current_duration is not None:
                duration = float(drev.current_duration)

        final_by_node[node] = final_id
        activities_out.append((node, final_id, duration, act.is_dummy, act))

    # Detect duplicate final IDs (two different nodes mapping to one ID).
    id_to_node: Dict[str, str] = {}
    for node, final_id, _, _, act in activities_out:
        if final_id in id_to_node:
            errors.append(GraphValidationIssue(
                "duplicate_activity_id",
                f"Activities at nodes '{id_to_node[final_id]}' and '{node}' "
                f"both resolve to ID '{final_id}' after review",
                [final_id],
            ))
        else:
            id_to_node[final_id] = node

    # Map raw activity ids -> final ids (unique only).
    for node, final_id, _, _, act in activities_out:
        raw = act.activity_id
        if raw in raw_to_final and raw_to_final[raw] != final_id:
            raw_to_final[raw] = ""  # ambiguous raw id
        else:
            raw_to_final[raw] = final_id

    return final_by_node, activities_out, removed_nodes, errors, raw_to_final


def _resolve_dependencies(
    reconstruction: Any,
    session: ReviewSession,
    final_by_node: Dict[str, str],
    activities_out: List[tuple],
    removed_nodes: set,
    raw_to_final: Dict[str, str],
) -> tuple:
    """
    Resolve the final dependency edge set from auto-accepted CV deps plus
    resolved review decisions. Returns (edges, warnings).
    """
    final_ids = {final_id for _, final_id, _, _, _ in activities_out}
    edges: List[tuple] = []  # (source, target, origin)
    warnings: List[str] = []

    def _final_id_from_node(node: Optional[str]) -> Optional[str]:
        if not node:
            return None
        return final_by_node.get(node)

    def _final_id_from_raw(raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        mapped = raw_to_final.get(raw)
        if mapped:
            return mapped
        if raw in final_ids:
            return raw
        return None

    # 1) Auto-accepted dependencies from the reconstruction.
    for dep in list(getattr(reconstruction, "dependencies", []) or []):
        s = _final_id_from_raw(dep.source_id)
        t = _final_id_from_raw(dep.target_id)
        if s is None or t is None:
            warnings.append(
                f"Dependency {dep.source_id}->{dep.target_id} references an "
                "activity that was removed or is unresolvable; dropped"
            )
            continue
        if s in final_ids and t in final_ids:
            edges.append((s, t, "auto"))

    # 2) Review decisions on dependencies.
    for drev in list(session.dependencies):
        if drev.status == ReviewStatus.PENDING:
            continue
        if drev.status == ReviewStatus.REJECTED:
            continue

        if drev.corrected_source_id and drev.corrected_source_id in final_ids:
            s = drev.corrected_source_id
        else:
            s = _final_id_from_node(drev.source_node_id) or _final_id_from_raw(
                drev.current_source_id
            )

        if drev.corrected_target_id and drev.corrected_target_id in final_ids:
            t = drev.corrected_target_id
        else:
            t = _final_id_from_node(drev.target_node_id) or _final_id_from_raw(
                drev.current_target_id
            )

        if s is None or t is None:
            warnings.append(
                f"Dependency review (arrow={drev.arrow_id}) could not be "
                "resolved to two activities; skipped"
            )
            continue
        if s not in final_ids or t not in final_ids:
            warnings.append(
                f"Dependency review {s}->{t} references unknown activities; skipped"
            )
            continue
        edges.append((s, t, "review"))

    # 3) Merge, dedupe, and flag duplicate edges among review decisions.
    #    Auto-accepted CV edges overlapping review decisions are deduped
    #    silently; only two separate review decisions resolving to the
    #    same edge produce a duplicate_edge error.
    final_edges: List[tuple] = []
    seen_origin: Dict[tuple, str] = {}
    dup_errors: List[tuple] = []
    for s, t, origin in edges:
        if (s, t) in seen_origin:
            if origin == "review" and seen_origin[(s, t)] == "review":
                dup_errors.append((s, t))
            continue
        seen_origin[(s, t)] = origin
        final_edges.append((s, t))

    return final_edges, warnings, dup_errors


def _build_graph_model(
    activities_out: List[tuple],
    edges: List[tuple],
    session: ReviewSession,
    dup_id_errors: List[GraphValidationIssue],
) -> tuple:
    """Build a GraphModel or return structural duplicate-id errors."""
    from pert_analyzer.core.models import Activity as CoreActivity
    from pert_analyzer.core.models import Dependency as CoreDependency
    from pert_analyzer.core.models import DiagramType, GraphModel

    if dup_id_errors:
        return None, dup_id_errors

    gm = GraphModel(diagram_type=DiagramType.AON)
    for node, final_id, duration, is_dummy, act in activities_out:
        drev = (
            session.get_duration_review(node)
            if session is not None
            else None
        )
        opt = pes = None
        # A-15: corrected three-point extremes land directly on the graph
        # activity; the engine reads them natively and they beat the
        # derived +/-25% fallback.
        if (
            drev is not None
            and drev.status == ReviewStatus.CORRECTED
            and (
                drev.corrected_optimistic is not None
                or drev.corrected_pessimistic is not None
            )
        ):
            opt = drev.corrected_optimistic
            pes = drev.corrected_pessimistic
        cact = CoreActivity(
            activity_id=final_id,
            name=getattr(act, "label", "") or final_id,
            duration=duration,
            optimistic_time=opt,
            most_likely_time=(
                duration if opt is not None or pes is not None else None
            ),
            pessimistic_time=pes,
            confidence=getattr(act, "confidence", 0.0) or 0.0,
            is_dummy=bool(is_dummy),
            metadata={
                "geometric_node_id": node,
                "provenance": "reconstruction+human_review",
                "raw_activity_id": act.activity_id,
            },
        )
        gm.add_activity(cact)

    decision_lookup = {}
    for d in session.decisions:
        decision_lookup.setdefault(d.item_id, d.decision)

    for s, t in edges:
        gm.add_dependency(CoreDependency(
            source=s,
            target=t,
            dependency_type="finish_to_start",
            confidence=1.0,
            metadata={
                "provenance": "human_review",
                "review_session_source": session.source_image_id,
            },
        ))
    return gm, []


def apply_review_decisions(
    source: Any,
    session: ReviewSession,
) -> ReviewedGraphCandidate:
    """
    Deterministically apply review decisions.

    Args:
        source: PipelineResult or ReconstructedDiagram (reconstruction
                that the session references; original stays immutable).
        session: ReviewSession with decisions.

    Returns:
        ReviewedGraphCandidate with graph (iff activity IDs are unique),
        structured validation, and CPM gated on validity.
    """
    reconstruction = getattr(source, "_reconstruction", None)
    if reconstruction is None:
        reconstruction = source

    final_by_node, activities_out, removed_nodes, dup_errors, raw_to_final = (
        _resolve_final_activities(reconstruction, session)
    )
    edges, warnings, dup_edge_errors = _resolve_dependencies(
        reconstruction, session, final_by_node, activities_out,
        removed_nodes, raw_to_final,
    )

    pre_errors = list(dup_errors)
    for s, t in dup_edge_errors:
        pre_errors.append(GraphValidationIssue(
            "duplicate_edge",
            f"Two separate review decisions resolved to the same edge {s}->{t}",
            [f"{s}->{t}"],
        ))

    graph, build_errors = _build_graph_model(
        activities_out, edges, session, [e for e in pre_errors if e.code == "duplicate_activity_id"]
    )
    if build_errors:
        pre_errors.extend(build_errors)

    if graph is None:
        validation = GraphValidationResult(
            status=GraphStatus.INVALID,
            errors=pre_errors,
            component_count=0,
            is_acyclic=False,
        )
        return ReviewedGraphCandidate(
            review_session=session,
            graph=None,
            validation=validation,
            cpm=None,
            cpm_gate=CpmGateStatus.BLOCKED_REVIEW,
            metadata={"warnings": warnings},
        )

    activity_ids = list(graph.activities.keys())
    durations = {
        aid: float(a.duration) for aid, a in graph.activities.items()
    }
    is_dummy = {aid for aid, a in graph.activities.items() if a.is_dummy}
    edge_list = [(d.source, d.target) for d in graph.dependencies]
    validation = validate_graph_structure(
        activity_ids=activity_ids,
        durations=durations,
        edges=edge_list,
        is_dummy=is_dummy,
    )
    validation.errors = list(pre_errors) + list(validation.errors)
    validation.status = (
        GraphStatus.INVALID
        if validation.errors or pre_errors
        else GraphStatus.VALID
    )

    candidate = ReviewedGraphCandidate(
        review_session=session,
        graph=graph,
        validation=validation,
        cpm=None,
        cpm_gate=CpmGateStatus.BLOCKED_REVIEW,
        metadata={"warnings": warnings},
    )

    # CPM gate: only when the graph is valid.
    if validation.is_valid:
        from pert_analyzer.analysis.cpm_engine import CPMEngine

        cpm = CPMEngine().analyze(graph)
        candidate.cpm = cpm
        candidate.cpm_gate = CpmGateStatus.RUNNABLE
        duration = float(cpm.project_duration)
        candidate.pure_critical_paths = [
            list(p) for p in cpm.critical_paths
            if abs(
                sum(float(graph.activities[n].duration) for n in p) - duration
            ) < 1e-6
        ]

    return candidate


# =============================================================================
# Convenience helpers
# =============================================================================


def run_cpm_on_graph(graph: Any) -> Any:
    """Run CPM only if the graph is valid (gate check)."""
    validation = validate_graph_model(graph)
    if not validation.is_valid:
        raise ValueError(
            "CPM blocked: graph is not valid "
            f"({validation.error_count} validation errors)"
        )
    from pert_analyzer.analysis.cpm_engine import CPMEngine

    return CPMEngine().analyze(graph)
