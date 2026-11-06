"""
Application session and explicit state model for the GUI.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from pert_analyzer.analysis.pert_engine import PertEstimate, PertStatus

logger = logging.getLogger(__name__)

_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _decide(workflow: Any, method: str, *args, **kwargs) -> bool:
    """Run a decision method on the workflow and return its success flag."""
    if workflow is None or not callable(getattr(workflow, method, None)):
        raise RuntimeError(f"No workflow available for {method}")
    return bool(getattr(workflow, method)(*args, **kwargs))


class AppState(Enum):
    """Single explicit state model for the application."""

    NO_PROJECT = "NO_PROJECT"
    IMAGE_SELECTED = "IMAGE_SELECTED"
    ANALYZING = "ANALYZING"
    ANALYSIS_COMPLETE = "ANALYSIS_COMPLETE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    VALIDATION_REQUIRED = "VALIDATION_REQUIRED"
    READY_FOR_RESULTS = "READY_FOR_RESULTS"
    RESULTS_AVAILABLE = "RESULTS_AVAILABLE"
    ERROR = "ERROR"


class ValidationCenterStatus(Enum):
    """Derived state of the Validation Center for the current graph."""

    NOT_AVAILABLE = "NOT_AVAILABLE"
    BLOCKED_REVIEW = "BLOCKED_REVIEW"
    INVALID = "INVALID"
    VALID = "VALID"


@dataclass
class GuiSession:
    """Owns all state for a single analysis session."""

    state: AppState = AppState.NO_PROJECT
    entry_mode: str = "NONE"
    current_image_path: Optional[str] = None
    image_size: Optional[Tuple[int, int]] = None
    workflow: Any = None
    error_message: str = ""
    review_summary: Optional[Dict[str, Any]] = None

    # Review-center state
    candidate: Any = None
    has_applied_reviews: bool = False
    reviews_dirty: bool = False

    # PERT-state (positioned: tests construct GuiSession() and never
    # reference these fields positionally)
    pert_estimates: Dict[str, PertEstimate] = field(default_factory=dict)
    pert_result: Any = None
    pert_status: PertStatus = PertStatus.NO_PERT_DATA

    @property
    def review_session(self) -> Any:
        """The ReviewSession backing the current workflow, or None."""
        workflow = self.workflow
        if workflow is None:
            return None
        return getattr(workflow, "review_session", None)

    @property
    def report(self) -> Any:
        """Authoritative ProjectReport for the current session.

        Built on demand from the applied candidate so the Export action
        shares one code path with the reporting subsystem.
        """
        from pert_analyzer.reporting.builder import ReportBuilder

        return ReportBuilder().build_report(self)

    def set_image(self, path: str) -> bool:
        """Select an image file. Returns True on success."""
        if not os.path.isfile(path):
            logger.warning("Image file not found: %s", path)
            return False
        ext = os.path.splitext(path)[1].lower()
        if ext not in _EXTENSIONS:
            logger.warning("Unsupported image type: %s", ext)
            return False
        self.current_image_path = path
        self.entry_mode = "IMAGE_ANALYSIS"
        self._reset_review_state()
        self.error_message = ""
        self.state = AppState.IMAGE_SELECTED
        logger.info("Image selected: %s", os.path.basename(path))
        return True

    def clear_image(self) -> None:
        """Clear the current image and reset state."""
        self.current_image_path = None
        self.entry_mode = "NONE"
        self.image_size = None
        self._reset_review_state()
        self.error_message = ""
        self.state = AppState.NO_PROJECT
        logger.info("Image cleared")

    def _reset_review_state(self) -> None:
        self.workflow = None
        self.review_summary = None
        self.candidate = None
        self.has_applied_reviews = False
        self.reviews_dirty = False
        self.pert_estimates = {}
        self.pert_result = None
        self.pert_status = PertStatus.NO_PERT_DATA

    def begin_analysis(self) -> None:
        """Transition to the ANALYZING state."""
        self._reset_review_state()
        self.error_message = ""
        self.state = AppState.ANALYZING
        logger.info("Analysis started")

    def complete_analysis(self, workflow: Any) -> None:
        """Process a completed analysis workflow."""
        self.workflow = workflow
        try:
            self.review_summary = workflow.summary()
        except Exception:
            self.review_summary = {}

        result = getattr(workflow, "pipeline_result", None)
        status = getattr(result, "status", None)

        try:
            from pert_analyzer.pipeline.result import AnalysisStatus

            if result is None or status == AnalysisStatus.FAILED:
                self.state = AppState.ERROR
                self.error_message = "; ".join(
                    getattr(result, "errors", []) or ["Analysis failed"]
                )
                logger.warning("Analysis failed: %s", self.error_message)
                return
        except ImportError:
            pass

        review_items = (
            (self.review_summary or {}).get("activity_reviews_pending", 0)
            + (self.review_summary or {}).get("pending_dependency_reviews", 0)
            + (self.review_summary or {}).get("duration_reviews_pending", 0)
        )

        if review_items or getattr(result, "review_required", False):
            self.state = AppState.REVIEW_REQUIRED
        else:
            self.state = AppState.VALIDATION_REQUIRED

        logger.info("Analysis completed: state=%s", self.state.value)

    def fail_analysis(self, message: str) -> None:
        """Record an analysis failure."""
        self.state = AppState.ERROR
        self.error_message = message
        logger.warning("Analysis failed: %s", message)

    # ------------------------------------------------------------------
    # Review-center session helpers
    # ------------------------------------------------------------------

    def pending_review_total(self) -> int:
        """Total number of pending review items across all categories."""
        session = self.review_session
        if session is None:
            return 0
        return sum((
            getattr(session, "pending_activity_count", 0),
            getattr(session, "pending_dependency_count", 0),
            getattr(session, "pending_duration_count", 0),
        ))

    def review_item_total(self) -> int:
        """Total number of review items across all categories."""
        session = self.review_session
        if session is None:
            return 0
        return sum((
            len(getattr(session, "activities", []) or []),
            len(getattr(session, "dependencies", []) or []),
            len(getattr(session, "durations", []) or []),
        ))

    def record_review_decision(self) -> None:
        """Mark the session as having unsaved review decisions."""
        self.reviews_dirty = True

    def decide_activity(
        self,
        geometric_node_id: str,
        decision: str,
        corrected_id: Optional[str] = None,
        reason: str = "",
        comment: str = "",
    ) -> bool:
        """Forward an activity decision to the backend workflow."""
        return _decide(
            self.workflow,
            "decide_activity",
            geometric_node_id,
            decision,
            corrected_id=corrected_id,
            reason=reason,
            comment=comment,
        )

    def decide_dependency(
        self,
        arrow_id: str,
        action: str,
        corrected_source_id: Optional[str] = None,
        corrected_target_id: Optional[str] = None,
        reason: str = "",
        comment: str = "",
    ) -> bool:
        """Forward a dependency decision to the backend workflow."""
        return _decide(
            self.workflow,
            "decide_dependency",
            arrow_id,
            action,
            corrected_source_id=corrected_source_id,
            corrected_target_id=corrected_target_id,
            reason=reason,
            comment=comment,
        )

    def decide_duration(
        self,
        geometric_node_id: str,
        decision: str,
        corrected_duration: Optional[float] = None,
        corrected_optimistic: Optional[float] = None,
        corrected_pessimistic: Optional[float] = None,
        reason: str = "",
        comment: str = "",
    ) -> bool:
        """Forward a duration decision to the backend workflow."""
        return _decide(
            self.workflow,
            "decide_duration",
            geometric_node_id,
            decision,
            corrected_duration=corrected_duration,
            corrected_optimistic=corrected_optimistic,
            corrected_pessimistic=corrected_pessimistic,
            reason=reason,
            comment=comment,
        )

    def begin_apply(self) -> None:
        """Transition into the apply-review-decisions operation."""
        self.state = AppState.VALIDATION_REQUIRED
        logger.info("Applying review decisions started")

    def complete_apply(self, candidate: Any) -> None:
        """Store the applied candidate and move state toward validation."""
        self.candidate = candidate
        self.has_applied_reviews = True
        self.reviews_dirty = False
        # A new reviewed graph means any previous PERT result is stale.
        self.pert_result = None
        self.pert_status = PertStatus.NO_PERT_DATA

        workflow = self.workflow
        if workflow is not None:
            try:
                self.review_summary = workflow.summary()
            except Exception:
                self.review_summary = self.review_summary or {}

        self.state = AppState.VALIDATION_REQUIRED
        logger.info("Review decisions applied: state=%s", self.state.value)

    def fail_apply(self, message: str) -> None:
        """Record a failure while applying review decisions."""
        self.error_message = message
        if self.pending_review_total() > 0:
            self.state = AppState.REVIEW_REQUIRED
        else:
            self.state = AppState.VALIDATION_REQUIRED
        logger.warning("Applying review decisions failed: %s", message)

    # ------------------------------------------------------------------
    # Validation-center session state
    # ------------------------------------------------------------------

    @property
    def current_candidate(self) -> Any:
        """The reviewed-graph candidate, or None when none exists yet.

        Prefers the GUI-applied candidate and falls back to the workflow's
        own applied candidate (e.g. a candidate auto-applied by summary()).
        """
        if self.candidate is not None:
            return self.candidate
        workflow = self.workflow
        if workflow is None:
            return None
        return getattr(workflow, "_candidate", None)

    @property
    def validation_result(self) -> Any:
        """The GraphValidationResult for the current candidate, or None."""
        candidate = self.current_candidate
        if candidate is None:
            return None
        return getattr(candidate, "validation", None)

    @property
    def validation_issue_count(self) -> int:
        """Number of validation errors plus warnings for the current graph."""
        result = self.validation_result
        if result is None:
            return 0
        return len(getattr(result, "errors", []) or []) + len(
            getattr(result, "warnings", []) or []
        )

    @property
    def graph_model(self) -> Any:
        """The reviewed GraphModel backing the current candidate, or None."""
        candidate = self.current_candidate
        if candidate is None:
            return None
        return getattr(candidate, "graph", None)

    @property
    def cpm_result(self) -> Any:
        """The backend CPM result for the current candidate, or None."""
        candidate = self.current_candidate
        if candidate is None:
            return None
        return getattr(candidate, "cpm", None)

    @property
    def validation_status(self) -> ValidationCenterStatus:
        """Derived Validation Center status for the current candidate.

        For the image workflow the workflow must be set first.
        For the manual builder the workflow may be None — the candidate
        itself carries all required state.
        """
        candidate = self.current_candidate
        if candidate is None:
            if self.workflow is None:
                return ValidationCenterStatus.NOT_AVAILABLE
            return ValidationCenterStatus.NOT_AVAILABLE
        if candidate is None:
            return ValidationCenterStatus.NOT_AVAILABLE
        validation = getattr(candidate, "validation", None)
        if validation is None:
            return ValidationCenterStatus.NOT_AVAILABLE
        if self.pending_review_total() > 0:
            return ValidationCenterStatus.BLOCKED_REVIEW
        if not getattr(validation, "is_valid", False):
            return ValidationCenterStatus.INVALID
        if getattr(self, "reviews_dirty", False):
            return ValidationCenterStatus.BLOCKED_REVIEW
        gate = getattr(candidate, "cpm_gate", None)
        gate_value = getattr(gate, "value", gate) if gate is not None else None
        if gate_value != "RUNNABLE":
            return ValidationCenterStatus.BLOCKED_REVIEW
        return ValidationCenterStatus.VALID

    @property
    def cpm_eligible(self) -> bool:
        """True when the graph is valid and CPM is ready to run."""
        return self.validation_status == ValidationCenterStatus.VALID
