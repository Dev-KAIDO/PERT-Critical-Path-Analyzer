"""
Programmatic review and correction API.

Provides mechanisms to correct reconstructed diagram elements
and re-run analysis without re-running CV operations.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from pert_analyzer.analysis.cpm_engine import CPMEngine
from pert_analyzer.core.models import DiagramType, GraphModel
from pert_analyzer.cv.reconstruction_models import (
    ReconstructedActivity,
    ReconstructedDependency,
    ReconstructedDiagram,
    ReconstructedEvent,
)
from pert_analyzer.pipeline.human_review import (
    apply_review_decisions,
    build_review_session,
    build_review_summary,
    CorrectionAction,
    HumanReviewItem,
    HumanReviewResult,
    ReviewedGraphCandidate,
    ReviewEvidence,
    ReviewIssue,
    ReviewIssueType,
    ReviewSession,
)
from pert_analyzer.pipeline.result import AnalysisStatus, PipelineResult

logger = logging.getLogger(__name__)


class ReviewCorrectionAPI:
    """
    Programmatic API for reviewing and correcting reconstructed diagrams.

    Allows correcting activity IDs, durations, arrow sources/targets,
    and re-running CPM without re-running the CV pipeline.

    Usage:
        api = ReviewCorrectionAPI(pipeline_result)
        api.correct_activity_duration("A1", 5.0)
        api.correct_dependency_source("dep_1", "B")
        new_result = api.reanalyze()
    """

    def __init__(self, pipeline_result: PipelineResult):
        """
        Initialize with a pipeline result.

        Args:
            pipeline_result: The result from EndToEndAnalyzer.analyze().
        """
        self._original_result = pipeline_result
        self._reconstruction = pipeline_result._reconstruction
        self._corrections: List[CorrectionAction] = []
        self._cpm_engine = CPMEngine()

    @property
    def reconstruction(self) -> Optional[ReconstructedDiagram]:
        """Get the current reconstructed diagram."""
        return self._reconstruction

    @property
    def corrections(self) -> List[CorrectionAction]:
        """Get all applied corrections."""
        return list(self._corrections)

    # =========================================================================
    # Correction Methods
    # =========================================================================

    def correct_activity_duration(
        self, activity_id: str, new_duration: float, reason: str = ""
    ) -> bool:
        """
        Correct the duration of an activity.

        Args:
            activity_id: The activity ID to correct.
            new_duration: The new duration value.
            reason: Reason for the correction.

        Returns:
            True if the correction was applied.
        """
        if self._reconstruction is None:
            return False

        act = self._reconstruction.get_activity_by_id(activity_id)
        if act is None:
            logger.warning("Activity '%s' not found", activity_id)
            return False

        old_duration = act.duration
        act.duration = new_duration
        act.confidence = 1.0
        act.warnings = [w for w in act.warnings if "duration" not in w.lower()]

        self._corrections.append(CorrectionAction(
            action_type="correct_duration",
            target_id=activity_id,
            field_name="duration",
            old_value=old_duration,
            new_value=new_duration,
            reason=reason,
        ))

        logger.info("Corrected '%s' duration: %s -> %s", activity_id, old_duration, new_duration)
        return True

    def correct_activity_id(
        self, old_id: str, new_id: str, reason: str = ""
    ) -> bool:
        """
        Correct the ID of an activity.

        Args:
            old_id: The current activity ID.
            new_id: The new activity ID.
            reason: Reason for the correction.

        Returns:
            True if the correction was applied.
        """
        if self._reconstruction is None:
            return False

        act = self._reconstruction.get_activity_by_id(old_id)
        if act is None:
            logger.warning("Activity '%s' not found", old_id)
            return False

        # Update dependencies
        for dep in self._reconstruction.dependencies:
            if dep.source_id == old_id:
                dep.source_id = new_id
            if dep.target_id == old_id:
                dep.target_id = new_id

        act.activity_id = new_id
        act.label = new_id if act.label == old_id else act.label
        act.confidence = 1.0

        self._corrections.append(CorrectionAction(
            action_type="correct_activity_id",
            target_id=old_id,
            field_name="activity_id",
            old_value=old_id,
            new_value=new_id,
            reason=reason,
        ))

        logger.info("Corrected activity ID: %s -> %s", old_id, new_id)
        return True

    def correct_activity_label(
        self, activity_id: str, new_label: str, reason: str = ""
    ) -> bool:
        """Correct the label of an activity."""
        if self._reconstruction is None:
            return False

        act = self._reconstruction.get_activity_by_id(activity_id)
        if act is None:
            return False

        old_label = act.label
        act.label = new_label

        self._corrections.append(CorrectionAction(
            action_type="correct_activity_label",
            target_id=activity_id,
            field_name="label",
            old_value=old_label,
            new_value=new_label,
            reason=reason,
        ))
        return True

    def correct_dependency_source(
        self, source_id: str, target_id: str, new_source: str, reason: str = ""
    ) -> bool:
        """
        Correct the source of a dependency.

        Args:
            source_id: Current source ID.
            target_id: Target ID of the dependency.
            new_source: New source ID.
            reason: Reason for the correction.

        Returns:
            True if correction was applied.
        """
        if self._reconstruction is None:
            return False

        for dep in self._reconstruction.dependencies:
            if dep.source_id == source_id and dep.target_id == target_id:
                old_source = dep.source_id
                dep.source_id = new_source
                self._corrections.append(CorrectionAction(
                    action_type="correct_dependency_source",
                    target_id=f"{source_id}->{target_id}",
                    field_name="source_id",
                    old_value=old_source,
                    new_value=new_source,
                    reason=reason,
                ))
                return True
        return False

    def correct_dependency_target(
        self, source_id: str, target_id: str, new_target: str, reason: str = ""
    ) -> bool:
        """Correct the target of a dependency."""
        if self._reconstruction is None:
            return False

        for dep in self._reconstruction.dependencies:
            if dep.source_id == source_id and dep.target_id == target_id:
                old_target = dep.target_id
                dep.target_id = new_target
                self._corrections.append(CorrectionAction(
                    action_type="correct_dependency_target",
                    target_id=f"{source_id}->{target_id}",
                    field_name="target_id",
                    old_value=old_target,
                    new_value=new_target,
                    reason=reason,
                ))
                return True
        return False

    def remove_false_detection(self, element_id: str, reason: str = "") -> bool:
        """Remove a falsely detected activity."""
        if self._reconstruction is None:
            return False

        act = self._reconstruction.get_activity_by_id(element_id)
        if act is None:
            return False

        self._reconstruction.activities = [
            a for a in self._reconstruction.activities
            if a.activity_id != element_id
        ]
        self._reconstruction.dependencies = [
            d for d in self._reconstruction.dependencies
            if d.source_id != element_id and d.target_id != element_id
        ]

        self._corrections.append(CorrectionAction(
            action_type="remove_false_detection",
            target_id=element_id,
            reason=reason,
        ))
        return True

    def add_missing_dependency(
        self, source_id: str, target_id: str, reason: str = ""
    ) -> bool:
        """Add a missing dependency between two activities."""
        if self._reconstruction is None:
            return False

        # Check if it already exists
        for dep in self._reconstruction.dependencies:
            if dep.source_id == source_id and dep.target_id == target_id:
                return False

        new_dep = ReconstructedDependency(
            source_id=source_id,
            target_id=target_id,
            confidence=1.0,
            evidence=[],
            warnings=["Added by human review"],
        )
        self._reconstruction.dependencies.append(new_dep)

        self._corrections.append(CorrectionAction(
            action_type="add_dependency",
            target_id=f"{source_id}->{target_id}",
            reason=reason,
        ))
        return True

    def mark_dummy_activity(self, activity_id: str, reason: str = "") -> bool:
        """Mark an activity as a dummy (zero-duration) activity."""
        if self._reconstruction is None:
            return False

        act = self._reconstruction.get_activity_by_id(activity_id)
        if act is None:
            return False

        act.is_dummy = True
        act.duration = 0.0

        self._corrections.append(CorrectionAction(
            action_type="mark_dummy",
            target_id=activity_id,
            reason=reason,
        ))
        return True

    # =========================================================================
    # Re-analysis
    # =========================================================================

    def reanalyze(self) -> PipelineResult:
        """
        Re-run graph construction, validation, and CPM on the
        corrected reconstruction.

        Does NOT re-run OCR or CV operations.

        Returns:
            Updated PipelineResult.
        """
        new_result = PipelineResult()
        new_result.image_metadata = self._original_result.image_metadata
        new_result.diagram_type = self._original_result.diagram_type
        new_result.diagram_confidence = self._original_result.diagram_confidence
        new_result.shape_count = self._original_result.shape_count
        new_result.arrow_count = self._original_result.arrow_count
        new_result.ocr_region_count = self._original_result.ocr_region_count
        new_result.warnings = list(self._original_result.warnings)
        new_result.errors = list(self._original_result.errors)

        # Copy counts from corrected reconstruction
        if self._reconstruction:
            new_result.reconstructed_activity_count = self._reconstruction.activity_count
            new_result.reconstructed_event_count = self._reconstruction.event_count
            new_result.dependency_count = self._reconstruction.dependency_count

        # Build graph from corrected reconstruction
        if self._reconstruction:
            try:
                graph = self._reconstruction.to_graph_model()
                new_result._graph_model = graph

                # Run validation and CPM
                if graph.diagram_type == DiagramType.AOA:
                    new_result.review_required = True
                    new_result.review_issues.append({
                        "type": "NOT_SUPPORTED_FOR_CPM",
                        "description": "AOA-to-CPM not yet supported",
                    })
                elif not graph.activities:
                    new_result.validation_passed = False
                    new_result.validation_errors.append("No activities in graph")
                else:
                    try:
                        cpm_result = self._cpm_engine.analyze(graph)
                        new_result._cpm_result = cpm_result
                        new_result.cpm_project_duration = cpm_result.project_duration
                        new_result.cpm_critical_path = cpm_result.critical_path
                        new_result.cpm_critical_paths = cpm_result.critical_paths
                        new_result.cpm_critical_activity_count = sum(
                            1 for aa in cpm_result.activity_analyses.values()
                            if aa.is_critical
                        )
                        new_result.validation_passed = True
                        new_result.status = AnalysisStatus.SUCCESS
                    except Exception as e:
                        new_result.validation_passed = False
                        new_result.validation_errors.append(str(e))
                        new_result.status = AnalysisStatus.FAILED

            except Exception as e:
                new_result.add_error(f"Graph rebuild failed: {e}")
                new_result.status = AnalysisStatus.FAILED

        new_result._reconstruction = self._reconstruction
        return new_result

    def get_review_result(self) -> HumanReviewResult:
        """
        Generate a HumanReviewResult from the current corrections.

        Returns:
            HumanReviewResult summarizing all corrections.
        """
        review = HumanReviewResult()
        review.corrections = list(self._corrections)
        review.is_fully_resolved = len(self._corrections) > 0

        for corr in self._corrections:
            item = HumanReviewItem(
                element_id=corr.target_id,
                element_type="activity" if "activity" in corr.action_type else "dependency",
                field_name=corr.field_name or corr.action_type,
                detected_value=corr.old_value,
                correction=corr,
                is_resolved=True,
            )
            review.items.append(item)

        return review


class ReviewWorkflow:
    """
    GUI-independent human-review workflow.

    Exposes the sequence:
        1. analyze image        (EndToEndAnalyzer)
        2. review session       (build_review_session)
        3. reviewer decisions   (decide_* on the session)
        4. apply                (apply_review_decisions)
        5. validate graph       (candidate.validation)
        6. run CPM              (gated on graph validity)

    The original reconstruction is never mutated; every decision is
    recorded in the session's audit trail.
    """

    def __init__(
        self,
        pipeline_result: Optional[PipelineResult] = None,
        review_session: Optional[ReviewSession] = None,
        analyzer: Any = None,
    ):
        """
        Initialize the workflow.

        Args:
            pipeline_result: An analyzed pipeline result (optional).
            review_session: An existing review session (optional).
            analyzer: An EndToEndAnalyzer instance for future analyses
                (optional).
        """
        self._pipeline_result = pipeline_result
        self._analyzer = analyzer
        self._session = review_session
        self._candidate: Optional[ReviewedGraphCandidate] = None

    # ------------------------------------------------------------------
    # Stage 1: analyze
    # ------------------------------------------------------------------

    @classmethod
    def analyze(
        cls,
        image_path: str,
        source_image_id: Optional[str] = None,
        tesseract_path: Optional[str] = None,
        ocr_languages: Optional[list[str]] = None,
        stage_callback: Optional[Callable[[Any], None]] = None,
    ) -> "ReviewWorkflow":
        """
        Analyze an image end-to-end and build a review session for it.

        ``stage_callback`` receives structured ``StageProgress`` events so
        GUI progress UI can show the real pipeline stages and the review
        preparation work that runs after analysis.
        """
        from pert_analyzer.pipeline.analyzer import EndToEndAnalyzer
        from pert_analyzer.pipeline.progress import StageProgress

        analyzer = EndToEndAnalyzer(
            tesseract_path=tesseract_path,
            ocr_languages=ocr_languages,
        )
        if stage_callback is not None:
            try:
                stage_callback(StageProgress("Review preparation", "RUNNING", 1.0))
            except Exception:
                pass
        result = analyzer.analyze(image_path, stage_callback=stage_callback)
        session = build_review_session(
            result, source_image_id=source_image_id or str(image_path)
        )
        if stage_callback is not None:
            try:
                review_items = sum((
                    len(getattr(session, "activities", []) or []),
                    len(getattr(session, "dependencies", []) or []),
                    len(getattr(session, "durations", []) or []),
                ))
                stage_callback(
                    StageProgress(
                        "Review preparation",
                        "COMPLETED",
                        1.0,
                        metrics={"review_items": review_items},
                    )
                )
            except Exception:
                pass
        return cls(
            pipeline_result=result,
            review_session=session,
            analyzer=analyzer,
        )

    @classmethod
    def from_pipeline_result(
        cls,
        pipeline_result: PipelineResult,
        source_image_id: Optional[str] = None,
    ) -> "ReviewWorkflow":
        """Build a workflow from an already-analyzed pipeline result."""
        session = build_review_session(
            pipeline_result, source_image_id=source_image_id
        )
        return cls(pipeline_result=pipeline_result, review_session=session)

    # ------------------------------------------------------------------
    # Stage 2: review items
    # ------------------------------------------------------------------

    @property
    def pipeline_result(self) -> Optional[PipelineResult]:
        return self._pipeline_result

    @property
    def review_session(self) -> Optional[ReviewSession]:
        return self._session

    def get_review_items(self) -> ReviewSession:
        """
        Get all review items (activities, dependencies, durations,
        ambiguities) for human inspection.
        """
        if self._session is None:
            raise RuntimeError("No review session; analyze an image first")
        return self._session

    def get_review_items_json(self) -> str:
        """Get review items serialized to JSON."""
        import json

        return json.dumps(self.get_review_items().to_dict(), indent=2)

    # ------------------------------------------------------------------
    # Stage 3: reviewer decisions
    # ------------------------------------------------------------------

    def decide_activity(
        self,
        geometric_node_id: str,
        decision: str,
        corrected_id: Optional[str] = None,
        reason: str = "",
        comment: str = "",
    ) -> bool:
        """Record an activity review decision (ACCEPT/CORRECT/REJECT/LEAVE_UNRESOLVED)."""
        if self._session is None:
            raise RuntimeError("No review session")
        ok, _ = self._session.decide_activity(
            geometric_node_id, decision,
            corrected_id=corrected_id, reason=reason,
            comment=comment, return_item=True,
        )
        return bool(ok)

    def decide_dependency(
        self,
        arrow_id: str,
        action: str,
        corrected_source_id: Optional[str] = None,
        corrected_target_id: Optional[str] = None,
        reason: str = "",
        comment: str = "",
    ) -> bool:
        """Record a dependency review decision."""
        if self._session is None:
            raise RuntimeError("No review session")
        ok, _ = self._session.decide_dependency(
            arrow_id, action,
            corrected_source_id=corrected_source_id,
            corrected_target_id=corrected_target_id,
            reason=reason, comment=comment, return_item=True,
        )
        return bool(ok)

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
        """Record a duration review decision (ACCEPT/CORRECT/LEAVE_UNRESOLVED)."""
        if self._session is None:
            raise RuntimeError("No review session")
        ok, _ = self._session.decide_duration(
            geometric_node_id, decision,
            corrected_duration=corrected_duration,
            corrected_optimistic=corrected_optimistic,
            corrected_pessimistic=corrected_pessimistic,
            reason=reason, comment=comment, return_item=True,
        )
        return bool(ok)

    # ------------------------------------------------------------------
    # Stage 4/5/6: apply, validate, CPM
    # ------------------------------------------------------------------

    def apply(self) -> ReviewedGraphCandidate:
        """
        Apply all review decisions deterministically.

        Returns a ReviewedGraphCandidate with structured validation and
        a CPM gate. CPM only runs when the reviewed graph is VALID.
        """
        if self._session is None:
            raise RuntimeError("No review session; analyze an image first")
        self._candidate = apply_review_decisions(
            self._pipeline_result or self._session.reconstruction,
            self._session,
        )
        return self._candidate

    def validate_graph(self) -> Any:
        """Validate the reviewed graph (run apply() first)."""
        if self._candidate is None:
            self.apply()
        return self._candidate.validation

    def run_cpm(self) -> Any:
        """
        Run CPM on the reviewed graph.

        Raises ValueError (CPM blocked) if the graph is not valid.
        """
        if self._candidate is None:
            self.apply()
        if not self._candidate.cpm_gate or self._candidate.cpm is None:
            raise ValueError("CPM blocked: graph validation failed")
        return self._candidate.cpm

    def summary(self) -> Dict[str, Any]:
        """Return the review summary counts (see section 10 of the spec)."""
        if self._candidate is None:
            self.apply()
        summary = build_review_summary(self._candidate)
        return summary.to_dict()

    def summary_string(self) -> str:
        """Return a human-readable review summary."""
        if self._candidate is None:
            self.apply()
        return str(build_review_summary(self._candidate))
