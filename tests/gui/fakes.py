"""
Fake backend and workflow objects for GUI tests (no Tesseract dependency).
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from pert_analyzer.gui.session import GuiSession
from pert_analyzer.pipeline.result import AnalysisStatus


class FakeReviewSession:
    """Minimal stand-in for ReviewSession with configurable pending counts."""

    def __init__(self, pa: int = 0, pd: int = 0, pdr: int = 0) -> None:
        self.pending_activity_count = pa
        self.pending_dependency_count = pd
        self.pending_duration_count = pdr


class FakePipelineResult:
    """Minimal stand-in for PipelineResult."""

    def __init__(
        self,
        status: AnalysisStatus = AnalysisStatus.SUCCESS,
        review_required: bool = False,
        errors: Optional[list[str]] = None,
    ) -> None:
        self.status = status
        self.review_required = review_required
        self.errors = errors or []


class FakeCandidate:
    """Minimal stand-in for ReviewedGraphCandidate."""

    def __init__(self) -> None:
        self._gate = type("G", (), {"value": "BLOCKED_REVIEW"})()
        self.cpm_gate = self._gate
        self.cpm_project_duration: Optional[float] = None
        self.critical_path_count: Optional[int] = None
        self.validation = None
        self.pure_critical_paths: list[list[str]] = []


class FakeWorkflow:
    """Minimal stand-in for ReviewWorkflow that is fully deterministic."""

    def __init__(
        self,
        pa: int = 0,
        pd: int = 0,
        pdr: int = 0,
        status: AnalysisStatus = AnalysisStatus.SUCCESS,
        review_required: bool = False,
        errors: Optional[list[str]] = None,
    ) -> None:
        self.pipeline_result = FakePipelineResult(status, review_required, errors)
        self.review_session = FakeReviewSession(pa, pd, pdr)
        self._candidate = None

    def summary(self) -> dict[str, Any]:
        return {
            "activity_reviews_pending": self.review_session.pending_activity_count,
            "pending_dependency_reviews": self.review_session.pending_dependency_count,
            "duration_reviews_pending": self.review_session.pending_duration_count,
            "total_activities": 5,
            "total_dependencies": 4,
            "graph_validation_status": "INVALID",
            "cpm_eligibility": "BLOCKED_REVIEW",
            "cpm_project_duration": None,
            "critical_path_count": None,
        }


class FakeBackend:
    """Configurable callable used as the analyze_fn for AnalysisWorker."""

    def __init__(
        self,
        workflow: Optional[FakeWorkflow] = None,
        fail: bool = False,
        msg: str = "backend error",
        block_event: Optional[threading.Event] = None,
    ) -> None:
        self._workflow = workflow or FakeWorkflow()
        self._fail = fail
        self._msg = msg
        self._block_event = block_event

    def __call__(self, image_path: str) -> Any:
        if self._block_event is not None:
            self._block_event.wait(timeout=10)
        if self._fail:
            raise RuntimeError(self._msg)
        return self._workflow


def fake_session() -> GuiSession:
    """Create a fresh GuiSession for tests."""
    return GuiSession()


# ---------------------------------------------------------------------------
# Human Review Center fakes
# ---------------------------------------------------------------------------

from types import SimpleNamespace  # noqa: E402


_UNSET = object()


class FakeValidationResult:
    """Minimal stand-in for GraphValidationResult."""

    def __init__(
        self,
        status: str = "INVALID",
        is_valid: bool = False,
        errors: Optional[list[Any]] = None,
        warnings: Optional[list[Any]] = None,
        component_count: int = 1,
        is_acyclic: bool = True,
    ) -> None:
        self.status = SimpleNamespace(value=status)
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.component_count = component_count
        self.is_acyclic = is_acyclic

    @property
    def error_count(self) -> int:
        return len(self.errors)


class FakeReviewedCandidate:
    """Stand-in for ReviewedGraphCandidate: valid by default."""

    def __init__(
        self,
        valid: bool = True,
        validation: Any = None,
        cpm_gate: str = "",
        cpm: Any = _UNSET,
        cpm_project_duration: Optional[float] = None,
        critical_path_count: Optional[int] = None,
        pure_critical_paths: Optional[list[list[str]]] = None,
        graph: Any = None,
        no_validation: bool = False,
    ) -> None:
        if no_validation:
            validation = None
        elif validation is None:
            validation = FakeValidationResult(
                status="VALID" if valid else "INVALID",
                is_valid=valid,
            )
        self.validation = validation
        if cpm_gate:
            gate_value = cpm_gate
        else:
            gate_value = "RUNNABLE" if valid else "BLOCKED_ERROR"
        self.cpm_gate = SimpleNamespace(value=gate_value)
        if cpm is _UNSET and valid:
            cpm = default_cpm_result()
        self.cpm = cpm
        self.cpm_project_duration = (
            cpm_project_duration if cpm_project_duration is not None else (14.0 if valid else None)
        )
        self.critical_path_count = (
            critical_path_count if critical_path_count is not None else (1 if valid else None)
        )
        self.pure_critical_paths = (
            pure_critical_paths if pure_critical_paths is not None else ([["A0", "A1"]] if valid else [])
        )
        self.graph = graph


class FakeReviewedWorkflow:
    """
    Workflow-like object that wraps a REAL ReviewSession with fabricated
    review items, plus delegate decide_* methods and a fake apply().
    """

    def __init__(
        self,
        review_session: Any,
        status: AnalysisStatus = AnalysisStatus.SUCCESS,
        review_required: bool = True,
    ) -> None:
        self.review_session = review_session
        self.pipeline_result = SimpleNamespace(
            status=status,
            review_required=review_required,
            errors=[],
            _reconstruction=getattr(review_session, "reconstruction", None),
        )
        self._candidate: Any = None

    def summary(self) -> dict[str, Any]:
        session = self.review_session
        return {
            "activity_reviews_pending": _pending("activities", session),
            "pending_dependency_reviews": _pending("dependencies", session),
            "duration_reviews_pending": _pending("durations", session),
            "total_activities": len(getattr(session, "activities", []) or []),
            "total_dependencies": len(getattr(session, "dependencies", []) or []),
            "graph_validation_status": "VALID",
            "cpm_eligibility": "RUNNABLE",
            "cpm_project_duration": None,
            "critical_path_count": None,
        }

    def decide_activity(self, geometric_node_id, decision, corrected_id=None, reason="", comment=""):
        ok, _ = self.review_session.decide_activity(
            geometric_node_id, decision,
            corrected_id=corrected_id, reason=reason, comment=comment, return_item=True,
        )
        return bool(ok)

    def decide_dependency(
        self, arrow_id, action, corrected_source_id=None, corrected_target_id=None,
        reason="", comment="",
    ):
        ok, _ = self.review_session.decide_dependency(
            arrow_id, action,
            corrected_source_id=corrected_source_id,
            corrected_target_id=corrected_target_id,
            reason=reason, comment=comment, return_item=True,
        )
        return bool(ok)

    def decide_duration(
        self, geometric_node_id, decision, corrected_duration=None,
        corrected_optimistic=None, corrected_pessimistic=None,
        reason="", comment="",
    ):
        ok, _ = self.review_session.decide_duration(
            geometric_node_id, decision,
            corrected_duration=corrected_duration,
            corrected_optimistic=corrected_optimistic,
            corrected_pessimistic=corrected_pessimistic,
            reason=reason, comment=comment, return_item=True,
        )
        return bool(ok)

    def apply(self) -> Any:
        candidate = FakeReviewedCandidate(valid=True)
        self._candidate = candidate
        return candidate


class FakeValidatedWorkflow:
    """
    Workflow-like object wrapping a fixed candidate; ``apply()`` returns a
    configurable candidate so tests can simulate revalidation outcomes.
    """

    def __init__(
        self,
        candidate: Any = None,
        next_candidate: Any = None,
        review_session: Any = None,
        status: Any = AnalysisStatus.SUCCESS,
    ) -> None:
        self._candidate = candidate or FakeReviewedCandidate(valid=True)
        self._next_candidate = next_candidate
        self.review_session = review_session or FakeReviewSession()
        self.pipeline_result = SimpleNamespace(
            status=status,
            review_required=False,
            errors=[],
            _reconstruction=None,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "activity_reviews_pending": 0,
            "pending_dependency_reviews": 0,
            "duration_reviews_pending": 0,
            "total_activities": 5,
            "total_dependencies": 4,
            "graph_validation_status": self._candidate.validation.status.value,
            "cpm_eligibility": self._candidate.cpm_gate.value,
            "cpm_project_duration": self._candidate.cpm_project_duration,
            "critical_path_count": self._candidate.critical_path_count,
        }

    def apply(self) -> Any:
        if self._next_candidate is not None:
            self._candidate = self._next_candidate
        return self._candidate


class FakeCpmResult:
    """Stand-in for backend CPMResult (authoritative values for the GUI)."""

    def __init__(
        self,
        project_duration: float = 14.0,
        critical_paths: Optional[list[list[str]]] = None,
        analyses: Optional[dict[str, Any]] = None,
    ) -> None:
        self.project_duration = float(project_duration)
        self.critical_paths = [list(p) for p in (critical_paths or [])]
        self.activity_analyses = dict(analyses or {})

    @property
    def critical_activity_count(self) -> int:
        return sum(
            1
            for analysis in self.activity_analyses.values()
            if bool(getattr(analysis, "is_critical", False))
        )

    @property
    def total_float(self) -> float:
        return 0.0


def make_analysis(
    activity_id: str,  # noqa: ARG001
    es: Optional[float] = 0.0,
    ef: Optional[float] = 7.0,
    ls: Optional[float] = 0.0,
    lf: Optional[float] = 7.0,
    total_float: Optional[float] = 0.0,
    free_float: Optional[float] = 0.0,
    is_critical: bool = True,
) -> SimpleNamespace:
    """Build a backend ActivityAnalysis stand-in with explicit CPM values."""
    return SimpleNamespace(
        early_start=es,
        early_finish=ef,
        late_start=ls,
        late_finish=lf,
        total_float=total_float,
        free_float=free_float,
        is_critical=is_critical,
    )


def default_cpm_result() -> FakeCpmResult:
    """Default valid CPM result used by FakeReviewedCandidate(valid=True)."""
    return FakeCpmResult(
        project_duration=14.0,
        critical_paths=[["A0", "A1"]],
        analyses={
            "A0": make_analysis("A0", 0.0, 7.0, 0.0, 7.0, 0.0, 0.0, True),
            "A1": make_analysis("A1", 7.0, 14.0, 7.0, 14.0, 0.0, 0.0, True),
        },
    )


class FakeGraphActivity:
    """Minimal stand-in for a backend GraphModel activity (PERT-aware)."""

    def __init__(
        self,
        activity_id: str,
        duration: float,
        name: str = "",
        optimistic_time: Optional[float] = None,
        most_likely_time: Optional[float] = None,
        pessimistic_time: Optional[float] = None,
    ) -> None:
        self.id = activity_id
        self.activity_id = activity_id
        self.duration = float(duration)
        self.name = name or activity_id
        self.optimistic_time = optimistic_time
        self.most_likely_time = most_likely_time
        self.pessimistic_time = pessimistic_time


class FakeGraphDependency:
    """Minimal stand-in for a backend GraphModel dependency edge."""

    def __init__(self, dependency_id: str, source: str, target: str) -> None:
        self.dependency_id = dependency_id
        self.source = source
        self.target = target


def make_fake_graph(
    activities: dict[str, Any],
    dependencies: list[Any],
) -> SimpleNamespace:
    """Build a GraphModel stand-in: activity id -> activity + edge list."""
    return SimpleNamespace(
        activities=dict(activities),
        dependencies=list(dependencies),
    )


# ---------------------------------------------------------------------------
# PERT fakes
# ---------------------------------------------------------------------------


class FakePertActivityResult:
    """Stand-in for a backend PertActivityResult row."""

    def __init__(
        self,
        activity_id: str,
        expected_time: float,
        variance: float,
        std_dev: float,
        total_float: float = 0.0,
        is_critical: bool = True,
        earliest_start: float = 0.0,
        earliest_finish: float = 0.0,
        latest_start: float = 0.0,
        latest_finish: float = 0.0,
    ) -> None:
        self.activity_id = activity_id
        self.expected_time = float(expected_time)
        self.variance = float(variance)
        self.std_dev = float(std_dev)
        self.total_float = float(total_float)
        self.is_critical = bool(is_critical)
        self.earliest_start = float(earliest_start)
        self.earliest_finish = float(earliest_finish)
        self.latest_start = float(latest_start)
        self.latest_finish = float(latest_finish)
        self.free_float = 0.0


class FakePertResult:
    """Stand-in for the backend PertResult (math stays in the engine)."""

    def __init__(
        self,
        project_duration: float = 10.0,
        project_variance: float = 0.8888888888888888,
        project_std_dev: Optional[float] = None,
        critical_paths: Optional[list[list[str]]] = None,
        analyses: Optional[dict[str, Any]] = None,
    ) -> None:
        self.project_duration = float(project_duration)
        self.project_variance = float(project_variance)
        if project_std_dev is None:
            project_std_dev = self.project_variance**0.5
        self.project_std_dev = float(project_std_dev)
        self.critical_paths = [list(p) for p in (critical_paths or [])]
        self.activity_results = dict(analyses or {})
        self.estimates = {}

    @property
    def critical_activity_count(self) -> int:
        return sum(
            1
            for result in self.activity_results.values()
            if bool(getattr(result, "is_critical", False))
        )

    @property
    def critical_path(self) -> list[str]:
        return list(self.critical_paths[0]) if self.critical_paths else []

    def probability_for(self, target: float) -> Any:
        from pert_analyzer.analysis.pert_engine import completion_probability

        result = completion_probability(
            float(target), self.project_duration, self.project_std_dev
        )
        return SimpleNamespace(
            target_duration=float(target),
            z_score=float(result.z_score),
            probability=float(result.probability),
        )


def make_pert_analysis(
    activity_id: str,
    expected_time: float,
    variance: float,
    is_critical: bool = True,
    total_float: float = 0.0,
) -> FakePertActivityResult:
    """Build a PERT activity-result stand-in with explicit values."""
    return FakePertActivityResult(
        activity_id=activity_id,
        expected_time=float(expected_time),
        variance=float(variance),
        std_dev=float(variance) ** 0.5,
        total_float=float(total_float),
        is_critical=is_critical,
    )


def default_pert_result() -> FakePertResult:
    """Default valid PERT result for the two-activity fixture."""
    return FakePertResult(
        project_duration=10.0,
        project_variance=32.0 / 36.0,
        project_std_dev=(32.0 / 36.0) ** 0.5,
        critical_paths=[["A0", "A1"]],
        analyses={
            "A0": make_pert_analysis("A0", 4.0, 16.0 / 36.0, True),
            "A1": make_pert_analysis("A1", 6.0, 16.0 / 36.0, True),
        },
    )


def make_pert_fixture_session(
    graph: Any = None,
    result: Any = None,
    estimates: Optional[dict[str, Any]] = None,
) -> GuiSession:
    """A GuiSession whose candidate carries a PERT-ready fake graph."""
    if graph is None:
        graph = SimpleNamespace(
            activities={
                "A0": FakeGraphActivity(
                    "A0", 5.0,
                    optimistic_time=2.0, most_likely_time=4.0, pessimistic_time=6.0,
                ),
                "A1": FakeGraphActivity(
                    "A1", 5.0,
                    optimistic_time=4.0, most_likely_time=6.0, pessimistic_time=8.0,
                ),
            },
            dependencies=[
                FakeGraphDependency(dependency_id="d1", source="A0", target="A1")
            ],
        )
    if result is None:
        result = default_pert_result()
    candidate = FakeReviewedCandidate(valid=True, graph=graph, cpm=_UNSET)
    session = GuiSession()
    session.workflow = FakeValidatedWorkflow(
        candidate=candidate, review_session=FakeReviewSession()
    )
    session.complete_apply(candidate)
    if estimates is not None:
        session.pert_estimates = dict(estimates)
    session.pert_result = result
    return session


def fake_issue(code: str, message: str = "", elements: Optional[list[Any]] = None) -> Any:
    """Build a backend GraphValidationIssue for tests."""
    from pert_analyzer.pipeline.human_review import GraphValidationIssue

    return GraphValidationIssue(
        code=code,
        message=message or f"{code} message",
        elements=list(elements or []),
    )


def _pending(attr: str, session: Any) -> int:
    items = getattr(session, attr, []) or []
    return sum(1 for it in items if _is_pending(it))


def _is_pending(item: Any) -> bool:
    status = getattr(item, "status", None)
    return status is not None and getattr(status, "value", "") == "PENDING"


class FakeReconstructionActivity:
    """Minimal ReconstructedActivity with geometry for tests."""

    def __init__(
        self,
        node: str,
        activity_id: str,
        x: float = 0,
        y: float = 0,
        w: float = 100,
        h: float = 40,
    ) -> None:
        self.source_node_id = node
        self.geometric_node_id = node
        self.activity_id = activity_id
        self.bounding_box = SimpleNamespace(x=x, y=y, width=w, height=h)
        self.position = None


def make_reconstruction(acts: list[tuple[str, str]]) -> Any:
    """Build a fake reconstruction from (node_id, activity_id) pairs."""
    return SimpleNamespace(
        activities=[FakeReconstructionActivity(n, a) for n, a in acts]
    )


def make_review_item(
    kind: str,
    geometric_node_id: str = "N0",
    activity_id: str = "A0",
    confidence: float = 0.8,
    status: str = "PENDING",
    duration: float = 3.0,
) -> Any:
    """Create a single fabricated review item with default evidence."""
    from pert_analyzer.pipeline.human_review import (
        ActivityReview,
        CandidateValue,
        DependencyReview,
        DurationReview,
        DurationStatus,
        ReviewEvidence,
        ReviewStatus,
    )

    evidence = [
        ReviewEvidence(
            source="ocr",
            reference_ids=["ref-1"],
            description="Boundary contact and line continuity",
            confidence=0.7,
        ),
        ReviewEvidence(
            source="heuristics",
            reference_ids=[],
            description="Weak arrowhead",
            confidence=0.2,
        ),
    ]
    st = ReviewStatus(status)
    if kind == "activity":
        return ActivityReview(
            geometric_node_id=geometric_node_id,
            current_activity_id=activity_id,
            proposed_activity_id=activity_id,
            alternatives=[CandidateValue("A1", 0.3, "ocr")],
            confidence=confidence,
            evidence=evidence,
            reason="OCR uncertainty on semantic id",
            status=st,
            raw_label="A?",
            raw_semantic_id=activity_id,
        )
    if kind == "dependency":
        return DependencyReview(
            arrow_id=f"E{geometric_node_id}",
            source_node_id=geometric_node_id,
            target_node_id="T",
            current_source_id=activity_id,
            current_target_id="B0",
            proposed_direction=f"{activity_id}->B0",
            alternative_direction="B0->" + activity_id,
            confidence=confidence,
            evidence=evidence,
            reason="Arrow direction uncertain",
            status=st,
        )
    return DurationReview(
        geometric_node_id=geometric_node_id,
        activity_id=activity_id,
        current_duration=duration,
        proposed_duration=duration + 1.0,
        duration_status=DurationStatus.REVIEW_REQUIRED,
        confidence=confidence,
        evidence=evidence,
        reason="Duration needs confirmation",
        status=st,
    )


def make_review_session(
    source_image_id: str = "review_fixture.png",
    n_activities: int = 2,
    n_dependencies: int = 1,
    n_durations: int = 1,
    reconstruction: Any = None,
) -> Any:
    """Build a real ReviewSession with fabricated pending review items."""
    from pert_analyzer.pipeline.human_review import ReviewSession

    activities = [
        make_review_item("activity", geometric_node_id=f"N{i}", activity_id=f"A{i}")
        for i in range(n_activities)
    ]
    dependencies = []
    for i in range(n_dependencies):
        src = f"N{2 * i}"
        tgt = f"N{2 * i + 1}"
        dependencies.append(
            make_review_item(
                "dependency",
                geometric_node_id=src,
                activity_id=f"A{2 * i}",
                confidence=0.6,
            )
        )
        dep = dependencies[-1]
        dep.target_node_id = tgt if n_activities > 2 * i + 1 else "N99"
        dep.current_target_id = f"A{2 * i + 1}" if n_activities > 2 * i + 1 else "AZ"
        dep.proposed_direction = f"{dep.current_source_id}->{dep.current_target_id}"
        dep.alternative_direction = f"{dep.current_target_id}->{dep.current_source_id}"
    durations = [
        make_review_item("duration", geometric_node_id=f"N{i}", activity_id=f"A{i}", duration=3.0)
        for i in range(n_durations)
    ]
    return ReviewSession(
        source_image_id=source_image_id,
        reconstruction=reconstruction,
        activities=activities,
        dependencies=dependencies,
        durations=durations,
    )


def reviewed_session(
    n_activities: int = 2,
    n_dependencies: int = 1,
    n_durations: int = 1,
    reconstruction: Any = None,
) -> GuiSession:
    """A GuiSession whose workflow wraps a fabricated review session."""
    rs = make_review_session(
        reconstruction=reconstruction,
        n_activities=n_activities,
        n_dependencies=n_dependencies,
        n_durations=n_durations,
    )
    session = GuiSession()
    session.complete_analysis(FakeReviewedWorkflow(rs, review_required=True))
    return session
