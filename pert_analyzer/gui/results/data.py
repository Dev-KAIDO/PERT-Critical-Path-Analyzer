"""
Results data snapshot: extracts a read-only view of the backend
CPM result and GraphModel.

The backend CPM result and graph are treated as authoritative. The GUI
never re-runs forward/backward passes, float math, or path enumeration.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pert_analyzer.analysis.pert_engine import (
    PertEngine,
    PertEstimate,
    PertStatus,
    resolve_estimates,
)


# Not-ready reasons (order matches the readiness derivation).
NO_ANALYSIS = "no_analysis"
REVIEW_REQUIRED = "review_required"
GRAPH_INVALID = "graph_invalid"
CPM_BLOCKED = "cpm_blocked"
RESULT_UNAVAILABLE = "result_unavailable"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActivityRow:
    """Display row for one activity in the Results dashboard."""

    activity_id: str
    name: str = ""
    duration: Optional[float] = None
    early_start: Optional[float] = None
    early_finish: Optional[float] = None
    late_start: Optional[float] = None
    late_finish: Optional[float] = None
    total_float: Optional[float] = None
    free_float: Optional[float] = None
    is_critical: bool = False
    predecessors: Tuple[str, ...] = ()
    successors: Tuple[str, ...] = ()


@dataclass(frozen=True)
class DependencyRow:
    """Display row for one dependency edge."""

    source: str
    target: str
    dependency_id: str = ""


@dataclass(frozen=True)
class ResultsData:
    """Immutable presentation snapshot for the Results dashboard."""

    project_duration: Optional[float] = None
    critical_path_count: Optional[int] = None
    critical_paths: List[List[str]] = field(default_factory=list)
    activities: List[ActivityRow] = field(default_factory=list)
    dependencies: List[DependencyRow] = field(default_factory=list)
    critical_activity_count: int = 0
    critical_edges: set = field(default_factory=set)
    ready: bool = False
    reason: str = NO_ANALYSIS

    # Keep the raw objects so views can render without another extraction.
    cpm: Any = None
    graph: Any = None

    @property
    def activity_count(self) -> int:
        return len(self.activities)

    @property
    def dependency_count(self) -> int:
        return len(self.dependencies)

    @property
    def critical_activity_ids(self) -> set:
        return {a.activity_id for a in self.activities if a.is_critical}


def _num(value: Any) -> Optional[float]:
    """Coerce a backend number to float, tolerating missing/malformed data."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _analysis_of(cpm: Any) -> Dict[str, Any]:
    """Return the activity_analyses mapping (defensively)."""
    if cpm is None:
        return {}
    analyses = getattr(cpm, "activity_analyses", None)
    if not analyses:
        return {}
    return analyses


def _critical_paths(candidate: Any, cpm: Any) -> List[List[str]]:
    """Prefer the backend cpm result, falling back to candidate data."""
    if cpm is not None:
        raw = getattr(cpm, "critical_paths", None)
        if isinstance(raw, list) and raw:
            return [list(p) for p in raw]
    candidate_paths = getattr(candidate, "pure_critical_paths", None)
    if isinstance(candidate_paths, list) and candidate_paths:
        return [list(p) for p in candidate_paths]
    return []


def _critical_edge_set(paths: List[List[str]]) -> set:
    """Consecutive (source, target) pairs along the backend critical paths."""
    edges: set = set()
    for path in paths:
        for i in range(len(path) - 1):
            edges.add((path[i], path[i + 1]))
    return edges


def _critical_activity_count(cpm: Any, analyses: Dict[str, Any],
                             activity_ids: List[str]) -> int:
    if cpm is not None:
        count = getattr(cpm, "critical_activity_count", None)
        if count is not None:
            try:
                return int(count)
            except (TypeError, ValueError):
                pass
    return sum(1 for aid in activity_ids
               if bool(getattr(analyses.get(aid), "is_critical", False)))


def describe_ready(session: Any) -> Tuple[bool, str]:
    """Derive the Results readiness state from the GuiSession.

    Ready requires: an applied candidate, a valid graph, an open CPM
    gate, and an existing CPM result.

    For the image workflow an analyzed workflow must exist first.
    For the manual builder the workflow may be None — the candidate
    itself carries all required state.
    """
    candidate = getattr(session, "current_candidate", None)
    if candidate is None:
        # No candidate at all — need a workflow-based analysis first.
        if session is None or getattr(session, "workflow", None) is None:
            logger.debug("[READYNESS] NO_ANALYSIS: no candidate, no workflow")
            return False, NO_ANALYSIS
        logger.debug("[READYNESS] REVIEW_REQUIRED: no candidate but workflow exists")
        return False, REVIEW_REQUIRED

    if getattr(session, "pending_review_total", lambda: 0)() > 0:
        logger.debug("[READYNESS] REVIEW_REQUIRED: pending reviews remain")
        return False, REVIEW_REQUIRED
    if getattr(session, "reviews_dirty", False):
        logger.debug("[READYNESS] REVIEW_REQUIRED: reviews dirty")
        return False, REVIEW_REQUIRED

    validation = getattr(candidate, "validation", None)
    if validation is None:
        logger.debug("[READYNESS] REVIEW_REQUIRED: no validation on candidate")
        return False, REVIEW_REQUIRED
    if not getattr(validation, "is_valid", False):
        logger.debug("[READYNESS] GRAPH_INVALID: validation failed")
        return False, GRAPH_INVALID

    gate = getattr(candidate, "cpm_gate", None)
    gate_value = getattr(gate, "value", gate) if gate is not None else None
    if gate_value != "RUNNABLE":
        logger.debug("[READYNESS] CPM_BLOCKED: gate=%s", gate_value)
        return False, CPM_BLOCKED

    if getattr(candidate, "cpm", None) is None:
        logger.debug("[READYNESS] RESULT_UNAVAILABLE: no CPM result")
        return False, RESULT_UNAVAILABLE

    logger.info("[READYNESS] READY: all checks passed")
    return True, ""


def extract(session: Any) -> ResultsData:
    """Build the full presentation snapshot for the current session."""
    ready, reason = describe_ready(session)
    if not ready:
        return ResultsData(ready=False, reason=reason)

    candidate = getattr(session, "current_candidate", None)
    cpm = getattr(candidate, "cpm", None)
    graph = getattr(candidate, "graph", None)
    analyses = _analysis_of(cpm)
    paths = _critical_paths(candidate, cpm)

    project_duration = _num(getattr(cpm, "project_duration", None))
    if project_duration is None:
        project_duration = _num(getattr(candidate, "cpm_project_duration", None))

    activity_ids: List[str] = []
    activities_data: Dict[str, Any] = {}
    graph_ref = graph
    if graph is not None:
        act_dict = getattr(graph, "activities", None) or {}
        activity_ids = list(act_dict.keys())
        activities_data = act_dict

    # Predecessor/successor lists come from the backend graph only.
    predecessors: Dict[str, List[str]] = defaultdict(list)
    successors: Dict[str, List[str]] = defaultdict(list)
    dependencies: List[DependencyRow] = []
    if graph is not None:
        for dep in getattr(graph, "dependencies", None) or []:
            src = getattr(dep, "source", None)
            tgt = getattr(dep, "target", None)
            if src is None or tgt is None:
                continue
            if src in activities_data and tgt in activities_data:
                successors[src].append(tgt)
                predecessors[tgt].append(src)
                dependencies.append(
                    DependencyRow(
                        source=src,
                        target=tgt,
                        dependency_id=getattr(dep, "dependency_id", "") or "",
                    )
                )

    activities: List[ActivityRow] = []
    for aid in sorted(activity_ids):
        backend_activity = activities_data.get(aid)
        analysis = analyses.get(aid)
        rows = ActivityRow(
            activity_id=aid,
            name=getattr(backend_activity, "name", "") or "",
            duration=_num(getattr(backend_activity, "duration", None)),
            early_start=_num(getattr(analysis, "early_start", None)),
            early_finish=_num(getattr(analysis, "early_finish", None)),
            late_start=_num(getattr(analysis, "late_start", None)),
            late_finish=_num(getattr(analysis, "late_finish", None)),
            total_float=_num(getattr(analysis, "total_float", None)),
            free_float=_num(getattr(analysis, "free_float", None)),
            is_critical=bool(getattr(analysis, "is_critical", False)),
            predecessors=tuple(predecessors.get(aid, [])),
            successors=tuple(successors.get(aid, [])),
        )
        activities.append(rows)

    return ResultsData(
        project_duration=project_duration,
        critical_path_count=len(paths) if paths else (
            _int_or_none(getattr(candidate, "critical_path_count", None))
        ),
        critical_paths=paths,
        activities=activities,
        dependencies=dependencies,
        critical_activity_count=_critical_activity_count(
            cpm, analyses, activity_ids
        ),
        critical_edges=_critical_edge_set(paths),
        ready=True,
        cpm=cpm,
        graph=graph_ref,
    )


def _int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# PERT presentation snapshot
#
# The PERT engine is authoritative for status and math; the GUI only maps
# the backend PertResult onto display rows and never invents values.
# ---------------------------------------------------------------------------

PERT_NO_DATA_TITLE = "PERT estimates are not available for this analysis."
PERT_NO_DATA_HINT = (
    "This project currently uses deterministic activity durations. "
    "Add three-point estimates in the table below, then run PERT."
)
# A-14: the derivation speaks for itself next to the ready status
PERT_DERIVED_HINT = (
    "Three-point estimates were derived automatically: most likely = accepted "
    "duration; the spread comes from the OCR duration candidates, or a default "
    "+/-25% band around it. Edit any value in the table below, re-run PERT."
)


# ---------------------------------------------------------------------------
# Derived three-point estimates (A-14)
#
# When no estimate exists anywhere (no manual table entries and no O/M/P
# fields on the activities) the accepted durations still support PERT:
# the accepted duration becomes the most-likely value and the spread is
# taken from the duration review candidates, defaulting to +/-25%.
# ---------------------------------------------------------------------------

_DERIVED_DEFAULT_BAND = 0.25


def _graph_carries_omp(graph: Any) -> bool:
    """True when at least one activity carries explicit O/M/P fields."""
    for act in (getattr(graph, "activities", None) or {}).values():
        if (
            getattr(act, "optimistic_time", None) is not None
            or getattr(act, "most_likely_time", None) is not None
            or getattr(act, "pessimistic_time", None) is not None
        ):
            return True
    return False


def _duration_reviews(session: Any) -> List[Any]:
    """Duration review items reachable from ``session`` (getattr only)."""
    review = getattr(session, "review_session", None)
    items = getattr(review, "durations", None) or []
    return list(items)


def _duration_item_candidates(item: Any) -> List[float]:
    """Distinct positive durations mentioned by one review item.

    Mirrors the review detail panel: the current/proposed durations plus
    every number found in the evidence descriptions.
    """
    raws: List[Any] = [
        getattr(item, "current_duration", None),
        getattr(item, "proposed_duration", None),
    ]
    for ev in getattr(item, "evidence", None) or []:
        desc = getattr(ev, "description", "") or ""
        raws.extend(desc.replace(",", " ").split())
    values: List[float] = []
    for raw in raws:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value > 0 and not (value != value):
            values.append(value)
    distinct: List[float] = []
    for value in values:
        if not any(abs(value - seen) < 1e-9 for seen in distinct):
            distinct.append(value)
    return distinct


def _activity_node_id(act: Any) -> Optional[str]:
    """The geometric node id baked into an activity's metadata, or None."""
    metadata = getattr(act, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    node = metadata.get("geometric_node_id")
    return node if node else None


def derived_pert_estimates(session: Any) -> Dict[str, PertEstimate]:
    """Derive a three-point estimate from every accepted duration.

    The accepted duration is the most-likely value ``m``. When the matching
    duration review mentions two or more distinct candidates, the extremes
    become optimistic/pessimistic, clamped to respect O <= M <= P; with
    fewer candidates a symmetric +/-25% band is used, for which the PERT
    expected time equals ``m`` exactly. Review items are matched by the
    activity's geometric node id, falling back to the activity id.
    """
    graph = getattr(session, "graph_model", None)
    if graph is None:
        return {}
    by_node: Dict[str, Any] = {}
    by_aid: Dict[str, Any] = {}
    for item in _duration_reviews(session):
        node = getattr(item, "geometric_node_id", None)
        if node and node not in by_node:
            by_node[node] = item
        aid = getattr(item, "activity_id", None)
        if aid and aid not in by_aid:
            by_aid[aid] = item
    out: Dict[str, PertEstimate] = {}
    for aid, act in (getattr(graph, "activities", None) or {}).items():
        duration = getattr(act, "duration", None)
        if duration is None:
            continue
        m = float(duration)
        if m == 0.0:
            # Dummy activities: the accepted zero stays exactly zero.
            out[aid] = PertEstimate(0.0, 0.0, 0.0)
            continue
        item = by_node.get(_activity_node_id(act)) or by_aid.get(aid)
        candidates = _duration_item_candidates(item) if item is not None else []
        if len(candidates) >= 2:
            a = min(min(candidates), m)
            p = max(max(candidates), m)
        else:
            a = m * (1.0 - _DERIVED_DEFAULT_BAND)
            p = m * (1.0 + _DERIVED_DEFAULT_BAND)
        out[aid] = PertEstimate(a, m, p)
    return out


@dataclass(frozen=True)
class PertRow:
    """Display row for one activity under PERT analysis."""

    activity_id: str
    optimistic: Optional[float] = None
    most_likely: Optional[float] = None
    pessimistic: Optional[float] = None
    expected_time: Optional[float] = None
    variance: Optional[float] = None
    std_dev: Optional[float] = None
    total_float: Optional[float] = None
    is_critical: bool = False


@dataclass(frozen=True)
class PertData:
    """Immutable presentation snapshot for the PERT tab."""

    status: PertStatus = PertStatus.NO_PERT_DATA
    rows: List[PertRow] = field(default_factory=list)
    result: Any = None
    estimates: Dict[str, PertEstimate] = field(default_factory=dict)
    project_duration: Optional[float] = None
    project_variance: Optional[float] = None
    project_std_dev: Optional[float] = None
    critical_paths: List[List[str]] = field(default_factory=list)
    critical_activity_count: int = 0
    estimates_source: str = ""  # "manual" / "graph" / "derived" / ""


def effective_pert_estimates(session: Any) -> Optional[Dict[str, PertEstimate]]:
    """The source of estimates for an analysis.

    Precedence: manual entries on the session win; otherwise ``None``
    tells the engine to read the O/M/P fields carried by the GraphModel
    activities; when no field exists anywhere, three-point estimates are
    derived automatically (see derived_pert_estimates).
    """
    manual = getattr(session, "pert_estimates", None) or {}
    if manual:
        return manual
    if _graph_carries_omp(getattr(session, "graph_model", None)):
        return None
    derived = derived_pert_estimates(session)
    return derived if derived else None


def pert_estimates_source(session: Any) -> str:
    """Which estimate supply wins: "manual", "graph", "derived" or ""."""
    manual = getattr(session, "pert_estimates", None) or {}
    if manual:
        return "manual"
    if _graph_carries_omp(getattr(session, "graph_model", None)):
        return "graph"
    if derived_pert_estimates(session):
        return "derived"
    return ""


def extract_pert(session: Any) -> PertData:
    """Build the PERT presentation snapshot for the current session."""
    graph = getattr(session, "graph_model", None)
    engine = PertEngine()
    estimates = effective_pert_estimates(session)
    status = engine.status(graph, estimates)
    result = getattr(session, "pert_result", None)

    resolved = resolve_estimates(graph, estimates)
    activity_results = (
        getattr(result, "activity_results", None) or {}
        if result is not None
        else {}
    )

    rows: List[PertRow] = []
    if status == PertStatus.PERT_READY:
        for aid in sorted(resolved):
            est = resolved[aid]
            analysis = activity_results.get(aid)
            rows.append(
                PertRow(
                    activity_id=aid,
                    optimistic=_num(est.optimistic),
                    most_likely=_num(est.most_likely),
                    pessimistic=_num(est.pessimistic),
                    expected_time=_num(getattr(analysis, "expected_time", None)),
                    variance=_num(getattr(analysis, "variance", None)),
                    std_dev=_num(getattr(analysis, "std_dev", None)),
                    total_float=_num(getattr(analysis, "total_float", None)),
                    is_critical=bool(getattr(analysis, "is_critical", False)),
                )
            )

    return PertData(
        status=status,
        rows=rows,
        result=result,
        estimates=resolved,
        project_duration=_num(getattr(result, "project_duration", None)),
        project_variance=_num(getattr(result, "project_variance", None)),
        project_std_dev=_num(getattr(result, "project_std_dev", None)),
        critical_paths=[
            list(p)
            for p in (getattr(result, "critical_paths", None) or [])
        ],
        critical_activity_count=_int_or_none(
            getattr(result, "critical_activity_count", 0) or 0
        ),
        estimates_source=pert_estimates_source(session),
    )