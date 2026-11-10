"""
Derived CPM ground truth for the generalization benchmark.

Given a parsed v1.0 annotation, this module rebuilds the reference project
graph the annotation describes and runs the production CPM engine on it.
The result is a *derived* expectation: it comes from the human-annotated
structure (activities, durations, dependencies) and is computed
deterministically by the same engine the pipeline uses. No annotation file
is modified and no value is fabricated.

Scope decisions (measured, honest):
- AON only: the CPM engine models activities as graph nodes with
  finish-to-start dependencies. AOA event graphs need the production
  event-activity transform, which is deliberately out of scope here and is
  reported as ``aoa_not_supported_in_compare``.
- An expectation is only produced when every comparable (non-uncertain)
  activity carries a duration; otherwise ``missing_durations`` is reported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["ExpectedCPM", "expected_cpm_from_annotation", "cpm_vs_gt_block"]


@dataclass
class ExpectedCPM:
    """Derived CPM expectation from a ground-truth annotation."""

    available: bool
    reason: Optional[str] = None
    diagram_type: Optional[str] = None
    project_duration: Optional[float] = None
    critical_paths: List[List[str]] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)

    # Aliases for JSON consumers.
    @property
    def critical_path_count(self) -> int:
        return len(self.critical_paths)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "diagram_type": self.diagram_type,
            "project_duration": self.project_duration,
            "critical_path_count": self.critical_path_count,
            "critical_paths": self.critical_paths,
            "detail": self.detail,
        }


def expected_cpm_from_annotation(annotation: Any) -> ExpectedCPM:
    """Derive the expected CPM result from a parsed v1.0 annotation.

    Returns an ``ExpectedCPM`` with ``available=False`` plus a machine
    reason whenever the expectation cannot be computed deterministically
    (AOA, missing durations, duplicates, graph cycles, empty graph).
    Never raises: measurement must not abort benchmarking.
    """
    diagram_type = str(getattr(annotation, "diagram_type", "") or "").upper()
    if diagram_type != "AON":
        return ExpectedCPM(
            available=False,
            reason="aoa_not_supported_in_compare",
            diagram_type=diagram_type,
        )
    if getattr(annotation, "annotation_status", "COMPLETE") == "UNCERTAIN":
        return ExpectedCPM(
            available=False, reason="annotation_uncertain", diagram_type=diagram_type
        )

    comparable = list(getattr(annotation, "comparable_activities", lambda: [])())
    if not comparable:
        return ExpectedCPM(
            available=False, reason="no_comparable_activities", diagram_type=diagram_type
        )

    missing = [a.id for a in comparable if a.duration is None]
    if missing:
        return ExpectedCPM(
            available=False,
            reason="missing_durations",
            diagram_type=diagram_type,
            detail={"missing_activity_ids": missing},
        )

    ids = [a.id for a in comparable]
    duplicates = sorted({aid for aid in ids if ids.count(aid) > 1})
    if duplicates:
        return ExpectedCPM(
            available=False,
            reason="duplicate_activity_ids",
            diagram_type=diagram_type,
            detail={"duplicate_ids": duplicates},
        )

    from pert_analyzer.core.models import (
        Activity,
        Dependency,
        DiagramType,
        GraphModel,
    )

    graph = GraphModel(diagram_type=DiagramType.AON)
    for act in comparable:
        graph.add_activity(
            Activity(activity_id=act.id, name=act.id, duration=float(act.duration))
        )

    dependency_refs = [
        (str(d[0]), str(d[1])) for d in (getattr(annotation, "comparable_dependencies", lambda: [])())
    ]
    activity_ids = {a.id for a in comparable}
    invalid_refs = sorted({p for edge in dependency_refs for p in edge if p not in activity_ids})
    for source, target in dependency_refs:
        if source in activity_ids and target in activity_ids:
            graph.dependencies.append(Dependency(source=source, target=target))

    from pert_analyzer.analysis.cpm_engine import CPMEngine

    try:
        analysis = CPMEngine().analyze(graph)
    except Exception as exc:  # noqa: BLE001 - GT graphs may be cyclic/incomplete
        return ExpectedCPM(
            available=False,
            reason=str(type(exc).__name__),
            diagram_type=diagram_type,
            detail={"error": str(exc)},
        )

    return ExpectedCPM(
        available=True,
        diagram_type=diagram_type,
        project_duration=float(analysis.project_duration),
        critical_paths=[list(p) for p in (analysis.critical_paths or [])],
    )


def cpm_vs_gt_block(
    annotation: Any,
    detected_project_duration: Any,
    detected_critical_path_count: Any,
) -> Dict[str, Any]:
    """Build the measured-vs-derived CPM comparison block for one image.

    ``detected_*`` come from the pipeline's own CPM stage (already
    surfaced by ``metrics.extract_stage_metrics``); they may be the
    sentinel ``UNAVAILABLE`` when CPM did not run (invalid graph).
    """
    expected = expected_cpm_from_annotation(annotation)
    block = {
        "expected": expected.to_dict(),
        "detected_project_duration": detected_project_duration,
        "detected_critical_path_count": detected_critical_path_count,
    }
    if (
        expected.available
        and isinstance(detected_project_duration, (int, float))
        and not isinstance(detected_project_duration, bool)
    ):
        block["project_duration_delta"] = round(
            float(detected_project_duration) - float(expected.project_duration),
            4,
        )
    if expected.available and isinstance(detected_critical_path_count, int):
        block["critical_path_count_delta"] = (
            detected_critical_path_count - expected.critical_path_count
        )
    return block
