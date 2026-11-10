# -*- coding: utf-8 -*-
"""Unit tests for Batch E derived-CPM compare and promotion-gate digest.

The canonical expectation is pinned to documented values in
``tests/test_data/ground_truth/README.md`` for ``1.json``:
22 activities, 28 dependencies, CPM duration 54, 16 critical paths.
No real CV/OCR pipeline runs here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pert_analyzer.benchmark.annotations import (
    AnnotatedActivity,
    GroundTruthAnnotation,
    from_dict,
)
from pert_analyzer.benchmark.cpm_compare import (
    ExpectedCPM,
    cpm_vs_gt_block,
    expected_cpm_from_annotation,
)
from pert_analyzer.benchmark.metrics import UNAVAILABLE
from pert_analyzer.cv.validated_dependency import (
    DependencyValidationReport,
    ValidatedDependency,
    ValidatedDependencyBuilder,
)
from pert_analyzer.cv.validated_dependency import DependencyEvidence


_GROUND_TRUTH_DIR = Path(__file__).resolve().parents[1] / "test_data" / "ground_truth"


def _load_annotation(stem: str):
    path = _GROUND_TRUTH_DIR / f"{stem}.json"
    if not path.is_file():
        pytest.skip(f"ground-truth corpus file missing: {path}")
    return from_dict(json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Derived expectation (AON)
# ---------------------------------------------------------------------------


def test_canonical_reference_matches_documented_cpm():
    annotation = _load_annotation("1")
    expected = expected_cpm_from_annotation(annotation)
    assert expected.available is True
    assert expected.reason is None
    assert pytest.approx(expected.project_duration, abs=1e-6) == 54.0
    assert expected.critical_path_count == 16


def test_all_complete_aon_images_derive_expectations():
    for stem in ("5", "11", "7"):
        annotation = _load_annotation(stem)
        expected = expected_cpm_from_annotation(annotation)
        assert expected.available is True, f"{stem}: {expected.reason}"
        assert expected.project_duration is not None
        assert expected.critical_path_count >= 1


def test_aoa_is_reported_as_unsupported_not_fabricated():
    for stem in ("2", "3", "4", "6"):
        annotation = _load_annotation(stem)
        expected = expected_cpm_from_annotation(annotation)
        assert expected.available is False
        assert expected.reason == "aoa_not_supported_in_compare"
        assert expected.project_duration is None


def test_missing_durations_block_expectation():
    annotation = GroundTruthAnnotation(
        image="synthetic.png",
        schema_version="1.0",
        annotation_status="COMPLETE",
        diagram_type="AON",
        activities=[
            AnnotatedActivity(id="A", duration=3.0),
            AnnotatedActivity(id="B", duration=None),
        ],
        dependencies=[("A", "B")],
    )
    expected = expected_cpm_from_annotation(annotation)
    assert expected.available is False
    assert expected.reason == "missing_durations"
    assert expected.detail.get("missing_activity_ids") == ["B"]


def test_uncertain_annotation_is_excluded():
    annotation = GroundTruthAnnotation(
        image="synthetic.png",
        schema_version="1.0",
        annotation_status="UNCERTAIN",
        diagram_type="AON",
        activities=[AnnotatedActivity(id="A", duration=1.0)],
    )
    expected = expected_cpm_from_annotation(annotation)
    assert expected.available is False
    assert expected.reason == "annotation_uncertain"


# ---------------------------------------------------------------------------
# Comparison block
# ---------------------------------------------------------------------------


def test_compare_block_reports_delta_when_detected_numbers_exist():
    annotation = _load_annotation("5")
    # Detected values come from the pipeline CPM stage; use a synthetic
    # pair (duration off by +4.5, one extra critical path).
    block = cpm_vs_gt_block(annotation, 60.0, 2)
    expected = block["expected"]
    assert expected["available"] is True
    assert block["project_duration_delta"] == pytest.approx(60.0 - expected["project_duration"], abs=1e-6)
    assert block["critical_path_count_delta"] == 2 - expected["critical_path_count"]


def test_compare_block_safe_when_cpm_unavailable():
    annotation = _load_annotation("5")
    block = cpm_vs_gt_block(annotation, UNAVAILABLE, UNAVAILABLE)
    assert block["expected"]["available"] is True
    assert block["detected_project_duration"] == UNAVAILABLE
    assert "project_duration_delta" not in block
    assert "critical_path_count_delta" not in block


# ---------------------------------------------------------------------------
# Promotion-gate digest (diagnostics only)
# ---------------------------------------------------------------------------


def _make_dep(confidence=0.9, arrowhead=0.9, boundary=0.9,
              direction=0.9, angular=0.9, route=0.9,
              crossing=0.0) -> ValidatedDependency:
    dep = ValidatedDependency(
        source_id="A",
        target_id="B",
        arrow_id="arrow_test",
    )
    dep.confidence_score = confidence
    dep.evidence = DependencyEvidence()
    dep.evidence.arrowhead_confidence = arrowhead
    dep.evidence.source_boundary_intersection = boundary
    dep.evidence.target_boundary_intersection = boundary
    dep.evidence.direction_consistency = direction
    dep.evidence.angular_consistency = angular
    dep.evidence.visual_route_support = route
    dep.evidence.shaft_visual_support = route
    dep.metadata["crossing_penalty"] = crossing
    return dep


def per_gate_values(dep):
    gates = ValidatedDependencyBuilder.evaluate_promotion_gates(dep)
    assert gates is not None
    return gates


def test_all_high_quality_candidate_passes_every_gate():
    gates = per_gate_values(_make_dep())
    assert gates["all_passed"] is True
    assert all(gates["passed"].values())


def test_low_confidence_blocks_promotion():
    gates = per_gate_values(_make_dep(confidence=0.55))
    assert gates["all_passed"] is False
    passed = gates["passed"]
    assert passed["confidence"] is False
    assert passed["arrowhead"] is True


def test_crossing_penalty_uses_lt_semantics():
    # Exactly at the threshold counts as failed (strictly-less relation).
    gates = per_gate_values(_make_dep(crossing=0.30))
    assert gates["passed"]["crossing_penalty"] is False
    gates = per_gate_values(_make_dep(crossing=0.20))
    assert gates["passed"]["crossing_penalty"] is True


def test_gate_digest_histogram_and_aggregates():
    builder = ValidatedDependencyBuilder()
    deps = [
        _make_dep(confidence=0.9),            # passes everything
        _make_dep(confidence=0.55),           # confidence fails first
        _make_dep(angular=0.2, direction=0.2) # direction fails before angular
    ]
    report = DependencyValidationReport()
    builder._build_debug_entries(deps, report)
    digest = report.gate_digest
    assert digest["candidates_evaluated"] == 3
    assert digest["all_gates_passed"] == 1
    assert digest["first_failing_gate"].get("confidence") == 1
    assert digest["first_failing_gate"].get("direction") == 1
    per_gate = digest["per_gate"]
    assert per_gate["confidence"]["failed"] >= 1
    # Each debug entry carries its per-candidate gate values.
    for entry in report.debug_entries:
        assert "promotion_gates" in entry


def test_candidate_without_evidence_fails_all_gates_deterministically():
    dep = ValidatedDependency(
        source_id="A", target_id="B", arrow_id="arrow_test"
    )
    gates = ValidatedDependencyBuilder.evaluate_promotion_gates(dep)
    assert gates["all_passed"] is False
    assert gates["values"]["confidence"] == 0.0
