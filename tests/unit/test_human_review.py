"""
Tests for the human-review model and the CPM gate.

Covers the review model (activities / dependencies / durations), the
review session, deterministic application of review decisions, graph
validation, and the CPM gate (RUNNABLE only when the reviewed graph is
VALID).  The CV pipeline and the CPM mathematics are never modified;
every decision is recorded in the audit trail and the original
reconstruction is never mutated.
"""

from __future__ import annotations

import copy

import pytest

from pert_analyzer.cv.reconstruction_models import (
    ReconstructedActivity,
    ReconstructedDependency,
    ReconstructedDiagram,
)
from pert_analyzer.pipeline.human_review import (
    ActivityReview,
    DependencyReview,
    DurationReview,
    DurationStatus,
    GraphStatus,
    ReviewSession,
    ReviewStatus,
    apply_review_decisions,
    build_review_session,
    build_review_summary,
    validate_graph_structure,
)
from pert_analyzer.pipeline.review_api import ReviewWorkflow
from pert_analyzer.pipeline.result import PipelineResult


# =============================================================================
# Helpers
# =============================================================================


def make_recon(activities, dependencies=None, diagram_type="AON"):
    """Build a small ReconstructedDiagram with geometric nodes."""
    recon = ReconstructedDiagram(diagram_type=diagram_type)
    for a in activities:
        recon.activities.append(
            ReconstructedActivity(
                activity_id=a["id"],
                label=a.get("label", a["id"]),
                duration=a.get("duration", 0.0),
                is_dummy=a.get("is_dummy", False),
                confidence=a.get("confidence", 0.9),
                source_node_id=a["node"],
                geometric_node_id=a["node"],
                needs_review=a.get("needs_review", False),
                semantic_activity_id=a.get("semantic_id", a["id"]),
            )
        )
    for dep in dependencies or []:
        recon.dependencies.append(
            ReconstructedDependency(
                source_id=dep["source"],
                target_id=dep["target"],
                confidence=0.9,
                source_arrow_id=dep.get("arrow_id"),
            )
        )
    return recon


def make_session(recon, source_image_id="test-image"):
    """Build a ReviewSession referencing an immutable reconstruction."""
    return ReviewSession(source_image_id=source_image_id, reconstruction=recon)


def add_activity_review(session, node, current_id, status=ReviewStatus.PENDING):
    item = ActivityReview(
        geometric_node_id=node,
        current_activity_id=current_id,
        proposed_activity_id=current_id,
        confidence=0.4,
        reason="needs review",
        status=status,
        raw_semantic_id=current_id,
    )
    session.activities.append(item)
    return item


def add_dependency_review(session, arrow_id, source_node, target_node, src_id, tgt_id):
    item = DependencyReview(
        arrow_id=arrow_id,
        source_node_id=source_node,
        target_node_id=target_node,
        current_source_id=src_id,
        current_target_id=tgt_id,
        proposed_direction=f"{src_id}->{tgt_id}",
        confidence=0.4,
        reason="ambiguous arrow",
        status=ReviewStatus.PENDING,
    )
    session.dependencies.append(item)
    return item


def add_duration_review(session, node, current_duration, status=ReviewStatus.PENDING):
    item = DurationReview(
        geometric_node_id=node,
        current_duration=current_duration,
        duration_status=DurationStatus.REVIEW_REQUIRED,
        reason="missing/invalid duration",
        status=status,
    )
    session.durations.append(item)
    return item


def graph_edges(candidate):
    if candidate.graph is None:
        return set()
    return {(d.source, d.target) for d in candidate.graph.dependencies}


# =============================================================================
# 1. Dependency: ACCEPT
# =============================================================================


class TestDependencyAccept:
    def test_accept_adds_edge_and_runs_cpm(self) -> None:
        recon = make_recon(
            activities=[
                {"id": "A", "node": "n_a", "duration": 1},
                {"id": "B", "node": "n_b", "duration": 2},
                {"id": "C", "node": "n_c", "duration": 3},
            ],
            dependencies=[{"source": "B", "target": "C"}],
        )
        session = make_session(recon)
        add_dependency_review(
            session, "ar1", "n_a", "n_b", "A", "B"
        )

        assert session.decide_dependency("ar1", "ACCEPT", reason="visually clear") is True
        candidate = apply_review_decisions(recon, session)

        assert ("A", "B") in graph_edges(candidate)
        assert ("B", "C") in graph_edges(candidate)
        assert candidate.validation.is_valid
        assert candidate.cpm_gate.value == "RUNNABLE"
        assert candidate.cpm_project_duration == 6.0
        assert candidate.critical_path_count == 1


# =============================================================================
# 2. Dependency: REJECT
# =============================================================================


class TestDependencyReject:
    def test_reject_removes_edge(self) -> None:
        recon = make_recon(
            activities=[
                {"id": "A", "node": "n_a", "duration": 1},
                {"id": "B", "node": "n_b", "duration": 2},
                {"id": "C", "node": "n_c", "duration": 3},
            ],
            dependencies=[{"source": "B", "target": "C"}],
        )
        session = make_session(recon)
        add_dependency_review(session, "ar1", "n_a", "n_b", "A", "B")

        session.decide_dependency("ar1", "REJECT", reason="false positive")
        candidate = apply_review_decisions(recon, session)

        assert ("A", "B") not in graph_edges(candidate)
        assert candidate.cpm is None
        assert candidate.cpm_gate.value == "BLOCKED_REVIEW"
        codes = [e.code for e in candidate.validation.errors]
        assert "isolated_activity" in codes


# =============================================================================
# 3. Dependency: REVERSE
# =============================================================================


class TestDependencyReverse:
    def test_reverse_swaps_source_and_target(self) -> None:
        recon = make_recon(
            activities=[
                {"id": "A", "node": "n_a", "duration": 1},
                {"id": "B", "node": "n_b", "duration": 2},
            ],
        )
        session = make_session(recon)
        add_dependency_review(session, "ar1", "n_a", "n_b", "A", "B")

        session.decide_dependency("ar1", "REVERSE", reason="arrow points the other way")
        candidate = apply_review_decisions(recon, session)

        assert ("B", "A") in graph_edges(candidate)
        assert ("A", "B") not in graph_edges(candidate)
        entry = session.decisions[0]
        assert entry.decision == "REVERSE"
        assert entry.original_value == "A->B"
        assert entry.corrected_value == "B->A"


# =============================================================================
# 4. Dependency: CHANGE_SOURCE
# =============================================================================


class TestDependencyChangeSource:
    def test_change_source_relinks_edge(self) -> None:
        recon = make_recon(
            activities=[
                {"id": "A", "node": "n_a", "duration": 1},
                {"id": "B", "node": "n_b", "duration": 2},
                {"id": "C", "node": "n_c", "duration": 3},
            ],
        )
        session = make_session(recon)
        add_dependency_review(session, "ar1", "n_a", "n_b", "A", "B")

        session.decide_dependency(
            "ar1", "CHANGE_SOURCE", corrected_source_id="C",
            reason="source should be C",
        )
        candidate = apply_review_decisions(recon, session)

        assert ("C", "B") in graph_edges(candidate)
        assert ("A", "B") not in graph_edges(candidate)


# =============================================================================
# 5. Dependency: CHANGE_TARGET
# =============================================================================


class TestDependencyChangeTarget:
    def test_change_target_relinks_edge(self) -> None:
        recon = make_recon(
            activities=[
                {"id": "A", "node": "n_a", "duration": 1},
                {"id": "B", "node": "n_b", "duration": 2},
                {"id": "C", "node": "n_c", "duration": 3},
            ],
        )
        session = make_session(recon)
        add_dependency_review(session, "ar1", "n_a", "n_b", "A", "B")

        session.decide_dependency(
            "ar1", "CHANGE_TARGET", corrected_target_id="C",
            reason="target should be C",
        )
        candidate = apply_review_decisions(recon, session)

        assert ("A", "C") in graph_edges(candidate)
        assert ("A", "B") not in graph_edges(candidate)


# =============================================================================
# 6. Activity: CORRECT ID
# =============================================================================


class TestCorrectActivityId:
    def test_corrected_id_flows_through_dependencies(self) -> None:
        recon = make_recon(
            activities=[
                {"id": "X1", "node": "n_x", "duration": 1},
                {"id": "B", "node": "n_b", "duration": 2},
            ],
            dependencies=[{"source": "X1", "target": "B"}],
        )
        session = make_session(recon)
        item = add_activity_review(session, "n_x", "X1")

        session.decide_activity("n_x", "CORRECT", corrected_id="A", reason="OCR misread")
        candidate = apply_review_decisions(recon, session)

        assert item.status == ReviewStatus.CORRECTED
        assert {"A", "B"} == set(candidate.graph.activities.keys())
        assert ("A", "B") in graph_edges(candidate)
        assert candidate.validation.is_valid
        assert candidate.cpm_project_duration == 3.0


# =============================================================================
# 7. Duration: CORRECT (Q = 0 -> 3) unblocks CPM
# =============================================================================


class TestCorrectDuration:
    def test_zero_duration_blocks_then_correction_unblocks(self) -> None:
        recon = make_recon(
            activities=[
                {"id": "P", "node": "n_p", "duration": 4},
                {"id": "Q", "node": "n_q", "duration": 0},
                {"id": "R", "node": "n_r", "duration": 2},
            ],
            dependencies=[
                {"source": "P", "target": "Q"},
                {"source": "Q", "target": "R"},
            ],
        )
        session = make_session(recon)
        add_duration_review(session, "n_q", 0.0)

        # Before correction: CPM must be blocked.
        blocked = apply_review_decisions(recon, session)
        assert blocked.cpm_gate.value == "BLOCKED_REVIEW"
        assert blocked.cpm is None
        codes = [e.code for e in blocked.validation.errors]
        assert "missing_duration" in codes

        assert session.decide_duration(
            "n_q", "CORRECT", corrected_duration=3, reason="OCR missed the 3"
        ) is True
        candidate = apply_review_decisions(recon, session)

        assert candidate.validation.is_valid
        assert candidate.cpm_gate.value == "RUNNABLE"
        assert candidate.cpm_project_duration == 9.0
        assert candidate.critical_path_count == 1


# =============================================================================
# 8. LEAVE_UNRESOLVED keeps item PENDING
# =============================================================================


class TestLeaveUnresolved:
    def test_leave_unresolved_keeps_pending_and_no_decision(self) -> None:
        recon = make_recon(
            activities=[
                {"id": "X1", "node": "n_x", "duration": 0},
                {"id": "B", "node": "n_b", "duration": 2},
            ],
            dependencies=[{"source": "X1", "target": "B"}],
        )
        session = make_session(recon)
        item = add_activity_review(session, "n_x", "X1")
        add_duration_review(session, "n_x", 0.0)

        session.decide_activity("n_x", "LEAVE_UNRESOLVED")
        session.decide_duration("n_x", "LEAVE_UNRESOLVED")

        assert item.status == ReviewStatus.PENDING
        assert session.decisions == []
        candidate = apply_review_decisions(recon, session)
        # Original ID/duration preserved; graph stays invalid (missing duration).
        assert candidate.cpm_gate.value == "BLOCKED_REVIEW"


# =============================================================================
# 9. Determinism
# =============================================================================


class TestDeterminism:
    def test_apply_is_deterministic(self) -> None:
        def _run():
            recon = make_recon(
                activities=[
                    {"id": "A", "node": "n_a", "duration": 1},
                    {"id": "B", "node": "n_b", "duration": 2},
                    {"id": "C", "node": "n_c", "duration": 3},
                ],
            )
            session = make_session(recon)
            add_dependency_review(session, "ar1", "n_a", "n_b", "A", "B")
            session.decide_dependency("ar1", "ACCEPT")
            candidate = apply_review_decisions(recon, session)
            return candidate

        c1, c2 = _run(), _run()
        assert c1.cpm_project_duration == c2.cpm_project_duration
        assert c1.critical_path_count == c2.critical_path_count
        assert graph_edges(c1) == graph_edges(c2)
        assert c1.cpm_gate == c2.cpm_gate


# =============================================================================
# 10. Graph validation reports structured errors
# =============================================================================


class TestGraphValidation:
    def test_isolated_activity_detected(self) -> None:
        recon = make_recon(
            activities=[{"id": "A", "node": "n_a", "duration": 5}],
        )
        session = make_session(recon)
        candidate = apply_review_decisions(recon, session)
        codes = [e.code for e in candidate.validation.errors]
        assert "isolated_activity" in codes
        assert candidate.validation.status == GraphStatus.INVALID

    def test_validate_graph_structure_checks_edges_and_cycles(self) -> None:
        result = validate_graph_structure(
            activity_ids=["A", "B"],
            durations={"A": 1.0, "B": 1.0},
            edges=[("A", "B"), ("B", "A")],
        )
        assert not result.is_valid
        codes = [e.code for e in result.errors]
        assert "cycle" in codes


# =============================================================================
# 11. CPM blocked when graph invalid
# =============================================================================


class TestCpmBlocked:
    def test_invalid_graph_never_runs_cpm(self) -> None:
        recon = make_recon(
            activities=[
                {"id": "A", "node": "n_a", "duration": 0},
                {"id": "B", "node": "n_b", "duration": 2},
            ],
            dependencies=[{"source": "A", "target": "B"}],
        )
        session = make_session(recon)
        candidate = apply_review_decisions(recon, session)
        assert candidate.cpm is None
        assert candidate.cpm_project_duration is None
        assert candidate.critical_path_count is None
        assert candidate.cpm_gate.value == "BLOCKED_REVIEW"


# =============================================================================
# 12. CPM allowed when graph valid
# =============================================================================


class TestCpmAllowed:
    def test_valid_graph_runs_cpm(self) -> None:
        recon = make_recon(
            activities=[
                {"id": "A", "node": "n_a", "duration": 1},
                {"id": "B", "node": "n_b", "duration": 2},
                {"id": "C", "node": "n_c", "duration": 3},
            ],
            dependencies=[
                {"source": "A", "target": "B"},
                {"source": "B", "target": "C"},
            ],
        )
        session = make_session(recon)
        candidate = apply_review_decisions(recon, session)
        assert candidate.validation.is_valid
        assert candidate.cpm_gate.value == "RUNNABLE"
        assert candidate.cpm_project_duration == 6.0
        assert candidate.critical_path_count == 1


# =============================================================================
# 13. Provenance / audit trail
# =============================================================================


class TestProvenance:
    def test_original_values_preserved_in_review_and_audit(self) -> None:
        recon = make_recon(
            activities=[
                {"id": "X1", "node": "n_x", "duration": 3},
                {"id": "B", "node": "n_b", "duration": 2},
            ],
            dependencies=[{"source": "X1", "target": "B"}],
        )
        session = make_session(recon)
        item = add_activity_review(session, "n_x", "X1")
        session.decide_activity("n_x", "CORRECT", corrected_id="A", reason="OCR misread")

        apply_review_decisions(recon, session)

        # Raw OCR id stays untouched; audit trail records original + corrected.
        assert item.current_activity_id == "X1"
        assert item.corrected_activity_id == "A"
        assert item.raw_semantic_id == "X1"
        entry = session.decisions[0]
        assert entry.item_type == "activity"
        assert entry.item_id == "n_x"
        assert entry.original_value == "X1"
        assert entry.corrected_value == "A"
        assert entry.reason == "OCR misread"
        assert entry.evidence_reference == "n_x"


# =============================================================================
# 14. Original reconstruction is immutable
# =============================================================================


class TestImmutability:
    def test_apply_never_mutates_reconstruction(self) -> None:
        recon = make_recon(
            activities=[
                {"id": "X1", "node": "n_x", "duration": 3},
                {"id": "B", "node": "n_b", "duration": 2},
            ],
            dependencies=[{"source": "X1", "target": "B"}],
        )
        snapshot_activities = copy.deepcopy(recon.activities)
        snapshot_dependencies = copy.deepcopy(recon.dependencies)

        session = make_session(recon)
        add_activity_review(session, "n_x", "X1")
        add_duration_review(session, "n_x", 3.0)
        session.decide_activity("n_x", "CORRECT", corrected_id="A")
        session.decide_duration("n_x", "ACCEPT")
        add_dependency_review(session, "ar1", "n_x", "n_b", "X1", "B")
        session.decide_dependency("ar1", "ACCEPT")

        apply_review_decisions(recon, session)

        assert recon.activities == snapshot_activities
        assert recon.dependencies == snapshot_dependencies
        assert recon.activities[0].activity_id == "X1"
        assert recon.activities[0].duration == 3.0


# =============================================================================
# 15. Duplicate decision handling
# =============================================================================


class TestDuplicateDecisions:
    def test_deciding_same_item_twice_is_noop(self) -> None:
        recon = make_recon(
            activities=[
                {"id": "A", "node": "n_a", "duration": 1},
                {"id": "B", "node": "n_b", "duration": 2},
            ],
        )
        session = make_session(recon)
        add_dependency_review(session, "ar1", "n_a", "n_b", "A", "B")

        assert session.decide_dependency("ar1", "ACCEPT") is True
        assert session.decide_dependency("ar1", "ACCEPT") is False
        assert len(session.decisions) == 1

    def test_two_review_decisions_for_same_edge_is_error(self) -> None:
        recon = make_recon(
            activities=[
                {"id": "A", "node": "n_a", "duration": 1},
                {"id": "B", "node": "n_b", "duration": 2},
                {"id": "C", "node": "n_c", "duration": 3},
            ],
            dependencies=[{"source": "B", "target": "C"}],
        )
        session = make_session(recon)
        add_dependency_review(session, "ar1", "n_a", "n_b", "A", "B")
        add_dependency_review(session, "ar2", "n_a", "n_b", "A", "B")
        session.decide_dependency("ar1", "ACCEPT")
        session.decide_dependency("ar2", "ACCEPT")

        candidate = apply_review_decisions(recon, session)
        codes = [e.code for e in candidate.validation.errors]
        assert "duplicate_edge" in codes
        assert candidate.cpm_gate.value == "BLOCKED_REVIEW"

    def test_auto_dep_overlapping_review_decision_is_deduped(self) -> None:
        recon = make_recon(
            activities=[
                {"id": "A", "node": "n_a", "duration": 1},
                {"id": "B", "node": "n_b", "duration": 2},
            ],
            dependencies=[{"source": "A", "target": "B"}],  # auto-accepted by CV
        )
        session = make_session(recon)
        add_dependency_review(session, "ar1", "n_a", "n_b", "A", "B")
        session.decide_dependency("ar1", "ACCEPT")

        candidate = apply_review_decisions(recon, session)
        codes = [e.code for e in candidate.validation.errors]
        # CV auto edge + human accept = same single edge, no error.
        assert "duplicate_edge" not in codes
        assert candidate.graph.dependency_count == 1


# =============================================================================
# 16. Session serialization round-trip
# =============================================================================


class TestSerialization:
    def test_to_dict_from_dict_roundtrip(self) -> None:
        recon = make_recon(
            activities=[
                {"id": "X1", "node": "n_x", "duration": 0},
                {"id": "B", "node": "n_b", "duration": 2},
            ],
        )
        session = make_session(recon)
        add_activity_review(session, "n_x", "X1")
        add_duration_review(session, "n_x", 0.0)
        add_dependency_review(session, "ar1", "n_x", "n_b", "X1", "B")
        session.decide_activity("n_x", "CORRECT", corrected_id="A")
        session.decide_duration("n_x", "CORRECT", corrected_duration=1)
        session.decide_dependency("ar1", "ACCEPT")

        data = session.to_dict()
        restored = ReviewSession.from_dict(data, reconstruction=recon)

        assert restored.source_image_id == session.source_image_id
        assert restored.pending_activity_count == session.pending_activity_count == 0
        assert len(restored.decisions) == len(session.decisions) == 3
        assert restored.decisions[0].original_value == "X1"
        assert restored.decisions[0].corrected_value == "A"
        assert restored.activities[0].corrected_activity_id == "A"
        assert restored.durations[0].corrected_duration == 1.0
        assert restored.dependencies[0].status == ReviewStatus.ACCEPTED

        candidate = apply_review_decisions(recon, restored)
        assert candidate.cpm_project_duration == 3.0


# =============================================================================
# Extra: ReviewSummary
# =============================================================================


class TestReviewSummary:
    def test_summary_counts(self) -> None:
        recon = make_recon(
            activities=[
                {"id": "A", "node": "n_a", "duration": 1},
                {"id": "B", "node": "n_b", "duration": 2},
                {"id": "C", "node": "n_c", "duration": 3},
            ],
            dependencies=[
                {"source": "A", "target": "B"},
                {"source": "B", "target": "C"},
            ],
        )
        session = make_session(recon)
        item = add_duration_review(session, "n_c", 3.0)
        session.decide_duration("n_c", "ACCEPT")

        candidate = apply_review_decisions(recon, session)
        summary = build_review_summary(candidate)

        assert summary.total_activities == 3
        assert summary.confirmed_activities == 3
        assert summary.activity_reviews_pending == 0
        assert summary.duration_reviews_pending == 0
        assert summary.duration_corrected == 0
        assert summary.graph_validation_status == GraphStatus.VALID.value
        assert summary.cpm_eligibility == "RUNNABLE"
        assert summary.accepted_dependencies == 2


# =============================================================================
# Extra: ReviewWorkflow (GUI-independent API)
# =============================================================================


class TestReviewWorkflow:
    def test_workflow_end_to_end_api(self) -> None:
        recon = make_recon(
            activities=[
                {"id": "X1", "node": "n_x", "duration": 0, "needs_review": True},
                {"id": "B", "node": "n_b", "duration": 2},
            ],
            dependencies=[{"source": "X1", "target": "B"}],
        )
        result = PipelineResult()
        result._reconstruction = recon

        workflow = ReviewWorkflow.from_pipeline_result(result, source_image_id="workflow-test")
        session = workflow.get_review_items()

        assert session.source_image_id == "workflow-test"
        assert session.pending_duration_count >= 1
        assert session.pending_activity_count >= 1

        assert workflow.decide_activity("n_x", "CORRECT", corrected_id="A", reason="OCR") is True
        assert workflow.decide_duration(
            "n_x", "CORRECT", corrected_duration=1, reason="missing"
        ) is True

        candidate = workflow.apply()
        assert candidate.validation.is_valid
        assert candidate.cpm_gate.value == "RUNNABLE"

        cpm = workflow.run_cpm()
        assert cpm.project_duration == 3.0

        summary = workflow.summary()
        assert summary["cpm_eligibility"] == "RUNNABLE"


# =============================================================================
# A-15: corrected three-point estimates (O/M/P) on duration decisions
# =============================================================================


class TestDurationThreePointDecision:
    def test_correct_stores_extremes_and_validates_order(self) -> None:
        recon = make_recon(activities=[{"id": "X1", "node": "n_x", "duration": 0}])
        session = make_session(recon)
        add_duration_review(session, "n_x", 0.0)

        session.decide_duration(
            "n_x", "CORRECT", corrected_duration=8.0,
            corrected_optimistic=5.0, corrected_pessimistic=10.0,
        )
        item = session.get_duration_review("n_x")
        assert item.corrected_duration == 8.0
        assert item.corrected_optimistic == 5.0
        assert item.corrected_pessimistic == 10.0

    def test_correct_rejects_optimistic_above_duration(self) -> None:
        recon = make_recon(activities=[{"id": "X1", "node": "n_x", "duration": 0}])
        session = make_session(recon)
        add_duration_review(session, "n_x", 0.0)

        with pytest.raises(ValueError, match="O must be <= M"):
            session.decide_duration(
                "n_x", "CORRECT", corrected_duration=8.0,
                corrected_optimistic=9.0,
            )

    def test_correct_rejects_pessimistic_below_duration(self) -> None:
        recon = make_recon(activities=[{"id": "X1", "node": "n_x", "duration": 0}])
        session = make_session(recon)
        add_duration_review(session, "n_x", 0.0)

        with pytest.raises(ValueError, match="P must be >= M"):
            session.decide_duration(
                "n_x", "CORRECT", corrected_duration=8.0,
                corrected_pessimistic=7.0,
            )

    def test_correct_rejects_negative_extreme(self) -> None:
        recon = make_recon(activities=[{"id": "X1", "node": "n_x", "duration": 0}])
        session = make_session(recon)
        add_duration_review(session, "n_x", 0.0)

        with pytest.raises(ValueError, match="non-negative"):
            session.decide_duration(
                "n_x", "CORRECT", corrected_duration=8.0,
                corrected_optimistic=-1.0,
            )

    def test_roundtrip_persists_extremes(self) -> None:
        recon = make_recon(activities=[{"id": "X1", "node": "n_x", "duration": 0}])
        session = make_session(recon)
        add_duration_review(session, "n_x", 0.0)
        session.decide_duration(
            "n_x", "CORRECT", corrected_duration=4.0,
            corrected_optimistic=3.0, corrected_pessimistic=6.0,
        )

        restored = ReviewSession.from_dict(session.to_dict(), reconstruction=recon)
        restored_item = restored.get_duration_review("n_x")
        assert restored_item.corrected_duration == 4.0
        assert restored_item.corrected_optimistic == 3.0
        assert restored_item.corrected_pessimistic == 6.0

    def test_apply_writes_extremes_onto_graph_activity(self) -> None:
        recon = make_recon(
            activities=[
                {"id": "X1", "node": "n_x", "duration": 0},
                {"id": "B", "node": "n_b", "duration": 2},
            ],
            dependencies=[{"source": "X1", "target": "B"}],
        )
        session = make_session(recon)
        add_duration_review(session, "n_x", 0.0)
        session.decide_duration(
            "n_x", "CORRECT", corrected_duration=8.0,
            corrected_optimistic=5.0, corrected_pessimistic=10.0,
        )

        candidate = apply_review_decisions(recon, session)
        graph = candidate.graph
        corrected_act = graph.activities["X1"]
        assert corrected_act.duration == 8.0
        assert corrected_act.optimistic_time == 5.0
        assert corrected_act.most_likely_time == 8.0
        assert corrected_act.pessimistic_time == 10.0
        # The other activity keeps no O/M/P fields (derivation fills them).
        other = graph.activities["B"]
        assert other.duration == 2.0
        assert other.optimistic_time is None
        assert other.pessimistic_time is None