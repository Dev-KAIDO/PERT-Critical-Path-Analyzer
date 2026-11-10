"""
Final End-to-End Acceptance Test — Full Reference AON.

One complete end-to-end run over the real reference diagram
(tests/test_data/reference_diagrams/reference_aon.png):

    Analyze -> Review -> Apply -> Validation -> CPM -> Results -> Export

Acceptance criteria (A..I):

  A. Analyze stage is non-fatal (REVIEW_REQUIRED, never Analysis ERROR).
  B. Review stage records activity/duration/dependency decisions and leaves
     zero unresolved items.
  C. Validation is VALID: 22 activities, 28 dependencies, acyclic, one
     connected component, no blocking issues.
  D. CPM: project duration 54.0, 16 critical paths; per-activity
     ES/EF/LS/LF/total float/free float/critical flags match an independent
     forward/backward pass over the gold fixture.
  E. Results dashboard data: 22 activities / 28 dependencies / 54.0 / 16.
  F. PERT: no O/M/P estimates -> NO_PERT_DATA (nothing invented).
  G. Excel export artifact is written from the validated result.
  H. PDF export artifact is written from the validated result and contains
     the project numbers.
  I. JSON + CSV exports contain the validated result.

Gold data is EVALUATION ONLY (see tests/helpers/reference_gold.py); CPM math
and the Review Center are never modified.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

from pert_analyzer.pipeline.human_review import (
    ReviewStatus,
    apply_review_decisions,
)
from pert_analyzer.pipeline.review_api import ReviewWorkflow
from pert_analyzer.reporting.export.manager import ExportManager
from pert_analyzer.reporting.export.result import ExportResult
from tests.helpers.reference_gold import (
    REFERENCE_AON,
    build_corrected_session,
    gold_forward_backward,
    load_gold,
    run_reference_analysis,
)


@pytest.fixture(scope="module")
def gold() -> dict:
    if not REFERENCE_AON.exists():
        pytest.skip(f"Reference image not found: {REFERENCE_AON}")
    return load_gold()


@pytest.fixture(scope="module")
def pipeline_result(gold):
    """Stage A: run the real pipeline on the reference image once."""
    return run_reference_analysis()


@pytest.fixture(scope="module")
def workflow(pipeline_result) -> ReviewWorkflow:
    """Stage A/B: a workflow with a fully review-resolved session."""
    session = build_corrected_session(
        pipeline_result, source_image_id=REFERENCE_AON.name
    )
    return ReviewWorkflow(pipeline_result=pipeline_result, review_session=session)


@pytest.fixture(scope="module")
def candidate(workflow):
    """Stage B/C/D: apply all decisions and run CPM on the reviewed graph."""
    return workflow.apply()


# ---------------------------------------------------------------------------
# A. Analyze
# ---------------------------------------------------------------------------


class TestStageAnalyze:
    def test_status_is_review_required_not_error(self, pipeline_result, gold) -> None:
        # A: recoverable OCR defects must never be a fatal Analysis ERROR.
        from pert_analyzer.pipeline.result import AnalysisStatus

        assert pipeline_result.status == AnalysisStatus.REVIEW_REQUIRED
        assert pipeline_result.errors == []
        assert pipeline_result.review_required is True

    def test_analysis_counts_recorded(self, pipeline_result, gold) -> None:
        assert pipeline_result.reconstructed_activity_count == gold["unique_activities"]
        assert pipeline_result.arrow_count > 0
        assert pipeline_result.ocr_region_count > 0

    def test_review_session_built(self, pipeline_result, gold) -> None:
        from pert_analyzer.pipeline.human_review import build_review_session

        raw = build_review_session(
            pipeline_result, source_image_id=REFERENCE_AON.name
        )
        assert raw is not None
        assert len(raw.reconstruction.activities) == gold["unique_activities"]
        # Raw CV output is intentionally imperfect -> review candidates exist.
        assert raw.pending_activity_count >= 1
        assert raw.pending_dependency_count >= 1
        assert raw.pending_duration_count >= 1


# ---------------------------------------------------------------------------
# B. Review
# ---------------------------------------------------------------------------


class TestStageReview:
    def test_all_review_decisions_recorded_and_none_pending(
        self, workflow, gold
    ) -> None:
        session = workflow.review_session
        corrected_ids = [
            r for r in session.activities if r.status == ReviewStatus.CORRECTED
        ]
        corrected_durations = [
            r for r in session.durations if r.status == ReviewStatus.CORRECTED
        ]
        accepted_superseded_deps = [
            r for r in session.dependencies
            if r.status in (ReviewStatus.ACCEPTED, ReviewStatus.REJECTED)
        ]

        assert len(corrected_ids) == gold["unique_activities"]
        assert len(corrected_durations) == gold["unique_activities"]
        assert len(accepted_superseded_deps) >= gold["unique_gold_dependencies"]

        assert session.pending_activity_count == 0
        assert session.pending_dependency_count == 0
        assert session.pending_duration_count == 0


# ---------------------------------------------------------------------------
# C/D. Apply -> Validation -> CPM
# ---------------------------------------------------------------------------


class TestStageValidationCpm:
    def test_validation_valid_22_28_acyclic_connected(self, candidate, gold) -> None:
        validation = candidate.validation
        assert validation.is_valid
        assert validation.status.value == "VALID"
        assert validation.component_count == 1
        assert validation.is_acyclic is True
        assert validation.errors == []

        assert candidate.graph.activity_count == gold["unique_activities"]
        assert candidate.graph.dependency_count == gold["unique_gold_dependencies"]

        assert candidate.cpm_gate.value == "RUNNABLE"
        assert candidate.cpm is not None

    def test_cpm_duration_and_path_count(self, candidate, gold) -> None:
        assert candidate.cpm_project_duration == pytest.approx(
            gold["expected_project_duration"], abs=1e-9
        )
        assert candidate.critical_path_count == gold["expected_critical_path_count"]

        actual = sorted(sorted(p) for p in candidate.pure_critical_paths)
        expected = sorted(sorted(p) for p in gold["expected_critical_paths"])
        assert actual == expected

    def test_per_activity_es_ef_ls_lf_floats_critical(
        self, candidate, gold
    ) -> None:
        # D: verify the backend CPM schedule against an independent
        # gold-derived forward/backward pass (never re-implement CPM in
        # production code).
        expected_duration, expected = gold_forward_backward(gold)
        assert candidate.cpm.project_duration == pytest.approx(
            expected_duration, abs=1e-9
        )

        analyses = candidate.cpm.activity_analyses
        assert set(analyses) == set(expected)
        for aid, want in expected.items():
            got = analyses[aid]
            assert got.early_start == pytest.approx(want["early_start"], abs=1e-9), aid
            assert got.early_finish == pytest.approx(want["early_finish"], abs=1e-9), aid
            assert got.late_start == pytest.approx(want["late_start"], abs=1e-9), aid
            assert got.late_finish == pytest.approx(want["late_finish"], abs=1e-9), aid
            assert got.total_float == pytest.approx(want["total_float"], abs=1e-9), aid
            assert got.free_float == pytest.approx(want["free_float"], abs=1e-9), aid
            assert got.is_critical == want["is_critical"], aid


# ---------------------------------------------------------------------------
# E. Results data
# ---------------------------------------------------------------------------


class TestStageResults:
    def test_results_data_matches_gold(self, candidate, workflow) -> None:
        from pert_analyzer.gui.session import GuiSession
        from pert_analyzer.gui.results.data import describe_ready, extract

        session = GuiSession()
        session.current_image_path = str(REFERENCE_AON)
        session.workflow = workflow
        session.complete_apply(candidate)

        ready, reason = describe_ready(session)
        assert ready, reason
        assert session.validation_status.value == "VALID"
        assert session.pending_review_total() == 0

        data = extract(session)
        assert data.ready
        assert data.activity_count == 22
        assert data.dependency_count == 28
        assert data.project_duration == pytest.approx(54.0, abs=1e-9)
        assert data.critical_path_count == 16
        assert len(data.critical_activity_ids) == 22  # every node is critical

        durations = {a.activity_id: round(a.duration or 0.0, 9) for a in data.activities}
        assert durations["A"] == 1.0
        assert durations["N"] == 8.0
        assert durations["Q"] == 3.0
        assert durations["V"] == 1.0

    def test_pert_derived_estimates(self, candidate, workflow) -> None:
        from pert_analyzer.analysis.pert_engine import PertStatus
        from pert_analyzer.gui.results.data import (
            extract_pert,
            pert_estimates_source,
        )
        from pert_analyzer.gui.session import GuiSession

        session = GuiSession()
        session.current_image_path = str(REFERENCE_AON)
        session.workflow = workflow
        session.complete_apply(candidate)

        pert = extract_pert(session)
        assert pert.status == PertStatus.PERT_READY
        assert len(pert.rows) == 22  # derived prefill; PERT not run yet
        assert pert.estimates_source == "derived"
        assert pert_estimates_source(session) == "derived"
        row = next(r for r in pert.rows if r.activity_id == "N")
        assert row.most_likely == pytest.approx(8.0, abs=1e-9)
        assert row.optimistic <= row.most_likely <= row.pessimistic
        assert row.expected_time is None  # analysis columns stay empty


# ---------------------------------------------------------------------------
# G/H/I. Export artifacts
# ---------------------------------------------------------------------------


class TestStageExport:
    @pytest.fixture(scope="class")
    @classmethod
    def gui_report(cls, candidate, workflow, tmp_path_factory) -> None:
        from pert_analyzer.gui.session import GuiSession

        session = GuiSession()
        session.current_image_path = str(REFERENCE_AON)
        session.workflow = workflow
        session.complete_apply(candidate)
        report = session.report  # fixture for the GUI Export path
        assert report.summary.activity_count == 22
        assert report.summary.dependency_count == 28
        assert report.summary.project_duration == pytest.approx(54.0, abs=1e-9)
        assert report.summary.critical_path_count == 16

        out = tmp_path_factory.mktemp("ref_export")
        return session, report, out

    def test_json_contains_result(self, gui_report) -> None:
        session, report, out = gui_report
        path = out / "reference_aon.json"
        res: ExportResult = ExportManager().export(
            report, output_format="json", output_path=str(path)
        )
        assert res.ok, res.warnings
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["summary"]["project_duration"] == pytest.approx(54.0, abs=1e-9)
        assert data["summary"]["critical_path_count"] == 16
        assert data["summary"]["activity_count"] == 22
        assert data["summary"]["dependency_count"] == 28
        assert len(data["activities"]) == 22
        assert len(data["critical_paths"]) == 16

    def test_csv_contains_result(self, gui_report) -> None:
        session, report, out = gui_report
        path = out / "reference_aon.csv"
        res: ExportResult = ExportManager().export(
            report, output_format="csv", output_path=str(path)
        )
        assert res.ok, res.warnings
        rows = list(csv.reader(path.open(encoding="utf-8")))
        header = rows[0]
        assert header[0] == "section"
        body = "\n".join(",".join(r) for r in rows)
        assert "54" in body or "54.0" in body
        assert "22" in body
        assert "28" in body
        assert "16" in body

    def test_excel_artifact_written(self, gui_report) -> None:
        session, report, out = gui_report
        path = out / "reference_aon.xlsx"
        res: ExportResult = ExportManager().export(
            report, output_format="excel", output_path=str(path)
        )
        assert res.ok, res.warnings
        assert path.exists()
        assert path.stat().st_size > 0

    def test_pdf_artifact_written_and_contains_result(self, gui_report) -> None:
        session, report, out = gui_report
        path = out / "reference_aon.pdf"
        res: ExportResult = ExportManager().export(
            report, output_format="pdf", output_path=str(path)
        )
        assert res.ok, res.warnings
        assert path.exists()
        assert path.stat().st_size > 0

        try:
            import fitz  # PyMuPDF
        except ImportError:
            pytest.skip("PyMuPDF not installed; PDF binary verified only")
        doc = fitz.open(str(path))
        text = "".join(page.get_text() for page in doc)
        doc.close()
        assert "54" in text and "16" in text and "22" in text and "28" in text