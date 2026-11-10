"""
Final End-to-End Acceptance Test — GUI workflow over the full reference AON.

Seeds a real GuiSession with the REAL pipeline workflow (not fakes) for the
reference diagram, drives the state progression, and verifies the Results
dashboard presents the authoritative numbers:

    22 activities / 28 dependencies / 54.0 days / 16 critical paths.

The user must never see an Analysis ERROR for this recoverable image; the
PERT tab must show the auto-derived three-point estimates (A-14).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from pert_analyzer.gui.pages.results_page import ResultsPage
from pert_analyzer.gui.results.data import PERT_DERIVED_HINT
from pert_analyzer.gui.session import AppState, GuiSession, ValidationCenterStatus
from pert_analyzer.pipeline.review_api import ReviewWorkflow
from tests.helpers.reference_gold import (
    REFERENCE_AON,
    build_corrected_session,
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
    return run_reference_analysis()


@pytest.fixture(scope="module")
def workflow(pipeline_result) -> ReviewWorkflow:
    session = build_corrected_session(
        pipeline_result, source_image_id=REFERENCE_AON.name
    )
    return ReviewWorkflow(pipeline_result=pipeline_result, review_session=session)


@pytest.fixture(scope="module")
def candidate(workflow):
    return workflow.apply()


@pytest.fixture()
def page(qapp: QApplication) -> ResultsPage:
    return ResultsPage()


def test_never_error_raw_session_progression(
    qapp, page: ResultsPage, pipeline_result
) -> None:
    """Analyze -> Review Required (recoverable OCR is never a fatal ERROR)."""
    session = GuiSession()
    assert session.state == AppState.NO_PROJECT
    assert session.set_image(str(REFERENCE_AON)) is True
    assert session.state == AppState.IMAGE_SELECTED

    raw_workflow = ReviewWorkflow.from_pipeline_result(
        pipeline_result, source_image_id=REFERENCE_AON.name
    )
    session.complete_analysis(raw_workflow)
    assert session.state != AppState.ERROR
    assert session.state == AppState.REVIEW_REQUIRED
    assert session.error_message == ""
    assert session.review_session is not None
    assert session.pending_review_total() > 0

    # Results must be gated until the review is finished.
    page.refresh(session)
    assert page.stack_index() == page._EMPTY
    assert "Review required" in page._empty_label.text()


def test_results_become_available_after_review(
    qapp, page: ResultsPage, workflow, candidate
) -> None:
    """Review -> Apply -> VALID -> Results available; never ERROR."""
    session = GuiSession()
    session.current_image_path = str(REFERENCE_AON)
    session.workflow = workflow
    session.complete_apply(candidate)

    assert session.state != AppState.ERROR
    assert session.state == AppState.VALIDATION_REQUIRED
    assert session.pending_review_total() == 0
    assert session.validation_status == ValidationCenterStatus.VALID
    assert session.cpm_eligible is True
    assert session.cpm_result is not None

    page.refresh(session)
    assert page.stack_index() == page._DASHBOARD
    assert "Project duration: 54 days" in page._summary_label.text()
    assert "Critical paths: 16" in page._summary_label.text()
    assert page._export_btn.isEnabled()
    assert page._report_btn.isEnabled()


def test_dashboard_kpis_real_reference(page: ResultsPage, workflow, candidate) -> None:
    session = GuiSession()
    session.current_image_path = str(REFERENCE_AON)
    session.workflow = workflow
    session.complete_apply(candidate)
    page.refresh(session)

    cards = page.overview()._kpi_cards
    assert cards["activities"].value() == "22"
    assert cards["dependencies"].value() == "28"
    assert cards["duration"].value() == "54 days"
    assert cards["critical_paths"].value() == "16"
    assert cards["critical_activities"].value() == "22"


def test_dashboard_views_contain_real_data(
    page: ResultsPage, workflow, candidate
) -> None:
    session = GuiSession()
    session.current_image_path = str(REFERENCE_AON)
    session.workflow = workflow
    session.complete_apply(candidate)
    page.refresh(session)

    assert page.activities().table().rowCount() == 22
    assert page.activities().table().horizontalHeaderItem(9) is not None  # header intact
    assert page.activities().table().item(0, 0) is not None

    assert len(page.network().node_items()) == 22
    assert len(page.network().edge_items()) == 28

    path_list = page.critical_paths().path_list()
    assert len(path_list._paths) == 16
    assert path_list._count_label.text() == "Critical paths: 16"


def test_pert_tab_shows_derived_estimates(page: ResultsPage, workflow, candidate) -> None:
    session = GuiSession()
    session.current_image_path = str(REFERENCE_AON)
    session.workflow = workflow
    session.complete_apply(candidate)
    page.refresh(session)

    tab = page.pert()
    status_text = tab.status_label().text()
    assert "ready" in status_text
    assert PERT_DERIVED_HINT in status_text
    assert tab.run_button().isEnabled() is True
    assert tab.results_table().rowCount() == 0
    # Derived three-point estimates show up in the editable table.
    table = tab.estimates_table()
    assert table.rowCount() == 22
    row = next(
        r for r in range(table.rowCount()) if table.item(r, 0).text() == "N"
    )
    assert float(table.item(row, 2).text()) == pytest.approx(8.0, abs=1e-9)