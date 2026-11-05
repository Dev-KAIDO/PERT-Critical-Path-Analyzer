"""
Phase 7.3 tests: Expert UI/UX redesign behaviors.

Covers the new Diagram Understanding page, click-to-upload activation,
the ANALYZING DIAGRAM progress panel summary rows, the adaptive Review
splitter, the Validation stat grid with expandable technical details,
and the embedded Results analytics surface with cross-highlighting.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from pert_analyzer.gui.analysis_progress import (
    AnalysisProgressModel,
    StageProgressPanel,
)
from pert_analyzer.gui.main_window import MainWindow
from pert_analyzer.gui.navigation import NavDestination
from pert_analyzer.gui.pages.analysis_page import AnalysisPage, ImageView
from pert_analyzer.gui.pages.results_page import ResultsPage
from pert_analyzer.gui.pages.review_page import ReviewPage
from pert_analyzer.gui.pages.validation_page import ValidationPage
from pert_analyzer.gui.session import GuiSession
from pert_analyzer.pipeline.progress import COMPLETED, RUNNING, StageProgress
from tests.gui.fakes import (
    FakeCpmResult,
    FakeGraphActivity,
    FakeGraphDependency,
    FakeReviewSession,
    FakeReviewedCandidate,
    FakeValidationResult,
    FakeValidatedWorkflow,
    fake_issue,
    make_analysis,
    make_fake_graph,
    make_review_session,
    make_reconstruction,
    reviewed_session,
)


@pytest.fixture()
def window(qapp: QApplication) -> MainWindow:
    w = MainWindow(backend=lambda path: None)
    yield w
    w.close()


@pytest.fixture()
def page(qapp: QApplication) -> ValidationPage:
    return ValidationPage()


def _sp(stage_id: str, state: str, progress: float, **kw) -> StageProgress:
    return StageProgress(stage_id=stage_id, state=state, progress=progress, **kw)


def make_session(candidate, review_session=None) -> GuiSession:
    session = GuiSession()
    session.workflow = FakeValidatedWorkflow(
        candidate=candidate,
        review_session=review_session or FakeReviewSession(),
    )
    session.complete_apply(candidate)
    return session


def two_activity_candidate() -> FakeReviewedCandidate:
    graph = make_fake_graph(
        activities={
            "A0": FakeGraphActivity("A0", 7.0, "Start"),
            "A1": FakeGraphActivity("A1", 7.0, "Finish"),
        },
        dependencies=[FakeGraphDependency("E0", "A0", "A1")],
    )
    cpm = FakeCpmResult(
        project_duration=14.0,
        critical_paths=[["A0", "A1"]],
        analyses={
            "A0": make_analysis("A0", 0.0, 7.0, 0.0, 7.0, 0.0, 0.0, True),
            "A1": make_analysis("A1", 7.0, 14.0, 7.0, 14.0, 0.0, 0.0, True),
        },
    )
    return FakeReviewedCandidate(valid=True, graph=graph, cpm=cpm)


# ---------------------------------------------------------------------------
# Navigation + workflow indicator (five stages)
# ---------------------------------------------------------------------------


def test_nav_has_understanding_between_analyze_and_review(window: MainWindow) -> None:
    destinations = [d for d in NavDestination]
    assert destinations == [
        NavDestination.ANALYZE,
        NavDestination.UNDERSTANDING,
        NavDestination.REVIEW,
        NavDestination.VALIDATE,
        NavDestination.RESULTS,
        NavDestination.NETWORK_BUILDER,
    ]
    understand = window._nav_buttons[NavDestination.UNDERSTANDING.value]
    assert understand.objectName() == "nav_understanding"
    assert "Understand" in understand.text()


def test_workflow_indicator_has_four_named_steps(window: MainWindow) -> None:
    window._update_workflow_indicator()
    text = window._workflow_indicator.text()
    # A-13: validation retired from the step flow
    for step in ("Analyze", "Understand", "Review", "Results"):
        assert step in text
    assert "Validate" not in text
    assert window._stack.count() == 6


# ---------------------------------------------------------------------------
# Diagram Understanding page
# ---------------------------------------------------------------------------


def test_understanding_page_title_banner_and_preview_canvas(window: MainWindow) -> None:
    page = window._understanding_page
    assert page.title == "Diagram Understanding"
    assert "PRELIMINARY RECONSTRUCTION" in page.banner_text
    assert page.canvas() is not None


def test_understanding_empty_state_without_analysis(window: MainWindow) -> None:
    page = window._understanding_page
    page.refresh(GuiSession())
    assert "No analysis yet" in page._empty_label.text()
    assert page._content.isHidden()


def test_understanding_no_reconstruction_data_state(window: MainWindow) -> None:
    page = window._understanding_page
    session = GuiSession()
    session.workflow = type("W", (), {"review_session": None})()
    page.refresh(session)
    assert "No reconstruction data" in page._empty_label.text()
    assert page._content.isHidden()


def test_understanding_metrics_and_canvas_from_review_session(window: MainWindow) -> None:
    page = window._understanding_page
    session = reviewed_session(
        n_activities=2,
        n_dependencies=1,
        n_durations=1,
        reconstruction=make_reconstruction([("N0", "A0"), ("N1", "A1")]),
    )
    page.refresh(session)
    values = {
        key: label.text()
        for key, (label, _caption) in page._metric_cells.items()
    }
    assert values["activities"] == "2"
    assert values["candidate_deps"] == "1"
    assert values["review_items"] == "4"
    assert values["pending"] == "4"
    assert page.canvas().node_count == 2
    assert page.canvas().pending_count == 2


def test_understanding_preliminary_canvas_counts(window: MainWindow) -> None:
    page = window._understanding_page
    session = make_review_session(
        n_activities=3,
        n_dependencies=2,
        n_durations=2,
        reconstruction=make_reconstruction(
            [("N0", "A0"), ("N1", "A1"), ("N2", "A2")]
        ),
    )
    from tests.gui.fakes import FakeReviewedWorkflow

    gui_session = GuiSession()
    gui_session.complete_analysis(FakeReviewedWorkflow(session, review_required=True))
    page.refresh(gui_session)
    assert page.canvas().node_count == 3
    assert page.canvas().pending_count == 3


# ---------------------------------------------------------------------------
# Click-to-upload on the Analyze page
# ---------------------------------------------------------------------------


def test_image_view_empty_click_emits_activate_requested(qapp: QApplication) -> None:
    view = ImageView()
    emitted: list[int] = []
    view.activate_requested.connect(lambda: emitted.append(1))
    QTest.mouseClick(view, Qt.MouseButton.LeftButton)
    assert emitted == [1]
    QTest.mouseClick(view, Qt.MouseButton.RightButton)
    assert emitted == [1]
    view.deleteLater()


def test_analysis_page_activation_opens_upload(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = AnalysisPage()
    calls: list[int] = []
    monkeypatch.setattr(page, "_on_upload", lambda: calls.append(1))
    QTest.mouseClick(page._image_view, Qt.MouseButton.LeftButton)
    assert calls == [1]
    page.deleteLater()


def test_image_view_click_with_image_does_not_activate(
    qapp: QApplication, tmp_path
) -> None:
    view = ImageView()
    emitted: list[int] = []
    view.activate_requested.connect(lambda: emitted.append(1))
    image_path = tmp_path / "dot.png"
    view.set_image(str(image_path))
    assert view._pixmap is None
    QTest.mouseClick(view, Qt.MouseButton.LeftButton)
    assert emitted == [1]
    view.deleteLater()


# ---------------------------------------------------------------------------
# ANALYZING DIAGRAM progress panel summary rows
# ---------------------------------------------------------------------------


def test_progress_panel_analyzing_diagram_summary(qapp: QApplication) -> None:
    model = AnalysisProgressModel()
    panel = StageProgressPanel()
    panel.set_model(model)
    assert panel._title.text() == "ANALYZING DIAGRAM"
    assert panel._percent_label.text() == "0%"

    model.handle_stage(_sp("Detecting shapes", RUNNING, 0.2, metrics={"shapes": 9}))
    assert "Current stage:" in panel._stage_row.text()

    model.handle_stage(
        _sp("Detecting shapes", COMPLETED, 0.2, metrics={"shapes": 9})
    )
    model.finish(ok=True)
    assert panel._percent_label.text() == "100%"
    assert "Complete" in panel._stage_row.text()
    assert "9 shapes detected" in panel._detected_row.text()
    assert "9 shapes detected" in panel._metrics.text()
    panel.deleteLater()


# ---------------------------------------------------------------------------
# Adaptive Review splitter
# ---------------------------------------------------------------------------


def test_review_splitter_rebalances_on_large_resize(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = ReviewPage()
    applied: list[list[int]] = []
    monkeypatch.setattr(
        page._splitter.__class__, "sizes", lambda self: [200, 300, 500]
    )
    monkeypatch.setattr(
        page._splitter.__class__,
        "setSizes",
        lambda self, sizes: applied.append(list(sizes)),
    )
    page.show()
    page.resize(1500, 900)
    QApplication.processEvents()
    assert applied, "rebalance should run on a large resize"
    assert applied[-1] == [200, 300, 500]
    page.deleteLater()


def test_review_splitter_skips_rebalance_after_user_drag(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = ReviewPage()
    applied: list[list[int]] = []
    monkeypatch.setattr(
        page._splitter.__class__, "sizes", lambda self: [200, 300, 500]
    )
    monkeypatch.setattr(
        page._splitter.__class__,
        "setSizes",
        lambda self, sizes: applied.append(list(sizes)),
    )
    page._splitter.splitterMoved.emit(10, 0)
    assert page._user_resized is True
    page.show()
    page.resize(1500, 900)
    QApplication.processEvents()
    assert applied == []
    page.deleteLater()


# ---------------------------------------------------------------------------
# Validation stat grid + technical details
# ---------------------------------------------------------------------------


def test_validation_stat_grid_six_cells(page: ValidationPage) -> None:
    assert list(page._stat_cells.keys()) == [
        "activities",
        "dependencies",
        "nodes",
        "components",
        "cycles",
        "pending_reviews",
    ]


def test_validation_stat_grid_valid_values(page: ValidationPage) -> None:
    page.refresh(make_session(FakeReviewedCandidate(valid=True)))
    assert page._stat_cells["cycles"].text() == "No"
    assert page._stat_cells["pending_reviews"].text() == "0"
    assert page._stat_cells["activities"].text() == "n/a"


def test_validation_stat_grid_invalid_cycle(page: ValidationPage) -> None:
    candidate = FakeReviewedCandidate(
        valid=False,
        validation=FakeValidationResult(
            is_valid=False,
            errors=[fake_issue("cycle")],
            is_acyclic=False,
        ),
    )
    page.refresh(make_session(candidate))
    assert page._stat_cells["cycles"].text() == "Yes"


def test_validation_technical_details_toggle(page: ValidationPage) -> None:
    page.refresh(make_session(FakeReviewedCandidate(valid=True)))
    assert page._technical.isHidden()
    assert page._tech_btn.text() == "Technical details"
    page._tech_btn.setChecked(True)
    assert not page._technical.isHidden()
    assert "Graph status" in page._tech_rows["graph_status"].text()
    assert "Blocking issues" in page._tech_rows["blocking_issues"].text()


# ---------------------------------------------------------------------------
# Embedded Results analytics surface
# ---------------------------------------------------------------------------


def _ready_page() -> tuple[ResultsPage, GuiSession]:
    page = ResultsPage()
    session = make_session(two_activity_candidate())
    page.refresh(session)
    return page, session


def test_results_overview_is_summary_only(qapp: QApplication) -> None:
    page, _session = _ready_page()
    overview = page.overview()
    assert overview._cards_row.count() == 6
    assert not hasattr(overview, "_embedded_network")
    assert not hasattr(overview, "_embedded_activities")
    assert "reviewed activities" in overview._composition.text()
    page.deleteLater()


def test_results_overview_has_health_and_confidence_summary(
    qapp: QApplication,
) -> None:
    page, _session = _ready_page()
    overview = page.overview()
    assert overview._health_status.text() == "Authoritative"
    assert "Backend CPM" in overview._confidence.text()
    assert "critical route" in overview._next_action.text()
    page.deleteLater()


def test_results_overview_actions_open_detail_tabs(
    qapp: QApplication,
) -> None:
    page, _session = _ready_page()
    overview = page.overview()
    overview._open_network_btn.click()
    assert page.tabs().currentWidget() is page.network()
    overview._paths_button.click()
    assert page.tabs().currentWidget() is page.critical_paths()
    page.deleteLater()


def test_results_overview_activity_action_switches_to_activities_tab(
    qapp: QApplication,
) -> None:
    page, _session = _ready_page()
    overview = page.overview()
    overview.open_activities_requested.emit()
    assert page.tabs().currentWidget() is page.activities()
    page.deleteLater()
