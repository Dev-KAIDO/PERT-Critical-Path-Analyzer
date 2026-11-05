"""
Results Dashboard (Phase 4) page tests.

Covers the readiness gate, not-ready reasons, dashboard population
(KPIs, tables, paths, network), cross-navigation, the Calculate-Results
worker flow, and the reference fixture (22 activities / 28 dependencies /
54.0 days / 16 critical paths).
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel

from pert_analyzer.gui.main_window import MainWindow
from pert_analyzer.gui.navigation import NavDestination
from pert_analyzer.gui.pages.results_page import ResultsPage
from pert_analyzer.gui.session import GuiSession, ValidationCenterStatus
from tests.gui.fakes import (
    FakeCpmResult,
    FakeGraphActivity,
    FakeGraphDependency,
    FakeReviewSession,
    FakeReviewedCandidate,
    FakeValidatedWorkflow,
    FakeWorkflow,
    make_analysis,
    make_fake_graph,
)


@pytest.fixture()
def page(qapp: QApplication) -> ResultsPage:
    return ResultsPage()


@pytest.fixture()
def window(qapp: QApplication) -> MainWindow:
    w = MainWindow(backend=lambda path: None)
    yield w
    w.close()


def wait_for(app: QApplication, condition, timeout: float = 10.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        for _ in range(20):
            app.processEvents()
        if condition():
            return True
        time.sleep(0.02)
    return condition()


def make_session(candidate) -> GuiSession:
    """A GuiSession that already applied a candidate through a workflow."""
    session = GuiSession()
    session.workflow = FakeValidatedWorkflow(
        candidate=candidate, review_session=FakeReviewSession()
    )
    session.complete_apply(candidate)
    return session


def two_activity_candidate() -> FakeReviewedCandidate:
    """A valid, CPM-ready candidate with a small A0->A1 network."""
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


def ready_page(qapp: QApplication) -> tuple[ResultsPage, GuiSession]:
    page = ResultsPage()
    session = make_session(two_activity_candidate())
    page.refresh(session)
    return page, session


# ---------------------------------------------------------------------------
# Readiness gate
# ---------------------------------------------------------------------------


def test_not_ready_no_project_analyzed(page: ResultsPage) -> None:
    page.refresh(GuiSession())
    assert page.stack_index() == page._EMPTY
    assert "No project analyzed" in page._empty_label.text()


def test_not_ready_review_required_without_candidate(page: ResultsPage) -> None:
    session = GuiSession()
    session.workflow = FakeWorkflow()
    page.refresh(session)
    assert "Results not ready" in page._empty_label.text()
    assert "Review required" in page._empty_label.text()
    assert page._empty_status_label.text() == "REVIEW REQUIRED"


def test_not_ready_graph_invalid(page: ResultsPage) -> None:
    candidate = FakeReviewedCandidate(valid=False)
    page.refresh(make_session(candidate))
    assert page.stack_index() == page._EMPTY
    assert "invalid" in page._empty_label.text()
    assert page._empty_status_label.text() == "GRAPH INVALID"


def test_not_ready_cpm_blocked(page: ResultsPage) -> None:
    candidate = FakeReviewedCandidate(valid=True, cpm_gate="BLOCKED_REVIEW", cpm=None)
    page.refresh(make_session(candidate))
    assert "CPM is blocked" in page._empty_label.text()


def test_not_ready_result_unavailable_shows_calculate(page: ResultsPage) -> None:
    candidate = FakeReviewedCandidate(
        valid=True,
        cpm=None,
        cpm_project_duration=None,
        critical_path_count=None,
        pure_critical_paths=[],
    )
    page.show()
    page.refresh(make_session(candidate))
    assert page.stack_index() == page._EMPTY
    assert "no CPM result" in page._empty_label.text()
    assert page._empty_status_label.text() == "READY TO CALCULATE"
    assert page._calculate_btn.isVisible()
    assert page._calculate_btn.isEnabled()
    page.hide()


def test_review_priority_summary_uses_pending_calibration_evidence() -> None:
    pending = SimpleNamespace(value="PENDING")

    def evidence(tier: str) -> SimpleNamespace:
        return SimpleNamespace(metadata={"review_calibration": {"tier": tier}})

    review_session = SimpleNamespace(
        activities=[
            SimpleNamespace(
                status=pending,
                evidence=[evidence("BLOCKING_REVIEW")],
            )
        ],
        dependencies=[
            SimpleNamespace(
                status=pending,
                evidence=[evidence("HIGH_PRIORITY_REVIEW")],
            )
        ],
        durations=[],
        metadata={},
    )
    session = SimpleNamespace(review_session=review_session)

    text = ResultsPage._review_priority_text(session)

    assert text == "Review priority — Blocking: 1 · High priority: 1"


def test_provenance_and_review_trace_are_explicit() -> None:
    pending = SimpleNamespace(value="PENDING")
    corrected = SimpleNamespace(value="CORRECTED")
    review_session = SimpleNamespace(
        source_image_id="7.jpeg",
        activities=[SimpleNamespace(status=pending, evidence=[])],
        dependencies=[SimpleNamespace(status=corrected, evidence=[])],
        durations=[],
        metadata={},
    )
    session = SimpleNamespace(
        review_session=review_session,
        current_image_path="/tmp/7.jpeg",
        review_summary={"diagram_type": "AON"},
        current_candidate=None,
        review_item_total=lambda: 2,
        pending_review_total=lambda: 1,
    )

    assert ResultsPage._provenance_text(session) == (
        "Source image: 7.jpeg  ·  Diagram type: AON"
    )
    assert ResultsPage._review_trace_text(session) == (
        "Review trace — 2 total · 1 pending · 1 resolved · 1 corrected"
    )
    assert ResultsPage._review_breakdown_text(session) == (
        "Review breakdown — Activities: 1 total · 1 pending · 0 accepted · 0 corrected"
        "  |  Dependencies: 1 total · 0 pending · 0 accepted · 1 corrected"
        "  |  Durations: 0 total · 0 pending · 0 accepted · 0 corrected"
    )


def test_ready_switches_to_dashboard(page: ResultsPage) -> None:
    page.show()
    session = make_session(two_activity_candidate())
    session.entry_mode = "NETWORK_BUILDER"
    page.refresh(session)
    assert page.stack_index() == page._DASHBOARD
    assert page._empty_label.isHidden()
    assert "Project duration: 14 days" in page._summary_label.text()
    assert page._context_mode_label.text() == "ENTRY: NETWORK BUILDER"
    assert page._context_trust_label.text() == "TRUST: AUTHORITATIVE"
    page.hide()


def test_image_review_and_network_builder_use_one_context_shell(page: ResultsPage) -> None:
    image_session = GuiSession()
    image_session.entry_mode = "IMAGE_ANALYSIS"
    image_session.current_image_path = "/tmp/diagram-7.jpeg"
    image_session.workflow = FakeWorkflow()
    page.refresh(image_session)
    assert page.stack_index() == page._EMPTY
    assert page._context_mode_label.text() == "ENTRY: IMAGE ANALYSIS"
    assert page._context_source_label.text() == "Source: diagram-7.jpeg"
    assert page._context_trust_label.text() == "TRUST: PRELIMINARY"

    builder_session = make_session(two_activity_candidate())
    builder_session.entry_mode = "NETWORK_BUILDER"
    page.refresh(builder_session)
    assert page.stack_index() == page._DASHBOARD
    assert page._context_mode_label.text() == "ENTRY: NETWORK BUILDER"
    assert page._context_source_label.text() == "Source: Manual Network Builder"
    assert page._context_trust_label.text() == "TRUST: AUTHORITATIVE"


def test_image_mode_renders_safe_detected_preview_without_cpm() -> None:
    activity_a = SimpleNamespace(
        geometric_node_id="geo-a", activity_id="A", duration=3.0,
        confidence=0.9, position=(120.0, 100.0),
    )
    activity_b = SimpleNamespace(
        geometric_node_id="geo-b", activity_id="B", duration=4.0,
        confidence=0.35, position=(340.0, 100.0),
    )
    dependency = SimpleNamespace(
        source_id="geo-a", target_id="geo-b", source_arrow_id="arrow-1",
        confidence=0.3,
    )
    reconstruction = SimpleNamespace(
        diagram_type="AON", activities=[activity_a, activity_b],
        dependencies=[dependency], events=[],
    )
    review_session = SimpleNamespace(
        source_image_id="diagram.png", reconstruction=reconstruction,
        activities=[], dependencies=[], durations=[], metadata={},
    )
    workflow = SimpleNamespace(
        review_session=review_session,
        _candidate=None,
        pipeline_result=SimpleNamespace(review_required=True),
        summary=lambda: {"total_activities": 2, "total_dependencies": 1},
    )
    session = GuiSession(
        entry_mode="IMAGE_ANALYSIS",
        current_image_path="/tmp/diagram.png",
        workflow=workflow,
    )
    page = ResultsPage()
    page.refresh(session)

    assert page.stack_index() == page._EMPTY
    assert not page._detected_preview.isHidden()
    assert set(page._detected_preview.node_items()) == {"geo-a", "geo-b"}
    assert page.data is None
    assert page._context_trust_label.text() == "TRUST: PRELIMINARY"


def test_dashboard_tabs_labels(page: ResultsPage) -> None:
    page.refresh(make_session(two_activity_candidate()))
    labels = [page.tabs().tabText(i) for i in range(page.tabs().count())]
    assert labels == ["Overview", "Network", "Activities", "Critical Paths", "PERT"]


def test_export_report_enabled(page: ResultsPage) -> None:
    page.refresh(make_session(two_activity_candidate()))
    assert page._export_btn.isEnabled()
    assert page._report_btn.isEnabled()
    emitted: list[str] = []
    page.export_requested.connect(lambda: emitted.append("export"))
    page.report_requested.connect(lambda: emitted.append("report"))
    page._export_btn.click()
    page._report_btn.click()
    assert emitted == ["export", "report"]

def test_calculate_requested_signal(page: ResultsPage) -> None:
    page.refresh(make_session(two_activity_candidate()))
    emitted: list[bool] = []
    page.calculate_requested.connect(lambda: emitted.append(True))
    page.refresh(make_session(FakeReviewedCandidate(valid=True, cpm=None)))
    page._calculate_btn.click()
    assert emitted == [True]


def test_not_ready_buttons_navigate(window: MainWindow) -> None:
    session = make_session(FakeReviewedCandidate(valid=True, cpm=None))
    window._session = session
    window._on_nav(NavDestination.RESULTS)
    window._results_page._go_review_btn.click()
    assert window._stack.currentIndex() == NavDestination.REVIEW.value
    # A-13: the results "Open Validation" action is retired
    assert not hasattr(window._results_page, "_go_validation_btn")


# ---------------------------------------------------------------------------
# Dashboard content
# ---------------------------------------------------------------------------


def test_kpi_cards_populated(page: ResultsPage) -> None:
    page.refresh(make_session(two_activity_candidate()))
    cards = page.overview()._kpi_cards
    assert cards["activities"].value() == "2"
    assert cards["dependencies"].value() == "1"
    assert cards["duration"].value() == "14 days"
    assert cards["critical_activities"].value() == "2"
    assert cards["critical_paths"].value() == "1"


def test_overview_snapshot_and_results_scroll_contract(qapp: QApplication) -> None:
    del qapp
    page = ResultsPage()
    page.refresh(make_session(reference_fixture_candidate()))
    overview = page.overview()

    assert overview._path_rows.count() == 3
    assert overview._more_paths.text().startswith("+ 13 more")
    assert overview._confidence_bar.value() == 3
    first_path_row = overview._path_rows.itemAt(0).widget()
    first_path_text = " ".join(label.text() for label in first_path_row.findChildren(QLabel))
    assert "A0" in first_path_text and "A21" in first_path_text
    assert "…" not in first_path_text
    assert page._stack.widget(page._DASHBOARD).__class__.__name__ == "QScrollArea"


def test_activities_table_rows(page: ResultsPage) -> None:
    page.refresh(make_session(two_activity_candidate()))
    table = page.activities().table()
    assert table.rowCount() == 2
    assert table.columnCount() == 11
    headers = [table.horizontalHeaderItem(i).text() for i in range(11)]
    assert headers[:9] == [
        "ID", "Duration", "ES", "EF", "LS", "LF",
        "Total Float", "Free Float", "Critical",
    ]


def test_activity_detail_panel(page: ResultsPage) -> None:
    page.refresh(make_session(two_activity_candidate()))
    page.activities().select_activity("A1")
    fields = page.activities().detail_fields()
    assert fields["id"].text() == "A1"
    assert fields["es"].text() == "7"
    assert fields["ef"].text() == "14"
    assert fields["critical"].text() == "Yes"


def test_critical_paths_list(page: ResultsPage) -> None:
    page.refresh(make_session(two_activity_candidate()))
    path_list = page.critical_paths().path_list()
    assert path_list._count_label.text() == "Critical paths: 1"
    assert "A0 \u2192 A1" in path_list._list.item(0).text()


# ---------------------------------------------------------------------------
# Cross-navigation
# ---------------------------------------------------------------------------


def test_activity_selection_highlights_network_node(page: ResultsPage) -> None:
    page.refresh(make_session(two_activity_candidate()))
    page.activities().select_activity("A1")
    assert page.network().selected_activity == "A1"
    assert page.network().node_items()["A1"]._selected


def test_network_node_click_switches_to_activities(page: ResultsPage) -> None:
    page.refresh(make_session(two_activity_candidate()))
    page.network().node_items()["A0"].clicked.emit("A0")
    assert page.tabs().currentWidget() is page.activities()
    assert page.activities().selected_activity_id() == "A0"


def test_path_selection_highlights_network(page: ResultsPage) -> None:
    page.refresh(make_session(two_activity_candidate()))
    path_list = page.critical_paths().path_list()
    path_list._list.setCurrentRow(-1)
    path_list.select_path(0)
    assert page.network().current_path() == ["A0", "A1"]
    assert page.network().node_items()["A0"]._path_highlight
    assert page.network().node_items()["A1"]._path_highlight


# ---------------------------------------------------------------------------
# Calculate Results (worker flow)
# ---------------------------------------------------------------------------


def test_calculate_results_worker_flow(window: MainWindow) -> None:
    candidate = FakeReviewedCandidate(
        valid=True,
        cpm=None,
        cpm_project_duration=None,
        critical_path_count=None,
        pure_critical_paths=[],
    )
    window._session = make_session(candidate)

    def run_cpm():
        return FakeCpmResult(
            project_duration=21.0,
            critical_paths=[["A0", "A1"]],
            analyses={
                "A0": make_analysis("A0", 0.0, 7.0, 0.0, 7.0, 0.0, 0.0, True),
                "A1": make_analysis("A1", 7.0, 21.0, 7.0, 21.0, 0.0, 0.0, True),
            },
        )

    window = MainWindow(backend=lambda path: None, run_cpm_fn=run_cpm)
    window._session = make_session(candidate)
    window._on_nav(NavDestination.RESULTS)
    assert window._results_page.stack_index() == window._results_page._EMPTY
    assert "no CPM result" in window._results_page._empty_label.text()

    window._results_page._calculate_btn.click()
    ok = wait_for(
        QApplication.instance(),
        lambda: window._results_page.stack_index()
        == window._results_page._DASHBOARD,
    )
    assert ok
    assert "Project duration: 21 days" in window._results_page._summary_label.text()
    assert window.statusBar().currentMessage() == "Results calculated"
    window.close()


def test_calculate_results_failure_reports_error(window: MainWindow) -> None:
    candidate = FakeReviewedCandidate(
        valid=True,
        cpm=None,
        cpm_project_duration=None,
        critical_path_count=None,
        pure_critical_paths=[],
    )

    def run_cpm():
        raise RuntimeError("boom")

    window = MainWindow(backend=lambda path: None, run_cpm_fn=run_cpm)
    window._session = make_session(candidate)
    window._on_nav(NavDestination.RESULTS)
    window._results_page._calculate_btn.click()
    ok = wait_for(
        QApplication.instance(),
        lambda: "failed" in window.statusBar().currentMessage().lower(),
    )
    assert ok
    assert window._results_page.stack_index() == window._results_page._EMPTY
    window.close()


# ---------------------------------------------------------------------------
# Reference fixture (22 activities / 28 dependencies / 54.0 / 16 paths)
# ---------------------------------------------------------------------------


def reference_fixture_candidate() -> FakeReviewedCandidate:
    ids = [f"A{i}" for i in range(22)]
    edges = [
        ("A0", "A1"), ("A0", "A2"), ("A1", "A3"), ("A1", "A4"),
        ("A2", "A4"), ("A2", "A5"), ("A3", "A6"), ("A4", "A6"),
        ("A5", "A6"), ("A6", "A7"), ("A6", "A8"), ("A7", "A9"),
        ("A8", "A9"), ("A8", "A10"), ("A9", "A11"), ("A10", "A11"),
        ("A11", "A12"), ("A11", "A13"), ("A11", "A14"), ("A13", "A15"),
        ("A14", "A15"), ("A12", "A16"), ("A15", "A16"), ("A16", "A17"),
        ("A16", "A18"), ("A17", "A19"), ("A19", "A20"), ("A20", "A21"),
    ]
    middles = [
        ("A3", "A6", "A7", "A9", "A11"),
        ("A4", "A6", "A7", "A9", "A11"),
        ("A3", "A6", "A8", "A9", "A11"),
        ("A4", "A6", "A8", "A9", "A11"),
        ("A3", "A6", "A8", "A10", "A11"),
        ("A4", "A6", "A8", "A10", "A11"),
        ("A2", "A5", "A6", "A7", "A9", "A11"),
        ("A2", "A5", "A6", "A8", "A9", "A11"),
    ]
    tails = [
        ("A12", "A16", "A17", "A19", "A20", "A21"),
        ("A13", "A15", "A16", "A17", "A19", "A20", "A21"),
    ]
    paths = [
        ["A0", "A1", *mid, *tail]
        for mid in middles
        for tail in tails
    ]
    assert len(edges) == 28
    assert len(paths) == 16

    on_path = {aid for path in paths for aid in path}
    analyses = {
        aid: (
            make_analysis(aid, 0.0, 3.0, 0.0, 3.0, 0.0, 0.0, True)
            if aid in on_path
            else make_analysis(aid, 0.0, 3.0, 18.0, 21.0, 18.0, 0.0, False)
        )
        for aid in ids
    }
    graph = make_fake_graph(
        activities={
            aid: FakeGraphActivity(aid, 3.0 if aid in on_path else 4.0)
            for aid in ids
        },
        dependencies=[
            FakeGraphDependency(f"E{i}", src, tgt)
            for i, (src, tgt) in enumerate(edges)
        ],
    )
    cpm = FakeCpmResult(project_duration=54.0, critical_paths=paths, analyses=analyses)
    return FakeReviewedCandidate(valid=True, graph=graph, cpm=cpm)


def test_reference_fixture_presented(page: ResultsPage) -> None:
    page.refresh(make_session(reference_fixture_candidate()))
    assert page.stack_index() == page._DASHBOARD
    assert "Project duration: 54 days" in page._summary_label.text()
    assert page.overview()._kpi_cards["activities"].value() == "22"
    assert page.overview()._kpi_cards["dependencies"].value() == "28"
    assert page.overview()._kpi_cards["critical_paths"].value() == "16"
    assert page.overview()._kpi_cards["critical_activities"].value() == "20"
    assert page.activities().table().rowCount() == 22
    assert len(page.network().node_items()) == 22
    assert len(page.network().edge_items()) == 28
    assert len(page.critical_paths().path_list()._paths) == 16
    assert page.network().edge_items()[("A0", "A1")].is_critical


def test_reference_fixture_path_selection(window: MainWindow, qapp) -> None:
    window._session = make_session(reference_fixture_candidate())
    window._on_nav(NavDestination.RESULTS)
    page = window._results_page
    page.critical_paths().path_list().select_path(3)
    expected = page.critical_paths().path_list()._paths[3]
    assert page.network().current_path() == expected
    assert len(page.network()._edge_items) == 28
