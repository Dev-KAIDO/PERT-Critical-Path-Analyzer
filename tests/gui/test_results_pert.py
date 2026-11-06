"""
PERT tab (Phase 5) GUI tests.

Covers:
- NO_PERT_DATA / READY states and status messages
- Estimates editor prefill and manual-entry write-back
- Signal emission when the run button is clicked
- Results rendering when a PERT result is stored
- Network metric toggle between CPM and PERT modes
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QTableWidgetItem

from pert_analyzer.analysis.pert_engine import PertStatus
from pert_analyzer.gui.pages.results_page import ResultsPage
from pert_analyzer.gui.results.data import (
    PERT_DERIVED_HINT,
    PERT_NO_DATA_HINT,
    PERT_NO_DATA_TITLE,
    derived_pert_estimates,
    effective_pert_estimates,
    extract_pert,
    pert_estimates_source,
)
from pert_analyzer.gui.results.formatting import format_duration, format_number
from pert_analyzer.gui.results.network import NetworkTab
from pert_analyzer.gui.session import GuiSession
from tests.gui.fakes import (
    FakeCpmResult,
    FakeGraphActivity,
    FakeGraphDependency,
    FakePertResult,
    FakeReviewedCandidate,
    FakeReviewSession,
    FakeValidatedWorkflow,
    _UNSET,
    default_pert_result,
    make_analysis,
    make_fake_graph,
    make_pert_analysis,
    make_pert_fixture_session,
)

from pert_analyzer.gui.results.data import PertData, PertRow  # noqa: E402
from pert_analyzer.analysis.pert_engine import PertEstimate  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def page(qapp: QApplication) -> ResultsPage:
    """A bare results page (QApplication already active)."""
    return ResultsPage()


def _make_session_no_estimates() -> GuiSession:
    """Durations with no O/M/P fields → reachable via derived estimates."""
    graph = make_fake_graph(
        activities={
            "A0": FakeGraphActivity("A0", 7.0, "Start"),
            "A1": FakeGraphActivity("A1", 7.0, "Finish"),
        },
        dependencies=[FakeGraphDependency("E0", "A0", "A1")],
    )
    candidate = FakeReviewedCandidate(valid=True, graph=graph, cpm=_UNSET)
    session = GuiSession()
    session.workflow = FakeValidatedWorkflow(
        candidate=candidate, review_session=FakeReviewSession(),
    )
    session.complete_apply(candidate)
    return session


def _make_session_no_durations() -> GuiSession:
    """Activities without any duration or O/M/P field → NO_PERT_DATA."""
    activities = {
        aid: SimpleNamespace(
            id=aid, activity_id=aid, duration=None, name=aid,
            optimistic_time=None, most_likely_time=None, pessimistic_time=None,
        )
        for aid in ("A0", "A1")
    }
    graph = make_fake_graph(activities=activities, dependencies=[])
    candidate = FakeReviewedCandidate(valid=True, graph=graph, cpm=_UNSET)
    session = GuiSession()
    session.workflow = FakeValidatedWorkflow(
        candidate=candidate, review_session=FakeReviewSession(),
    )
    session.complete_apply(candidate)
    return session


def _ready_page(page: ResultsPage) -> tuple[ResultsPage, GuiSession]:
    session = make_pert_fixture_session()
    page.refresh(session)
    return page, session


# ---------------------------------------------------------------------------
# State / messages
# ---------------------------------------------------------------------------


def test_derived_state_messages(page: ResultsPage) -> None:
    """Durations without O/M/P derive a READY three-point set (A-14)."""
    session = _make_session_no_estimates()
    page.refresh(session)
    tab = page.pert()
    text = tab.status_label().text()
    assert "ready" in text
    assert PERT_DERIVED_HINT in text
    assert tab.run_button().isEnabled()
    assert tab.data.estimates_source == "derived"
    # The editor prefills derived values; touching a cell adopts them.
    table = tab.estimates_table()
    assert table.rowCount() == 2
    assert table.item(0, 2).text() == format_number(7.0)


def test_no_pert_data_state_messages(page: ResultsPage) -> None:
    """Without any duration there is nothing to derive -> NO_PERT_DATA."""
    session = _make_session_no_durations()
    page.refresh(session)
    tab = page.pert()
    text = tab.status_label().text()
    assert PERT_NO_DATA_TITLE in text
    assert PERT_NO_DATA_HINT in text
    assert not tab.run_button().isEnabled()


def test_manual_estimates_win_over_derivation(page: ResultsPage) -> None:
    session = _make_session_no_estimates()
    estimates = {
        "A0": PertEstimate(1.0, 2.0, 3.0),
        "A1": PertEstimate(4.0, 5.0, 6.0),
    }
    session.pert_estimates = estimates
    assert effective_pert_estimates(session) is estimates
    assert pert_estimates_source(session) == "manual"


def test_graph_fields_win_over_derivation(page: ResultsPage) -> None:
    session = make_pert_fixture_session()
    assert effective_pert_estimates(session) is None
    assert pert_estimates_source(session) == "graph"


def test_derived_spread_from_review_candidates() -> None:
    """>= 2 distinct candidates -> extremes, clamped to O <= M <= P."""
    graph = SimpleNamespace(
        activities={
            "A1": SimpleNamespace(
                id="A1", activity_id="A1", duration=3.0,
                metadata={"geometric_node_id": "n1"},
            ),
            "A2": SimpleNamespace(
                id="A2", activity_id="A2", duration=4.0,
                metadata={"geometric_node_id": "n2"},
            ),
            "A3": SimpleNamespace(
                id="A3", activity_id="A3", duration=0.0, metadata={},
            ),
            "A4": SimpleNamespace(
                id="A4", activity_id="A4", duration=None, metadata=None,
            ),
        },
        dependencies=[],
    )
    reviews = [
        SimpleNamespace(
            geometric_node_id="n1", activity_id=None,
            current_duration=3.0, proposed_duration=None,
            evidence=[
                SimpleNamespace(description="cost label reads '2 days': 2"),
                SimpleNamespace(description="see the '7 days' cell too: 7"),
            ],
        ),
        SimpleNamespace(
            geometric_node_id="n2", activity_id="A2",
            current_duration=4.0, proposed_duration=9.0, evidence=[],
        ),
    ]
    session = SimpleNamespace(
        graph_model=graph,
        review_session=SimpleNamespace(durations=reviews),
        pert_estimates={},
    )

    est = derived_pert_estimates(session)
    assert est["A1"] == PertEstimate(2.0, 3.0, 7.0)
    assert est["A2"] == PertEstimate(4.0, 4.0, 9.0)
    assert est["A3"] == PertEstimate(0.0, 0.0, 0.0)
    assert "A4" not in est
    assert pert_estimates_source(session) == "derived"


def test_pert_ready_state_enables_run_button(page: ResultsPage) -> None:
    page, session = _ready_page(page)
    tab = page.pert()
    assert tab.run_button().isEnabled()
    assert "ready" in tab.status_label().text()
    assert tab.data is not None
    assert tab.data.status == PertStatus.PERT_READY


# ---------------------------------------------------------------------------
# Estimates editor
# ---------------------------------------------------------------------------


def test_estimates_table_prefilled_from_graph_fields(page: ResultsPage) -> None:
    page, session = _ready_page(page)
    tab = page.pert()
    table = tab.estimates_table()
    assert table.rowCount() == 2
    ids = [table.item(row, 0).text() for row in range(2)]
    assert ids == ["A0", "A1"]
    assert table.item(0, 1).text() == format_number(2.0)
    assert table.item(0, 2).text() == format_number(4.0)
    assert table.item(0, 3).text() == format_number(6.0)


def test_editing_cell_to_empty_triggers_review_and_clears_result(
    page: ResultsPage,
) -> None:
    page, session = _ready_page(page)
    tab = page.pert()
    assert tab.data.status == PertStatus.PERT_READY
    assert session.pert_result is not None
    table = tab.estimates_table()
    table.setItem(0, 1, QTableWidgetItem(""))
    table.cellChanged.emit(0, 1)
    assert session.pert_estimates["A0"].optimistic is None
    assert session.pert_result is None
    tab.refresh(session)
    assert tab.data.status == PertStatus.PERT_REVIEW_REQUIRED
    assert not tab.run_button().isEnabled()


# ---------------------------------------------------------------------------
# Run signal
# ---------------------------------------------------------------------------


def test_pert_run_requested_signal_emitted(page: ResultsPage) -> None:
    page, session = _ready_page(page)
    tab = page.pert()
    received: list[bool] = []
    tab.pert_run_requested.connect(lambda: received.append(True))
    tab._on_run_clicked()
    assert received == [True]


# ---------------------------------------------------------------------------
# Results rendering
# ---------------------------------------------------------------------------


def test_results_table_shows_backend_values(page: ResultsPage) -> None:
    page, session = _ready_page(page)
    tab = page.pert()
    assert tab.data.status == PertStatus.PERT_READY
    assert tab.data.result is not None
    table = tab.results_table()
    assert table.rowCount() == 2
    # Sort-independent: build a map activity_id -> row
    id_to_row: dict[str, int] = {}
    for r in range(table.rowCount()):
        item = table.item(r, 0)
        if item is not None:
            id_to_row[item.text()] = r
    row_a0 = id_to_row["A0"]
    row_a1 = id_to_row["A1"]
    assert table.item(row_a0, 4).text() == format_number(4.0)
    assert table.item(row_a0, 5).text() == format_number(16.0 / 36.0)
    assert table.item(row_a0, 7).text() == "Yes"
    assert table.item(row_a1, 7).text() == "Yes"


def test_summary_cards_populated(page: ResultsPage) -> None:
    page, session = _ready_page(page)
    cards = page.pert().summary_cards()
    assert cards["expected_duration"].value() == format_duration(10.0)
    assert cards["project_variance"].value() == format_number(
        32.0 / 36.0
    )
    std_val = (32.0 / 36.0) ** 0.5
    assert cards["project_std_dev"].value() == format_number(std_val)
    prob_text = cards["probability"].value()
    assert prob_text != "Unavailable"
    assert "%" in prob_text


# ---------------------------------------------------------------------------
# Network metric toggle
# ---------------------------------------------------------------------------


def _sample_results_data_and_pert_data():
    graph = make_fake_graph(
        activities={
            "A0": FakeGraphActivity("A0", 5.0, "Start"),
            "A1": FakeGraphActivity("A1", 3.0, "Middle"),
            "A2": FakeGraphActivity("A2", 4.0, "End"),
        },
        dependencies=[
            FakeGraphDependency("E0", "A0", "A1"),
            FakeGraphDependency("E1", "A1", "A2"),
        ],
    )
    cpm = FakeCpmResult(
        project_duration=12.0,
        critical_paths=[["A0", "A1", "A2"]],
        analyses={
            "A0": make_analysis("A0", 0.0, 5.0, 0.0, 5.0, 0.0, 0.0, True),
            "A1": make_analysis("A1", 5.0, 8.0, 5.0, 8.0, 0.0, 0.0, True),
            "A2": make_analysis("A2", 8.0, 12.0, 8.0, 12.0, 0.0, 0.0, True),
        },
    )
    from pert_analyzer.gui.results.data import ResultsData

    results = ResultsData(
        project_duration=12.0,
        critical_path_count=1,
        critical_paths=[["A0", "A1", "A2"]],
        critical_activity_count=3,
        critical_edges={("A0", "A1"), ("A1", "A2")},
        ready=True,
        reason="ok",
        cpm=cpm,
        graph=graph,
    )
    pert_data = PertData(
        status=PertStatus.PERT_READY,
        rows=[
            PertRow("A0", 2.0, 4.0, 6.0, 4.0, 16.0 / 36.0, 4.0 / 6.0, 0.0, True),
            PertRow("A1", 1.0, 3.0, 5.0, 3.0, 16.0 / 36.0, 4.0 / 6.0, 0.0, True),
            PertRow("A2", 2.0, 4.0, 6.0, 4.0, 16.0 / 36.0, 4.0 / 6.0, 0.0, True),
        ],
        critical_paths=[["A0", "A1", "A2"]],
        critical_activity_count=3,
    )
    return results, pert_data


def test_network_toggle_renders_pert_expected_times(qapp: QApplication) -> None:
    del qapp
    results, pert_data = _sample_results_data_and_pert_data()
    network = NetworkTab()
    network.set_data(results)
    assert network.metric_mode() == "CPM"
    assert network.node_items()["A0"].duration == 5.0
    network.set_pert_data(pert_data)
    network.set_metric_mode("PERT")
    assert network.metric_mode() == "PERT"
    assert network.node_items()["A0"].duration == 4.0
    assert network.node_items()["A0"].metric_label == "Expected"
    network.set_metric_mode("CPM")
    assert network.node_items()["A0"].duration == 5.0
    assert network.node_items()["A0"].metric_label == "Duration"