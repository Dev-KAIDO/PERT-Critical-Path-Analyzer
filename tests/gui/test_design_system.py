"""
Phase 7 — design-system tests.

These tests verify behaviour and key style assignments (not exact pixels):
theme loading, navigation visual state, workflow indicator, and focusability.
"""

from __future__ import annotations

import pytest

from pert_analyzer.gui.main_window import MainWindow
from pert_analyzer.gui.navigation import NavDestination
from pert_analyzer.gui.themes.palette import COLORS, NETWORK_COLORS
from pert_analyzer.gui.themes.components import (
    badge_color,
    badge_qss,
    card_qss,
    primary_button_qss,
)
from pert_analyzer.gui.themes.style import apply_theme


@pytest.fixture()
def window(qapp: "QApplication") -> "MainWindow":  # noqa: F821
    w = MainWindow(backend=lambda path: None)
    yield w
    w.close()


# ── 1. Theme loading ──────────────────────────────────────────────
def test_theme_applies(qapp: "QApplication") -> None:  # noqa: F821
    apply_theme(qapp)
    qss = qapp.styleSheet()
    assert "#0F1117" in qss          # bg token
    assert "#4F8CFF" in qss          # accent token
    assert "QPushButton" in qss
    assert "QTabWidget::pane" in qss


def test_colors_dict_has_required_keys() -> None:
    required = {
        "bg", "surface", "surface_light", "elevated", "text",
        "text_secondary", "muted", "accent", "accent_hover",
        "success", "danger", "warning", "info", "border",
    }
    assert required.issubset(COLORS.keys())
    for value in COLORS.values():
        assert value.startswith("#")


def test_network_colors_has_required_keys() -> None:
    required = {
        "canvas", "node_fill", "node_fill_critical", "node_border",
        "node_border_critical", "node_text", "node_text_muted",
        "node_selected", "edge", "edge_critical", "path_highlight",
    }
    assert required.issubset(NETWORK_COLORS.keys())


# ── 2. Badge / card components ────────────────────────────────────
@pytest.mark.parametrize(
    "status,expected",
    [
        ("VALID", COLORS["success"]),
        ("READY", COLORS["success"]),
        ("REVIEW_REQUIRED", COLORS["warning"]),
        ("INVALID", COLORS["danger"]),
        ("ERROR", COLORS["danger"]),
        ("BLOCKED", COLORS["danger"]),
        ("CRITICAL", COLORS["danger"]),
    ],
)
def test_badge_color_semantics(status: str, expected: str) -> None:
    assert badge_color(status) == expected


def test_badge_qss_uses_semantic_color() -> None:
    assert COLORS["success"] in badge_qss("VALID")
    assert COLORS["danger"] in badge_qss("INVALID")


def test_card_qss_references_surface_and_radius() -> None:
    assert COLORS["surface"] in card_qss()
    assert "border-radius" in card_qss()


def test_primary_button_qss_references_accent() -> None:
    assert COLORS["accent"] in primary_button_qss()


# ── 3. Navigation visual state ────────────────────────────────────
@pytest.mark.parametrize("dest", list(NavDestination))
def test_nav_button_checked_state(
    window: MainWindow, dest: NavDestination
) -> None:
    window._on_nav(dest)
    for idx, btn in window._nav_buttons.items():
        assert btn.isChecked() == (idx == dest.value)


def test_nav_buttons_have_object_names(window: MainWindow) -> None:
    names = {
        btn.objectName()
        for btn in window._nav_buttons.values()
    }
    assert names == {
        "nav_analyze",
        "nav_understanding",
        "nav_review",
        "nav_results",
        "nav_network_builder",
    }  # A-13: nav_validate retired


def test_nav_buttons_styled_for_active_state(window: MainWindow) -> None:
    btn = window._nav_buttons[NavDestination.ANALYZE.value]
    assert ":checked" in btn.styleSheet()
    assert "font-weight" in btn.styleSheet()


def test_workflow_indicator_exists_and_updates(window: MainWindow) -> None:
    assert window._workflow_indicator is not None
    window._on_nav(NavDestination.REVIEW)
    text = window._workflow_indicator.text()
    assert "Review" in text
    assert "\u25CF" in text          # current step marker


# ── 4. Review Center design-system surface ────────────────────────
from PySide6.QtWidgets import QLabel  # noqa: E402

from pert_analyzer.gui.pages.review_page import ReviewPage  # noqa: E402
from pert_analyzer.gui.themes.palette import (  # noqa: E402
    TEXT_MUTED,
    TEXT_SECONDARY,
)


def _labels(widget) -> list[QLabel]:
    return widget.findChildren(QLabel)


def test_review_page_header_uses_design_tokens(qapp: "QApplication") -> None:  # noqa: F821
    page = ReviewPage()
    labels = _labels(page)
    assert any(lbl.text() == "Review Center" for lbl in labels)
    assert any(
        "Confirm or correct detected activities" in lbl.text() and TEXT_SECONDARY in lbl.styleSheet()
        for lbl in labels
    )
    assert page._detail_placeholder.text() == "Select a review item from the list to begin."
    assert "color: " in page._detail_placeholder.styleSheet()
    page.deleteLater()


def test_review_page_empty_state_has_correct_tokens(qapp: "QApplication") -> None:  # noqa: F821
    page = ReviewPage()
    assert "Run an analysis" in page.empty_message
    assert TEXT_MUTED in page._empty_state._message.styleSheet()
    assert page._empty_state._go_btn.objectName() == "primary"
    page.deleteLater()


# ── 5. Validation Center design-system surface ────────────────────
from pert_analyzer.gui.pages.validation_page import ValidationPage  # noqa: E402


def test_validation_page_header_uses_design_tokens(qapp: "QApplication") -> None:  # noqa: F821
    page = ValidationPage()
    labels = _labels(page)
    assert any(lbl.text() == "Validation Center" for lbl in labels)
    assert any(
        "Validate the reviewed graph" in lbl.text() and TEXT_SECONDARY in lbl.styleSheet()
        for lbl in labels
    )
    page.deleteLater()


def test_validation_page_empty_state_uses_design_tokens(qapp: "QApplication") -> None:  # noqa: F821
    page = ValidationPage()
    assert "No project analyzed" in page._empty_label.text()
    assert TEXT_MUTED in page._empty_label.styleSheet()
    assert page._go_analyze_btn.objectName() == "primary"
    assert page._continue_btn.objectName() == "primary"
    page.deleteLater()


# ── 6. Results dashboard + PERT status semantics ──────────────────
from pert_analyzer.gui.themes.palette import SUCCESS  # noqa: E402


def test_pert_tab_status_label_color_ready(qapp: "QApplication") -> None:  # noqa: F821
    from tests.gui.fakes import make_pert_fixture_session

    from pert_analyzer.gui.pages.results_page import ResultsPage

    page = ResultsPage()
    session = make_pert_fixture_session()
    page.refresh(session)
    assert page.tabs().tabText(4) == "PERT"
    pert = page.pert()
    assert SUCCESS in pert._status_label.styleSheet()
    page.deleteLater()


def test_pert_tab_status_label_color_missing(qapp: "QApplication") -> None:  # noqa: F821
    from pert_analyzer.gui.pages.results_page import ResultsPage
    from pert_analyzer.gui.session import GuiSession

    page = ResultsPage()
    session = GuiSession()
    page.refresh(session)
    pert = page.pert()
    assert TEXT_MUTED in pert._status_label.styleSheet()
    page.deleteLater()


def test_results_export_and_report_buttons_present(
    window: MainWindow,
) -> None:
    window._on_nav(NavDestination.RESULTS)
    page = window._results_page
    assert page._export_btn.text() == "Export"
    assert page._report_btn.text() == "Report"
    assert "Export".lower() in page._export_btn.toolTip().lower()
    assert "Report".lower() in page._report_btn.toolTip().lower()


def test_header_status_badge_styled_for_valid(window: MainWindow) -> None:
    assert window._header_status.objectName() == "headerStatus"
    assert window._header_status.styleSheet()
    assert "font-weight" in window._header_status.styleSheet()