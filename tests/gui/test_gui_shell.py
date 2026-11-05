"""
Tests for the GUI shell: window launch, 5 pages, navigation, empty states.
"""

from __future__ import annotations

import pytest

from pert_analyzer.gui.main_window import MainWindow
from pert_analyzer.gui.navigation import NavDestination


@pytest.fixture()
def window(qapp: "QApplication") -> "MainWindow":  # noqa: F821
    w = MainWindow(backend=lambda path: None)
    yield w
    w.close()


def test_window_launches(window: MainWindow) -> None:
    assert window.windowTitle() == "PERT & Critical Path Analyzer"
    assert window.minimumWidth() >= 1100
    assert window.minimumHeight() >= 720


def test_five_pages_exist(window: MainWindow) -> None:
    assert window._stack.count() == 6


@pytest.mark.parametrize("dest", list(NavDestination))
def test_navigation_works(window: MainWindow, dest: NavDestination) -> None:
    window._on_nav(dest)
    assert window._stack.currentIndex() == dest.value
    for idx, btn in window._nav_buttons.items():
        assert btn.isChecked() == (idx == dest.value)


def test_review_page_empty_state(window: MainWindow) -> None:
    window._on_nav(NavDestination.REVIEW)
    assert "No analysis available" in window._review_page.empty_message


def test_results_page_empty_state(window: MainWindow) -> None:
    window._on_nav(NavDestination.RESULTS)
    assert "No project analyzed" in window._results_page._empty_label.text()
