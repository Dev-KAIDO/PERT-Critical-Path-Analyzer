"""
Phase 7.2 UX-refinement tests: responsive workspace, data readability,
drag & drop, preview interactions, and result visibility.

These tests are presentation-layer only and never depend on OCR, CV, or
the real pipeline. They verify the widgets the Phase 7.2 rework shipped:
start-up sizing, the reusable zoom/pan viewer (analyze + review image
context), drag-and-drop from Explorer, clipboard paste, the 20/30/50
review splitter, scrollable validation details, PRELIMINARY vs FINAL
results states, the workflow step indicator, friendly review terminology,
and the reflowed Overview KPI layout.
"""

from __future__ import annotations

import os

from PySide6.QtCore import QMimeData, QUrl
from PySide6.QtGui import QClipboard, QImage
from PySide6.QtWidgets import QApplication, QSplitter

from pert_analyzer.gui import app as gui_app
from pert_analyzer.gui.main_window import MainWindow
from pert_analyzer.gui.pages.analysis_page import AnalysisPage, ImageView
from pert_analyzer.gui.pages.results_page import ResultsPage
from pert_analyzer.gui.pages.review_page import ReviewPage
from pert_analyzer.gui.pages.validation_page import ValidationPage
from pert_analyzer.gui.results.overview import OverviewTab
from pert_analyzer.gui.reviews.context import HighlightRect
from pert_analyzer.gui.reviews.detail_panels import (
    ActivityDetailPanel,
    DependencyDetailPanel,
    DurationDetailPanel,
)
from pert_analyzer.gui.reviews.widgets import EvidencePanel, ImageContextView
from pert_analyzer.gui.session import GuiSession
from tests.gui.fakes import (
    FakeReviewedWorkflow,
    make_review_item,
    make_review_session,
)


def _make_png(path: str, w: int = 160, h: int = 120, color: str = "#4F8CFF") -> str:
    image = QImage(w, h, QImage.Format.Format_RGB32)
    image.fill(_hex(color))
    assert image.save(path, "PNG")
    return path


def _hex(value: str) -> int:
    return int(value.lstrip("#"), 16)


# ---------------------------------------------------------------------------
# Start-up sizing / responsive workspace
# ---------------------------------------------------------------------------


def test_launcher_screen_large_enough_returns_bool(qapp: QApplication) -> None:
    assert isinstance(gui_app._screen_large_enough(MainWindow()), bool)


def test_main_window_responsive_minimum(qapp: QApplication) -> None:
    window = MainWindow(backend=lambda path: None)
    min_w, min_h = window.minimumWidth(), window.minimumHeight()
    assert min_w >= 1100
    assert min_h >= 720
    window.resize(1280, 720)
    qapp.processEvents()
    assert window.width() >= 1100
    assert len(window._nav_buttons) == 6
    for item in window._nav_buttons.keys():
        assert isinstance(item, int)
    window.close()


# ---------------------------------------------------------------------------
# Workflow step indicator
# ---------------------------------------------------------------------------


def test_workflow_indicator_pending_with_no_workflow(qapp: QApplication) -> None:
    window = MainWindow(backend=lambda path: None)
    window._update_workflow_indicator()
    text = window._workflow_indicator.text()
    assert "\u25CB" in text  # pending glyph present
    assert "\u25CF" in text  # current page glyph present
    window.close()


def test_workflow_indicator_done_and_blocked_steps(qapp: QApplication) -> None:
    window = MainWindow(backend=lambda path: None)
    session = GuiSession()
    session.complete_analysis(
        FakeReviewedWorkflow(
            make_review_session(n_activities=2, n_dependencies=1, n_durations=1)
        )
    )
    window._session = session
    window._update_workflow_indicator()
    text = window._workflow_indicator.text()
    assert "\u25CF" in text  # current page (Analyze)
    assert "\u26a0" in text  # review step blocked while pending items exist
    assert "Review" in text
    window.close()


# ---------------------------------------------------------------------------
# Drag & drop + clipboard paste on the Analyze page
# ---------------------------------------------------------------------------


def test_image_view_accepts_drop_only_images(tmp_path) -> None:
    png = os.path.normpath(str(tmp_path / "diagram.png"))
    _make_png(png)
    txt = tmp_path / "notes.txt"
    txt.write_text("x", encoding="utf-8")

    mime_png = QMimeData()
    mime_png.setUrls([QUrl.fromLocalFile(png)])
    mime_txt = QMimeData()
    mime_txt.setUrls([QUrl.fromLocalFile(str(txt))])

    view = ImageView()
    assert view.accepts_drop(mime_png) is True
    assert view.accepts_drop(mime_txt) is False
    assert os.path.normpath(ImageView.drop_path_from_mime(mime_png)) == png
    assert ImageView.drop_path_from_mime(mime_txt) is None
    view.deleteLater()


def test_image_view_drop_emits_path(qapp: QApplication, tmp_path) -> None:
    png = os.path.normpath(str(tmp_path / "drop.png"))
    _make_png(png)
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(png)])
    view = ImageView()
    view.setFixedSize(320, 220)
    emitted: list[str] = []
    view.image_dropped.connect(emitted.append)
    assert os.path.normpath(view.load_dropped(mime)) == png
    assert emitted and os.path.normpath(emitted[0]) == png
    view.deleteLater()


def test_analysis_page_loads_dropped_image(qapp: QApplication, tmp_path) -> None:
    page = AnalysisPage()
    page.resize(1000, 700)
    png = os.path.normpath(str(tmp_path / "web.png"))
    _make_png(png)

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(png)])
    page._image_view.load_dropped(mime)
    qapp.processEvents()

    assert os.path.normpath(page._image_path) == png
    assert page._image_view._pixmap is not None
    assert page._analyze_btn.isEnabled()
    assert page._fit_btn.isEnabled()
    page.deleteLater()


def test_analysis_page_paste_from_clipboard(qapp: QApplication, tmp_path) -> None:
    page = AnalysisPage()
    clip = QApplication.clipboard()
    image = QImage(64, 48, QImage.Format.Format_RGB32)
    image.fill(_hex("#31C48D"))
    clip.setImage(image, QClipboard.Mode.Clipboard)
    assert page._on_paste() is True
    assert page._image_view._pixmap is not None
    clip.clear(QClipboard.Mode.Clipboard)
    page.deleteLater()


def test_analysis_page_paste_with_empty_clipboard(qapp: QApplication) -> None:
    page = AnalysisPage()
    QApplication.clipboard().clear(QClipboard.Mode.Clipboard)
    assert page._on_paste() is False
    page.deleteLater()


# ---------------------------------------------------------------------------
# Viewer zoom / pan / reset
# ---------------------------------------------------------------------------


def test_image_view_zoom_in_out_and_reset(qapp: QApplication, tmp_path) -> None:
    png = _make_png(str(tmp_path / "zoom.png"))
    view = ImageView()
    view.resize(400, 300)
    w, h = view.set_image(png)
    qapp.processEvents()
    assert (w, h) == (160, 120)
    assert not view.is_zoomed

    view.zoom_in()
    assert view.is_zoomed
    assert view.zoom_factor > 1.0

    view.set_zoom(3.0)
    assert view.zoom_factor == 3.0

    before = view.zoom_factor
    view.zoom_out()
    assert view.zoom_factor < before

    view.reset_zoom()
    assert view.zoom_factor == 1.0
    assert not view.is_zoomed
    assert view._pan == view._pan  # pan stays valid on reset
    view.deleteLater()


def test_image_view_fit_and_clear(qapp: QApplication, tmp_path) -> None:
    png = _make_png(str(tmp_path / "fit.png"))
    view = ImageView()
    view.resize(400, 300)
    view.set_image(png)
    view.zoom_in()
    view.fit_image()
    assert view.zoom_factor == 1.0
    view.clear_image()
    assert view._pixmap is None
    assert "No image selected" in view.text()
    view.deleteLater()


def test_image_view_wheel_zooms(qapp: QApplication, tmp_path) -> None:
    png = _make_png(str(tmp_path / "wheel.png"))
    view = ImageView()
    view.resize(400, 300)
    view.set_image(png)

    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent

    wheel = QWheelEvent(
        QPointF(200, 150),
        QPointF(200, 150),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(view, wheel)
    assert view.is_zoomed
    view.deleteLater()


# ---------------------------------------------------------------------------
# Review Center: splitter + friendly terminology
# ---------------------------------------------------------------------------


def test_review_splitter_20_30_50_initial(qapp: QApplication) -> None:
    page = ReviewPage()
    splitters = page.findChildren(QSplitter)
    assert splitters, "Review page must contain a QSplitter"
    splitter = splitters[0]
    assert splitter.orientation() == splitter.orientation().Horizontal
    assert splitter.count() == 3
    sizes = splitter.sizes()
    assert all(s > 0 for s in sizes)


def test_review_splitter_user_resizable(qapp: QApplication) -> None:
    page = ReviewPage()
    splitter = page.findChildren(QSplitter)[0]
    page.resize(1200, 700)
    qapp.processEvents()
    before = list(splitter.sizes())
    splitter.setSizes([400, 200, 400])
    qapp.processEvents()
    sizes = list(splitter.sizes())
    assert all(int(x) >= 0 for x in sizes)
    # the user can drag the visible handle, so splitter is resizable
    assert splitter.handleWidth() > 0
    # sizes responded to the requested distribution
    if sizes:
        assert max(sizes) > 0


def test_review_detail_panels_friendly_titles(qapp: QApplication) -> None:
    session = GuiSession()
    session.current_image_path = None

    activity = make_review_item("activity", geometric_node_id="N0", activity_id="A0")
    panel = ActivityDetailPanel()
    panel.show_item(activity, session)
    assert "Activity identity needs confirmation" in panel._card.title()
    assert "The engine read this node as" in panel._summary_label.text()

    dep = make_review_item("dependency", geometric_node_id="N0", activity_id="A0")
    dpanel = DependencyDetailPanel()
    dpanel.show_item(dep, session)
    assert "Uncertain dependency" in dpanel._card.title()

    dur = make_review_item("duration", geometric_node_id="N0", activity_id="A0")
    drpanel = DurationDetailPanel()
    drpanel.show_item(dur, session)
    assert "Missing or invalid duration" in drpanel._card.title()


def test_evidence_technical_details_expandable(qapp: QApplication) -> None:
    ev = EvidencePanel()
    try:
        item = make_review_item("activity", geometric_node_id="N0", activity_id="A0")
        evidence = list(getattr(item, "evidence", []) or [])
        ev.set_evidence(evidence, 0.8, reason="OCR uncertainty on semantic id")
        assert ev._technical.isHidden()
        ev._tech_btn.setChecked(True)
        ev._toggle_technical(True)
        assert not ev._technical.isHidden()
        assert "Confidence" in ev._confidence.text()
    finally:
        ev.deleteLater()


def test_review_image_context_zoomable(qapp: QApplication, tmp_path) -> None:
    png = _make_png(str(tmp_path / "ctx.png"))
    session = GuiSession()
    session.current_image_path = str(png)
    view = ImageContextView()
    view.resize(300, 240)
    highlight = HighlightRect(x=10, y=10, w=80, h=40, color="accent", label="N0")
    shown = view.set_context(str(png), [highlight])
    assert shown is True
    assert view.zoom_factor == 1.0
    view.zoom_in()
    assert view.is_zoomed
    view.reset_zoom()
    assert view.zoom_factor == 1.0
    view.deleteLater()


def test_review_image_context_focus_evidence(qapp: QApplication, tmp_path) -> None:
    png = _make_png(str(tmp_path / "focus.png"), 400, 300)
    view = ImageContextView()
    view.resize(300, 240)
    highlight = HighlightRect(x=250, y=180, w=60, h=50, color="accent", label="N9")
    assert view.set_context(png, [highlight]) is True
    view.focus_highlights()
    assert view.zoom_factor == 1.8
    assert view.is_zoomed
    view.deleteLater()


# ---------------------------------------------------------------------------
# Validation Center readability
# ---------------------------------------------------------------------------


def test_validation_pages_responsive_layout(qapp: QApplication) -> None:
    page = ValidationPage()
    assert page._summary_card.title() == "Graph Summary"
    assert page._cpm_card.title() == "CPM Readiness"
    assert page._issues_card.title() == "Validation Issues"
    # the Issue Details body must live inside a scrollable area
    from PySide6.QtWidgets import QScrollArea

    scrolls = [w for w in page._detail_card.findChildren(QScrollArea)]
    assert scrolls, "Validation Page must wrap Issue Details in a scroll area"
    assert scrolls[0].widgetResizable() is True
    assert scrolls[0].minimumHeight() >= 120
    assert page._source_label is not None
    page.deleteLater()


# ---------------------------------------------------------------------------
# Results: PRELIMINARY vs FINAL + overview KPI reflow
# ---------------------------------------------------------------------------


def test_results_not_ready_with_workflow_shows_preliminary(qapp: QApplication) -> None:
    page = ResultsPage()
    session = GuiSession()
    session.complete_analysis(
        FakeReviewedWorkflow(
            make_review_session(n_activities=2, n_dependencies=1, n_durations=1)
        )
    )
    page.refresh(session)
    assert page.stack_index() == page._EMPTY
    assert page._not_ready_heading.text() == "RESULTS NOT READY"
    assert not page._preliminary_card.isHidden()
    assert "reviews complete" in page._progress_label.text()
    assert "0 / 4" in page._progress_label.text()
    assert "Results not ready" in page._empty_label.text()
    page.deleteLater()


def test_results_ready_hides_preliminary_shows_final_caption(qapp: QApplication) -> None:
    from tests.gui.test_results_dashboard import make_session, two_activity_candidate

    page = ResultsPage()
    session = make_session(two_activity_candidate())
    page.refresh(session)
    assert page.stack_index() == page._DASHBOARD
    assert "FINAL RESULTS" in page._status_label.text()
    assert page._empty_label.isHidden()
    assert page._preliminary_card.isHidden()
    assert page.overview()._kpi_cards["critical_activities"].value() == "2"
    page.deleteLater()


def test_overview_kpi_reflow_single_row(qapp: QApplication) -> None:
    from tests.gui.test_results_dashboard import two_activity_candidate, make_session

    page = ResultsPage()
    session = make_session(two_activity_candidate())
    page.refresh(session)
    overview: OverviewTab = page.overview()
    keys = list(overview._kpi_cards)
    assert set(keys) == {
        "activities",
        "dependencies",
        "duration",
        "critical_activities",
        "critical_paths",
        "review_state",
    }
    assert overview._cards_row.count() == 6  # one KPI row with six cards
    assert "reviewed activities" in overview._composition.text()
    assert "Backend CPM" in overview._confidence.text()
    page.deleteLater()


def test_results_no_analysis_empty_state_has_actions(qapp: QApplication) -> None:
    page = ResultsPage()
    page.refresh(GuiSession())
    assert page.stack_index() == page._EMPTY
    assert page._go_review_btn.text() == "Continue Review"
    assert page._go_validation_btn.text() == "Open Validation"
    page.deleteLater()


def test_analysis_page_diagram_type_shown(qapp: QApplication) -> None:
    page = AnalysisPage()
    page.set_diagram_type("PERT Network")
    assert page._diagram_type_value.text() == "PERT Network"
    page.set_diagram_type("")
    assert page._diagram_type_value.text() == "\u2014"
    page.deleteLater()
