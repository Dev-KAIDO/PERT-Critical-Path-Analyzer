from PySide6.QtCore import Qt

from pert_analyzer.gui.locale import UiLanguage, locale_for, nav_label, normalize_language
from pert_analyzer.gui.main_window import MainWindow


def test_locale_normalization_and_direction() -> None:
    assert normalize_language("ar") is UiLanguage.ARABIC
    assert normalize_language("unknown") is UiLanguage.ENGLISH
    assert locale_for("ar").is_rtl is True
    assert locale_for("en").is_rtl is False


def test_arabic_navigation_labels() -> None:
    assert nav_label("analyze", "ar") == "تحليل"
    assert nav_label("results", "ar") == "النتائج"
    assert nav_label("results", "en") == "Results"


def test_main_window_switches_language_and_direction(qapp) -> None:
    window = MainWindow(backend=lambda path: None)
    try:
        assert window.language is UiLanguage.ENGLISH
        window.set_language("ar")
        assert window.language is UiLanguage.ARABIC
        # A-8 policy: the shell (sidebar) never flips - Arabic only
        # rewrites texts, so the window always stays left-to-right.
        assert window.layoutDirection() == Qt.LayoutDirection.LeftToRight
        assert window._locale_btn.text() == "English"
        assert window._nav_buttons[0].text().endswith("تحليل")
        window.set_language("en")
        assert window.layoutDirection() == Qt.LayoutDirection.LeftToRight
        assert window._locale_btn.text() == "عربي"
        assert window._nav_buttons[0].text().endswith("Analyze")
    finally:
        window.close()
