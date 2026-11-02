"""Small, explicit UI-locale layer for English and Arabic layouts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QLabel, QPushButton, QWidget


class UiLanguage(StrEnum):
    ENGLISH = "en"
    ARABIC = "ar"


@dataclass(frozen=True)
class UiLocale:
    language: UiLanguage
    direction: Qt.LayoutDirection

    @property
    def is_rtl(self) -> bool:
        return self.direction == Qt.LayoutDirection.RightToLeft


LOCALES = {
    UiLanguage.ENGLISH: UiLocale(UiLanguage.ENGLISH, Qt.LayoutDirection.LeftToRight),
    UiLanguage.ARABIC: UiLocale(UiLanguage.ARABIC, Qt.LayoutDirection.RightToLeft),
}


def normalize_language(value: str | UiLanguage | None) -> UiLanguage:
    try:
        return UiLanguage(value or UiLanguage.ENGLISH)
    except ValueError:
        return UiLanguage.ENGLISH


def locale_for(value: str | UiLanguage | None) -> UiLocale:
    return LOCALES[normalize_language(value)]


def apply_ui_locale(widget: QWidget, value: str | UiLanguage | None) -> UiLocale:
    """Apply the locale WITHOUT flipping layout direction (A-8 policy).

    Arabic only rewrites texts; the window keeps its stable left
    sidebar regardless of language (user decision - the whole-shell
    flip felt confusing).  ``locale.direction`` stays available as the
    locale's natural-direction marker, the properties record the
    active locale, and the style re-polish still runs.
    """
    locale = locale_for(value)
    widget.setProperty("uiLanguage", locale.language.value)
    widget.setProperty("uiDirection", "rtl" if locale.is_rtl else "ltr")
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    return locale


def translate_text(source: str, value: str | UiLanguage | None) -> str:
    """Translate one UI string while keeping backend identifiers untouched."""
    return UI_TEXT.get(source, source) if normalize_language(value) == UiLanguage.ARABIC else source


def widget_language(widget: QWidget) -> UiLanguage:
    """Return the language currently applied to a widget subtree."""
    return normalize_language(widget.property("uiLanguage"))


NAV_LABELS = {
    UiLanguage.ENGLISH: {
        "analyze": "Analyze",
        "understanding": "Understand",
        "review": "Review",
        "validate": "Validate",
        "results": "Results",
        "network_builder": "Build Network",
    },
    UiLanguage.ARABIC: {
        "analyze": "تحليل",
        "understanding": "فهم النتائج",
        "review": "المراجعة",
        "validate": "التحقق",
        "results": "النتائج",
        "network_builder": "بناء الشبكة",
    },
}

UI_TEXT = {
    # Navigation and page chrome
    "Analyze": "تحليل",
    "Understand": "فهم النتائج",
    "Review": "مراجعة",
    "Validate": "تحقق",
    "Results": "النتائج",
    "Build Network": "بناء الشبكة",
    "Review Center": "مركز المراجعة",
    "Validation Center": "مركز التحقق",
    "Results Dashboard": "لوحة النتائج",
    "Network Builder": "منشئ الشبكة",
    "Analyze Diagram": "تحليل المخطط",
    "Understanding": "فهم النتائج",
    "Confirm or correct detected activities, dependencies, and durations.": "أكد أو صحح الأنشطة والعلاقات والمدد المكتشفة.",
    "Critical path analysis of the reviewed graph.": "تحليل المسار الحرج للرسم البياني الذي تمت مراجعته.",
    "Upload a PERT/CPM network diagram to begin": "ارفع مخطط شبكة PERT/CPM للبدء",
    "No image selected": "لم يتم اختيار صورة",
    "Select a review item from the list to begin.": "اختر عنصر مراجعة من القائمة للبدء.",
    "Confidence filter": "فلتر الثقة",
    "All items": "كل العناصر",
    "Low confidence": "ثقة منخفضة",
    "Medium confidence": "ثقة متوسطة",
    "High confidence": "ثقة عالية",
    "Focus evidence": "تركيز الأدلة",
    "Apply & Validate": "تطبيق والتحقق",
    "Leave Unresolved": "ترك دون حل",
    "Undo last decision": "التراجع عن آخر قرار",
    "Accept": "قبول",
    "Reject": "رفض",
    "Correct...": "تصحيح...",
    "Reverse": "عكس الاتجاه",
    "Export": "تصدير",
    "Report": "تقرير",
    "Calculate Results": "حساب النتائج",
    "Continue Review": "متابعة المراجعة",
    "Open Validation": "فتح التحقق",
    "FINAL RESULTS": "النتائج النهائية",
    "PRELIMINARY": "أولية",
    "RESULTS NOT READY": "النتائج غير جاهزة",
    "Ready": "جاهز",
    "Apply": "تطبيق",
    "Cancel": "إلغاء",
    "Go to Validation": "الانتقال إلى التحقق",
    "Review Complete": "اكتملت المراجعة",
    "All reviews resolved and decisions applied.": "تم حل جميع عناصر المراجعة وتطبيق القرارات.",
    "All reviews resolved. Apply decisions to continue.": "تم حل جميع عناصر المراجعة. طبّق القرارات للمتابعة.",
    "Activity reviews": "مراجعات الأنشطة",
    "Dependency reviews": "مراجعات العلاقات",
    "Duration reviews": "مراجعات المدد",
    "Activities": "الأنشطة",
    "Dependencies": "العلاقات",
    "Durations": "المدد",
    "Select a review item from the list to begin.": "اختر عنصر مراجعة من القائمة للبدء.",
    "No analysis available. Run an analysis on the Analyze Diagram page first.": "لا يتوفر تحليل. شغّل التحليل أولًا من صفحة تحليل المخطط.",
    "Leave Unresolved": "ترك دون حل",
    "Coming in a later phase": "ستتوفر هذه الميزة في مرحلة لاحقة",
    "Export the current results (PDF, Excel, JSON, CSV)": "تصدير النتائج الحالية (PDF وExcel وJSON وCSV)",
    "Generate a detailed project report": "إنشاء تقرير مفصل عن المشروع",
    "Run the existing CPM service for this valid graph": "تشغيل خدمة CPM الحالية لهذا الرسم الصحيح",
    # Analysis page
    "Ready": "جاهز",
    "No image selected": "لم يتم اختيار صورة",
    "Upload a PERT/CPM network diagram to begin": "ارفع مخطط شبكة PERT/CPM للبدء",
    "Drop diagram here": "أفلت المخطط هنا",
    "Release to load this image": "حرّر الزر لتحميل الصورة",
    "Choose Image": "اختيار صورة",
    "Analyze Diagram": "تحليل المخطط",
    "Zoom in": "تكبير",
    "Zoom out": "تصغير",
    "Fit image": "ملاءمة الصورة",
    "Reset zoom": "إعادة التكبير",
    # Results page and tabs
    "Overview": "نظرة عامة",
    "Network": "الشبكة",
    "Critical Paths": "المسارات الحرجة",
    "PERT": "PERT",
    "RESULTS NOT READY": "النتائج غير جاهزة",
    "PRELIMINARY": "أولية",
    "Detected from image — may change after review.": "مكتشفة من الصورة — قد تتغير بعد المراجعة.",
    "Activities detected": "الأنشطة المكتشفة",
    "Dependencies detected": "العلاقات المكتشفة",
    "Review items": "عناصر المراجعة",
    "Graph status": "حالة الرسم البياني",
    "CPM readiness": "جاهزية CPM",
    "Continue Review": "متابعة المراجعة",
    "Open Validation": "فتح التحقق",
    "Calculate Results": "حساب النتائج",
    "FINAL RESULTS": "النتائج النهائية",
    "All review decisions applied. Graph validated. CPM computed.": "تم تطبيق جميع قرارات المراجعة. تم التحقق من الرسم وحساب CPM.",
    "Authoritative result  ·  reviewed graph  ·  CPM values are not recomputed in the UI": "نتيجة معتمدة · رسم تمت مراجعته · لا يعاد حساب قيم CPM في الواجهة",
    "Export": "تصدير",
    "Report": "تقرير",
    "Copy selected": "نسخ المحدد",
    "Copy the selected path to the clipboard": "نسخ المسار المحدد إلى الحافظة",
    "No critical paths found.": "لم يتم العثور على مسارات حرجة.",
}


def translate_widget_tree(root: QWidget, value: str | UiLanguage | None) -> None:
    """Translate current visible widget text while preserving English for reversal."""
    language = normalize_language(value)
    for widget in [root, *root.findChildren(QWidget)]:
        widget.setProperty("uiLanguage", language.value)
        if isinstance(widget, (QLabel, QPushButton)):
            source = widget.property("sourceText")
            if source is None:
                source = widget.text()
                widget.setProperty("sourceText", source)
            widget.setText(UI_TEXT.get(source, source) if language == UiLanguage.ARABIC else source)
        elif isinstance(widget, QComboBox):
            for index in range(widget.count()):
                source = widget.itemData(index, Qt.ItemDataRole.UserRole + 1)
                if source is None:
                    source = widget.itemText(index)
                    widget.setItemData(index, source, Qt.ItemDataRole.UserRole + 1)
                widget.setItemText(index, UI_TEXT.get(source, source) if language == UiLanguage.ARABIC else source)


def nav_label(key: str, value: str | UiLanguage | None) -> str:
    language = normalize_language(value)
    return NAV_LABELS[language].get(key, key)


__all__ = [
    "UiLanguage",
    "UiLocale",
    "apply_ui_locale",
    "locale_for",
    "nav_label",
    "normalize_language",
    "translate_text",
    "widget_language",
]
