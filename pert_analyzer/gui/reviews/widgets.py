"""
Reusable widgets for the Human Review Center.

Includes: badges, section cards, the review categories panel, the
pending-item list, the compact evidence panel, and the non-destructive
image context overlay.
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pert_analyzer.gui.reviews.categories import (
    ReviewCategory,
    item_display_summary,
    item_key,
)
from pert_analyzer.gui.reviews.confidence import confidence_band, confidence_reason
from pert_analyzer.gui.reviews.context import (
    HighlightLine,
    HighlightRect,
)
from pert_analyzer.gui.themes.components import badge_qss
from pert_analyzer.gui.themes.palette import (
    ACCENT,
    BORDER,
    DANGER,
    SUCCESS,
    SURFACE,
    SURFACE_LIGHT,
    TEXT_MUTED,
    WARNING,
)
from pert_analyzer.gui.themes.typography import LABEL_FONT, MUTED_FONT, STATUS_FONT

logger = logging.getLogger(__name__)

BADGE_COLORS = {
    "PENDING": WARNING,
    "ACCEPTED": SUCCESS,
    "CORRECTED": ACCENT,
    "REJECTED": DANGER,
    "ERROR": DANGER,
    "WARNING": WARNING,
    "INFO": TEXT_MUTED,
    "OK": SUCCESS,
}

HIGHLIGHT_COLORS = {
    "accent": ACCENT,
    "success": SUCCESS,
    "danger": DANGER,
    "warning": WARNING,
}


def _pixmap_from_pil(img: Any) -> QPixmap:
    """Convert a PIL RGBA image to a QPixmap."""
    from PySide6.QtGui import QImage as _QImage

    data = img.tobytes("raw", "RGBA")
    qimg = _QImage(data, img.width, img.height, _QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


class BadgeLabel(QLabel):
    """Small colored status badge."""

    def __init__(self, text: str = "", status: str = "", parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFont(QFont(*MUTED_FONT))
        self.set_status(status or text)

    def set_status(self, status: str, text: str | None = None) -> None:
        label = text if text is not None else status
        self.setText(label)
        self.setStyleSheet(badge_qss(status))


class SectionCard(QFrame):
    """A card with a section header and a content body."""

    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(
            f"background-color: {SURFACE};"
            f" border: 1px solid {BORDER}; border-radius: 8px;"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(6)
        header = QLabel(title)
        header.setFont(QFont(*STATUS_FONT))
        header.setStyleSheet(f"color: {ACCENT}; border: none; background: transparent;")
        layout.addWidget(header)
        self._header = header
        self._body = QVBoxLayout()
        self._body.setSpacing(6)
        layout.addLayout(self._body, stretch=1)

    def set_title(self, title: str) -> None:
        self._header.setText(title)

    def title(self) -> str:
        return self._header.text()

    def add_widget(self, widget: QWidget, stretch: int = 0) -> None:
        self._body.addWidget(widget, stretch)

    def add_widgets(self, widgets: List[QWidget]) -> None:
        for w in widgets:
            self._body.addWidget(w)


class ReviewCategoriesPanel(QWidget):
    """Left-hand category selector with live pending counts."""

    category_selected = Signal(object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumWidth(150)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header = QLabel("Review Categories")
        header.setFont(QFont(*STATUS_FONT))
        header.setStyleSheet(f"color: {TEXT_MUTED};")
        layout.addWidget(header)

        self._buttons: dict[ReviewCategory, QPushButton] = {}
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        for category in ReviewCategory:
            btn = QPushButton()
            btn.setCheckable(True)
            btn.setFont(QFont(*LABEL_FONT))
            btn.clicked.connect(
                lambda checked=False, cat=category: self.category_selected.emit(cat)
            )
            self._buttons[category] = btn
            self._group.addButton(btn)
            layout.addWidget(btn)

        self._category: ReviewCategory = ReviewCategory.ACTIVITIES
        layout.addStretch()

    def _button_label(self, category: ReviewCategory, count: int) -> str:
        return REVIEW_LABELS[category] + (f" ({count})" if count else "")

    def set_counts(self, activities: int, dependencies: int, durations: int) -> None:
        self._buttons[ReviewCategory.ACTIVITIES].setText(self._button_label(ReviewCategory.ACTIVITIES, activities))
        self._buttons[ReviewCategory.DEPENDENCIES].setText(self._button_label(ReviewCategory.DEPENDENCIES, dependencies))
        self._buttons[ReviewCategory.DURATIONS].setText(self._button_label(ReviewCategory.DURATIONS, durations))

    def set_category(self, category: ReviewCategory, emit: bool = False) -> None:
        self._category = category
        self._buttons[category].setChecked(True)
        if emit:
            self.category_selected.emit(category)

    def current(self) -> ReviewCategory:
        return self._category


REVIEW_LABELS = {
    ReviewCategory.ACTIVITIES: "Activities",
    ReviewCategory.DEPENDENCIES: "Dependencies",
    ReviewCategory.DURATIONS: "Durations",
}


class ReviewItemList(QListWidget):
    """Center list of pending review items for the selected category."""

    item_picked = Signal(object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAlternatingRowColors(False)
        self.setStyleSheet(
            f"QListWidget {{ background-color: {SURFACE};"
            f" border: 1px solid {BORDER}; border-radius: 8px;}}"
            f"QListWidget::item {{ padding: 6px 8px; border-bottom: 1px solid {BORDER};}}"
            f"QListWidget::item:selected {{ background-color: {ACCENT}; color: white;}}"
            f"QListWidget::item:hover:!selected {{ background-color: {SURFACE_LIGHT};}}"
        )
        self._category: Optional[ReviewCategory] = None
        self._items: list[Any] = []
        self._visible_items: list[Any] = []
        self._confidence_filter: str | None = None

    def _on_current_row_changed(self, row: int) -> None:
        if 0 <= row < len(self._visible_items):
            self.item_picked.emit(self._visible_items[row])

    def populate(self, items: list[Any], category: ReviewCategory) -> None:
        self._category = category
        self._items = list(items)
        self._render_items()

    def set_confidence_filter(self, band: str | None) -> None:
        """Show all items or only items in a selected confidence band."""
        self._confidence_filter = band or None
        self._render_items()

    def _render_items(self) -> None:
        self.clear()
        visible_items = [
            item for item in self._items
            if self._confidence_filter is None
            or confidence_band(getattr(item, "confidence", 0.0)).value == self._confidence_filter
        ]
        self._visible_items = visible_items
        for item in visible_items:
            summary = item_display_summary(item, self._category)
            band = confidence_band(getattr(item, "confidence", 0.0)).value
            text = (
                f"{summary['id']}\n"
                f"{summary['label']}  ·  {summary['confidence']}  ·  {band}"
            )
            li = QListWidgetItem(text)
            li.setToolTip(confidence_reason(item))
            li.setData(Qt.ItemDataRole.UserRole, item_key(item, self._category))
            self.addItem(li)
        if visible_items:
            self.setCurrentRow(0)

    def connect_selection(self) -> None:
        self.currentRowChanged.connect(self._on_current_row_changed)

    def select_row(self, row: int) -> None:
        if 0 <= row < self.count():
            self.setCurrentRow(row)


class EvidencePanel(QWidget):
    """Compact evidence summary with confidence and technical details."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._compact = QLabel("")
        self._compact.setWordWrap(True)
        self._compact.setFont(QFont(*MUTED_FONT))
        layout.addWidget(self._compact)

        self._confidence = QLabel("")
        self._confidence.setFont(QFont(*STATUS_FONT))
        layout.addWidget(self._confidence)

        self._reason = QLabel("")
        self._reason.setWordWrap(True)
        self._reason.setStyleSheet(f"color: {TEXT_MUTED};")
        layout.addWidget(self._reason)

        self._tech_btn = QPushButton("Technical details")
        self._tech_btn.setCheckable(True)
        self._tech_btn.setFont(QFont(*MUTED_FONT))
        self._tech_btn.clicked.connect(self._toggle_technical)
        layout.addWidget(self._tech_btn)

        self._technical = QPlainTextEdit()
        self._technical.setReadOnly(True)
        self._technical.setMaximumHeight(160)
        self._technical.hide()
        layout.addWidget(self._technical)

    def _toggle_technical(self, checked: bool) -> None:
        self._technical.setVisible(checked)

    def clear(self) -> None:
        self._compact.setText("")
        self._confidence.setText("")
        self._reason.setText("")
        self._technical.clear()
        self._technical.hide()
        self._tech_btn.setChecked(False)

    def set_evidence(
        self,
        evidence: list[Any],
        confidence: float,
        reason: str = "",
        provenance: str = "",
        technical_text: str = "",
    ) -> None:
        """Populate the panel from ReviewEvidence objects."""
        self.clear()
        if not evidence and not reason:
            self._compact.setText("No evidence recorded for this item.")
            return

        lines: list[str] = []
        for ev in evidence:
            conf = float(getattr(ev, "confidence", 0.0) or 0.0)
            desc = getattr(ev, "description", "") or getattr(ev, "source", "evidence")
            lines.append(f"{_evidence_marker(conf)} {desc}")
        self._compact.setText("\n".join(lines))
        self._confidence.setText(f"Confidence:  {confidence:.0%}")
        self._reason.setText(reason or "Review the highlighted evidence before deciding.")

        if not technical_text:
            parts = []
            if provenance:
                parts.append(f"provenance: {provenance}")
            for ev in evidence:
                try:
                    parts.append(json.dumps(ev.to_dict(), indent=2, default=str))
                except Exception:
                    parts.append(f"- {getattr(ev, 'source', '?')}: {getattr(ev, 'description', '')}")
            technical_text = "\n".join(parts)

        self._technical.setPlainText(technical_text)


def _evidence_marker(confidence: float) -> str:
    if confidence >= 0.6:
        return "\u2713"
    if confidence >= 0.35:
        return "\u26a0"
    return "\u2717"


class ImageContextView(QWidget):
    """Shows the original image with a non-destructive highlight overlay.

    The overlay is drawn directly in ``paintEvent`` so the scaled image never
    participates in layout size-hint negotiation. The previous implementation
    set the scaled image on a layout-managed ``QLabel`` from ``resizeEvent``;
    once the review page became visible after an asynchronous analysis
    completed, the pixmap's size hint drove a synchronous relayout that re-fed
    ``resizeEvent``, producing unbounded native re-entrancy
    (``0xC00000FD`` stack overflow).
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._highlights: list[Any] = []
        self._placeholder_text = "No image loaded"
        self._zoom: float = 1.0
        self._pan = QPoint(0, 0)
        self._panning = False
        self._press_pos: Optional[QPoint] = None
        self._pan_origin = QPoint(0, 0)
        self.setMinimumHeight(220)

    @property
    def highlight_count(self) -> int:
        return len(self._highlights)

    @property
    def current_highlights(self) -> list[Any]:
        return list(self._highlights)

    @property
    def zoom_factor(self) -> float:
        return self._zoom

    @property
    def is_zoomed(self) -> bool:
        return self._zoom > 1.0

    def _set_placeholder(self, text: str) -> None:
        self._placeholder_text = text
        self._zoom = 1.0
        self._pan = QPoint(0, 0)
        self._panning = False
        self.update()

    def clear(self) -> None:
        self._pixmap = None
        self._highlights = []
        self._set_placeholder("No image context available")

    def set_context(
        self,
        image_path: str,
        highlights: list[Any],
        image_size: Optional[tuple[int, int]] = None,
        crop_to_highlights: bool = True,
    ) -> bool:
        """Render the original image with overlay highlights. Returns True if shown.

        ``crop_to_highlights=False`` keeps the whole drawing in frame - the
        review panels want the item read ON its symbol against the full
        drawing, not a tight sliver that loses the surrounding context.
        """
        self._highlights = [h for h in highlights if _has_geometry(h)]
        if not self._highlights:
            self.clear()
            return False
        pix = _render_overlay(
            image_path,
            self._highlights,
            image_size,
            crop_to_highlights=crop_to_highlights,
        )
        if pix is None:
            self.clear()
            return False
        self._pixmap = pix
        self._zoom = 1.0
        self._pan = QPoint(0, 0)
        self._panning = False
        self.update()
        return True

    def zoom_in(self) -> None:
        self._zoom_around(self.width() / 2, self.height() / 2, 1.25)

    def zoom_out(self) -> None:
        self._zoom_around(self.width() / 2, self.height() / 2, 1.0 / 1.25)

    def reset_zoom(self) -> None:
        self._zoom = 1.0
        self._pan = QPoint(0, 0)
        self._panning = False
        self.update()

    def focus_highlights(self) -> None:
        """Zoom and center the view on the current evidence geometry."""
        if self._pixmap is None or self._pixmap.isNull() or not self._highlights:
            return
        points: list[tuple[float, float]] = []
        for highlight in self._highlights:
            if isinstance(highlight, HighlightRect):
                points.extend([
                    (highlight.x, highlight.y),
                    (highlight.x + highlight.w, highlight.y + highlight.h),
                ])
            elif isinstance(highlight, HighlightLine):
                points.extend([(highlight.x1, highlight.y1), (highlight.x2, highlight.y2)])
        if not points:
            return
        min_x = min(p[0] for p in points)
        max_x = max(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_y = max(p[1] for p in points)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        self._zoom = 1.8
        view_w, view_h = max(1, self.width()), max(1, self.height())
        fit = min(view_w / self._pixmap.width(), view_h / self._pixmap.height())
        total = fit * self._zoom
        content_w = total * self._pixmap.width()
        content_h = total * self._pixmap.height()
        self._pan = QPoint(
            int(round(view_w / 2 - (view_w - content_w) / 2 - center_x * total)),
            int(round(view_h / 2 - (view_h - content_h) / 2 - center_y * total)),
        )
        self._clamp_pan()
        self.update()

    def _zoom_around(self, cx: float, cy: float, factor: float) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        pix_w = self._pixmap.width()
        pix_h = self._pixmap.height()
        view_w, view_h = max(1, self.width()), max(1, self.height())
        fit = min(view_w / pix_w, view_h / pix_h)
        new_zoom = min(8.0, max(1.0, self._zoom * factor))
        if abs(new_zoom - self._zoom) < 1e-6:
            return
        old_w, old_h = self._content_size()
        new_total = fit * new_zoom
        new_w, new_h = new_total * pix_w, new_total * pix_h
        if old_w > 1 and old_h > 1:
            fx = (cx - (view_w - old_w) / 2 - self._pan.x()) / old_w
            fy = (cy - (view_h - old_h) / 2 - self._pan.y()) / old_h
        else:
            fx = fy = 0.5
        self._zoom = new_zoom
        self._pan = QPoint(
            int(round((view_w - new_w) / 2 + fx * new_w - cx)),
            int(round((view_h - new_h) / 2 + fy * new_h - cy)),
        )
        self._clamp_pan()
        self.update()

    def _content_size(self) -> tuple[float, float]:
        if self._pixmap is None or self._pixmap.isNull():
            return (0, 0)
        view_w, view_h = max(1, self.width()), max(1, self.height())
        fit = min(view_w / self._pixmap.width(), view_h / self._pixmap.height())
        total = fit * self._zoom
        return total * self._pixmap.width(), total * self._pixmap.height()

    def _clamp_pan(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            self._pan = QPoint(0, 0)
            return
        view_w, view_h = max(1, self.width()), max(1, self.height())
        disp_w, disp_h = self._content_size()
        if disp_w <= view_w and disp_h <= view_h:
            self._pan = QPoint(
                int(round((view_w - disp_w) / 2)),
                int(round((view_h - disp_h) / 2)),
            )
            return
        lo_x, hi_x = -max(0, (disp_w - view_w) / 2), max(0, (disp_w - view_w) / 2)
        lo_y, hi_y = -max(0, (disp_h - view_h) / 2), max(0, (disp_h - view_h) / 2)
        self._pan.setX(max(int(lo_x), min(int(hi_x), self._pan.x())))
        self._pan.setY(max(int(lo_y), min(int(hi_y), self._pan.y())))

    def wheelEvent(self, event) -> None:  # noqa: N802
        if self._pixmap is None or self._pixmap.isNull():
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else 1.0 / 1.25
        pos = event.position().toPoint()
        self._zoom_around(pos.x(), pos.y(), factor)
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self.is_zoomed and event.button() == Qt.MouseButton.LeftButton:
            self._panning = True
            self._press_pos = event.position().toPoint()
            self._pan_origin = QPoint(self._pan)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._panning and self._press_pos is not None:
            pos = event.position().toPoint()
            delta = pos - self._press_pos
            self._pan = QPoint(self._pan_origin.x() + delta.x(), self._pan_origin.y() + delta.y())
            self._clamp_pan()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._panning and event.button() == Qt.MouseButton.LeftButton:
            self._panning = False
            self._press_pos = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.reset_zoom()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event  # unused
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(SURFACE))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(1, 1, -1, -1)
        if self._pixmap is None or self._pixmap.isNull():
            pen = QPen(QColor(BORDER))
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRoundedRect(rect, 8, 8)
            painter.setPen(QColor(TEXT_MUTED))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._placeholder_text)
            painter.end()
            return
        painter.setClipRect(rect)
        disp_w, disp_h = self._content_size()
        target_w = max(1, int(round(disp_w)))
        target_h = max(1, int(round(disp_h)))
        scaled = self._pixmap.scaled(
            target_w,
            target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = rect.x() + (rect.width() - target_w) / 2 + self._pan.x()
        y = rect.y() + (rect.height() - target_h) / 2 + self._pan.y()
        painter.drawPixmap(int(round(x)), int(round(y)), scaled)
        painter.end()


def _has_geometry(highlight: Any) -> bool:
    if isinstance(highlight, HighlightRect):
        return highlight.w > 0 and highlight.h > 0
    if isinstance(highlight, HighlightLine):
        return True
    return False


def _render_overlay(
    image_path: str,
    highlights: list[Any],
    image_size: Optional[tuple[int, int]] = None,
    crop_to_highlights: bool = True,
) -> Optional[QPixmap]:
    """Render the image with translucent highlight geometry (non-destructive)."""
    try:
        from PIL import Image, ImageDraw

        img = Image.open(image_path).convert("RGBA")
    except Exception as exc:
        logger.warning("Could not load image for context overlay: %s", exc)
        return None

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    rects: list[HighlightRect] = []
    for highlight in highlights:
        if isinstance(highlight, HighlightRect):
            rects.append(highlight)
            color = _rgba(HIGHLIGHT_COLORS.get(highlight.color, ACCENT), 40)
            outline = HIGHLIGHT_COLORS.get(highlight.color, ACCENT)
            draw.rectangle(
                [highlight.x, highlight.y, highlight.x + highlight.w, highlight.y + highlight.h],
                fill=color,
                outline=outline,
                width=2,
            )
            if highlight.label:
                _safe_text(
                    draw,
                    highlight.x,
                    max(0, highlight.y - 12),
                    highlight.label,
                    outline,
                )
        elif isinstance(highlight, HighlightLine):
            color = HIGHLIGHT_COLORS.get(highlight.color, ACCENT)
            draw.line([(highlight.x1, highlight.y1), (highlight.x2, highlight.y2)], fill=color, width=2)
            _draw_arrowhead(
                draw,
                highlight.x1,
                highlight.y1,
                highlight.x2,
                highlight.y2,
                color,
            )

    combined = Image.alpha_composite(img, overlay)

    # Crop to the relevant region when the highlights are small.
    if rects and crop_to_highlights:
        crop = _crop_box(img.size, rects)
        if crop is not None:
            combined = combined.crop(crop)

    if image_size is not None:
        combined = combined.resize(image_size, Image.Resampling.LANCZOS)

    return _pixmap_from_pil(combined)


def _rgba(hex_color: str, alpha: int) -> tuple:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)


def _safe_text(draw: Any, x: float, y: float, text: str, fill: str) -> None:
    try:
        draw.text((x + 1, y + 1), text, fill="black")
        draw.text((x, y), text, fill=fill)
    except Exception:
        pass


def _draw_arrowhead(draw: Any, x1: float, y1: float, x2: float, y2: float, fill: str) -> None:
    import math

    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    ux, uy = dx / length, dy / length
    size = 10.0
    px, py = -uy, ux
    tip = (x2 + ux * 2, y2 + uy * 2)
    base = (x2 - ux * size, y2 - uy * size)
    left = (base[0] + px * size / 2, base[1] + py * size / 2)
    right = (base[0] - px * size / 2, base[1] - py * size / 2)
    draw.polygon([tip, left, right], fill=fill)


def _crop_box(
    image_size: tuple[int, int],
    rects: list[HighlightRect],
    margin: float = 24.0,
) -> Optional[tuple[int, int, int, int]]:
    """Return a crop box around the highlights, or None if already large."""
    img_w, img_h = image_size
    min_x = min(r.x for r in rects)
    min_y = min(r.y for r in rects)
    max_x = max(r.x + r.w for r in rects)
    max_y = max(r.y + r.h for r in rects)

    crop_w = max_x - min_x + 2 * margin
    crop_h = max_y - min_y + 2 * margin
    if crop_w <= 0 or crop_h <= 0:
        return None
    if crop_w / img_w > 0.55 and crop_h / img_h > 0.55:
        return None

    left = max(0, int(min_x - margin))
    top = max(0, int(min_y - margin))
    right = min(img_w, int(max_x + margin))
    bottom = min(img_h, int(max_y + margin))
    if right - left < 8 or bottom - top < 8:
        return None
    return (left, top, right, bottom)
