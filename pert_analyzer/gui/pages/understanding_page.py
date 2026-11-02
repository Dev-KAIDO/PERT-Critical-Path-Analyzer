"""
Understanding page: preliminary reconstruction of the analyzed diagram.

The page shows the original image alongside a spatial preview of the detected
nodes and edges (driven by the review session) before the user moves into the
detailed Human Review Center.  When the reconstruction carries the detected
image coordinates, the map mirrors the drawing (A-10b); otherwise it falls
back to a layered left-to-right network map. It only renders state; all logic lives
in the backend pipeline and the review session.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from pert_analyzer.gui.pages.analysis_page import ImageView
from pert_analyzer.gui.themes.palette import (
    ACCENT,
    BORDER,
    DANGER,
    SUCCESS,
    SURFACE,
    SURFACE_LIGHT,
    TEXT,
    TEXT_MUTED,
    TEXT_SECONDARY,
    WARNING,
    qrgba,
)
from pert_analyzer.gui.themes.spacing import LG, MD, RADIUS_LG, RADIUS_MD, SM, XS
from pert_analyzer.gui.themes.typography import (
    BODY_FONT,
    BODY_SMALL_FONT,
    KPI_FONT,
    LABEL_FONT,
    MUTED_FONT,
    STATUS_FONT,
    SUBTITLE_FONT,
    TITLE_FONT,
)

# A-10c drawn nodes: the mirror reproduces the drawing's own node
# symbols - a small circle with the detected id inside and the duration
# written beneath it (brief, direct, painted like a real network).
_NODE_R = 16      # node circle radius
_NODE_W = 46      # collision footprint (circle + side air)
_NODE_H = 48      # collision footprint (circle + duration label below)


def _is_resolved(item: Any) -> bool:
    """A review item has been resolved when its status leaves PENDING."""
    status = getattr(item, "status", None)
    value = getattr(status, "value", status)
    return str(value).lower() != "pending"


def _alpha(hex_color: str, alpha: int) -> QColor:
    """Painter-side translucent colour (QSS must use qrgba instead)."""
    color = QColor(hex_color)
    color.setAlpha(max(0, min(255, int(alpha))))
    return color


class ReconstructionCanvas(QWidget):
    """Machine map of the detected reconstruction (A-10).

    Nodes are drawn as the reconstruction of the uploaded drawing: a
    small circle per node carrying its detected id, with the duration
    written beneath it and soft bezier edges with tiny direction
    arrowheads (confirmed = accent, pending = dashed amber).  A-10c
    paints the durations directly on this page (the A-6 tooltips-only
    rule still governs every other canvas).

    A-10b: the layout mirrors the uploaded drawing - nodes sit at their
    real detected image coordinates when available - and falls back to
    a layered left-to-right network map when geometry is missing.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._nodes: list[dict[str, Any]] = []
        self._edges: list[tuple[int, int, bool]] = []
        self._durations: dict[int, str] = {}
        self._positions: list[tuple[float, float]] = []
        self._id_to_index: dict[str, int] = {}
        self._mode = "empty"  # "mirror" | "layers" | "empty"
        self._t = 0.0
        self._halo = True
        self._beat = QTimer(self)
        self._beat.setInterval(34)
        self._beat.timeout.connect(self._tick)
        self.setMouseTracking(True)  # hover provides the d= tooltips
        self.setMinimumHeight(230)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_reconstruction(
        self,
        activities: list[Any],
        dependencies: list[Any],
        durations: list[Any],
        geometry: dict[str, tuple[float, float]] | None = None,
    ) -> int:
        """Rebuild the preview from review items. Returns node count.

        ``geometry`` carries real image coordinates (A-10b mirror of the
        drawing); when a point exists for every chip the map reproduces
        the detected positions, otherwise it falls back to layers.
        """
        self._nodes = []
        self._edges = []
        self._durations = {}
        self._id_to_index = {}

        for act in activities or []:
            geometric = getattr(act, "geometric_node_id", None)
            activity_id = (
                getattr(act, "activity_id", None)
                or getattr(act, "current_activity_id", None)
                or getattr(act, "proposed_activity_id", None)
            )
            node_id = activity_id or geometric or "?"
            index = len(self._nodes)
            self._nodes.append(
                {
                    "id": str(node_id),
                    "label": str(node_id),
                    "resolved": _is_resolved(act),
                    "point": (geometry or {}).get(
                        str(geometric) if geometric is not None else str(node_id)
                    ),
                }
            )
            for key in (geometric, activity_id):
                if key is not None:
                    self._id_to_index.setdefault(str(key), index)

        for dep in dependencies or []:
            source = getattr(dep, "current_source_id", None) or getattr(
                dep, "source_node_id", None
            )
            target = getattr(dep, "current_target_id", None) or getattr(
                dep, "target_node_id", None
            )
            if source is None or target is None:
                continue
            si = self._id_to_index.get(str(source))
            ti = self._id_to_index.get(str(target))
            if si is not None and ti is not None:
                self._edges.append((si, ti, _is_resolved(dep)))  # type: ignore[arg-type]

        for dur in durations or []:
            geometric = getattr(dur, "geometric_node_id", None)
            activity_id = (
                getattr(dur, "activity_id", None)
                or getattr(dur, "current_activity_id", None)
                or getattr(dur, "proposed_activity_id", None)
            )
            index = None
            for key in (geometric, activity_id):
                if key is not None:
                    index = self._id_to_index.get(str(key))
                    if index is not None:
                        break
            if index is None:
                continue
            current = getattr(dur, "current_duration", None)
            if _is_resolved(dur) and current is not None:
                self._durations[index] = f"d={current:g}"
            else:
                self._durations[index] = "d=?"

        self._t = 0.0
        self._layout()
        self.setToolTip("")
        self._sync_beat()
        self.update()
        return len(self._nodes)

    def clear(self) -> None:
        self._nodes = []
        self._edges = []
        self._durations = {}
        self._positions = []
        self._id_to_index = {}
        self._mode = "empty"
        self.setToolTip("")
        self._layout()
        self._sync_beat()
        self.update()

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def pending_count(self) -> int:
        return sum(1 for node in self._nodes if not node["resolved"])

    # ------------------------------------------------------------------
    # Breathing halo (A-10 chamber language)
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        self._t = (self._t + 0.034) % 3.4
        self.update()

    def _sync_beat(self) -> None:
        """The halo breathes only while the map has something to show."""
        if self.isVisible() and bool(self._nodes):
            self._beat.start()
        else:
            self._beat.stop()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._sync_beat()

    def hideEvent(self, event) -> None:  # noqa: N802
        super().hideEvent(event)
        self._sync_beat()

    # ------------------------------------------------------------------
    # Hover: the only place where d= durations surface
    # ------------------------------------------------------------------

    def _hit_index(self, pos: QPointF) -> Optional[int]:
        px, py = pos.x(), pos.y()
        for index, (x, y) in enumerate(self._positions):
            if math.hypot(px - x, py - y) <= _NODE_R + 2.0:
                return index
        return None

    def _tooltip_text(self, index: int) -> str:
        node = self._nodes[index]
        lines = [node["id"]]
        duration = self._durations.get(index)
        if duration is not None:
            lines.append(duration)
        elif not node["resolved"]:
            lines.append("d=?")
        return "\n".join(lines)

    def hoverMoveEvent(self, event) -> None:  # noqa: N802
        index = self._hit_index(event.position())
        tooltip = self._tooltip_text(index) if index is not None else ""
        if self.toolTip() != tooltip:
            self.setToolTip(tooltip)
        super().hoverMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self.setToolTip("")
        super().leaveEvent(event)

    # ------------------------------------------------------------------
    # Layout + painting
    # ------------------------------------------------------------------

    def _layout(self) -> None:
        """A-10b: place chips as a mirror of the drawing when real
        detected coordinates exist, else a layered network map."""
        self._positions = []
        count = len(self._nodes)
        if count == 0:
            self._mode = "empty"
            return
        if self._layout_mirror():
            self._mode = "mirror"
        else:
            self._mode = "layers"
            self._layout_layers()
        self._relax()

    def _layout_mirror(self) -> bool:
        """Scale real drawing coordinates into the canvas with a true
        aspect ratio (y grows down exactly like the uploaded image),
        keeping the map in frame.  False when the coordinates carry no
        spatial value (everything inside one chip footprint)."""
        points: list[tuple[float, float]] = []
        for node in self._nodes:
            point = node.get("point")
            if point is None:
                return False
            points.append((float(point[0]), float(point[1])))
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        if span_x <= _NODE_W and span_y <= _NODE_H:
            return False
        pad = 56.0  # node footprint + breathing room in the frame: the
        # healthy headroom relaxation needs to unstack tight columns
        avail_x = max(self.width() - 2.0 * pad, _NODE_W)
        avail_y = max(self.height() - 2.0 * pad, _NODE_W)
        candidates: list[float] = []
        if span_x > 1e-6:
            candidates.append(avail_x / span_x)
        if span_y > 1e-6:
            candidates.append(avail_y / span_y)
        scale = min(candidates) if candidates else 1.0
        cx, cy = self.width() / 2.0, self.height() / 2.0
        center_x = (max(xs) + min(xs)) / 2.0
        center_y = (max(ys) + min(ys)) / 2.0
        self._positions = [
            (cx + (x - center_x) * scale, cy + (y - center_y) * scale)
            for x, y in points
        ]
        return True

    def _layout_layers(self) -> None:
        """Fallback map: columns follow network depth (Kahn levels),
        long levels wrap into half-step sub-columns."""
        levels = self._depth_levels()
        columns: list[list[int]] = [[] for _ in range(max(1, len(levels)))]
        for index, level in enumerate(levels):
            columns[max(0, min(level, len(columns) - 1))].append(index)
        pad = 56.0
        row_step = _NODE_H * 1.66
        cap_rows = max(3, int((self.height() - 2.0 * pad) // row_step) or 3)
        width_units = 0.0
        shape: dict[int, tuple[int, int]] = {}  # level -> (subcols, rows)
        for column, indexes in enumerate(columns):
            if not indexes:
                continue
            subcols = max(1, int(math.ceil(len(indexes) / cap_rows)))
            shape[column] = (subcols, min(len(indexes), cap_rows))
            width_units = max(width_units, column + 0.42 * (subcols - 1))
        cx, cy = self.width() / 2.0, self.height() / 2.0
        unit_x = max(
            (self.width() - 2.0 * pad - _NODE_W) / max(width_units, 1.0),
            _NODE_W,
        )
        points: dict[int, tuple[float, float]] = {}
        for column, indexes in enumerate(columns):
            shape_here = shape.get(column)
            if shape_here is None:
                continue
            subcols, _rows = shape_here
            for k, index in enumerate(indexes):
                sub = min(k // cap_rows, subcols - 1)
                row = k % cap_rows
                rows_here = min(len(indexes) - sub * cap_rows, cap_rows)
                x = cx + ((column + 0.42 * sub) - width_units / 2.0) * unit_x
                if rows_here > 1:
                    step = min(
                        row_step, (self.height() - 2.0 * pad) / (rows_here - 1)
                    )
                    y = cy + (row - (rows_here - 1) / 2.0) * step
                else:
                    y = cy
                points[index] = (x, y)
        self._positions = [
            points.get(i, (cx, cy)) for i in range(len(self._nodes))
        ]

    def _depth_levels(self) -> list[int]:
        """Column depth per node: Kahn levels; closed loops settle for
        the predecessor-based approximation."""
        count = len(self._nodes)
        preds: list[set[int]] = [set() for _ in range(count)]
        succs: list[set[int]] = [set() for _ in range(count)]
        for si, ti, _resolved in self._edges:
            if si < count and ti < count and si != ti:
                succs[si].add(ti)
                preds[ti].add(si)
        indegree = [len(preds[i]) for i in range(count)]
        level = [-1] * count
        frontier = [i for i in range(count) if indegree[i] == 0]
        depth = 0
        while frontier:
            nxt: list[int] = []
            for node in frontier:
                level[node] = depth
            for node in frontier:
                for child in sorted(succs[node]):
                    indegree[child] -= 1
                    if indegree[child] == 0:
                        nxt.append(child)
            frontier = nxt
            depth += 1
        for node in range(count):
            if level[node] >= 0:
                continue
            options = [level[p] + 1 for p in sorted(preds[node]) if level[p] >= 0]
            level[node] = max(options, default=0)
        return [max(lv, 0) for lv in level]

    def _relax(self) -> None:
        """Deterministic nudges so chips never overlap; positions stay
        pinned inside the frame (plain pixel space, A-10b).  Each sweep
        splits EVERY violating pair once, so cascades - a split shoving
        a neighbour onto someone else - settle across sweeps instead of
        exhausting a restart budget.  Splits part horizontally when the
        frame has room, which keeps the drawing's vertical reading
        intact and ends the stacked-column bounce."""
        if not self._positions:
            return
        pad_x = _NODE_R + 11.0   # clearance for the node circle
        pad_y = _NODE_R + 20.0   # clearance for the duration label below
        min_x = float(_NODE_W)
        min_y = float(_NODE_H)
        n = len(self._positions)
        for _ in range(400):
            violated = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = self._positions[i][0] - self._positions[j][0]
                    dy = self._positions[i][1] - self._positions[j][1]
                    if abs(dx) < min_x and abs(dy) < min_y:
                        violated = True
                        self._split_pair(i, j, dx, dy)
            self._clamp_frame(pad_x, pad_y)
            if not violated:
                return
        # Worst case: leave the layout as-is; painting remains valid.

    def _clamp_frame(self, pad_x: float, pad_y: float) -> None:
        """Keep every chip fully inside the drawn frame."""
        width, height = self.width(), self.height()
        self._positions = [
            (
                min(max(x, pad_x), max(width - pad_x, pad_x)),
                min(max(y, pad_y), max(height - pad_y, pad_y)),
            )
            for (x, y) in self._positions
        ]

    def _split_pair(self, i: int, j: int, dx: float, dy: float) -> None:
        """Push one too-close pair apart in a single midpoint move, on
        the axis that actually has room inside the frame.  Horizontal
        parting is preferred: it preserves the mirror's vertical
        reading of the drawing and never fights the top/bottom clamps
        where stacked columns previously bounced for ever."""
        pad_x = _NODE_R + 11.0
        pad_y = _NODE_R + 20.0
        xi, yi = self._positions[i]
        xj, yj = self._positions[j]
        need_x = (_NODE_W - abs(dx)) / 2.0 + 0.5
        need_y = (_NODE_H - abs(dy)) / 2.0 + 0.5
        width, height = self.width(), self.height()
        if dx >= 0:
            room_x = (width - pad_x - xi) + (xj - pad_x)
        else:
            room_x = (xi - pad_x) + (width - pad_x - xj)
        if dy >= 0:
            room_y = (height - pad_y - yi) + (yj - pad_y)
        else:
            room_y = (yi - pad_y) + (height - pad_y - yj)
        if room_x >= 2.0 * need_x:
            sign = 1.0 if dx >= 0 else -1.0
            self._positions[i] = (xi + need_x * sign, yi)
            self._positions[j] = (xj - need_x * sign, yj)
        elif room_y >= 2.0 * need_y:
            sign = 1.0 if dy >= 0 else -1.0
            self._positions[i] = (xi, yi + need_y * sign)
            self._positions[j] = (xj, yj - need_y * sign)
        else:
            # both axes are pinned against the frame: fall back to the
            # axis that clears in the fewest nudges
            if need_x / 8.0 <= need_y / 7.0:
                sign = 1.0 if dx >= 0 else -1.0
                self._positions[i] = (xi + need_x * sign, yi)
                self._positions[j] = (xj - need_x * sign, yj)
            else:
                sign = 1.0 if dy >= 0 else -1.0
                self._positions[i] = (xi, yi + need_y * sign)
                self._positions[j] = (xj, yj - need_y * sign)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._layout()

    def paintEvent(self, event) -> None:  # noqa: N802
        del event  # unused
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(SURFACE))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        frame = self.rect().adjusted(1, 1, -1, -1)
        painter.setPen(QPen(QColor(BORDER), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(frame, RADIUS_LG, RADIUS_LG)

        if not self._nodes:
            painter.setFont(QFont(*BODY_FONT))
            painter.setPen(QColor(TEXT_MUTED))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "No reconstruction available yet",
            )
            painter.end()
            return

        painter.setClipRect(frame)

        # A-10 chamber: a quiet pool of light the machine map floats in.
        self._paint_halo(painter)

        # soft bezier arcs first, so chips cover their endpoints
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for index, (si, ti, resolved) in enumerate(self._edges):
            if si >= len(self._positions) or ti >= len(self._positions):
                continue
            color = ACCENT if resolved else WARNING
            pen = QPen(_alpha(color, 200 if resolved else 215), 1.5)
            if not resolved:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(self._edge_path(si, ti, index))
            # A-10b: tiny direction arrowhead at the target end
            painter.setBrush(_alpha(color, 225))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(self._arrow_head(si, ti))

        painter.setFont(QFont(*BODY_SMALL_FONT))
        for index, node in enumerate(self._nodes):
            self._paint_node(painter, index, node)

        painter.end()

    def _paint_halo(self, painter: QPainter) -> None:
        """Breathing pool of light behind the map (no orbit guides in
        the mirror layout - the drawing itself is the reference)."""
        if not self._halo:
            return
        breathe = 0.5 + 0.5 * math.sin(2.0 * math.pi * self._t / 3.4)
        cx, cy = self.width() / 2.0, self.height() / 2.0
        glow_radius = max(self.width(), self.height()) * 0.62
        grad = QRadialGradient(QPointF(cx, cy), glow_radius)
        grad.setColorAt(0.0, _alpha(ACCENT, int(20 + 16 * breathe)))
        grad.setColorAt(1.0, _alpha(ACCENT, 0))
        painter.fillRect(self.rect(), grad)

    def _arrow_head(self, si: int, ti: int) -> QPolygonF:
        """Tiny direction triangle just outside the target chip, aimed
        along the straight source-to-target line (A-10b)."""
        sx, sy = self._positions[si]
        tx, ty = self._positions[ti]
        dx, dy = tx - sx, ty - sy
        dist = math.hypot(dx, dy) or 1.0
        ux, uy = dx / dist, dy / dist
        # retreat until the triangle clears the target's footprint;
        # the duration label hangs below, so flowing up needs more air
        wall_x = (_NODE_R + 4.0) / abs(ux) if abs(ux) > 1e-9 else math.inf
        wall_y = (
            (_NODE_R + 18.0) if uy < 0.0 else (_NODE_R + 4.0)
        ) / abs(uy) if abs(uy) > 1e-9 else math.inf
        retreat = min(wall_x, wall_y)
        base_x = tx - ux * retreat
        base_y = ty - uy * retreat
        px, py = -uy, ux  # perpendicular of the line
        return QPolygonF(
            [
                QPointF(base_x, base_y),
                QPointF(
                    base_x - ux * 6.0 + px * 4.0, base_y - uy * 6.0 + py * 4.0
                ),
                QPointF(
                    base_x - ux * 6.0 - px * 4.0, base_y - uy * 6.0 - py * 4.0
                ),
            ]
        )

    def _edge_path(self, si: int, ti: int, index: int) -> QPainterPath:
        """Alternating quad-bezier arc so parallel connections split."""
        sx, sy = self._positions[si]
        tx, ty = self._positions[ti]
        dx, dy = tx - sx, ty - sy
        dist = math.hypot(dx, dy) or 1.0
        bow = dist * 0.16 * (1.0 if index % 2 == 0 else -1.0)
        mx = (sx + tx) / 2.0 - dy / dist * bow
        my = (sy + ty) / 2.0 + dx / dist * bow
        path = QPainterPath(QPointF(sx, sy))
        path.quadTo(QPointF(mx, my), QPointF(tx, ty))
        return path

    def _paint_node(
        self, painter: QPainter, index: int, node: dict[str, Any]
    ) -> None:
        """Draw one node as the drawing's own symbol: a small circle
        with the detected id inside and the duration beneath (A-10c)."""
        x, y = self._positions[index]
        center = QPointF(x, y)
        resolved = bool(node["resolved"])
        family = ACCENT if resolved else WARNING

        # opaque base + a whisper of the family wash, ring on top
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(SURFACE_LIGHT))
        painter.drawEllipse(center, _NODE_R, _NODE_R)
        painter.setBrush(_alpha(family, 24))
        painter.drawEllipse(center, _NODE_R, _NODE_R)
        pen = QPen(QColor(family), 1.4)
        if not resolved:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(center, _NODE_R - 0.7, _NODE_R - 0.7)

        # detected id inside the symbol, elided like any drawing label
        label = QFontMetrics(painter.font()).elidedText(
            node["label"], Qt.TextElideMode.ElideMiddle, _NODE_R * 2.0 - 10.0
        )
        painter.setPen(QColor(TEXT))
        painter.drawText(
            QRectF(
                x - _NODE_R + 5.0,
                y - _NODE_R + 2.0,
                _NODE_R * 2.0 - 10.0,
                _NODE_R * 2.0 - 4.0,
            ),
            Qt.AlignmentFlag.AlignCenter,
            label,
        )

        # duration beneath the symbol, painted like a real network label
        duration = self._durations.get(index)
        if duration is None and not resolved:
            duration = "d=?"
        if duration is not None:
            painter.setFont(QFont(*MUTED_FONT))
            painter.setPen(
                QColor(TEXT_MUTED)
                if duration == "d=?"
                else _alpha(family, 235)
            )
            painter.drawText(
                QRectF(
                    x - _NODE_R,
                    y + _NODE_R + 1.0,
                    _NODE_R * 2.0,
                    14.0,
                ),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                duration,
            )


def _status_color(status: Any) -> str:
    value = getattr(status, "value", status)
    return {
        "VALIDATED": SUCCESS,
        "VALID": SUCCESS,
        "INVALID": DANGER,
        "REVIEW REQUIRED": WARNING,
        "BLOCKED_REVIEW": WARNING,
    }.get(str(value), TEXT_MUTED)


class UnderstandingPage(QWidget):
    """Diagram understanding / preliminary reconstruction workspace."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._session: Any = None
        self._loaded_image_path: Optional[str] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(SM)

        self._empty_label = QLabel("")
        self._empty_label.setWordWrap(True)
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setFont(QFont(*LABEL_FONT))
        self._empty_label.setStyleSheet(f"color: {TEXT_MUTED};")
        root.addWidget(self._empty_label, stretch=1)
        self._empty_label.hide()

        self._content = QWidget()
        self._content.setStyleSheet("background: transparent; border: none;")
        content = QVBoxLayout(self._content)
        content.setContentsMargins(24, 24, 24, 24)
        content.setSpacing(MD)
        root.addWidget(self._content, stretch=1)

        # header row: title + subtitle + reconstruction banner
        header = QHBoxLayout()
        header.setSpacing(MD)
        header.setContentsMargins(0, 0, 0, 0)
        title_col = QVBoxLayout()
        title_col.setSpacing(2)

        self._title_label = QLabel("Diagram Understanding")
        self._title_label.setFont(QFont(*TITLE_FONT))
        self._title_label.setStyleSheet(f"color: {TEXT}; border: none; background: transparent;")
        title_col.addWidget(self._title_label)

        subtitle = QLabel(
            "Review the preliminary reconstruction of the detected diagram before final review."
        )
        subtitle.setFont(QFont(*SUBTITLE_FONT))
        subtitle.setStyleSheet(
            f"color: {TEXT_MUTED}; border: none; background: transparent;"
        )
        title_col.addWidget(subtitle)
        header.addLayout(title_col, stretch=1)

        self._banner = QLabel("  \u25C8  PRELIMINARY RECONSTRUCTION  ")
        self._banner.setFont(QFont(*BODY_SMALL_FONT))
        self._banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._banner.setStyleSheet(
            f"color: {ACCENT}; background-color: {qrgba(ACCENT, 24)};"
            f" border: 1px solid {qrgba(ACCENT, 68)}; border-radius: 6px;"
            f" padding: 4px 10px; font-weight: 600;"
        )
        header.addWidget(self._banner, alignment=Qt.AlignmentFlag.AlignTop)
        content.addLayout(header)
        content.addSpacing(SM)

        # body: original image left, metrics card right
        body = QWidget()
        body.setStyleSheet("background: transparent; border: none;")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(MD)

        left = QVBoxLayout()
        left.setSpacing(MD)
        self._image_view = ImageView()
        self._image_view.setMinimumHeight(200)
        left.addWidget(self._image_view, stretch=1)
        body_layout.addLayout(left, stretch=3)

        self._metrics_card = QFrame()
        self._metrics_card.setObjectName("understandingMetrics")
        self._metrics_card.setStyleSheet(
            f"QFrame#understandingMetrics {{ background-color: {SURFACE_LIGHT};"
            f" border: 1px solid {BORDER}; border-radius: {RADIUS_MD}px; }}"
        )
        metrics_layout = QVBoxLayout(self._metrics_card)
        metrics_layout.setContentsMargins(LG, MD, LG, MD)
        metrics_layout.setSpacing(MD)

        grid = QGridLayout()
        grid.setHorizontalSpacing(MD)
        grid.setVerticalSpacing(LG)
        self._metric_cells: dict[str, tuple[QLabel, QLabel]] = {}
        for col, (key, caption) in enumerate(
            (("activities", "Activities detected"), ("candidate_deps", "Candidate connections"))
        ):
            grid.addLayout(self._metric_cell(key, caption), 0, col)
        for col, (key, caption) in enumerate(
            (("confirmed_deps", "Confirmed dependencies"), ("review_required", "Review required"))
        ):
            grid.addLayout(self._metric_cell(key, caption), 1, col)
        for col, (key, caption) in enumerate(
            (("review_items", "Total review items"), ("pending", "Pending resolution"))
        ):
            grid.addLayout(self._metric_cell(key, caption), 2, col)
        metrics_layout.addLayout(grid)

        metrics_layout.addSpacing(XS)
        metrics_layout.addWidget(self._separator())

        self._validation_row = QLabel("")
        self._validation_row.setWordWrap(True)
        self._validation_row.setTextFormat(Qt.TextFormat.RichText)
        metrics_layout.addWidget(self._validation_row)

        self._cpm_row = QLabel("")
        self._cpm_row.setWordWrap(True)
        self._cpm_row.setTextFormat(Qt.TextFormat.RichText)
        metrics_layout.addWidget(self._cpm_row)

        metrics_layout.addStretch()
        self._metrics_card.setFixedWidth(300)
        body_layout.addWidget(self._metrics_card, alignment=Qt.AlignmentFlag.AlignTop)
        content.addWidget(body, stretch=5)

        # bottom: reconstruction canvas with legend
        canvas_section = QWidget()
        canvas_section.setStyleSheet("background: transparent; border: none;")
        canvas_layout = QVBoxLayout(canvas_section)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(SM)

        canvas_header = QHBoxLayout()
        canvas_header.setSpacing(SM)
        canvas_title = QLabel("Detected structure")
        canvas_title.setFont(QFont(*LABEL_FONT))
        canvas_title.setStyleSheet(f"color: {TEXT}; border: none; background: transparent;")
        canvas_header.addWidget(canvas_title)
        canvas_header.addStretch()
        # A-10: chip-family legend, same visual language as the map chips.
        self._legend_chips: list[QLabel] = [
            self._legend_chip("\u25CF", "Confirmed", ACCENT),
            self._legend_chip("\u25CB", "Pending review", WARNING),
        ]
        for chip in self._legend_chips:
            canvas_header.addWidget(chip)
        canvas_layout.addLayout(canvas_header)

        self._canvas = ReconstructionCanvas()
        self._canvas.setMinimumHeight(200)
        canvas_layout.addWidget(self._canvas, stretch=1)
        content.addWidget(canvas_section, stretch=4)

        self._content.hide()
        self._show_empty(
            "No analysis yet. Run an analysis on the Analyze Diagram page."
        )

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _metric_cell(self, key: str, caption: str) -> QVBoxLayout:
        cell = QVBoxLayout()
        cell.setSpacing(2)
        value = QLabel("\u2014")
        value.setFont(QFont(*KPI_FONT))
        value.setStyleSheet("border: none; background: transparent;")
        cell.addWidget(value)
        label = QLabel(caption)
        label.setFont(QFont(*BODY_SMALL_FONT))
        label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; border: none; background: transparent;"
        )
        cell.addWidget(label)
        self._metric_cells[key] = (value, label)
        return cell

    def _separator(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"color: {BORDER}; background-color: {BORDER};")
        line.setFixedHeight(1)
        return line

    def _legend_chip(self, glyph: str, text: str, color: str) -> QLabel:
        chip = QLabel(f"{glyph}  {text}")
        chip.setFont(QFont(*BODY_SMALL_FONT))
        chip.setStyleSheet(
            f"color: {TEXT}; background-color: {qrgba(color, 20)};"
            f" border: 1px solid {qrgba(color, 78)}; border-radius: 10px;"
            f" padding: 2px 10px;"
        )
        return chip

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def title(self) -> str:
        return self._title_label.text()

    @property
    def banner_text(self) -> str:
        return self._banner.text()

    def canvas(self) -> ReconstructionCanvas:
        return self._canvas

    @staticmethod
    def _geometry_map(review_session: Any) -> dict[str, tuple[float, float]]:
        """A-10b: real detected coordinates per geometric node id,
        taken from ``position`` or the bounding-box centre."""
        reconstruction = getattr(review_session, "reconstruction", None)
        activities = getattr(reconstruction, "activities", None) or []
        found: dict[str, tuple[float, float]] = {}
        for act in activities:
            identity = getattr(act, "geometric_node_id", None)
            if identity is None:
                continue
            point = getattr(act, "position", None)
            if point is None:
                box = getattr(act, "bounding_box", None)
                if box is not None:
                    try:
                        point = (
                            float(box.x) + float(box.width) / 2.0,
                            float(box.y) + float(box.height) / 2.0,
                        )
                    except (AttributeError, TypeError, ValueError):
                        point = None
            if point is not None:
                found[str(identity)] = (float(point[0]), float(point[1]))
        return found

    def set_reconstruction(self, session: Any) -> None:
        """Render content for a session with a workflow + review session."""
        review_session = session.review_session
        activities = getattr(review_session, "activities", None) or []
        dependencies = getattr(review_session, "dependencies", None) or []
        durations = getattr(review_session, "durations", None) or []
        self._canvas.set_reconstruction(
            activities,
            dependencies,
            durations,
            self._geometry_map(review_session),
        )

        image_path = getattr(session, "current_image_path", None)
        if image_path and image_path != self._loaded_image_path:
            self._image_view.set_image(image_path)
            self._loaded_image_path = image_path

        summary = getattr(session, "review_summary", None) or {}
        total_activities = summary.get("total_activities")
        if total_activities is None:
            total_activities = len(activities)
        total_dependencies = summary.get("total_dependencies")
        if total_dependencies is None:
            total_dependencies = len(dependencies)

        # Count confirmed (accepted/corrected) vs pending dependencies
        confirmed_deps = 0
        review_required_deps = 0
        for dep in dependencies:
            status_val = getattr(dep, "status", None)
            val = getattr(status_val, "value", status_val) if status_val is not None else "pending"
            if str(val).lower() in ("accepted", "corrected"):
                confirmed_deps += 1
            else:
                review_required_deps += 1

        # Count confirmed vs pending activities
        confirmed_activities = 0
        review_required_activities = 0
        for act in activities:
            status_val = getattr(act, "status", None)
            val = getattr(status_val, "value", status_val) if status_val is not None else "pending"
            if str(val).lower() in ("accepted", "corrected"):
                confirmed_activities += 1
            else:
                review_required_activities += 1

        self._metric_cells["activities"][0].setText(str(total_activities))
        self._metric_cells["candidate_deps"][0].setText(str(total_dependencies))
        self._metric_cells["confirmed_deps"][0].setText(str(confirmed_deps))
        self._metric_cells["review_required"][0].setText(str(review_required_deps))
        self._metric_cells["review_items"][0].setText(str(session.review_item_total()))
        self._metric_cells["pending"][0].setText(str(session.pending_review_total()))

        status = session.validation_status
        status_value = getattr(status, "value", status)
        self._validation_row.setText(
            f"<b>Graph validation:</b> "
            f'<font color="{_status_color(status)}">{status_value}</font>'
        )
        eligible = bool(getattr(session, "cpm_eligible", False))
        self._cpm_row.setText(
            f"<b>CPM ready:</b> "
            f'<font color="{SUCCESS if eligible else TEXT_MUTED}">'
            f'{"Yes" if eligible else "Not yet"}</font>'
        )

    # ------------------------------------------------------------------
    # Refresh API
    # ------------------------------------------------------------------

    def refresh(self, session: Any) -> None:
        self._session = session
        if session is None or getattr(session, "workflow", None) is None:
            self._content.hide()
            self._empty_label.show()
            self._empty_label.setText(
                "No analysis yet. Run an analysis on the Analyze Diagram page."
            )
            return
        review_session = getattr(session, "review_session", None)
        if review_session is None:
            self._content.hide()
            self._empty_label.show()
            self._empty_label.setText(
                "No reconstruction data is available for this analysis."
            )
            return
        self._empty_label.hide()
        self._content.show()
        self.set_reconstruction(session)

    def _show_empty(self, message: str) -> None:
        self._content.hide()
        self._empty_label.show()
        self._empty_label.setText(message)