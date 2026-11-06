"""
Detail panels for activity, dependency, and duration review items.

Each panel renders item details, evidence, image context overlay, and
review actions. All decisions are emitted as a ReviewActionRequest and
applied by the ReviewPage through the backend ReviewSession; no review
logic is duplicated here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pert_analyzer.gui.reviews import context as review_context
from pert_analyzer.gui.reviews.actions import (
    geometric_node_label,
    is_duplicate_activity_id,
    validate_activity_id,
    validate_duration_triplet,
)
from pert_analyzer.gui.reviews.categories import ReviewCategory
from pert_analyzer.gui.results.data import duration_item_candidates
from pert_analyzer.gui.reviews.widgets import (
    BadgeLabel,
    EvidencePanel,
    ImageContextView,
    SectionCard,
)
from pert_analyzer.gui.themes.palette import (
    ACCENT,
    BORDER,
    DANGER,
    SUCCESS,
    SURFACE_LIGHT,
    TEXT_MUTED,
)
from pert_analyzer.gui.themes.typography import (
    LABEL_FONT,
    MUTED_FONT,
    SUBTITLE_FONT,
    TITLE_FONT,
)


@dataclass
class ReviewActionRequest:
    """A user decision on a single review item, ready for the backend."""

    category: ReviewCategory
    item: Any
    action: str
    corrected_value: Any = None
    # A-15: three-point corrections ride the request for duration items.
    corrected_optimistic: Any = None
    corrected_pessimistic: Any = None
    comment: str = ""
    reason: str = ""

    @property
    def item_id(self) -> str:
        if self.category == ReviewCategory.ACTIVITIES:
            return getattr(self.item, "geometric_node_id", "?")
        if self.category == ReviewCategory.DEPENDENCIES:
            return getattr(self.item, "arrow_id", "?")
        return getattr(self.item, "geometric_node_id", getattr(self.item, "activity_id", "?"))


def _status_name(item: Any) -> str:
    status = getattr(item, "status", None)
    return getattr(status, "value", str(status))


def _info_row(layout: QVBoxLayout, key: str, value: str, value_color: str = "") -> QLabel:
    """Add a "Key: value" row and return the value label."""
    value_label = QLabel(value)
    value_label.setFont(QFont(*LABEL_FONT))
    value_label.setWordWrap(True)
    if value_color:
        value_label.setStyleSheet(f"color: {value_color};")
    key_label = QLabel(f"<b>{key}:</b>")
    layout.addWidget(_bordered_row(key_label, value_label))
    return value_label


def _bordered_row(key_label: QLabel, value_label: QLabel) -> QWidget:
    row = QWidget()
    row.setStyleSheet(
        f"background-color: {SURFACE_LIGHT};"
        f" border-radius: 4px; padding: 4px 8px;"
    )
    layout = QHBoxLayout(row)
    layout.setContentsMargins(8, 4, 8, 4)
    layout.setSpacing(8)
    key_label.setFont(QFont(*MUTED_FONT))
    key_label.setStyleSheet(f"color: {TEXT_MUTED};")
    layout.addWidget(key_label)
    layout.addWidget(value_label, stretch=1)
    return row


def _section_action_row(
    parent: QWidget,
    actions: List["tuple[str, str]"],  # (text, action_name)
    on_click,
) -> QWidget:
    """Build a row of action buttons wired to *on_click(action_name, text)."""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    for text, name in actions:
        btn = QPushButton(text)
        btn.setFont(QFont(*LABEL_FONT))
        btn.clicked.connect(lambda checked=False, n=name: on_click(n, text))
        layout.addWidget(btn)
    layout.addStretch()
    return row


def _geometry_highlights(item: Any, category: ReviewCategory, session: Any) -> list[Any]:
    recon = _reconstruction(session)
    if recon is None:
        return []
    return review_context.highlights_for_item(item, category, recon)


def _reconstruction(session: Any) -> Any:
    """Resolve the immutable reconstruction referenced by the session."""
    workflow = getattr(session, "workflow", None)
    if workflow is None:
        return None
    pipeline_result = getattr(workflow, "pipeline_result", None)
    recon = getattr(pipeline_result, "_reconstruction", None) if pipeline_result is not None else None
    if recon is None:
        review_session = getattr(workflow, "review_session", None)
        recon = getattr(review_session, "reconstruction", None) if review_session is not None else None
    return recon


class _BaseReviewPanel(QWidget):
    """Shared layout for detail panels.

    A-11 decision-first order: the drawing strip (your own image with the
    item highlighted) sits at the top of the card, right under the title;
    the single decision row follows it; technical details live below the
    fold.  ``_render_context`` is shared by all panels and auto-focuses
    the strip on the item's highlight when one is rendered.
    """

    decision_made = Signal(object)

    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._item: Any = None
        self._session: Any = None
        self._decode_commit = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # We keep the inner card simple; title is rendered by parent.
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        root.addWidget(self._scroll)

        card = SectionCard(title)
        self._scroll.setWidget(card)
        self._card = card
        self._body = card._body

    def _set_friendly_title(self, text: str) -> None:
        self._card.set_title(text)

    def _add_summary_hint(self, text: str) -> QLabel:
        """Decision copy, spoken in drawing language.  Inserted directly
        under the drawing strip (body index 1, after the strip itself)."""
        old = getattr(self, "_summary_label", None)
        if old is not None:
            old.setParent(None)
            old.deleteLater()
        label = QLabel(text)
        label.setWordWrap(True)
        label.setFont(QFont(*MUTED_FONT))
        label.setStyleSheet(f"color: {ACCENT};")
        self._summary_label = label
        self._body.insertWidget(1, label)
        return label

    def _add_divider(self) -> QWidget:
        """A thin rule between the decision zone and the details."""
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(f"background-color: {BORDER}; border: none;")
        self._body.addWidget(divider)
        return divider

    def _render_context(self) -> None:
        highlights = _geometry_highlights(self._item, self._category(), self._session)
        image_path = getattr(self._session, "current_image_path", None)
        if image_path:
            # whole drawing in frame (A-11): the item must be read on its
            # symbol against the user's own drawing, not a cropped sliver
            self._context.set_context(
                image_path, highlights, crop_to_highlights=False
            )
            if self._context.highlight_count:
                # step straight to the item's own geometry: strip per
                # item switch, wheel zooms back out for the whole view
                self._context.focus_highlights()

    def set_busy(self, busy: bool) -> None:
        """Disable action buttons while an operation is in progress."""
        for w in self.findChildren(QPushButton):
            if w is not self._busy_safe():
                pass
        self.setEnabled(not busy)

    def _busy_safe(self) -> Optional[QPushButton]:
        return None

    def _emit(
        self,
        action: str,
        corrected_value: Any = None,
        corrected_optimistic: Any = None,
        corrected_pessimistic: Any = None,
    ) -> None:
        self.decision_made.emit(
            ReviewActionRequest(
                category=self._category(),
                item=self._item,
                action=action,
                corrected_value=corrected_value,
                corrected_optimistic=corrected_optimistic,
                corrected_pessimistic=corrected_pessimistic,
            )
        )

    def _category(self) -> ReviewCategory:
        return ReviewCategory.ACTIVITIES


def _clear_children(widget: QWidget) -> None:
    for child in list(widget.children()):
        if isinstance(child, QWidget):
            child.setParent(None)
            child.deleteLater()


class ActivityDetailPanel(_BaseReviewPanel):
    """Details + actions for an ActivityReview item."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__("Activity Review", parent)

        # A-11: the drawing strip opens the card - the decision is made
        # on your own drawing, then the single action row follows it.
        self._status_badge = BadgeLabel()
        self._context = ImageContextView()
        self._context.setStyleSheet("border: none;")
        self._body.addWidget(self._context)

        self._evidence = EvidencePanel()

        self._actions = _section_action_row(
            self,
            [
                ("Accept", "ACCEPT"),
                ("Correct...", "CORRECT"),
            ],
            self._on_action,
        )
        self._body.addWidget(self._actions)

        # Correction editor (hidden until CORRECT is chosen)
        self._edit_row = QWidget()
        ed_layout = QHBoxLayout(self._edit_row)
        ed_layout.setContentsMargins(0, 0, 0, 0)
        ed_layout.setSpacing(6)
        self._id_edit = QLineEdit()
        self._id_edit.setPlaceholderText("Enter corrected activity ID...")
        ed_layout.addWidget(self._id_edit, stretch=1)
        self._apply_edit = QPushButton("Apply")
        self._apply_edit.clicked.connect(self._apply_correction)
        ed_layout.addWidget(self._apply_edit)
        self._cancel_edit = QPushButton("Cancel")
        self._cancel_edit.clicked.connect(self._hide_editor)
        ed_layout.addWidget(self._cancel_edit)
        self._edit_row.hide()
        self._body.addWidget(self._edit_row)

        self._feedback = QLabel("")
        self._feedback.setWordWrap(True)
        self._feedback.setFont(QFont(*MUTED_FONT))
        self._body.addWidget(self._feedback)

        # details zone: identity rows + evidence, behind a thin rule
        self._add_divider()
        self._node_label = _info_row(self._body, "Geometric Node", "")
        self._current_label = _info_row(self._body, "Current ID", "")
        self._proposed_label = _info_row(self._body, "Proposed", "")
        self._confidence_label = _info_row(self._body, "OCR Confidence", "")
        self._alternatives_label = _info_row(self._body, "Alternatives", "None", TEXT_MUTED)
        self._body.addWidget(self._status_badge, alignment=Qt.AlignmentFlag.AlignLeft)
        self._body.addWidget(self._evidence)

    def _category(self) -> ReviewCategory:
        return ReviewCategory.ACTIVITIES

    def show_item(self, item: Any, session: Any) -> None:
        self._item = item
        self._session = session
        self._hide_editor()
        node_label = str(getattr(item, "geometric_node_id", "?"))
        proposed = str(getattr(item, "proposed_activity_id", "")) or (
            str(getattr(item, "current_activity_id", "?")) or "?"
        )
        pct = f"{float(getattr(item, 'confidence', 0.0) or 0.0):.0%}"
        self._set_friendly_title("Activity identity needs confirmation")
        self._add_summary_hint(
            f"The engine read this node as {proposed} ({pct}).  "
            f"Accept its reading, or correct the ID for {node_label}."
        )
        self._node_label.setText(node_label)
        self._current_label.setText(str(getattr(item, "current_activity_id", "?")) or "None")
        self._proposed_label.setText(str(getattr(item, "proposed_activity_id", "")) or "None")
        self._confidence_label.setText(f"{float(getattr(item, 'confidence', 0.0) or 0.0):.0%}")
        alts = getattr(item, "alternatives", []) or []
        if alts:
            self._alternatives_label.setText("; ".join(
                f"{getattr(a, 'value', '?')} ({float(getattr(a, 'confidence', 0.0) or 0.0):.0%})"
                for a in alts[:5]
            ))
        else:
            self._alternatives_label.setText("None")
        self._status_badge.set_status(_status_name(item))
        self._evidence.set_evidence(
            list(getattr(item, "evidence", []) or []),
            float(getattr(item, "confidence", 0.0) or 0.0),
            reason=getattr(item, "reason", "") or "",
            provenance=getattr(item, "provenance", "") or "",
        )
        self._render_context()
        self._set_feedback("")

    def _on_action(self, action: str, _text: str) -> None:
        if action == "CORRECT":
            self._id_edit.setEnabled(True)
            self._edit_row.show()
            self._id_edit.setFocus()
            self._set_feedback("")
            return
        if action == "ACCEPT":
            self._contains_dirty = True
        self._emit(action)

    def _apply_correction(self) -> None:
        result = validate_activity_id(self._id_edit.text())
        if not result.ok:
            self._set_feedback(result.message, error=True)
            return
        review_session = getattr(self._session, "review_session", None)
        duplicate = is_duplicate_activity_id(
            result.value,
            review_session,
            exclude_node_id=getattr(self._item, "geometric_node_id", None),
        )
        if duplicate:
            self._set_feedback(f"Duplicate ID: already used by geometric node {duplicate}", error=True)
            return
        self._hide_editor()
        self._set_feedback("")
        self._emit("CORRECT", result.value)

    def _hide_editor(self) -> None:
        self._edit_row.hide()
        self._id_edit.clear()

    def _set_feedback(self, text: str, error: bool = False) -> None:
        color = DANGER if error else SUCCESS
        self._feedback.setText(text)
        self._feedback.setStyleSheet(f"color: {color};")


class DependencyDetailPanel(_BaseReviewPanel):
    """Details + actions for a DependencyReview item."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__("Dependency Review", parent)

        # A-11: drawing strip first, decisions second, details below
        self._status_badge = BadgeLabel()
        self._context = ImageContextView()
        self._body.addWidget(self._context)

        self._evidence = EvidencePanel()

        self._actions = _section_action_row(
            self,
            [
                ("Accept", "ACCEPT"),
                ("Reject", "REJECT"),
                ("Reverse", "REVERSE"),
                ("Change Source...", "CHANGE_SOURCE"),
                ("Change Target...", "CHANGE_TARGET"),
            ],
            self._on_action,
        )
        self._body.addWidget(self._actions)

        self._change_row = QWidget()
        ch_layout = QHBoxLayout(self._change_row)
        ch_layout.setContentsMargins(0, 0, 0, 0)
        ch_layout.setSpacing(6)
        self._change_label = QLabel("Choose node:")
        self._change_label.setFont(QFont(*MUTED_FONT))
        ch_layout.addWidget(self._change_label)
        self._node_combo = QComboBox()
        self._node_combo.setFont(QFont(*LABEL_FONT))
        ch_layout.addWidget(self._node_combo, stretch=1)
        self._node_apply = QPushButton("Apply")
        self._node_apply.clicked.connect(self._apply_change)
        ch_layout.addWidget(self._node_apply)
        self._node_cancel = QPushButton("Cancel")
        self._node_cancel.clicked.connect(self._hide_change)
        ch_layout.addWidget(self._node_cancel)
        self._change_row.hide()
        self._body.addWidget(self._change_row)

        self._feedback = QLabel("")
        self._feedback.setWordWrap(True)
        self._feedback.setFont(QFont(*MUTED_FONT))
        self._body.addWidget(self._feedback)

        # details zone: live review rows + evidence, behind a thin rule
        self._add_divider()
        self._direction_label = _info_row(self._body, "Current", "")
        self._proposed_label = _info_row(self._body, "Proposed Direction", "")
        self._alternative_label = _info_row(self._body, "Reverse Alternative", "")
        self._confidence_label = _info_row(self._body, "Confidence", "")
        self._body.addWidget(self._status_badge, alignment=Qt.AlignmentFlag.AlignLeft)
        self._body.addWidget(self._evidence)

    def _category(self) -> ReviewCategory:
        return ReviewCategory.DEPENDENCIES

    def show_item(self, item: Any, session: Any) -> None:
        self._item = item
        self._session = session
        self._hide_change()
        source = str(getattr(item, "current_source_id", "?"))
        target = str(getattr(item, "current_target_id", "?"))
        pct = f"{float(getattr(item, 'confidence', 0.0) or 0.0):.0%}"
        self._set_friendly_title("Uncertain dependency")
        self._add_summary_hint(
            f"The engine drew the arrow {source} -> {target} ({pct}).  "
            f"Accept it, or fix the direction or the ends it connects."
        )
        self._direction_label.setText(str(getattr(item, "current_direction", "?")) or "?")
        self._proposed_label.setText(str(getattr(item, "proposed_direction", "")) or "None")
        self._alternative_label.setText(str(getattr(item, "alternative_direction", "")) or "None")
        self._confidence_label.setText(f"{float(getattr(item, 'confidence', 0.0) or 0.0):.0%}")
        self._status_badge.set_status(_status_name(item))
        self._evidence.set_evidence(
            list(getattr(item, "evidence", []) or []),
            float(getattr(item, "confidence", 0.0) or 0.0),
            reason=getattr(item, "reason", "") or "",
            provenance=getattr(item, "provenance", "") or "",
        )
        self._render_context()
        self._set_feedback("")

    def _node_options(self) -> List[tuple[Any, Any]]:
        """[(geometric_node_id, semantic_id)] for change-source/target selection."""
        options: List[tuple[Any, Any]] = []
        recon = _reconstruction(self._session)
        if recon is None:
            return options
        for act in getattr(recon, "activities", []) or []:
            node = getattr(act, "source_node_id", None) or getattr(act, "geometric_node_id", None)
            if node is None:
                continue
            semantic = getattr(act, "activity_id", node)
            options.append((node, semantic))
        return options

    def _on_action(self, action: str, _text: str) -> None:
        if action in ("CHANGE_SOURCE", "CHANGE_TARGET"):
            self._open_change(action)
            return
        self._emit(action)

    def _open_change(self, action: str) -> None:
        self._pending_change = action
        self._node_combo.clear()
        options = self._node_options()
        if not options:
            self._set_feedback("No geometric nodes available for this review.", error=True)
            return
        for node, semantic in options:
            self._node_combo.addItem(geometric_node_label(node, semantic), semantic or node)
        self._change_label.setText("Change source:" if action == "CHANGE_SOURCE" else "Change target:")
        self._change_row.show()
        self._set_feedback("")

    def _apply_change(self) -> None:
        if not hasattr(self, "_pending_change"):
            return
        value = self._node_combo.currentData()
        if value is None:
            value = self._node_combo.currentText()
        action = self._pending_change
        self._hide_change()
        self._set_feedback("")
        self._emit(action, value)

    def _hide_change(self) -> None:
        self._change_row.hide()

    def _set_feedback(self, text: str, error: bool = False) -> None:
        color = DANGER if error else SUCCESS
        self._feedback.setText(text)
        self._feedback.setStyleSheet(f"color: {color};")


class DurationDetailPanel(_BaseReviewPanel):
    """Details + actions for a DurationReview item."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__("Duration Review", parent)

        # A-11: drawing strip first, decisions second, details below
        self._status_badge = BadgeLabel()
        self._context = ImageContextView()
        self._body.addWidget(self._context)

        self._evidence = EvidencePanel()

        self._actions = _section_action_row(
            self,
            [
                ("Accept", "ACCEPT"),
                ("Correct...", "CORRECT"),
            ],
            self._on_action,
        )
        self._body.addWidget(self._actions)

        # A-15: three-point correction editor (O <= M <= P)
        self._edit_row = QWidget()
        ed_layout = QHBoxLayout(self._edit_row)
        ed_layout.setContentsMargins(0, 0, 0, 0)
        ed_layout.setSpacing(4)
        self._triplet_hint = QLabel("O <= M <= P")
        self._triplet_hint.setFont(QFont(*MUTED_FONT))
        self._triplet_hint.setStyleSheet(f"color: {TEXT_MUTED};")
        ed_layout.addWidget(self._triplet_hint)
        self._o_edit = QLineEdit()
        self._o_edit.setPlaceholderText("Opt. (O)")
        self._m_edit = QLineEdit()
        self._m_edit.setPlaceholderText("Most likely (M)")
        self._p_edit = QLineEdit()
        self._p_edit.setPlaceholderText("Pess. (P)")
        for edit in (self._o_edit, self._m_edit, self._p_edit):
            ed_layout.addWidget(edit, stretch=1)
        self._apply_edit = QPushButton("Apply")
        self._apply_edit.clicked.connect(self._apply_correction)
        ed_layout.addWidget(self._apply_edit)
        self._cancel_edit = QPushButton("Cancel")
        self._cancel_edit.clicked.connect(self._hide_editor)
        ed_layout.addWidget(self._cancel_edit)
        self._edit_row.hide()
        self._body.addWidget(self._edit_row)

        self._feedback = QLabel("")
        self._feedback.setWordWrap(True)
        self._feedback.setFont(QFont(*MUTED_FONT))
        self._body.addWidget(self._feedback)

        # details zone: live review rows + evidence, behind a thin rule
        self._add_divider()
        self._activity_label = _info_row(self._body, "Activity", "")
        self._current_label = _info_row(self._body, "Current Duration", "")
        self._proposed_label = _info_row(self._body, "Candidate Durations", "")
        self._confidence_label = _info_row(self._body, "Confidence", "")
        self._body.addWidget(self._status_badge, alignment=Qt.AlignmentFlag.AlignLeft)
        self._body.addWidget(self._evidence)

    def _category(self) -> ReviewCategory:
        return ReviewCategory.DURATIONS

    def show_item(self, item: Any, session: Any) -> None:
        self._item = item
        self._session = session
        self._hide_editor()
        activity = str(getattr(item, "activity_id", "?"))
        current = getattr(item, "current_duration", None)
        current_text = f"{current:g}" if current is not None and not _is_nan(current) else "None"
        pct = f"{float(getattr(item, 'confidence', 0.0) or 0.0):.0%}"
        self._set_friendly_title("Missing or invalid duration")
        self._add_summary_hint(
            f"The engine read a duration of {current_text} ({pct}) for "
            f"{activity}.  Accept it, or type the correct value."
        )
        self._activity_label.setText(activity)
        self._current_label.setText(current_text)
        self._proposed_label.setText(
            "; ".join(_format_duration_candidates(item)) or "None"
        )
        self._confidence_label.setText(f"{float(getattr(item, 'confidence', 0.0) or 0.0):.0%}")
        self._status_badge.set_status(_status_name(item))
        self._evidence.set_evidence(
            list(getattr(item, "evidence", []) or []),
            float(getattr(item, "confidence", 0.0) or 0.0),
            reason=getattr(item, "reason", "") or "",
            provenance=getattr(item, "provenance", "") or "",
        )
        self._render_context()
        self._set_feedback("")

    def _on_action(self, action: str, _text: str) -> None:
        if action == "CORRECT":
            self._show_triplet_editor()
            self._set_feedback("")
            return
        self._emit(action)

    def _show_triplet_editor(self) -> None:
        """Prefill O/M/P so touching Apply adopts the derivation (A-14/A-15).

        ``m`` is the accepted duration (or the CV-proposed fallback); the
        spread reuses the derivation rules: review-candidate extremes when
        two or more distinct numbers exist, otherwise a +/-25% band.
        """
        item = self._item
        m = getattr(item, "current_duration", None)
        if m is None or _is_nan(m):
            m = getattr(item, "proposed_duration", None)
            if m is not None and _is_nan(m):
                m = None
        a_text = p_text = m_text = ""
        if m is not None and m > 0:
            candidates = duration_item_candidates(item) or []
            if len(candidates) >= 2:
                a_val = min(min(candidates), m)
                p_val = max(max(candidates), m)
            else:
                a_val = m * 0.75
                p_val = m * 1.25
            a_text = f"{a_val:g}"
            m_text = f"{m:g}"
            p_text = f"{p_val:g}"
        self._o_edit.setText(a_text)
        self._m_edit.setText(m_text)
        self._p_edit.setText(p_text)
        self._edit_row.show()
        self._m_edit.setFocus()

    def _apply_correction(self) -> None:
        result = validate_duration_triplet(
            self._o_edit.text(), self._m_edit.text(), self._p_edit.text()
        )
        if not result.ok:
            self._set_feedback(result.message, error=True)
            return
        o, m, p = result.value
        self._hide_editor()
        self._set_feedback("")
        self._emit(
            "CORRECT",
            m,
            corrected_optimistic=o,
            corrected_pessimistic=p,
        )

    def _hide_editor(self) -> None:
        self._edit_row.hide()
        for edit in (self._o_edit, self._m_edit, self._p_edit):
            edit.clear()

    def _set_feedback(self, text: str, error: bool = False) -> None:
        color = DANGER if error else SUCCESS
        self._feedback.setText(text)
        self._feedback.setStyleSheet(f"color: {color};")


def _is_nan(value: float) -> bool:
    return value != value


def _format_duration_candidates(item: Any) -> list[str]:
    candidates: list[float] = []
    proposed = getattr(item, "proposed_duration", None)
    if proposed is not None and not _is_nan(proposed):
        candidates.append(proposed)
    for ev in getattr(item, "evidence", []) or []:
        desc = getattr(ev, "description", "") or ""
        for chunk in desc.replace(",", " ").split():
            try:
                v = float(chunk)
                if v > 0 and not _is_nan(v):
                    candidates.append(v)
            except ValueError:
                continue
    seen: list[float] = []
    for c in candidates:
        if not any(abs(c - s) < 1e-9 for s in seen):
            seen.append(c)
    return [f"{c:g}" for c in seen[:5]]
