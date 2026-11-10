"""
A-15: three-point duration editor inside the review detail panel.

Covers the editor prefill (band + review-candidate spread), the O <= M <= P
validation, and the emitted ReviewActionRequest carrying the triple.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication

from pert_analyzer.gui.reviews.categories import ReviewCategory
from pert_analyzer.gui.reviews.detail_panels import DurationDetailPanel


def _duration_item(current=8.0, proposed=None, evidence=()) -> SimpleNamespace:
    return SimpleNamespace(
        geometric_node_id="n1",
        activity_id="A1",
        current_duration=current,
        proposed_duration=proposed,
        evidence=list(evidence),
        confidence=0.9,
        reason="",
        provenance="",
    )


@pytest.fixture()
def panel(qapp: QApplication) -> DurationDetailPanel:
    return DurationDetailPanel()


def _show(panel: DurationDetailPanel, item: SimpleNamespace) -> None:
    panel.show_item(item, SimpleNamespace(current_image_path=None))


def _emitted(panel: DurationDetailPanel) -> list:
    captured: list = []
    panel.decision_made.connect(lambda req: captured.append(req))
    return captured


def test_prefill_uses_default_band(panel) -> None:
    _show(panel, _duration_item(current=8.0))
    panel._on_action("CORRECT", "Correct...")
    assert not panel._edit_row.isHidden()
    assert panel._m_edit.text() == "8"
    assert panel._o_edit.text() == "6"
    assert panel._p_edit.text() == "10"


def test_prefill_falls_back_to_proposed(panel) -> None:
    _show(panel, _duration_item(current=None, proposed=6.0))
    panel._on_action("CORRECT", "Correct...")
    assert panel._m_edit.text() == "6"
    assert panel._o_edit.text() == "4.5"
    assert panel._p_edit.text() == "7.5"


def test_prefill_adopts_review_candidates(panel) -> None:
    item = _duration_item(
        current=4.0,
        evidence=[
            SimpleNamespace(description="cost label reads '2 days': 2"),
            SimpleNamespace(description="see the '7 days' cell too: 7"),
        ],
    )
    _show(panel, item)
    panel._on_action("CORRECT", "Correct...")
    assert panel._o_edit.text() == "2"
    assert panel._m_edit.text() == "4"
    assert panel._p_edit.text() == "7"


def test_invalid_order_shows_feedback_and_blocks(panel) -> None:
    _show(panel, _duration_item(current=8.0))
    panel._on_action("CORRECT", "Correct...")
    captured = _emitted(panel)
    panel._o_edit.setText("9")
    panel._p_edit.setText("10")
    panel._apply_edit.click()
    assert "exceeds" in panel._feedback.text().lower()
    assert captured == []
    assert not panel._edit_row.isHidden()


def test_valid_triplet_emits_request(panel) -> None:
    item = _duration_item(current=8.0)
    _show(panel, item)
    panel._on_action("CORRECT", "Correct...")
    captured = _emitted(panel)
    panel._o_edit.setText("5")
    panel._m_edit.setText("8")
    panel._p_edit.setText("10")
    panel._apply_edit.click()
    assert len(captured) == 1
    request = captured[0]
    assert request.action == "CORRECT"
    assert request.category is ReviewCategory.DURATIONS
    assert request.corrected_value == 8
    assert request.corrected_optimistic == 5
    assert request.corrected_pessimistic == 10
    assert panel._edit_row.isHidden()  # hidden after apply
