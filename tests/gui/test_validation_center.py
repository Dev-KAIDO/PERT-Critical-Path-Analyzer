"""
Tests for the Validation Center (Phase 3).

Covers status derivation, issue presentation (severity, blocking vs
non-blocking), revalidation, Review-issue navigation, the Continue gate,
Results gating, and session state updates.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication

from pert_analyzer.gui.main_window import MainWindow
from pert_analyzer.gui.navigation import NavDestination
from pert_analyzer.gui.pages.validation_page import ValidationPage
from pert_analyzer.gui.reviews.categories import ReviewCategory
from pert_analyzer.gui.reviews.validation_presentation import (
    ValidationSeverity,
    present_issues,
    review_target,
)
from pert_analyzer.gui.session import GuiSession, ValidationCenterStatus
from tests.gui.fakes import (
    FakeReviewSession,
    FakeReviewedCandidate,
    FakeValidationResult,
    FakeValidatedWorkflow,
    fake_issue,
    make_review_session,
)


@pytest.fixture()
def page(qapp: QApplication) -> ValidationPage:
    return ValidationPage()


@pytest.fixture()
def window(qapp: QApplication) -> MainWindow:
    w = MainWindow(backend=lambda path: None)
    yield w
    w.close()


def make_session(candidate, review_session=None) -> GuiSession:
    """A GuiSession that already applied a candidate through a workflow."""
    session = GuiSession()
    session.workflow = FakeValidatedWorkflow(
        candidate=candidate, review_session=review_session or FakeReviewSession()
    )
    session.complete_apply(candidate)
    return session


def valid_candidate(**kwargs) -> FakeReviewedCandidate:
    return FakeReviewedCandidate(valid=True, **kwargs)


def invalid_candidate(**kwargs) -> FakeReviewedCandidate:
    return FakeReviewedCandidate(
        valid=False,
        cpm_gate="BLOCKED_ERROR",
        cpm_project_duration=None,
        critical_path_count=None,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Presentation mapping (unit)
# ---------------------------------------------------------------------------


def test_present_issues_maps_errors_and_warnings() -> None:
    validation = FakeValidationResult(
        is_valid=False,
        errors=[fake_issue("cycle"), fake_issue("missing_duration")],
        warnings=[fake_issue("disconnected_components")],
    )
    issues = present_issues(validation)
    severities = [i.severity for i in issues]
    assert severities == [
        ValidationSeverity.ERROR,
        ValidationSeverity.ERROR,
        ValidationSeverity.WARNING,
    ]
    assert [i.code for i in issues] == [
        "cycle",
        "missing_duration",
        "disconnected_components",
    ]


def test_present_issues_unknown_code_gets_generic_meta() -> None:
    issues = present_issues(
        FakeValidationResult(is_valid=False, errors=[fake_issue("mystery_code")])
    )
    assert issues[0].title == "Validation issue"


def test_present_issues_none_returns_empty() -> None:
    assert present_issues(None) == []


def test_review_target_duration_mapping() -> None:
    rs = make_review_session(n_activities=0, n_dependencies=0, n_durations=1)
    target = review_target(fake_issue("missing_duration", elements=["A0"]), rs)
    assert target == (ReviewCategory.DURATIONS, "dur:N0")


def test_review_target_activity_mapping() -> None:
    rs = make_review_session(n_activities=2, n_dependencies=0, n_durations=0)
    target = review_target(fake_issue("duplicate_activity_id", elements=["A1"]), rs)
    assert target == (ReviewCategory.ACTIVITIES, "act:N1")


def test_review_target_dependency_mapping() -> None:
    rs = make_review_session(n_activities=2, n_dependencies=1, n_durations=0)
    target = review_target(fake_issue("self_loop", elements=["A0"]), rs)
    assert target == (ReviewCategory.DEPENDENCIES, "dep:EN0")


def test_review_target_unknown_reference_alias() -> None:
    rs = make_review_session(n_activities=2, n_dependencies=1, n_durations=0)
    target = review_target(fake_issue("unknown_dependency_reference", elements=["A0"]), rs)
    assert target == (ReviewCategory.DEPENDENCIES, "dep:EN0")


def test_review_target_no_match_returns_none() -> None:
    rs = make_review_session(n_activities=1, n_dependencies=0, n_durations=0)
    assert review_target(fake_issue("missing_duration", elements=["Z9"]), rs) is None
    assert review_target(fake_issue("cycle", elements=[]), rs) is None
    assert review_target(fake_issue("missing_duration", elements=["A0"]), None) is None


# ---------------------------------------------------------------------------
# Empty / unavailable states
# ---------------------------------------------------------------------------


def test_validation_page_no_analysis_empty_state(page: ValidationPage) -> None:
    page.refresh(GuiSession())
    assert "No project analyzed" in page._empty_label.text()
    assert page.stack_index == page._EMPTY
    assert not page._go_analyze_btn.isHidden()


def test_validation_page_no_reviewed_graph_state(page: ValidationPage) -> None:
    candidate = FakeReviewedCandidate(valid=True, no_validation=True)
    session = make_session(candidate)
    page.refresh(session)
    assert "No reviewed graph" in page._empty_label.text()
    assert page.stack_index == page._EMPTY
    assert not page._go_review_btn.isHidden()
    assert not page._empty_revalidate_btn.isHidden()


# ---------------------------------------------------------------------------
# Status rendering
# ---------------------------------------------------------------------------


def test_valid_status_shown(page: ValidationPage) -> None:
    session = make_session(valid_candidate())
    page.refresh(session)
    assert page.stack_index == page._WORKSPACE
    assert page._status_label.text() == "VALIDATED"
    assert page._status_hint.text()
    assert page._continue_btn.isEnabled()


def test_invalid_status_shown(page: ValidationPage) -> None:
    candidate = invalid_candidate(
        validation=FakeValidationResult(
            is_valid=False, errors=[fake_issue("cycle")]
        )
    )
    page.refresh(make_session(candidate))
    assert page._status_label.text() == "INVALID"
    assert page._status_hint.text()
    assert not page._continue_btn.isEnabled()


def test_blocked_review_status_when_pending(page: ValidationPage) -> None:
    rs = make_review_session()
    session = make_session(valid_candidate(), review_session=rs)
    page.refresh(session)
    assert page._status_label.text() == "REVIEW REQUIRED"
    assert "pending" in page._status_hint.text().lower()
    assert not page._continue_btn.isEnabled()


def test_blocked_review_status_when_reviews_dirty(page: ValidationPage) -> None:
    session = make_session(valid_candidate())
    session.reviews_dirty = True
    page.refresh(session)
    assert page._status_label.text() == "REVIEW REQUIRED"


def test_not_available_status_when_validation_missing(page: ValidationPage) -> None:
    session = make_session(FakeReviewedCandidate(valid=True, no_validation=True))
    page.refresh(session)
    assert page.stack_index == page._EMPTY


def test_blocked_status_when_cpm_gate_not_runnable(page: ValidationPage) -> None:
    candidate = valid_candidate(cpm_gate="BLOCKED_REVIEW")
    session = make_session(candidate)
    page.refresh(session)
    assert page._status_label.text() == "REVIEW REQUIRED"


# ---------------------------------------------------------------------------
# Issue list + details
# ---------------------------------------------------------------------------


def test_issue_list_renders_errors_and_warnings(page: ValidationPage) -> None:
    candidate = invalid_candidate(
        validation=FakeValidationResult(
            is_valid=False,
            errors=[fake_issue("missing_duration", elements=["A0"]), fake_issue("cycle")],
            warnings=[fake_issue("disconnected_components")],
        )
    )
    page.refresh(make_session(candidate))
    assert page._issue_list.count() == 3
    texts = "\n".join(page._issue_list.item(i).text() for i in range(3))
    assert "missing_duration" in texts
    assert "cycle" in texts
    assert "disconnected_components" in texts


def test_issue_detail_shows_severity_message_and_elements(page: ValidationPage) -> None:
    candidate = invalid_candidate(
        validation=FakeValidationResult(
            is_valid=False,
            errors=[fake_issue("missing_duration", elements=["A0"])],
        )
    )
    session = make_session(
        candidate,
        review_session=make_review_session(n_activities=0, n_dependencies=0, n_durations=1),
    )
    page.refresh(session)
    assert page._current_issue is not None
    assert page._severity_badge.text() == "Blocking"
    assert page._blocking_label.text() == "Must be resolved before CPM can run"
    assert "Affected: A0" in page._detail_elements.text()
    assert page._review_issue_btn.isEnabled()


def test_blocking_and_nonblocking_issues_distinguished(page: ValidationPage) -> None:
    candidate = invalid_candidate(
        validation=FakeValidationResult(
            is_valid=False,
            errors=[fake_issue("cycle")],
            warnings=[fake_issue("disconnected_components")],
        )
    )
    page.refresh(make_session(candidate))
    page._issue_list.setCurrentRow(0)
    assert page._severity_badge.text() == "Blocking"
    assert page._blocking_label.text() == "Must be resolved before CPM can run"

    page._issue_list.setCurrentRow(1)
    assert page._severity_badge.text() == "Warning"
    assert page._blocking_label.text() == "Non-blocking warning"


def test_issue_without_match_disables_review_button(page: ValidationPage) -> None:
    candidate = invalid_candidate(
        validation=FakeValidationResult(
            is_valid=False, errors=[fake_issue("cycle", elements=[])]
        )
    )
    page.refresh(make_session(candidate, review_session=make_review_session()))
    page._issue_list.setCurrentRow(0)
    assert not page._review_issue_btn.isEnabled()
    assert "No matching review item" in page._target_hint.text()


def test_clear_issue_detail_when_no_issues(page: ValidationPage) -> None:
    page.refresh(make_session(valid_candidate()))
    assert page._issue_list.count() == 0
    assert page._detail_title.text() == "No issues detected"


# ---------------------------------------------------------------------------
# Buttons: revalidate + continue
# ---------------------------------------------------------------------------


def test_revalidate_button_states(page: ValidationPage) -> None:
    page.refresh(make_session(valid_candidate()))
    assert page._revalidate_btn.isEnabled()

    page.refresh(make_session(valid_candidate(), review_session=make_review_session()))
    assert not page._revalidate_btn.isEnabled()


def test_continue_button_only_enabled_when_valid(page: ValidationPage) -> None:
    page.refresh(make_session(valid_candidate()))
    assert page._continue_btn.isEnabled()

    page.refresh(make_session(invalid_candidate()))
    assert not page._continue_btn.isEnabled()

    page.refresh(make_session(valid_candidate(), review_session=make_review_session()))
    assert not page._continue_btn.isEnabled()


# A-13: window-flow tests for the retired validation page (revalidate,
# review-issue navigation, continue gate, back/analyze) were removed with
# the wiring itself. The page class keeps its own unit tests above.

# ---------------------------------------------------------------------------
# Window integration: results readiness
# ---------------------------------------------------------------------------


def test_results_page_not_ready_when_invalid(window: MainWindow) -> None:
    window._session = make_session(invalid_candidate())
    window._on_nav(NavDestination.RESULTS)
    assert "Results not ready" in window._results_page._empty_label.text()


def test_results_page_shows_results_when_valid(window: MainWindow) -> None:
    window._session = make_session(valid_candidate())
    window._on_nav(NavDestination.RESULTS)
    assert window._results_page._empty_label.isHidden()
    assert "Project duration" in window._results_page._summary_label.text()
