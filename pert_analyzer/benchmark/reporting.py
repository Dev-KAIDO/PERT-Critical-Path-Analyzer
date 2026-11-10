"""
Deterministic report writers for the generalization benchmark.

Produces a human-readable Markdown report and the machine-readable JSON
bundle (``docs/GENERALIZATION_BENCHMARK.md`` + ``.json``).  Every table
row is derived from the measured results summary; nothing is hardcoded.

The accuracy section is added whenever v1.0 ground-truth annotations
exist (see ``pert_analyzer.benchmark.annotations``).  Aggregate averages
are simple means over images whose metric is comparable (``N/A`` images
are excluded); the aggregation method is stated in the report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pert_analyzer.benchmark.metrics import UNAVAILABLE


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if value is UNAVAILABLE or (isinstance(value, str) and value.upper() == "UNAVAILABLE"):
        return "_unavail_"
    return str(value)


def _fmt_float(value: Any) -> str:
    """Display helper for float metrics; keeps computation callers raw."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.4f}".rstrip("0").rstrip(".") or "0"
    return _cell(value)


def _metric(entry: Dict[str, Any], key: str) -> Any:
    return ((entry.get("stage_metrics") or {}).get(key))


# ---------------------------------------------------------------------------
# Accuracy (ground-truth) table helpers
# ---------------------------------------------------------------------------


def _acc(entry: Dict[str, Any]) -> Dict[str, Any]:
    return entry.get("accuracy_metrics") or {}


def _fmt_p(value: Any) -> str:
    if value is None:
        return "-"
    for prefix in ("0.", "1."):
        if str(value).startswith(prefix):
            return f"{value:.4f}"
    return str(value)


def _activity_row(acc: Dict[str, Any]) -> List[str]:
    counts = acc.get("activity_counts") or {}
    acts = acc.get("activities") or {}
    return [
        _cell(counts.get("expected")),
        _cell(counts.get("detected")),
        _fmt_p(acts.get("precision")),
        _fmt_p(acts.get("recall")),
        _fmt_p(acts.get("f1")),
    ]


def _dependency_row(acc: Dict[str, Any]) -> List[str]:
    deps = acc.get("dependencies") or {}
    directed = deps.get("directed") or {}
    return [
        _cell(acc.get("expected_dependencies")),
        _cell(acc.get("detected_dependencies")),
        _fmt_p(directed.get("precision")),
        _fmt_p(directed.get("recall")),
        _fmt_p(directed.get("f1")),
        _fmt_p((deps.get("direction") or {}).get("direction_accuracy")),
    ]


def _duration_cell(acc: Dict[str, Any]) -> str:
    dur = acc.get("durations") or {}
    exact = dur.get("exact_match_accuracy")
    if exact is None:
        return "-"
    mae = dur.get("mean_abs_error")
    return f"{_fmt_p(exact)} (MAE {_fmt_p(mae)})"


def _event_row(acc: Dict[str, Any]) -> List[str]:
    counts = acc.get("event_counts") or {}
    events = acc.get("events") or {}
    return [
        _cell(counts.get("expected")),
        _cell(counts.get("detected")),
        _fmt_p(events.get("precision")),
        _fmt_p(events.get("recall")),
        _fmt_p(events.get("f1")),
    ]


def _accuracy_table_rows(images: List[Dict[str, Any]]) -> List[List[str]]:
    rows: List[List[str]] = []
    for entry in images:
        acc = _acc(entry)
        if not acc.get("available"):
            continue
        diagram = acc.get("diagram_type")
        name = f"`{entry.get('relative_path', '')}`"
        deps = acc.get("dependencies") or {}
        directed = deps.get("directed") or {}
        if diagram == "AON":
            base = _activity_row(acc)
            dep_cols = _dependency_row(acc)
            row = [
                name, "AON",
                base[0], base[1], base[2], base[3], base[4],
                dep_cols[0], dep_cols[1], dep_cols[2], dep_cols[3], dep_cols[4],
                dep_cols[5], _duration_cell(acc),
                _cell(_metric(entry, "graph_is_valid")),
                _cell(entry.get("outcome")),
            ]
        else:
            ev = _event_row(acc)
            row = [
                name, "AOA",
                ev[0], ev[1], ev[2], ev[3], ev[4],
                _cell(acc.get("expected_dependencies")),
                _cell(acc.get("detected_dependencies")),
                _fmt_p(directed.get("precision")),
                _fmt_p(directed.get("recall")),
                _fmt_p(directed.get("f1")),
                _fmt_p((deps.get("direction") or {}).get("direction_accuracy")),
                _duration_cell(acc),
                _cell(_metric(entry, "graph_is_valid")),
                _cell(entry.get("outcome")),
            ]
        rows.append(row)
    return rows


def _aggregate(
    images: List[Dict[str, Any]],
    diagram_type: str,
    metric_path: List[str],
    field: str,
) -> Dict[str, Any]:
    """Simple per-image mean of ``field`` at ``metric_path`` in the accuracy
    dict over images with a comparable value (weights each image equally)."""
    values: List[float] = []
    for entry in images:
        acc = _acc(entry)
        if acc.get("diagram_type") != diagram_type or not acc.get("available"):
            continue
        node: Any = acc
        for key in metric_path:
            node = (node or {}).get(key)
            if not isinstance(node, dict):
                node = None
                break
        value = node.get(field) if isinstance(node, dict) else None
        if value is None:
            continue
        values.append(float(value))
    if not values:
        return {"count": 0, "mean": None}
    return {"count": len(values), "mean": round(sum(values) / len(values), 4)}


def _write_accuracy_section(out: List[str], images: List[Dict[str, Any]]) -> None:
    rows = _accuracy_table_rows(images)
    out.append("## Accuracy benchmark (v1.0 ground truth)")
    out.append("")
    out.append(
        "Reported per-image metrics are computed from v1.0 annotations in "
        "`tests/test_data/ground_truth/` (see the annotation schema there). "
        "Nodes are matched geometrically (bounding-box IoU / centre distance), "
        "never by array position or OCR label alone. Dependencies are compared "
        "as normalized directed pairs; direction accuracy is tracked separately "
        "from the undirected pair. Durations are never rounded. `-` = not "
        "computed (`N/A`) for that image."
    )
    out.append("")
    if not rows:
        out.append("_No accuracy annotations available for this run._")
        out.append("")
        return
    out.append("| Image | Type | ExpActs | DetActs | ActP | ActR | ActF1 | ExpDeps | DetDeps | DepP | DepR | DepF1 | DirAcc | DurationAcc | GraphValid | Review |")
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    out.append("")

    # Aggregates (documented aggregation method: per-image mean).
    out.append("### Aggregates (simple per-image mean over comparable metrics)")
    out.append("")
    out.append("| Diagram | Component | Metric | Images compared | Mean |")
    out.append("| --- | --- | --- | --- | --- |")
    for diagram, label in (("AON", "AON"), ("AOA", "AOA")):
        if diagram == "AON":
            combinations = [
                ("activities", "detection", "activity precision",
                 ["activities"], "precision"),
                ("activities", "detection", "activity recall",
                 ["activities"], "recall"),
                ("dependencies", "directed", "dependency precision",
                 ["dependencies", "directed"], "precision"),
                ("dependencies", "directed", "dependency recall",
                 ["dependencies", "directed"], "recall"),
                ("dependencies", "direction", "direction accuracy",
                 ["dependencies", "direction"], "direction_accuracy"),
                ("durations", "durations", "duration exact-match accuracy",
                 ["durations"], "exact_match_accuracy"),
            ]
        else:
            combinations = [
                ("events", "detection", "event precision",
                 ["events"], "precision"),
                ("events", "detection", "event recall",
                 ["events"], "recall"),
                ("arrows", "directed", "arrow precision",
                 ["arrows", "directed"], "precision"),
                ("arrows", "directed", "arrow recall",
                 ["arrows", "directed"], "recall"),
                ("arrows", "direction", "arrow direction accuracy",
                 ["arrows", "direction"], "direction_accuracy"),
            ]
        for tag, _, metric_name, path, field in combinations:
            aggr = _aggregate(images, diagram, path, field)
            out.append(
                f"| {label} | {tag} | {metric_name} | "
                f"{aggr['count']} | {_fmt_p(aggr['mean'])} |"
            )
        out.append("")
    out.append("_Aggregation method: unweighted mean of the per-image metric over images where that metric is comparable (uncertain/N-A images excluded). Different components are never averaged together._")
    out.append("")


def _summary_row(entry: Dict[str, Any]) -> List[str]:
    classification = entry.get("failure_classification") or {}
    return [
        f"`{entry.get('relative_path', '')}`",
        _cell(entry.get("file_format")),
        f"{_cell(entry.get('width'))}x{_cell(entry.get('height'))}",
        _cell(_metric(entry, "reconstructed_activities")),
        _cell(_metric(entry, "validated_dependencies")),
        _cell(_metric(entry, "ocr_labels")),
        _cell(entry.get("diagram_type")),
        _cell(entry.get("outcome")),
        _cell(entry.get("final_status")),
        _cell(classification.get("failure_class")),
        f"{_cell(entry.get('duration_seconds'))} s",
    ]


def _write_summary_table(out: List[str], images: List[Dict[str, Any]]) -> None:
    out.append("| Image | Fmt | Size | Activities | Valid deps | OCR labels | Diagram | Outcome | Status | Bottleneck | Time |")
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for entry in images:
        row = _summary_row(entry)
        out.append("| " + " | ".join(row) + " |")


def _write_per_image_sections(out: List[str], images: List[Dict[str, Any]]) -> None:
    for idx, entry in enumerate(images, start=1):
        rel = entry.get("relative_path", f"image-{idx}")
        outcome = entry.get("outcome")
        status = entry.get("final_status")
        error = entry.get("error")
        out.append(f"### {idx}. `{rel}`")
        out.append("")
        out.append(
            f"- **Outcome:** `{outcome}` · **Final status:** `{status}`"
        )
        out.append(
            f"- **Format/Size:** {_cell(entry.get('file_format'))} "
            f"{_cell(entry.get('width'))}x{_cell(entry.get('height'))} "
            f"({_cell(entry.get('file_size_bytes'))} bytes) "
            f"· **Elapsed:** {_cell(entry.get('duration_seconds'))} s"
        )
        out.append(
            f"- **Diagram type:** {_cell(entry.get('diagram_type'))} "
            f"(confidence {_cell(entry.get('diagram_confidence'))})"
        )
        out.append("")
        out.append("**Measured pipeline progression:**")
        out.append("")
        out.append("| Stage | Counts |")
        out.append("| --- | --- |")
        for row in entry.get("stage_progression") or []:
            stage = row.get("stage", "")
            counts = " · ".join(
                f"{k}={_cell(v)}" for k, v in (row.get("counts") or {}).items()
            )
            out.append(f"| {stage} | {counts} |")
        out.append("")
        classification = entry.get("failure_classification") or {}
        failure_class = classification.get("failure_class")
        out.append(f"**First abnormal stage:** {_cell(classification.get('stage'))}")
        out.append("")
        out.append(f"**Failure classification:** `{failure_class}`")
        if classification.get("evidence"):
            evidence = " · ".join(
                f"{k}={_cell(v)}"
                for k, v in classification.get("evidence", {}).items()
            )
            out.append(f"**Measured evidence:** {evidence}")
        out.append("")
        if entry.get("gt_compare"):
            out.append("**Ground-truth comparison:**")
            out.append("")
            gt = entry.get("gt_compare") or {}
            for key in sorted(gt):
                out.append(f"- `{key}` = {_cell(gt[key])}")
            out.append("")
        acc = _acc(entry)
        if acc.get("available"):
            out.append("**Accuracy (v1.0 ground truth):**")
            out.append("")
            diagram = acc.get("diagram_type")
            status = acc.get("annotation_status")
            out.append(f"- Diagram type: `{diagram}` · Annotation status: `{status}`")
            if status != "COMPLETE":
                excluded = (acc.get("excluded") or {}).get("count", 0)
                out.append(f"- Partially/uncertain annotation: {excluded} reference items excluded from metrics.")
            if diagram == "AON":
                counts = acc.get("activity_counts") or {}
                acts = acc.get("activities") or {}
                out.append(
                    f"- Activities: expected `{counts.get('expected')}`, "
                    f"detected `{counts.get('detected')}`, "
                    f"precision `{_fmt_p(acts.get('precision'))}`, "
                    f"recall `{_fmt_p(acts.get('recall'))}`, "
                    f"F1 `{_fmt_p(acts.get('f1'))}`"
                )
            else:
                counts = acc.get("event_counts") or {}
                events = acc.get("events") or {}
                out.append(
                    f"- Events: expected `{counts.get('expected')}`, "
                    f"detected `{counts.get('detected')}`, "
                    f"precision `{_fmt_p(events.get('precision'))}`, "
                    f"recall `{_fmt_p(events.get('recall'))}`, "
                    f"F1 `{_fmt_p(events.get('f1'))}`"
                )
            deps = acc.get("dependencies") or {}
            directed = deps.get("directed") or {}
            out.append(
                f"- Dependencies/arrows: expected `{acc.get('expected_dependencies')}`, "
                f"detected `{acc.get('detected_dependencies')}`, "
                f"precision `{_fmt_p(directed.get('precision'))}`, "
                f"recall `{_fmt_p(directed.get('recall'))}`, "
                f"F1 `{_fmt_p(directed.get('f1'))}`"
            )
            direction = deps.get("direction") or {}
            out.append(
                f"- Direction accuracy: `{_fmt_p(direction.get('direction_accuracy'))}` "
                f"(`{_cell(direction.get('correct_pairs'))}` correct · "
                f"`{_cell(direction.get('reversed_pairs'))}` reversed)"
            )
            out.append(f"- Duration accuracy: `{_duration_cell(acc)}`")
            ids = acc.get("ids") if diagram == "AON" else None
            if ids:
                out.append(f"- ID accuracy: `{_fmt_p(ids.get('id_accuracy'))}` over `{_cell(ids.get('compared'))}` compared nodes")
            conflicts = entry.get("accuracy_metrics", {}).get("conflicts", [])
            if conflicts:
                out.append("- **Conflicts:**")
                for c in conflicts:
                    out.append(f"  - {c}")
            out.append("")
        if entry.get("warnings"):
            out.append("**Warnings:**")
            for warning in entry.get("warnings", []):
                out.append(f"- {_cell(warning)}")
            out.append("")
        if error:
            out.append(f"**Error:** `{error}`")
            out.append("")


def build_markdown(summary: Dict[str, Any]) -> str:
    images = summary.get("images", [])
    s = summary.get("summary", {})
    out: List[str] = []
    out.append("# Generalization Benchmark")
    out.append("")
    out.append(
        f"- **PERT Analyzer version:** {summary.get('pert_analyzer_version', '-')}"
    )
    out.append(f"- **Generated at (UTC):** {summary.get('generated_at', '-')}")
    out.append(f"- **Dataset root:** `{summary.get('dataset_root', '-')}`")
    out.append(
        f"- **Images discovered:** {summary.get('images_discovered', '-')} · "
        f"**Images analyzed:** {summary.get('images_analyzed', '-')}"
    )
    out.append("")
    out.append("## Summary")
    out.append("")
    out.append(f"- **Automatic success:** {s.get('automatic_success', 0)}")
    out.append(f"- **Automatic review required:** {s.get('automatic_review_required', 0)}")
    out.append(f"- **Fatal failures:** {s.get('fatal_failures', 0)}")
    out.append(f"- **With ground truth:** {s.get('with_ground_truth', 0)} · "
               f"**Without ground truth:** {s.get('without_ground_truth', 0)}")
    out.append(
        f"- **With accuracy annotation:** "
        f"{s.get('with_accuracy_annotation', 0)} · "
        f"**Statuses:** {s.get('annotation_statuses', {})}"
    )
    out.append(
        f"- **Largest activity-count inflation:** "
        f"{s.get('largest_activity_count_inflation', 0)} "
        f"(`{s.get('largest_activity_count_inflation_image', 'n/a')}`)"
    )
    out.append("")
    out.append("## Bottleneck summary (first abnormal stage per image)")
    out.append("")
    out.append("| Failure class | Images |")
    out.append("| --- | --- |")
    for row in summary.get("bottleneck_summary", []):
        out.append(
            f"| `{row.get('failure_class')}` | {row.get('image_count')} |"
        )
    out.append("")
    out.append("## Per-image results")
    out.append("")
    out.append("### Summary table")
    out.append("")
    _write_summary_table(out, images)
    out.append("")
    _write_cpm_compare_section(out, images)
    _write_accuracy_section(out, images)
    out.append("### Detail")
    out.append("")
    _write_per_image_sections(out, images)
    return "\n".join(out)


def _write_cpm_compare_section(out: List[str], images: List[Dict[str, Any]]) -> None:
    """Measured (pipeline CPM) vs derived (GT graph CPM), one row per image."""
    rows = []
    for entry in images:
        extra = entry.get("extra") or {}
        block = extra.get("cpm_vs_gt")
        if not block:
            continue
        expected = block.get("expected") or {}
        if expected.get("available"):
            rows.append(
                [
                    f"`{(entry.get('relative_path') or '?')}`",
                    "yes",
                    _fmt_float(expected.get("project_duration")),
                    _fmt_float(block.get("detected_project_duration")),
                    _fmt_float(block.get("project_duration_delta")),
                    _cell(expected.get("critical_path_count")),
                    _cell(block.get("detected_critical_path_count")),
                    _cell(block.get("critical_path_count_delta")),
                    "-",
                ]
            )
        else:
            rows.append(
                [
                    f"`{(entry.get('relative_path') or '?')}`",
                    "no",
                    "-",
                    _cell(block.get("detected_project_duration")),
                    "-",
                    "-",
                    _cell(block.get("detected_critical_path_count")),
                    "-",
                    str(expected.get("reason") or "-"),
                ]
            )
    if not rows:
        return
    out.append("### CPM vs ground truth (derived expectation)")
    out.append("")
    out.append("_Expectation is computed live by the production CPM engine on the")
    out.append("annotated graph (AON images with complete durations). Deltas are")
    out.append("``detected - expected``; empty detected cells mean CPM did not run")
    out.append("(invalid pipeline graph). AOA is reported honestly as unsupported._")
    out.append("")
    out.append("| Image | Expected | Exp.duration | Det.duration | Delta | Exp.paths | Det.paths | Delta paths | Note |")
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    out.append("")


def write_reports(
    summary: Dict[str, Any],
    md_path: str | Path,
    json_path: str | Path,
    debug: bool = False,
) -> None:
    md = Path(md_path)
    jf = Path(json_path)
    md.parent.mkdir(parents=True, exist_ok=True)
    jf.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(build_markdown(summary), encoding="utf-8")
    jf.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if debug:
        _write_debug_reports(summary, md, jf)


def _write_debug_reports(
    summary: Dict[str, Any],
    md_path: Path,
    json_path: Path,
) -> None:
    """Emit the optional per-image debug digest files (Batch E ``--debug``)."""
    images = summary.get("images") or []
    debug_entries = [
        entry for entry in images if (entry.get("extra") or {}).get("debug")
    ]
    if not debug_entries:
        return
    debug_md = md_path.with_suffix(".debug.md")
    debug_json = json_path.with_suffix(".debug.json")
    debug_md.write_text(_build_debug_markdown(debug_entries), encoding="utf-8")
    debug_json.write_text(
        json.dumps(
            {"schema_version": "debug-1", "images": debug_entries},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _build_debug_markdown(entries: List[Dict[str, Any]]) -> str:
    out: List[str] = []
    out.append("# Benchmark debug digest (--debug)")
    out.append("")
    out.append("_Read-only diagnostics: promotion-gate pass/fail aggregation for")
    out.append("dependency review routing, plus Batch A duration sub-crop outreach")
    out.append("counters. Nothing here changes pipeline behavior._")
    out.append("")
    for entry in entries:
        out.append(f"## `{entry.get('relative_path') or '?'}`")
        out.append("")
        debug = (entry.get("extra") or {}).get("debug") or {}
        digest = debug.get("gate_digest") or {}
        if digest:
            out.append("**Promotion-gate digest** (per-gate pass counts and")
            out.append("first-failing-gate histogram across every evaluated")
            out.append("dependency candidate):")
            out.append("")
            out.append("| Gate | Passed | Failed |")
            out.append("| --- | --- | --- |")
            for name, stats in (digest.get("per_gate") or {}).items():
                out.append(
                    f"| `{name}` | {stats.get('passed', 0)} "
                    f"| {stats.get('failed', 0)} |"
                )
            out.append("")
            out.append(
                f"- candidates evaluated: **{digest.get('candidates_evaluated', 0)}**"
                f" · all gates passed: **{digest.get('all_gates_passed', 0)}**"
            )
            blockers = digest.get("first_failing_gate") or {}
            if blockers:
                summary_text = ", ".join(
                    f"`{name}`={count}" for name, count in blockers.items()
                )
                out.append(f"- first-failing-gate histogram: {summary_text}")
            out.append("")
        routing = debug.get("review_routing") or {}
        if routing:
            out.append("**Review routing counts:**")
            out.append("")
            out.append(
                f"raw_arrows={routing.get('raw_arrows')} → "
                f"deduplicated={routing.get('deduplicated_arrows')} → "
                f"accepted={routing.get('accepted')} · review="
                f"{routing.get('review')} · rejected={routing.get('rejected')}"
                f" (self_loops={routing.get('self_loops')}, "
                f"duplicate_edges={routing.get('duplicate_edges')})"
            )
            out.append("")
        sub_crop = debug.get("duration_sub_crop") or {}
        if sub_crop:
            out.append("**Duration sub-crop outreach (Batch A pass):**")
            out.append("")
            out.append(
                f"nodes with sub-crop regions: "
                f"**{sub_crop.get('nodes_with_regions', 0)}** · "
                f"attached numeric parses: "
                f"**{sub_crop.get('attached_parses', 0)}**"
            )
            out.append("")
    return "\n".join(out)