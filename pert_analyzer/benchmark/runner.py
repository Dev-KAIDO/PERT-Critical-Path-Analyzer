"""
Benchmark runner: drives the real pipeline over a dataset corpus.

Each image is analyzed sequentially with the authoritative production
entry point (``ReviewWorkflow.analyze``), exactly as the GUI does; the
benchmark only *measures* what the pipeline produced.  Per-image results
are written incrementally to JSON so a slow or long-running corpus never
loses completed work, and one failing image never aborts the run.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pert_analyzer.benchmark import classification as _classification
from pert_analyzer.benchmark.accuracy import compute_accuracy
from pert_analyzer.benchmark.annotations import (
    AnnotationError,
    default_ground_truth_dir,
    load_annotation,
)
from pert_analyzer.benchmark.discovery import (
    discover_images,
    relative_image_path,
)
from pert_analyzer.benchmark.ground_truth import GroundTruthError, load_ground_truth
from pert_analyzer.benchmark.metrics import (
    classify_outcome,
    extract_stage_metrics,
    extract_stage_progression,
    extract_stage_timings,
)

logger = logging.getLogger("pert_analyzer.benchmark")

#: Canonical pipeline stage ids surfaced in progress logs (see progress.py).
PIPELINE_STAGE_IDS: Tuple[str, ...] = (
    "Loading image",
    "Preprocessing",
    "Detecting shapes",
    "Classifying diagram",
    "Detecting arrows",
    "Extracting text (OCR)",
    "Associating text",
    "Reconstructing diagram",
    "Building graph",
    "Validating and analyzing",
    "Complete",
    "Review preparation",
)


class BenchmarkRunError(Exception):
    """Fatal runner-level error (e.g. unreadable image file)."""


@dataclass
class ImageRunResult:
    """Measured outcome for a single benchmark image."""

    relative_path: str
    width: Optional[int] = None
    height: Optional[int] = None
    file_format: str = ""
    file_size_bytes: Optional[int] = None
    duration_seconds: Optional[float] = None
    diagram_type: Optional[str] = None
    diagram_confidence: Optional[float] = None
    final_status: str = "UNKNOWN"
    outcome: str = _classification.OUTCOME_REVIEW_REQUIRED
    stage_metrics: Dict[str, Any] = field(default_factory=dict)
    stage_progression: List[Dict[str, Any]] = field(default_factory=list)
    stage_timings: Dict[str, Any] = field(default_factory=dict)
    failure_classification: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    ground_truth: Optional[Dict[str, Any]] = None
    gt_compare: Optional[Dict[str, Any]] = None
    accuracy_metrics: Optional[Dict[str, Any]] = None
    activity_ids: Optional[List[str]] = field(default_factory=list)
    dependency_edges: Optional[List[List[str]]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _default_analyze(image_path: str, stage_callback: Any = None) -> Any:
    """Authoritative production analysis entry point (same as the GUI)."""
    from pert_analyzer.pipeline.review_api import ReviewWorkflow

    return ReviewWorkflow.analyze(
        image_path,
        source_image_id=os.path.basename(image_path),
        stage_callback=stage_callback,
    )


def _image_info(image_path: Path) -> Tuple[Optional[int], Optional[int], str, Optional[int]]:
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            return (
                int(img.width),
                int(img.height),
                str((img.format or "").upper()),
                image_path.stat().st_size,
            )
    except Exception:
        return None, None, "", None


def _gt_compare(
    gt: Any, result: Any, metrics: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Compare measured counts against ground truth, only for present fields."""
    compare: Dict[str, Any] = {"available": True}
    recon = getattr(result, "_reconstruction", None) if result is not None else None
    measured_activities = metrics.get("reconstructed_activities")
    measured_deps = metrics.get("reconstructed_dependencies")

    if gt.diagram_type:
        compare["diagram_type_expected"] = gt.diagram_type
        compare["diagram_type_observed"] = (
            getattr(result, "diagram_type", None) or None
        )
        compare["diagram_type_match"] = (
            compare["diagram_type_observed"] is not None
            and compare["diagram_type_observed"].upper()
            == gt.diagram_type.upper()
        )
    if gt.activity_count is not None:
        compare["activity_count_expected"] = gt.activity_count
        compare["activity_count_observed"] = (
            _as_int(measured_activities)
        )
        if compare["activity_count_observed"] is not None:
            compare["activity_ratio"] = round(
                compare["activity_count_observed"] / gt.activity_count, 3
            )
            compare["activity_count_match"] = (
                compare["activity_count_observed"] == gt.activity_count
            )
    if gt.dependency_count is not None:
        compare["dependency_count_expected"] = gt.dependency_count
        compare["dependency_count_observed"] = _as_int(measured_deps)
        if compare["dependency_count_observed"] is not None:
            compare["dependency_count_match"] = (
                compare["dependency_count_observed"] == gt.dependency_count
            )
    if gt.durations is not None and recon is not None:
        observed = {}
        for act in getattr(recon, "activities", []) or []:
            aid = act.activity_id or act.semantic_activity_id
            if aid and act.duration is not None:
                observed[aid] = float(act.duration)
        expected_ids = set(gt.durations)
        observed_ids = set(observed)
        matched = expected_ids & observed_ids
        compare["duration_expected_ids"] = len(expected_ids)
        compare["duration_observed_ids"] = len(observed_ids)
        compare["duration_id_matches"] = len(matched)
    return compare


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _make_stage_logger(relative: str) -> Callable[[Any], None]:
    """Return a stage_callback that logs real pipeline stage transitions."""

    def _cb(stage_progress: Any) -> None:
        try:
            stage = (
                getattr(stage_progress, "stage_name", None)
                or getattr(stage_progress, "stage", None)
            )
            status = getattr(stage_progress, "status", None)
            if status is None:
                return
            status = str(status)
            if status == "RUNNING" and stage in PIPELINE_STAGE_IDS:
                logger.info("Benchmark STAGE image=%s stage=%s", relative, stage)
            elif status in ("COMPLETED", "FAILED"):
                logger.debug(
                    "Benchmark STAGE-END image=%s stage=%s status=%s",
                    relative,
                    stage,
                    status,
                )
        except Exception:  # pragma: no cover - never let logging break analysis
            return

    return _cb


def _collect_debug_digest(
    workflow: Any,
    result: Any,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Collect the per-image debug digest (Batch E ``--debug``).

    Read-only aggregation of intermediates the pipeline already produced:
    the dependency-promotion gate digest, review routing counts and the
    OCR duration sub-crop attach counters from Batch A's wired pass.
    Never raises outwards; all accessors are defensive.
    """
    digest: Dict[str, Any] = {}
    recon = getattr(result, "_reconstruction", None)
    report = (getattr(recon, "metadata", None) or {}).get(
        "dependency_validation"
    )
    if report is not None:
        digest["gate_digest"] = getattr(report, "gate_digest", None) or {}
        digest["review_routing"] = {
            "raw_arrows": _as_int(getattr(report, "raw_arrow_count", None)),
            "deduplicated_arrows": _as_int(
                getattr(report, "deduplicated_arrow_count", None)
            ),
            "accepted": _as_int(getattr(report, "accepted_count", None)),
            "review": _as_int(getattr(report, "review_count", None)),
            "rejected": _as_int(getattr(report, "rejected_count", None)),
            "self_loops": _as_int(getattr(report, "self_loop_count", None)),
            "duplicate_edges": _as_int(
                getattr(report, "duplicate_edge_count", None)
            ),
        }
    # Duration sub-crop outreach (Batch A pass wiring health): how many
    # node-local numeric parses the new pass produced.
    region_results = getattr(result, "_region_ocr_results", None) or []
    nodes_with_duration_regions = 0
    attached_parses = 0
    for nr in region_results:
        regions = getattr(nr, "duration_sub_crop_regions", None) or []
        if regions:
            nodes_with_duration_regions += 1
        for region in regions:
            if getattr(region, "parsed_value", None) is not None:
                attached_parses += 1
    digest["duration_sub_crop"] = {
        "nodes_with_regions": nodes_with_duration_regions,
        "attached_parses": attached_parses,
    }
    return digest


def analyze_single(
    image_path: str | Path,
    dataset_root: str | Path | None = None,
    analyze_fn: Optional[Callable[[str], Any]] = None,
    stage_callback: Any = None,
    load_gt: bool = True,
    ground_truth_dir: str | Path | None = None,
    debug: bool = False,
) -> ImageRunResult:
    """Analyze one image with the real pipeline and measure everything.

    ``analyze_fn`` defaults to ``ReviewWorkflow.analyze``, the same
    entry point the production GUI uses.  ``stage_callback`` is forwarded
    to the pipeline so progress events can be observed/logged.
    """
    path = Path(image_path)
    rel = (
        relative_image_path(dataset_root, path)
        if dataset_root is not None
        else path.name
    )
    run = ImageRunResult(relative_path=rel)
    width, height, fmt, size = _image_info(path)
    run.width, run.height, run.file_format, run.file_size_bytes = (
        width,
        height,
        fmt,
        size,
    )
    if width is None:
        run.outcome = _classification.OUTCOME_FAILURE
        run.final_status = "FAILED"
        run.error = f"Unreadable or unsupported image file: {path}"
        return run

    if analyze_fn is None:
        callback = stage_callback or _make_stage_logger(rel)
        analyze = lambda p: _default_analyze(str(p), callback)  # noqa: E731
    else:
        analyze = analyze_fn

    started = time.perf_counter()
    workflow = None
    error: Optional[str] = None
    try:
        workflow = analyze(str(path))
    except Exception as exc:  # noqa: BLE001 - capture any pipeline failure
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("Benchmark FAILURE: %s", rel)
    finally:
        run.duration_seconds = round(time.perf_counter() - started, 3)

    result = None
    if workflow is not None:
        result = getattr(workflow, "pipeline_result", None)
        if result is None:
            result = getattr(workflow, "result", None)

    if error is not None:
        run.outcome = _classification.OUTCOME_FAILURE
        run.final_status = "FAILED"
        run.error = error
        run.stage_metrics = extract_stage_metrics(workflow) if workflow else {}
        run.stage_progression = []
        run.stage_timings = extract_stage_timings(result)
        return run

    try:
        metrics = extract_stage_metrics(workflow)
    except Exception as exc:  # pragma: no cover - defensive
        metrics = {}
        run.warnings.append(f"Metric extraction failed: {exc}")

    run.stage_metrics = metrics
    run.stage_progression = extract_stage_progression(metrics)
    run.stage_timings = extract_stage_timings(result)

    if result is not None:
        status = getattr(result, "status", None)
        run.final_status = (
            status.value if hasattr(status, "value") else str(status or "UNKNOWN")
        )
        run.diagram_type = getattr(result, "diagram_type", None)
        if hasattr(run.diagram_type, "value"):
            run.diagram_type = run.diagram_type.value
        if run.diagram_type is not None:
            metrics["diagram_type"] = run.diagram_type
        run.diagram_confidence = _float_or_none(
            getattr(result, "diagram_confidence", None)
        )
        for w in getattr(result, "warnings", None) or []:
            text = w.get("description", str(w)) if isinstance(w, dict) else str(w)
            run.warnings.append(text)

    run.outcome = classify_outcome(workflow)

    # Reviewed candidate graph surface (activities + dependency edges).
    candidate = None
    apply = getattr(workflow, "apply", None)
    if callable(apply):
        try:
            candidate = apply()
        except Exception:
            candidate = None
    if candidate is not None:
        graph = getattr(candidate, "graph", None)
        if graph is not None:
            run.activity_ids = sorted(
                {str(a) for a in (getattr(graph, "activities", None) or [])}
            )
            edges = getattr(graph, "dependencies", None) or []
            selected: List[List[str]] = []
            for dep in edges:
                source = getattr(dep, "source", None) or getattr(
                    dep, "source_id", None
                )
                target = getattr(dep, "target", None) or getattr(
                    dep, "target_id", None
                )
                if source is not None and target is not None:
                    selected.append([str(source), str(target)])
            run.dependency_edges = selected

    # First abnormal stage + failure class (measured evidence only).
    stage, cls, evidence = _classification.classify_first_abnormal_stage(metrics)
    run.failure_classification = {
        "stage": stage,
        "failure_class": cls,
        "evidence": evidence,
    }

    # Optional ground truth (mode B evaluation).
    if load_gt:
        try:
            gt = load_ground_truth(path)
        except GroundTruthError as exc:
            run.ground_truth = {"error": str(exc)}
            run.warnings.append(f"Ground truth ignored: {exc}")
            gt = None
        if gt is not None:
            run.ground_truth = gt.to_dict()
            try:
                run.gt_compare = _gt_compare(gt, result, metrics)
            except Exception as exc:  # noqa: BLE001 - never abort on GT issues
                run.gt_compare = {"available": True, "compare_error": str(exc)}

        # v1.0 accuracy annotations (dedicated ground-truth folder).
        try:
            gt_dir = ground_truth_dir
            if gt_dir is None and dataset_root is not None:
                gt_dir = default_ground_truth_dir(dataset_root)
            annotation = load_annotation(path, ground_truth_dir=gt_dir)
        except AnnotationError as exc:
            run.warnings.append(f"Annotation ignored: {exc}")
            annotation = None
        if annotation is not None:
            try:
                run.accuracy_metrics = compute_accuracy(
                    annotation, result, run.dependency_edges
                )
            except Exception as exc:  # noqa: BLE001 - measurement must not abort
                run.accuracy_metrics = {"available": False, "error": str(exc)}
            # Derived CPM expectation (Batch E): computed live from the
            # annotation's own graph; never fabricates values and appears
            # on every report run (cheap, deterministic).
            try:
                from pert_analyzer.benchmark.cpm_compare import cpm_vs_gt_block

                expected_block = cpm_vs_gt_block(
                    annotation,
                    metrics.get("cpm_project_duration"),
                    metrics.get("critical_path_count"),
                )
                run.extra["cpm_vs_gt"] = expected_block
            except Exception as exc:  # noqa: BLE001 - never abort on GT issues
                run.extra["cpm_vs_gt"] = {"available": False, "error": str(exc)}
            if debug:
                try:
                    run.extra["debug"] = _collect_debug_digest(
                        workflow, result, metrics
                    )
                except Exception as exc:  # noqa: BLE001 - diagnostics only
                    run.extra["debug"] = {"error": str(exc)}
    return run


def _float_or_none(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _now_iso() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run_benchmark(
    dataset_root: str | Path,
    analyze_fn: Optional[Callable[[str], Any]] = None,
    output_path: Optional[str | Path] = None,
    load_gt: bool = True,
    ground_truth_dir: str | Path | None = None,
    limit: Optional[int] = None,
    progress_callback: Optional[Callable[[str, ImageRunResult], None]] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """Run the real pipeline over every image in ``dataset_root``.

    Returns a deterministic summary dict with one entry per image.
    When ``output_path`` is given, completed per-image results are
    appended to that JSON file after every image (incremental).
    ``limit`` caps the number of processed images (smoke tests).
    """
    import pert_analyzer

    root = Path(dataset_root).resolve()
    discovered = discover_images(root)
    images = discovered if limit is None else discovered[:limit]
    logger.info("Benchmark START dataset_root=%s images=%d", root, len(images))

    results: List[Dict[str, Any]] = []
    for path in images:
        rel = relative_image_path(root, path)
        logger.info("Benchmark START image=%s", rel)
        run = analyze_single(
            path,
            dataset_root=root,
            analyze_fn=analyze_fn,
            load_gt=load_gt,
            ground_truth_dir=ground_truth_dir,
            debug=debug,
        )
        results.append(run.to_dict())
        logger.info(
            "Benchmark COMPLETION image=%s outcome=%s final_status=%s "
            "activities=%s deps=%s duration=%ss",
            rel,
            run.outcome,
            run.final_status,
            run.stage_metrics.get("reconstructed_activities"),
            run.stage_metrics.get("reconstructed_dependencies"),
            run.duration_seconds,
        )
        if progress_callback is not None:
            try:
                progress_callback(rel, run)
            except Exception:
                pass
        if output_path is not None:
            _write_incremental(Path(output_path), results)

    summary = _build_summary(root, discovered, images, results)
    logger.info(
        "Benchmark COMPLETED images=%d success=%d review=%d failed=%d",
        summary["images_analyzed"],
        summary["summary"]["automatic_success"],
        summary["summary"]["automatic_review_required"],
        summary["summary"]["fatal_failures"],
    )
    return summary


def _write_incremental(output_path: Path, results: List[Dict[str, Any]]) -> None:
    payload = {
        "partial": True,
        "written_at": _now_iso(),
        "results_count": len(results),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _build_summary(
    root: Path,
    discovered: List[Path],
    processed: List[Path],
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    success = review = failed = 0
    with_gt = without_gt = 0
    with_annotation = 0
    annotation_statuses: Dict[str, int] = {}
    largest_inflation = 0
    largest_inflation_image = None
    for entry in results:
        outcome = entry.get("outcome")
        if outcome == _classification.OUTCOME_SUCCESS:
            success += 1
        elif outcome == _classification.OUTCOME_FAILURE:
            failed += 1
        else:
            review += 1
        if entry.get("ground_truth") is None:
            without_gt += 1
        else:
            with_gt += 1
        acc = entry.get("accuracy_metrics") or {}
        if acc.get("available"):
            with_annotation += 1
            status = acc.get("annotation_status") or "UNCERTAIN"
            annotation_statuses[status] = annotation_statuses.get(status, 0) + 1
        activities = _as_int(
            (entry.get("stage_metrics") or {}).get("reconstructed_activities")
        )
        if activities is not None and activities > largest_inflation:
            largest_inflation = activities
            largest_inflation_image = entry.get("relative_path")
    return {
        "schema_version": "1.0",
        "generated_at": _now_iso(),
        "pert_analyzer_version": getattr(
            __import__("pert_analyzer", fromlist=["__version__"]), "__version__", ""
        ),
        "dataset_root": str(root),
        "images_discovered": len(discovered),
        "images_analyzed": len(results),
        "bottleneck_summary": _classification.summarize_bottlenecks(results),
        "summary": {
            "images_discovered": len(discovered),
            "images_analyzed": len(results),
            "automatic_success": success,
            "automatic_review_required": review,
            "fatal_failures": failed,
            "with_ground_truth": with_gt,
            "without_ground_truth": without_gt,
            "with_accuracy_annotation": with_annotation,
            "annotation_statuses": annotation_statuses,
            "largest_activity_count_inflation": largest_inflation,
            "largest_activity_count_inflation_image": (
                largest_inflation_image
            ),
        },
        "images": results,
    }