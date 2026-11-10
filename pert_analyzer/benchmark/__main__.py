"""
CLI entry point for the generalization benchmark.

Run the real production pipeline over a dataset corpus and write
``docs/GENERALIZATION_BENCHMARK.md`` + ``.json``.

Example::

    python -m pert_analyzer.benchmark \\
        --dataset "tests/test_data/Imag PERT" \\
        --output docs/GENERALIZATION_BENCHMARK
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, List, Optional

from pert_analyzer.benchmark.reporting import write_reports
from pert_analyzer.benchmark.runner import (
    ImageRunResult,
    run_benchmark,
)


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m pert_analyzer.benchmark",
        description="Generalization benchmark for the PERT & Critical Path Analyzer.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset root directory scanned recursively for images.",
    )
    parser.add_argument(
        "--output",
        default="docs/GENERALIZATION_BENCHMARK",
        help="Output stem for the generated .md and .json reports.",
    )
    parser.add_argument(
        "--ground-truth",
        default=None,
        help=(
            "Directory holding v1.0 annotation JSONs (one <image stem>.json "
            "per diagram). Defaults to the sibling 'ground_truth' folder of "
            "the dataset root."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N discovered images (smoke test).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG level logging (very noisy OCR console output).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Collect per-image diagnostics into <output>.debug.md/.json "
            "(dependency promotion-gate digest, review routing counts, "
            "duration sub-crop outreach). Main report gains the derived "
            "CPM-vs-ground-truth comparison either way."
        ),
    )
    return parser.parse_args(argv)


def _progress_callback(relative: str, run: ImageRunResult) -> None:
    metrics = run.stage_metrics or {}
    vals = " ".join(
        f"{key}={metrics.get(key)}"
        for key in ("reconstructed_activities", "validated_dependencies", "ocr_labels")
    )
    print(
        f"[benchmark] {relative}: outcome={run.outcome} status={run.final_status} "
        f"({vals}) {run.duration_seconds}s",
        flush=True,
    )


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("pert_analyzer.benchmark")

    root = Path(args.dataset).resolve()
    out_stem = Path(args.output)
    print(
        f"[benchmark] dataset_root={root} limit={args.limit} output={out_stem}",
        flush=True,
    )

    incremental_path: Optional[Path] = out_stem.parent / (
        out_stem.name + ".incremental.json"
    )
    summary = run_benchmark(
        root,
        output_path=str(incremental_path),
        limit=args.limit,
        ground_truth_dir=args.ground_truth,
        progress_callback=_progress_callback,
        debug=args.debug,
    )
    if incremental_path.exists():
        incremental_path.unlink()

    md_path = out_stem.with_suffix(".md")
    json_path = out_stem.with_suffix(".json")
    write_reports(summary, md_path, json_path)
    print(f"[benchmark] wrote {md_path}", flush=True)
    print(f"[benchmark] wrote {json_path}", flush=True)

    s = summary["summary"]
    print(
        f"[benchmark] DONE images={s['images_analyzed']} "
        f"success={s['automatic_success']} "
        f"review={s['automatic_review_required']} "
        f"failed={s['fatal_failures']} "
        f"worst_inflation={s['largest_activity_count_inflation']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())