<div align="center">

<img src="docs/pert-cover.svg" alt="PERT Analyzer blueprint cover" width="100%">

# PERT & Critical Path Analyzer

### Turn a network-diagram image into a reviewable project plan

<p><strong>Computer vision · OCR · graph reconstruction · human review · CPM/PERT · audit-ready reports</strong></p>

<a href="https://github.com/Dev-KAIDO/PERT-Critical-Path-Analyzer/actions"><img src="https://img.shields.io/github/actions/workflow/status/Dev-KAIDO/PERT-Critical-Path-Analyzer/ci.yml?branch=main&style=for-the-badge&logo=github-actions&logoColor=white&label=CI" alt="CI status"></a>
<a href="https://github.com/Dev-KAIDO/PERT-Critical-Path-Analyzer"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+"></a>
<a href="https://github.com/Dev-KAIDO/PERT-Critical-Path-Analyzer/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-16a34a?style=for-the-badge" alt="MIT license"></a>
<a href="#review-gate"><img src="https://img.shields.io/badge/Uncertain%20graphs-review--gated-d97706?style=for-the-badge" alt="Review gated"></a>

[Pipeline](#pipeline) · [Quick start](#quick-start) · [Automation](#automation) · [Review gate](#review-gate) · [Reports](#reports) · [Architecture](#architecture)

</div>

---

## What this project does

**PERT & Critical Path Analyzer** reads a project network diagram from a PNG or JPEG image and turns it into a structured, reviewable analysis. It combines computer vision, OCR, graph reconstruction, project-network mathematics, a Qt desktop interface, and a CLI.

The important design decision is that automation is **evidence-aware**. The application does not silently calculate CPM from a graph that is incomplete, contradictory, or uncertain. It pauses for review, exposes the evidence, and lets the user correct activities, durations, or dependencies before authoritative analysis continues.

> **بالعربي:** المشروع يحوّل صورة مخطط PERT/CPM إلى شبكة قابلة للتحليل، لكن الأتمتة لا تتجاوز الشكوك بصمت؛ إذا كانت القراءة أو العلاقات غير مؤكدة، تظهر بوابة مراجعة بشرية قبل حساب المسار الحرج.

## The product promise

| From | To |
| --- | --- |
| AON or AOA network diagram image | Structured activity/dependency graph |
| Text inside nodes and labels | OCR results with confidence and bounding boxes |
| Visual arrows and geometry | Reconciled dependencies with validation issues |
| Validated project graph | CPM/PERT duration, float, critical activities, and paths |
| Analysis session | JSON, CSV, Excel, and PDF exports |

## Pipeline at a glance

<img src="docs/pert-pipeline-light.svg" alt="PERT Analyzer evidence-first workflow" width="100%">

```text
Image
  → preprocessing and diagram classification
  → shape and arrow detection
  → region OCR with confidence
  → activity and dependency reconstruction
  → graph validation and reconciliation
  → review gate when evidence is uncertain
  → CPM / PERT analysis
  → visualization and JSON / CSV / Excel / PDF reports
```

## Why the review gate matters

OCR can confuse `C` with `Z`. Dense arrows can hide a dependency. A duplicate label can represent two different activities or a false positive. A plausible-looking graph can still produce mathematically wrong project dates.

This project treats uncertainty as a first-class state:

- confirmed evidence continues automatically;
- ambiguous evidence is marked for review;
- invalid graphs are blocked from authoritative CPM output;
- corrections remain visible in the review workflow;
- the final report can be traced back to the analyzed session.

## Capabilities

### Image understanding

- Detect AON and AOA network diagrams from PNG/JPEG inputs.
- Preprocess, threshold, denoise, and deskew diagram images.
- Detect shapes, activity boxes, event nodes, arrows, and arrowheads.
- Associate text regions with nearby visual elements.
- Extract activity identifiers and durations through Tesseract OCR.
- Route English and Arabic OCR where the environment supports both languages.

### Graph reconstruction

- Reconstruct activities and dependencies from visual evidence.
- Build validated NetworkX-compatible graph models.
- Reconcile OCR, geometry, arrow direction, and topology evidence.
- Detect duplicates, self-arrows, missing relationships, and suspicious nodes.
- Keep inferred or low-confidence values distinct from confirmed values.

### Project analysis

- Calculate earliest and latest event/activity times.
- Calculate total float and critical activities.
- Identify critical paths and expected project duration.
- Support CPM and PERT-oriented analysis models.
- Refuse to publish misleading results when graph validation fails.

### Desktop and delivery

- PySide6 Qt desktop GUI with analysis, progress, review, validation, results, and visualization screens.
- CLI for headless and scripted analysis.
- Export JSON, CSV, Excel, and PDF reports.
- Debug exports and evidence artifacts for benchmark and acceptance workflows.

## Quick start

### 1. System prerequisites

- Python **3.11 or newer**.
- Tesseract OCR on `PATH` for OCR-enabled analysis.
- Qt runtime for the desktop GUI.

On Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y \
  tesseract-ocr \
  tesseract-ocr-eng \
  tesseract-ocr-ara \
  libegl1 \
  libgl1 \
  libxkbcommon-x11-0
```

On Windows or macOS, install Tesseract separately and confirm that the executable is discoverable from the terminal.

### 2. Install the complete feature set

```bash
python -m venv .venv

# Linux/macOS
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -e '.[all]'

# Windows PowerShell
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\pip install -e ".[all]"
```

Optional extras are available when you need a smaller installation:

| Extra | Adds |
| --- | --- |
| `gui` | PySide6 desktop interface |
| `ocr` | pytesseract and Pillow |
| `reports` | Excel and PDF exporters |
| `dev` | pytest, coverage, Ruff, and mypy |
| `all` | All of the above |

### 3. Launch the GUI

```bash
.venv/bin/pert-analyzer
```

### 4. Analyze an image from the CLI

```bash
.venv/bin/pert-analyzer \
  --cli \
  --image path/to/network-diagram.png
```

The CLI reports project duration, critical path, activity count, and critical activity count only when the graph passes the configured validation policy. Sessions requiring human review are intentionally not promoted to authoritative CPM output.

## Automation

The project is designed for a repeatable automation loop rather than a one-off desktop demo.

### Local automation loop

```bash
# Install all project and development dependencies
python -m pip install -e '.[all]'

# Run the complete headless test suite
QT_QPA_PLATFORM=offscreen PYTHONPATH=. python -m pytest

# Run static checks
ruff check pert_analyzer tests

# Run a focused analysis from the command line
.venv/bin/pert-analyzer --cli --image path/to/diagram.png
```

### CI automation

Every push and pull request runs GitHub Actions on Python 3.11 and 3.12. The workflow:

1. installs Tesseract and Qt runtime libraries;
2. installs the complete package with `.[all]`;
3. runs the Ruff baseline check;
4. runs the test suite with `QT_QPA_PLATFORM=offscreen`;
5. exercises GUI logic without requiring a visible desktop.

[View the CI workflow](.github/workflows/ci.yml) · [View recent CI runs](https://github.com/Dev-KAIDO/PERT-Critical-Path-Analyzer/actions)

### Benchmark and evidence automation

The `pert_analyzer.benchmark` package and the `docs/` reports support repeatable validation of:

- OCR diagnostics;
- AON/AOA classification;
- activity and dependency reconstruction;
- review outcomes;
- generalization across reference images;
- GUI acceptance and screenshot evidence.

Example diagnostic entry points:

```bash
PYTHONPATH=. python -m pert_analyzer.benchmark --help
PYTHONPATH=. python -m pert_analyzer.benchmark.review_outcomes \
  --input path/to/review-session.json
```

The benchmark corpus and historical acceptance reports are kept in `docs/` and `artifacts/reference_validation/` so that accuracy claims remain inspectable instead of being hidden behind a single success badge.

## Review gate

The review flow is a safety boundary, not an error screen.

<img src="docs/review-gate.svg" alt="PERT Analyzer confidence review gate" width="100%">

```text
AUTO-DETECT
    ↓
CONFIDENCE + TOPOLOGY CHECK
    ├── sufficient evidence → CPM / PERT
    └── uncertainty detected → HUMAN REVIEW
                                   ↓
                         correct / accept / reject
                                   ↓
                            validate again
                                   ↓
                              CPM / PERT
```

Typical review triggers include:

- missing or ambiguous activity duration;
- OCR label confidence below the configured threshold;
- duplicate or implausible activity IDs;
- missing, contradictory, or self-referential dependencies;
- disconnected or cyclic graph structure;
- disagreement between visual geometry and OCR evidence.

The application deliberately prefers **“needs review”** over a confident-looking but incorrect project schedule.

## Reports and exports

The reporting layer can produce:

| Format | Best for |
| --- | --- |
| JSON | Machine-readable session and graph data |
| CSV | Spreadsheet inspection and downstream analysis |
| Excel | Project review, filtering, and handoff |
| PDF | Formal delivery and archival |

A report should preserve enough context to answer:

- Which image was analyzed?
- Which activities and dependencies were detected?
- Which values were confirmed, inferred, or reviewed?
- What duration and critical path were calculated?
- What issues remain unresolved?

<details>
<summary><strong>Open screenshots and UI evidence</strong></summary>

## Screenshots and evidence

### Analysis workspace

![PERT Analyzer analysis workspace](docs/screenshots/analyze_1920x1080.png)

### Review workflow

![PERT Analyzer review workflow](docs/screenshots/phase73_review_1920x1080.png)

### Results dashboard

![PERT Analyzer results dashboard](docs/screenshots/results_1920x1080.png)

More UI evidence, phase reports, benchmark outputs, and acceptance artifacts are available in [`docs/`](docs/).

</details>

## Architecture

<img src="docs/pert-pipeline-light.svg" alt="PERT Analyzer layered workflow" width="100%">

The implementation is organized into layers:

| Layer | Modules | Responsibility |
| --- | --- | --- |
| **Presentation** | `pert_analyzer.gui` | Qt windows, progress, review, validation, results, and visualization |
| **Application** | `pert_analyzer.pipeline` | Orchestration, callbacks, state, review API, and end-to-end flow |
| **Domain** | `pert_analyzer.analysis`, `pert_analyzer.graph`, `pert_analyzer.core` | CPM/PERT math, graph models, contracts, and validation rules |
| **Computer vision** | `pert_analyzer.cv`, `pert_analyzer.ocr` | Preprocessing, shapes, arrows, OCR, association, reconstruction, and reconciliation |
| **Reporting** | `pert_analyzer.reporting`, `pert_analyzer.persistence` | Result models, serialization, CSV, Excel, PDF, and session data |
| **Benchmarking** | `pert_analyzer.benchmark` | Accuracy, diagnostics, ground truth, generalization, and review metrics |

### Project layout

```text
PERT-Critical-Path-Analyzer/
├── pert_analyzer/
│   ├── analysis/          # CPM and PERT engines
│   ├── benchmark/         # diagnostics and repeatable benchmarks
│   ├── core/              # typed models and interfaces
│   ├── cv/                # vision, arrows, OCR association, reconstruction
│   ├── graph/             # graph building and adaptation
│   ├── gui/               # PySide6 application and review screens
│   ├── pipeline/          # orchestration and review workflow
│   ├── reporting/         # JSON, CSV, Excel, and PDF outputs
│   └── validation/        # structural and confidence checks
├── tests/                 # unit, integration, GUI, reference, and benchmark tests
├── docs/                  # acceptance reports, benchmarks, screenshots, and evidence
├── artifacts/             # reference-validation artifacts
├── scripts/               # capture and verification helpers
├── pyproject.toml         # package metadata, extras, CLI, and tooling
└── .github/workflows/     # automated CI
```

## Testing

Run the complete headless suite:

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=. python -m pytest
```

Run focused areas:

```bash
# CPM and PERT mathematics
QT_QPA_PLATFORM=offscreen PYTHONPATH=. python -m pytest tests -k 'cpm or pert'

# GUI and review behavior
QT_QPA_PLATFORM=offscreen PYTHONPATH=. python -m pytest tests/gui -q

# Reference acceptance
QT_QPA_PLATFORM=offscreen PYTHONPATH=. python -m pytest tests -k 'reference or acceptance'
```

The repository contains unit, integration, GUI, OCR, reconstruction, review-confidence, reference-acceptance, and benchmark-oriented tests. Historical acceptance reports record runs such as **1,242 passed** for the full suite; run the commands above in the current environment to verify the exact result for your checkout.

## Current limitations

Automation quality depends on the input diagram and the evidence available in it. In particular:

- dense or overlapping arrows may require review;
- OCR can misread visually similar labels;
- diagram quality, scale, contrast, and font affect detection;
- AON and AOA conventions can vary between organizations;
- Arabic UI support does not imply that every mathematical algorithm supports Arabic activity labels;
- the benchmark corpus does not prove perfect accuracy for every real-world diagram;
- Tesseract and Qt are runtime dependencies for their respective features.

## Responsible use

This tool supports project planning and education. Review the detected graph before making scheduling, contractual, safety, or financial decisions. The safest workflow is to preserve the original diagram, inspect the review queue, validate the graph, and archive the exported report with the evidence used to create it.

## Contributing

A useful contribution should improve one of the following:

- detection or OCR evidence;
- graph reconstruction and validation;
- review ergonomics;
- CPM/PERT correctness;
- export fidelity;
- benchmark coverage;
- documentation and reproducibility.

Before opening a pull request:

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=. python -m pytest
ruff check pert_analyzer tests
```

Please include the input class, expected behavior, test coverage, and any benchmark or screenshot evidence relevant to the change.

## License

MIT

---

<div align="center">

**Automate the reading · review the uncertainty · trust the validated graph**

<a href="https://github.com/Dev-KAIDO"><img src="https://img.shields.io/badge/Maintained%20by-Eng_Abdulrhaman-0f766e?style=flat-square&logo=github&logoColor=white" alt="Maintained by Eng Abdulrhaman"></a>

</div>
