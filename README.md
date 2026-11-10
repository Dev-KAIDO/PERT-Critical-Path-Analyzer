<div align="center">

<img src="docs/pert-mark.svg" alt="PERT Analyzer mark" width="92">

# PERT & Critical Path Analyzer

### From network diagram to a validated critical path

<p><strong>Read the diagram · review the evidence · calculate with confidence</strong></p>

<a href="https://github.com/Dev-KAIDO/PERT-Critical-Path-Analyzer/actions"><img src="https://img.shields.io/github/actions/workflow/status/Dev-KAIDO/PERT-Critical-Path-Analyzer/ci.yml?branch=main&style=flat-square&logo=github-actions&logoColor=white&label=CI" alt="CI status"></a>
<a href="https://github.com/Dev-KAIDO/PERT-Critical-Path-Analyzer"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11+"></a>
<a href="https://github.com/Dev-KAIDO/PERT-Critical-Path-Analyzer/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-16a34a?style=flat-square" alt="MIT license"></a>

<img src="docs/pert-cover.svg" alt="PERT Analyzer blueprint cover" width="100%">

[Quick start](#quick-start) · [How it works](#how-it-works) · [Review gate](#review-gate) · [Architecture](#architecture) · [Testing](#testing)

</div>

---

## What it does

PERT Analyzer turns a PNG/JPEG project network diagram into a structured graph, checks the evidence, and calculates CPM/PERT only when the graph is trustworthy.

> **الفكرة الأساسية:** إذا كانت القراءة أو العلاقة غير مؤكدة، لا يخفيها المشروع داخل نتيجة جميلة؛ يرسلها إلى المراجعة أولًا.

<img src="docs/pert-feature-icons.svg" alt="PERT Analyzer feature icons" width="100%">

## How it works

<img src="docs/pert-pipeline-light.svg" alt="PERT Analyzer workflow" width="100%">

1. **Read** the diagram with computer vision and OCR.
2. **Rebuild** activities, nodes, arrows, and dependencies.
3. **Review** ambiguous evidence instead of guessing.
4. **Calculate** CPM/PERT, float, critical activities, and critical paths.
5. **Export** JSON, CSV, Excel, or PDF reports.

## Review gate

The review gate is the project’s signature feature: it keeps automation fast without allowing uncertain data to become an authoritative schedule.

<img src="docs/review-gate.svg" alt="PERT Analyzer review gate" width="100%">

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install -e '.[all]'
.venv/bin/pert-analyzer
.venv/bin/pert-analyzer --cli --image path/to/diagram.png
```

For OCR on Debian/Ubuntu:

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-ara
```

## Architecture

| Layer | Responsibility |
| --- | --- |
| `cv` + `ocr` | Image preprocessing, shapes, arrows, text, confidence |
| `pipeline` | End-to-end orchestration and review workflow |
| `graph` + `analysis` | Graph validation, CPM, PERT, float, and paths |
| `gui` | PySide6 analysis, review, results, and visualization |
| `reporting` | JSON, CSV, Excel, and PDF exports |

## Verification

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=. python -m pytest
ruff check pert_analyzer tests
```

<details>
<summary><strong>Open UI screenshots and benchmark evidence</strong></summary>

The full repository contains the acceptance reports, benchmark artifacts, and UI screenshots under [`docs/`](docs/).

</details>

---

<div align="center">

<img src="docs/pert-mark.svg" alt="PERT Analyzer mark" width="48">

**Read the evidence. Respect the uncertainty. Trust the validated graph.**

</div>
