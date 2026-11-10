"""
Region-based OCR processor for AON activity nodes.

Crops each detected candidate rectangle, preprocesses for OCR,
runs multi-pass Tesseract with different PSM modes, and merges
results with full-image OCR.

Supports targeted ID-specific and numeric-specific sub-crop OCR
for improved accuracy on difficult nodes.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from pert_analyzer.core.models import BoundingBox, _generate_id
from pert_analyzer.cv.models import CandidateNode, ShapeType
from pert_analyzer.cv.numeric_extraction import NumericExtractor
from pert_analyzer.cv.ocr_engine import OCREngineBase, OCREngineError
from pert_analyzer.cv.ocr_models import OCRConfig, OCRTextRegion, OCRProcessingResult, TextType

logger = logging.getLogger(__name__)

# Common OCR character confusions for single-letter activity IDs
AMBIGUITY_MAP: Dict[str, List[str]] = {
    "O": ["0", "D", "Q"],
    "0": ["O", "D", "Q"],
    "S": ["5", "8"],
    "5": ["S"],
    "V": ["Y", "U"],
    "Y": ["V"],
    "I": ["1", "l"],
    "1": ["I", "l"],
    "Z": ["2"],
    "2": ["Z"],
    "B": ["8"],
    "8": ["B"],
    "G": ["6", "C"],
    "6": ["G"],
    "Q": ["O", "0"],
    "D": ["O", "0"],
}

# Characters that look like letters but aren't
NON_ALPHA_CONFUSIONS: Dict[str, str] = {
    "0": "O",
    "5": "S",
    "8": "B",
    "1": "I",
    "2": "Z",
    "6": "G",
    "3": "B",
}


@dataclass
class RegionOCRConfig:
    """Configuration for region-based OCR."""

    # Crop settings
    padding_px: int = 8
    min_crop_size: int = 20

    # Upscaling
    scale_factor: int = 3
    interpolation: int = cv2.INTER_CUBIC

    # PSM modes for full-node OCR
    psm_modes: List[int] = field(default_factory=lambda: [6, 11])
    # PSM modes for ID sub-region (single word/character)
    id_psm_modes: List[int] = field(default_factory=lambda: [8, 11])
    # PSM modes for duration sub-region
    duration_psm_modes: List[int] = field(default_factory=lambda: [8])
    oem: int = 3

    # Confidence thresholds
    min_confidence: float = 0.0
    min_text_length: int = 1

    # Activity ID character whitelist
    activity_id_chars: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Duration numeric whitelist
    duration_chars: str = "0123456789"

    # Sub-crop relative positions (fraction of node dimensions)
    # ID is typically in the top-left corner of AON nodes
    id_crop_top: float = 0.0
    id_crop_bottom: float = 0.45
    id_crop_left: float = 0.0
    id_crop_right: float = 1.0

    # Duration is typically in the bottom-center of AON nodes
    duration_crop_top: float = 0.55
    duration_crop_bottom: float = 1.0
    duration_crop_left: float = 0.0
    duration_crop_right: float = 1.0

    # Sub-crop upscaling
    sub_crop_scale: int = 3


@dataclass
class NodeOCRResult:
    """OCR result for a single node region."""

    node_id: str
    candidate_node: Optional[CandidateNode] = None
    crop_bbox: Optional[BoundingBox] = None
    regions: List[OCRTextRegion] = field(default_factory=list)
    best_activity_id: Optional[str] = None
    best_activity_id_confidence: float = 0.0
    activity_id_alternatives: List[Tuple[str, float]] = field(default_factory=list)
    numeric_candidates: List[Tuple[float, str, float]] = field(default_factory=list)
    best_numeric: Optional[Tuple[float, str, float]] = None
    processing_time: float = 0.0
    warnings: List[str] = field(default_factory=list)
    # Sub-crop OCR results
    id_sub_crop_regions: List[OCRTextRegion] = field(default_factory=list)
    duration_sub_crop_regions: List[OCRTextRegion] = field(default_factory=list)


class RegionOCRProcessor:
    """
    Region-based OCR processor for AON activity nodes.

    For each detected candidate rectangle:
    1. Crops the region from the original image
    2. Adds configurable padding
    3. Creates ID and duration sub-crops
    4. Runs multi-pass Tesseract with different PSM modes
    5. Returns per-node OCR results
    """

    def __init__(
        self,
        ocr_engine: OCREngineBase,
        config: Optional[RegionOCRConfig] = None,
    ):
        self.ocr_engine = ocr_engine
        self.config = config or RegionOCRConfig()

    def process_node_regions(
        self,
        image: np.ndarray,
        candidate_nodes: List[CandidateNode],
    ) -> List[NodeOCRResult]:
        """
        Process each candidate node region through OCR.

        Args:
            image: Original image (BGR or grayscale).
            candidate_nodes: Detected candidate nodes from shape detection.

        Returns:
            List of NodeOCRResult, one per candidate node.
        """
        results = []
        for node in candidate_nodes:
            result = self._process_single_node(image, node)
            results.append(result)
        return results

    def _process_single_node(
        self,
        image: np.ndarray,
        node: CandidateNode,
    ) -> NodeOCRResult:
        """Process a single node through region OCR."""
        start_time = time.time()
        result = NodeOCRResult(
            node_id=node.node_id,
            candidate_node=node,
        )

        try:
            # 1. Crop region with padding
            crop, crop_bbox = self._crop_region(image, node.bounding_box)
            if crop is None:
                result.warnings.append("Crop failed: region outside image bounds")
                return result
            result.crop_bbox = crop_bbox

            # 2. Create OCR-ready representations
            representations = self._prepare_representations(crop)

            # 3. Multi-pass OCR on each representation (full node)
            all_regions = []
            for repr_name, repr_image in representations.items():
                regions = self._run_ocr_pass(repr_image, crop_bbox, repr_name)
                all_regions.extend(regions)

            # 4. ID sub-crop OCR (top-left region where activity ID lives)
            id_crop = self._extract_sub_crop(crop, self.config, is_id=True)
            if id_crop is not None:
                id_regions = self._run_sub_crop_ocr(
                    id_crop, crop_bbox, "id_sub", self.config.id_psm_modes
                )
                result.id_sub_crop_regions = id_regions
                all_regions.extend(id_regions)

            # 5. Duration sub-crop OCR (bottom-center region). The regions
            #    stay node-local (numeric extraction only): the designed
            #    pass was configured but never invoked, so digits read from
            #    the ID area became phantom durations. Parsed values are
            #    attached so the semantic resolver's sub-crop boost fires.
            duration_crop = self._extract_sub_crop(crop, self.config, is_id=False)
            if duration_crop is not None:
                duration_regions = self._run_sub_crop_ocr(
                    duration_crop,
                    crop_bbox,
                    "duration_sub",
                    self.config.duration_psm_modes,
                )
                result.duration_sub_crop_regions = duration_regions
                numeric_extractor = NumericExtractor()
                for region in duration_regions:
                    parsed = numeric_extractor.extract_from_region(region)
                    if parsed:
                        region.is_numeric = True
                        region.parsed_value = parsed[0].value

            # 6. Deduplicate and rank results
            result.regions = self._deduplicate_regions(all_regions)

            # 7. Extract activity ID (with ambiguity handling)
            self._extract_activity_id_with_alternatives(result)

            # 8. Extract numeric candidates from duration sub-crop
            self._extract_numeric_candidates(result)

        except Exception as e:
            result.warnings.append(f"Node OCR error: {e}")
            logger.debug("Node %s OCR error: %s", node.node_id, e)

        result.processing_time = time.time() - start_time
        return result

    def _crop_region(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
    ) -> Tuple[Optional[np.ndarray], BoundingBox]:
        """Crop a region from the image with padding."""
        h, w = image.shape[:2]
        pad = self.config.padding_px

        x1 = max(0, int(bbox.x) - pad)
        y1 = max(0, int(bbox.y) - pad)
        x2 = min(w, int(bbox.x + bbox.width) + pad)
        y2 = min(h, int(bbox.y + bbox.height) + pad)

        # Ensure minimum size
        if (x2 - x1) < self.config.min_crop_size or (y2 - y1) < self.config.min_crop_size:
            return None, BoundingBox(0, 0, 0, 0)

        crop = image[y1:y2, x1:x2]
        crop_bbox = BoundingBox(x=float(x1), y=float(y1), width=float(x2 - x1), height=float(y2 - y1))
        return crop, crop_bbox

    def _extract_sub_crop(
        self,
        crop: np.ndarray,
        config: RegionOCRConfig,
        is_id: bool = True,
    ) -> Optional[np.ndarray]:
        """Extract a sub-region from the crop (ID or duration)."""
        h, w = crop.shape[:2]
        if h < 10 or w < 10:
            return None

        if is_id:
            top = int(h * config.id_crop_top)
            bottom = int(h * config.id_crop_bottom)
            left = int(w * config.id_crop_left)
            right = int(w * config.id_crop_right)
        else:
            top = int(h * config.duration_crop_top)
            bottom = int(h * config.duration_crop_bottom)
            left = int(w * config.duration_crop_left)
            right = int(w * config.duration_crop_right)

        # Ensure valid region
        if bottom <= top or right <= left:
            return None

        sub = crop[top:bottom, left:right]
        if sub.size == 0:
            return None

        # Upscale
        sh, sw = sub.shape[:2]
        upscaled = cv2.resize(
            sub,
            (sw * config.sub_crop_scale, sh * config.sub_crop_scale),
            interpolation=config.interpolation,
        )
        return upscaled

    def _run_sub_crop_ocr(
        self,
        sub_image: np.ndarray,
        crop_bbox: BoundingBox,
        repr_name: str,
        psm_modes: List[int],
    ) -> List[OCRTextRegion]:
        """Run OCR on a sub-crop image and map coordinates back."""
        regions = []

        # Convert to grayscale if needed
        if len(sub_image.shape) == 3:
            gray = cv2.cvtColor(sub_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = sub_image

        # Also try Otsu
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        representations = {
            f"{repr_name}_gray": gray,
        }

        for rep_name, rep_img in representations.items():
            for psm in psm_modes:
                try:
                    config = OCRConfig(
                        psm=psm,
                        oem=self.config.oem,
                        min_confidence=self.config.min_confidence,
                    )
                    raw_regions = self.ocr_engine.recognize(rep_img, config)

                    # Map coordinates: sub-crop -> crop -> original
                    sub_h, sub_w = rep_img.shape[:2]
                    crop_h = crop_bbox.height * self.config.sub_crop_scale
                    crop_w = crop_bbox.width * self.config.sub_crop_scale

                    for region in raw_regions:
                        # Scale from sub-crop to crop coordinates
                        scale_x = crop_bbox.width / max(sub_w, 1)
                        scale_y = crop_bbox.height / max(sub_h, 1)

                        new_x = crop_bbox.x + region.bounding_box.x * scale_x
                        new_y = crop_bbox.y + region.bounding_box.y * scale_y
                        new_w = region.bounding_box.width * scale_x
                        new_h = region.bounding_box.height * scale_y

                        region.bounding_box = BoundingBox(x=new_x, y=new_y, width=new_w, height=new_h)
                        region.center = region.bounding_box.center
                        region.source_engine = f"subcrop_{rep_name}_psm{psm}"
                        region.metadata["repr"] = rep_name
                        region.metadata["psm"] = psm
                        region.metadata["is_sub_crop"] = True
                        regions.append(region)

                except (OCREngineError, Exception):
                    continue

        return regions

    def _prepare_representations(
        self, crop: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Create multiple preprocessed representations for OCR."""
        representations = {}

        # 1. Grayscale (if color)
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        # 2. Upscaled grayscale
        h, w = gray.shape[:2]
        upscaled = cv2.resize(gray, (w * self.config.scale_factor, h * self.config.scale_factor), interpolation=self.config.interpolation)
        representations["upscaled"] = upscaled

        # 3. Upscaled Otsu
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_up = cv2.resize(otsu, (w * self.config.scale_factor, h * self.config.scale_factor), interpolation=self.config.interpolation)
        representations["upscaled_otsu"] = otsu_up

        return representations

    def _run_ocr_pass(
        self,
        image: np.ndarray,
        crop_bbox: BoundingBox,
        repr_name: str,
    ) -> List[OCRTextRegion]:
        """Run Tesseract OCR on an image representation."""
        regions = []

        for psm in self.config.psm_modes:
            try:
                config = OCRConfig(
                    psm=psm,
                    oem=self.config.oem,
                    min_confidence=self.config.min_confidence,
                )
                raw_regions = self.ocr_engine.recognize(image, config)

                # Map coordinates back to original image
                for region in raw_regions:
                    mapped = self._map_coordinates(region, crop_bbox, image.shape)
                    mapped.source_engine = f"region_{repr_name}_psm{psm}"
                    mapped.metadata["repr"] = repr_name
                    mapped.metadata["psm"] = psm
                    mapped.metadata["crop_bbox"] = {
                        "x": crop_bbox.x, "y": crop_bbox.y,
                        "width": crop_bbox.width, "height": crop_bbox.height,
                    }
                    regions.append(mapped)

            except OCREngineError:
                continue
            except Exception:
                continue

        return regions

    def _map_coordinates(
        self,
        region: OCRTextRegion,
        crop_bbox: BoundingBox,
        crop_shape: Tuple[int, ...],
    ) -> OCRTextRegion:
        """Map OCR coordinates from crop space to original image space."""
        scale_x = crop_bbox.width / max(crop_shape[1], 1)
        scale_y = crop_bbox.height / max(crop_shape[0], 1)

        new_x = crop_bbox.x + region.bounding_box.x * scale_x
        new_y = crop_bbox.y + region.bounding_box.y * scale_y
        new_w = region.bounding_box.width * scale_x
        new_h = region.bounding_box.height * scale_y

        region.bounding_box = BoundingBox(x=new_x, y=new_y, width=new_w, height=new_h)
        region.center = region.bounding_box.center
        return region

    def _deduplicate_regions(
        self, regions: List[OCRTextRegion]
    ) -> List[OCRTextRegion]:
        """Deduplicate OCR regions by text similarity and bounding box overlap."""
        if not regions:
            return regions

        # Sort by confidence descending
        sorted_regions = sorted(regions, key=lambda r: r.confidence, reverse=True)

        unique = []
        seen_texts: Set[str] = set()

        for region in sorted_regions:
            text = region.text.strip()
            if not text:
                continue
            if len(text) < self.config.min_text_length:
                continue

            # Check for exact duplicate text
            if text in seen_texts:
                continue

            # Check for near-duplicate bounding boxes
            is_dup = False
            for existing in unique:
                overlap = self._compute_overlap(region.bounding_box, existing.bounding_box)
                if overlap > 0.5 and text.lower() == existing.text.lower():
                    is_dup = True
                    break

            if not is_dup:
                unique.append(region)
                seen_texts.add(text)

        return unique

    def _compute_overlap(self, a: BoundingBox, b: BoundingBox) -> float:
        """Compute IoU overlap between two bounding boxes."""
        x1 = max(a.x, b.x)
        y1 = max(a.y, b.y)
        x2 = min(a.x + a.width, b.x + b.width)
        y2 = min(a.y + a.height, b.y + b.height)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area_a = a.width * a.height
        area_b = b.width * b.height
        union = area_a + area_b - intersection

        return intersection / max(union, 1.0)

    def _extract_activity_id_with_alternatives(self, result: NodeOCRResult) -> None:
        """
        Extract best activity ID candidate with ambiguity handling.

        When OCR returns confusable characters (e.g., 'V' for 'O'),
        generates alternatives based on character confusion patterns.
        """
        candidates: List[Tuple[str, float, str]] = []  # (id, confidence, source)

        for region in result.regions:
            text = region.text.strip()
            source = region.source_engine or "unknown"

            # Single character A-Z
            if len(text) == 1 and text.upper() in self.config.activity_id_chars:
                candidates.append((text.upper(), region.confidence, source))
                continue

            # Pattern: letter + digits (A1, B12, etc.)
            if len(text) >= 2 and len(text) <= 6:
                if text[0].upper() in self.config.activity_id_chars and text[1:].isdigit():
                    candidates.append((text.upper(), region.confidence, source))

        # Also check for digit-to-letter confusions
        for region in result.regions:
            text = region.text.strip()
            if len(text) == 1 and text in NON_ALPHA_CONFUSIONS:
                mapped_letter = NON_ALPHA_CONFUSIONS[text]
                candidates.append((mapped_letter, region.confidence * 0.7, f"{source}_ambiguous"))

        # Sort by confidence
        candidates.sort(key=lambda x: x[1], reverse=True)

        if candidates:
            best = candidates[0]
            result.best_activity_id = best[0]
            result.best_activity_id_confidence = best[1]

            # Store alternatives (excluding the best)
            seen = {best[0]}
            for cid, cconf, _ in candidates[1:]:
                if cid not in seen and len(cid) == 1:
                    result.activity_id_alternatives.append((cid, cconf))
                    seen.add(cid)

    def _extract_numeric_candidates(self, result: NodeOCRResult) -> None:
        """Extract numeric candidates from OCR regions using NumericExtractor."""
        extractor = NumericExtractor()

        # Extract from all regions (full-node + sub-crop)
        all_regions = result.regions + result.duration_sub_crop_regions
        seen_values: Set[Tuple[float, str]] = set()

        for region in all_regions:
            candidates = extractor.extract_from_region(region)
            for cand in candidates:
                key = (cand.value, cand.raw_text)
                if key not in seen_values:
                    seen_values.add(key)
                    entry = (cand.value, cand.raw_text, cand.confidence)
                    result.numeric_candidates.append(entry)
                    if result.best_numeric is None or cand.confidence > result.best_numeric[2]:
                        result.best_numeric = entry


def merge_ocr_results(
    full_image_result: OCRProcessingResult,
    region_results: List[NodeOCRResult],
) -> OCRProcessingResult:
    """
    Merge full-image OCR with region-based OCR results.

    Prefers region-based results for text inside activity nodes.
    Keeps full-image results for START, FINISH, titles, etc.
    """
    merged_regions = list(full_image_result.regions)
    region_region_ids = set()

    # Collect all region-based regions
    for node_result in region_results:
        for region in node_result.regions:
            region_region_ids.add(region.region_id)
            merged_regions.append(region)

    # Deduplicate: prefer region-based for overlaps
    deduped = _deduplicate_merged(merged_regions)

    # Build new result
    merged = OCRProcessingResult(
        regions=deduped,
        groups=full_image_result.groups,
        numeric_candidates=full_image_result.numeric_candidates,
        engine=full_image_result.engine,
        image_dimensions=full_image_result.image_dimensions,
        processing_time=full_image_result.processing_time,
    )

    # Add region numeric candidates
    for node_result in region_results:
        for value, raw_text, conf in node_result.numeric_candidates:
            from pert_analyzer.cv.ocr_models import NumericCandidate
            from pert_analyzer.core.models import BoundingBox as BB
            nc = NumericCandidate(
                value=value,
                raw_text=raw_text,
                confidence=conf,
                bounding_box=BB(0, 0, 0, 0),
            )
            merged.numeric_candidates.append(nc)

    return merged


def _deduplicate_merged(regions: List[OCRTextRegion]) -> List[OCRTextRegion]:
    """Deduplicate overlapping OCR readings while preserving alternatives.

    Full-image OCR and node-region OCR intentionally produce different text
    readings for the same pixels. Exact-text matching alone therefore leaves
    duplicates such as ``5.17``/``9.17`` in the merged result. This pass uses
    conservative geometry plus source priority, and stores discarded readings
    as provenance instead of treating them as independent labels.
    """
    if not regions:
        return regions

    def source_priority(region: OCRTextRegion) -> int:
        source = (region.source_engine or "").lower()
        return 2 if source.startswith(("region_", "subcrop_")) else 1

    def quality(region: OCRTextRegion) -> tuple[int, float, int]:
        return (source_priority(region), float(region.confidence), len(region.text.strip()))

    def near_duplicate(candidate: OCRTextRegion, existing: OCRTextRegion) -> bool:
        candidate_text = candidate.text.strip()
        existing_text = existing.text.strip()
        candidate_numeric = bool(re.fullmatch(r"[0-9OolI.,]+", candidate_text))
        existing_numeric = bool(re.fullmatch(r"[0-9OolI.,]+", existing_text))
        candidate_id = len(candidate_text) == 1 and candidate_text.isalpha()
        existing_id = len(existing_text) == 1 and existing_text.isalpha()
        # A node's ID and duration can have oversized OCR boxes that overlap;
        # they must remain separate semantic candidates.
        if (candidate_numeric and existing_id) or (existing_numeric and candidate_id):
            return False
        overlap = _compute_iou(candidate.bounding_box, existing.bounding_box)
        # Do not use center proximity alone: an activity ID and its duration
        # are intentionally close but occupy different text boxes.
        return overlap >= 0.30

    sorted_regions = sorted(regions, key=quality, reverse=True)
    unique = []

    for region in sorted_regions:
        text = region.text.strip()
        if not text:
            continue

        duplicate_of = None
        for existing in unique:
            if near_duplicate(region, existing):
                duplicate_of = existing
                break

        if duplicate_of is None:
            unique.append(region)
            region.metadata.setdefault("ocr_dedup", {
                "duplicate_count": 0,
                "alternative_readings": [],
            })
        else:
            details = duplicate_of.metadata.setdefault("ocr_dedup", {
                "duplicate_count": 0,
                "alternative_readings": [],
            })
            details["duplicate_count"] += 1
            alternatives = details.setdefault("alternative_readings", [])
            alternative = {
                "text": text,
                "confidence": round(float(region.confidence), 3),
                "source_engine": region.source_engine,
            }
            if alternative not in alternatives:
                alternatives.append(alternative)

    return unique


def _compute_iou(a: BoundingBox, b: BoundingBox) -> float:
    """Compute IoU between two bounding boxes."""
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.width, b.x + b.width)
    y2 = min(a.y + a.height, b.y + b.height)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area_a = a.width * a.height
    area_b = b.width * b.height
    union = area_a + area_b - intersection

    return intersection / max(union, 1.0)
